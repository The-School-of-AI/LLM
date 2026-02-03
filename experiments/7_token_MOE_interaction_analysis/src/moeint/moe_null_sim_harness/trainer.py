import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_sentencepiece import SentencePieceBackend
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from moeint.moe_null_sim_harness.model import DeepSeekIsh, DeepSeekMoE
from moeint.routing_health_metrics import RoutingAnalyzer, TokenGroups


class Trainer:
    def __init__(self, batch_size: int = 8, seq_len: int = 512, lr: float = 1e-4):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        self.topk = 2
        self.num_real_experts = 4

        self.tokenizer: TokenizersBackend | SentencePieceBackend = (
            AutoTokenizer.from_pretrained("gpt2")
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = (
            load_dataset(
                "HuggingFaceFW/fineweb-edu",
                "sample-10BT",
                split="train",
                streaming=True,
            )
            .map(
                lambda sample: self.tokenizer(
                    sample["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=seq_len,
                ),
                batched=True,
            )
            .with_format("torch")
        )

        self.dataloader = DataLoader(dataset, batch_size=batch_size)  # type: ignore
        self.model = DeepSeekIsh(
            vocab_size=self.tokenizer.vocab_size,
            num_hidden_layers=4,
            hidden_size=288,
            num_attention_heads=9,
            num_routed_experts=self.num_real_experts,
            num_null_experts=2,
            num_experts_per_tok=self.topk,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.routing_analyzer = RoutingAnalyzer(
            vocab_size=self.tokenizer.vocab_size,
            token_id_group_mapping=self._get_token_id_group_mapping(self.tokenizer),
            device=self.device,
        )

    def train(self, max_steps: int = 100):
        self.model.train()

        for step, batch in enumerate(self.dataloader):
            input_ids = batch["input_ids"].to(device=self.device)
            logits = self.model(input_ids)

            # loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            moe_router_logits = self._collect_moe_router_logits()
            stats = self.routing_analyzer.analyze(
                input_ids,
                moe_router_logits,
                num_real_experts=self.num_real_experts,
                top_k=self.topk,
            )
            print(
                f"Step {step} | Loss: {loss.item():.4f} | L0 junk to null rate: {stats.null_expert_stats[0].junk_to_null_rate:.2f} | L0 null got junk rate: {stats.null_expert_stats[0].null_junk_rate:.2f}"
            )

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step > max_steps:
                break

    def _update_moe_router_bias(self):
        with torch.no_grad():
            for layer in self.model.layers:
                moe: DeepSeekMoE | None = getattr(layer, "moe", None)
                if moe is None:
                    continue

                expert_load = moe.last_expert_load
                if expert_load is not None:
                    moe.update_bias_terms(expert_load)

    def _collect_moe_router_logits(self) -> list[Tensor]:
        logits = []
        for layer in self.model.layers:
            moe: DeepSeekMoE | None = getattr(layer, "moe", None)
            if moe is None:
                continue
            logits.append(moe.last_routing_logits)

        return logits

    def _get_token_id_group_mapping(
        self, tokenizer: TokenizersBackend | SentencePieceBackend
    ) -> dict[int, int]:
        mapping = {}
        for token_str, token_id in tokenizer.get_vocab().items():
            # 1. Convert the byte-level representation back to a normal string
            # GPT-2 uses a specific byte-map; decoding is the safest way to check
            decoded_token = tokenizer.convert_tokens_to_string([token_str])

            # 2. Check if it's "Pure Junk"
            # This catches standalone spaces, newlines, tabs, and strings of them
            if decoded_token.isspace():
                mapping[token_id] = TokenGroups.whitespace
            else:
                mapping[token_id] = TokenGroups.content

        return mapping
