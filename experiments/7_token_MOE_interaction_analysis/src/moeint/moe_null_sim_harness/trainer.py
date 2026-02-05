import torch
from datasets import load_dataset
from moeint.moe_null_sim_harness.model import DeepSeekIsh, MoERouter
from moeint.routing_health_metrics import RouterHealthAnalyzer, TokenGroups
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_sentencepiece import SentencePieceBackend
from transformers.tokenization_utils_tokenizers import TokenizersBackend


class Trainer:
    def __init__(self, batch_size: int = 8, seq_len: int = 512, lr: float = 1e-4):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        )
        self.topk = 2
        self.num_experts = 4
        self.data_sparsity = 0.5

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
            num_experts=self.num_experts,
            data_sparsity=self.data_sparsity,
            topk=self.topk,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.router_health_analyzer = RouterHealthAnalyzer(
            vocab_size=self.tokenizer.vocab_size,
            token_id_group_mapping=self._get_token_id_group_mapping(self.tokenizer),
            num_experts=self.num_experts,
            data_sparsity=self.data_sparsity,
            topk=self.topk,
            device=self.device,
        )

    def train(self, max_steps: int = 100):
        self.model.train()

        for step, batch in enumerate(self.dataloader):
            input_ids = batch["input_ids"].to(device=self.device)
            logits, aux_loss = self.model(input_ids)

            # loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = loss + aux_loss

            moe_router_logits = self._collect_moe_router_logits()
            stats = self.router_health_analyzer.analyze(input_ids, moe_router_logits)
            print(
                f"Step {step} | Loss: {loss.item():.4f} | L0 junk to null rate: {stats.null_experts_stats[0].junk_to_null_rate:.2f} | L0 null got junk rate: {stats.null_experts_stats[0].null_junk_rate:.2f}"
            )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step > max_steps:
                break

    def _collect_moe_router_logits(self) -> list[Tensor]:
        logits = []
        for _, module in self.model.named_modules():
            if isinstance(module, MoERouter):
                logits.append(module.last_router_logits)

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
