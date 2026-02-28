import json

import torch
from datasets import load_dataset
from moeint.moe_null_sim_harness.model import DeepSeekIsh, MoERouter
from moeint.routing_health_metrics import RouterHealthAnalyzer, TokenGroups
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_sentencepiece import SentencePieceBackend
from transformers.tokenization_utils_tokenizers import TokenizersBackend
from collections import defaultdict

class Trainer:
    def __init__(self, data_file: str, tokenizer_dir: str, batch_size: int = 8, seq_len: int = 512, lr: float = 1e-4):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        )
        self.topk = 2
        self.num_experts = 4
        self.data_sparsity = 0.5

        if tokenizer_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                use_fast=True
            )
        else:
            self.tokenizer: TokenizersBackend | SentencePieceBackend = (
                AutoTokenizer.from_pretrained("gpt2")
            )

        print("Vocab size:", self.tokenizer.vocab_size)
        print(f"Tokenizer pad token: {self.tokenizer.pad_token}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if data_file:
            dataset = load_dataset(
                "parquet",
                data_files=data_file,
                split="train",
            )
            dataset = dataset.shuffle(seed=42)
        else:
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                "sample-10BT",
                split="train",
                streaming=True,
            )

        dataset = dataset.map(
                lambda sample: self.tokenizer(
                    sample["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=seq_len,
                ),
                batched=True,
            ).with_format("torch")

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


    def print_expert_tokens(self, expert_token_counts):
        """
        Prints:
          - Top 20 tokens per real expert
          - Collapsed NULL expert
          - % junk vs content per expert
        """

        vocab = self.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}

        num_real = self.num_experts
        num_null = int(num_real * (1 - self.data_sparsity) / self.data_sparsity)

        print("\n==============================")
        print("TOP 20 TOKENS PER EXPERT (REAL + NULL)")
        print("==============================")

        # ---- Helper to print one expert ----
        def print_single_expert(label, token_counts):
            if not token_counts:
                print(f"\n--- {label} ---")
                print("No tokens routed.")
                return

            sorted_tokens = sorted(
                token_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]

            total = sum(token_counts.values())

            # Compute junk vs content %
            junk_count = 0
            content_count = 0
            for token_id, count in token_counts.items():
                group = self.router_health_analyzer.group_map[token_id].item()
                if group == TokenGroups.junk:
                    junk_count += count
                else:
                    content_count += count

            junk_pct = 100 * junk_count / total if total > 0 else 0
            content_pct = 100 * content_count / total if total > 0 else 0

            print(f"\n--- {label} ---")
            print(f"Total tokens: {total}")
            print(f"Junk: {junk_pct:.2f}% | Content: {content_pct:.2f}%")
            print("-" * 50)

            for token_id, count in sorted_tokens:
                token_str = id_to_token.get(token_id, f"<UNK:{token_id}>")
                try:
                    decoded = self.tokenizer.convert_tokens_to_string([token_str])
                except:
                    decoded = token_str

                pct = 100 * count / total
                print(f"{decoded!r:20s} | count={count:5d} | {pct:5.2f}%")

        # ---- Print Real Experts ----
        for expert_id in range(num_real):
            token_counts = expert_token_counts.get(expert_id, {})
            print_single_expert(f"Real Expert {expert_id}", token_counts)

        # ---- Collapse NULL Experts ----
        null_token_counts = {}

        for expert_id in range(num_real, num_real + num_null):
            token_counts = expert_token_counts.get(expert_id, {})
            for token_id, count in token_counts.items():
                null_token_counts[token_id] = null_token_counts.get(token_id, 0) + count

        print_single_expert("NULL (collapsed)", null_token_counts)

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

            # ===============================================
            # Aggregate routing across ALL MoE layers
            # ===============================================
            expert_token_counts = defaultdict(lambda: defaultdict(int))

            for routing_logits in moe_router_logits:
                routing_probs = torch.softmax(routing_logits, dim=-1)
                _, topk_indices = torch.topk(routing_probs, self.topk, dim=-1)

                flat_ids = input_ids.flatten()
                flat_experts = topk_indices[..., 0].flatten()  # top-1 per layer

                for token_id, expert_id in zip(flat_ids.tolist(), flat_experts.tolist()):
                    expert_token_counts[expert_id][token_id] += 1

            self.print_expert_tokens(expert_token_counts)

            stats = self.router_health_analyzer.analyze_logits(
                input_ids, moe_router_logits
            )

            json_str = json.dumps([stat._asdict() for stat in stats], indent=2)
            print(f"step: {step}")
            print(json_str)
            print("-" * 100)

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

        # dummy token grouping logic, just to test things out:
        for token_str, token_id in tokenizer.get_vocab().items():
            # 1. Convert the byte-level representation back to a normal string
            # GPT-2 uses a specific byte-map; decoding is the safest way to check
            decoded_token = tokenizer.convert_tokens_to_string([token_str])

            # 2. Check if it's "Pure Junk"
            # This catches standalone spaces, newlines, tabs, and strings of them
            if decoded_token.isspace():
                mapping[token_id] = TokenGroups.junk
            else:
                mapping[token_id] = TokenGroups.content

        return mapping
