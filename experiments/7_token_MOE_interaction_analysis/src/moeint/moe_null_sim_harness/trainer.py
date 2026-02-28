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
import string

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
        self.num_null_experts = int(self.num_experts * (1 - self.data_sparsity) / self.data_sparsity)

        # Collapse all null experts into ONE bucket for stats
        self.num_total_experts = self.num_experts + 1

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
        self.group_map_cpu = self.router_health_analyzer.group_map.cpu()
        self.global_stats_ema = None
        self.ema_decay = 0.95  # higher = smoother
        self.global_expert_token_counts = {}
        self.print_every = 50

        vocab = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.junk_mask = (self.group_map_cpu == TokenGroups.junk)
        self._count_buffer = torch.zeros(
            self.num_total_experts * self.tokenizer.vocab_size,
            dtype=torch.int32
        )


    def _init_layer_counter(self, layer_idx: int):
        if layer_idx not in self.global_expert_token_counts:
            self.global_expert_token_counts[layer_idx] = torch.zeros(
                self.num_total_experts,
                self.tokenizer.vocab_size,
                dtype=torch.int32,
                device="cpu"  # keep on CPU to avoid GPU memory blowup
            )


    def print_expert_tokens(self, expert_token_counts, step, layer_idx):
        """
        Prints:
          - Top 20 tokens per real expert
          - Collapsed NULL expert
          - % junk vs content per expert
        """

        num_null = self.num_null_experts
        num_real = self.num_experts

        print("\n==============================")
        print(f"STEP {step} - Layer {layer_idx}: TOP 20 TOKENS PER EXPERT (REAL + NULL)")
        print("==============================")

        # ---- Helper to print one expert ----
        def print_single_expert(label, token_counts):
            total = token_counts.sum().item()

            if total == 0:
                print(f"\n--- {label} ---")
                print("No tokens routed.")
                return

            # Vectorized junk vs content % computation
            # junk_mask = (self.group_map_cpu == TokenGroups.junk)
            junk_count = token_counts[self.junk_mask].sum().item()
            content_count = total - junk_count
            junk_pct = 100 * junk_count / total if total > 0 else 0
            content_pct = 100 * content_count / total if total > 0 else 0

            print(f"\n--- {label} ---")
            print(f"Total tokens: {total}")
            print(f"Junk: {junk_pct:.2f}% | Content: {content_pct:.2f}%")
            print("-" * 50)

            # top_vals, top_ids = torch.topk(token_counts, 20)
            nonzero_ids = torch.nonzero(token_counts).squeeze(-1)
            nonzero_vals = token_counts[nonzero_ids]
            if len(nonzero_vals) > 20:
                top_vals, idx = torch.topk(nonzero_vals, 20)
                top_ids = nonzero_ids[idx]
            else:
                top_vals = nonzero_vals
                top_ids = nonzero_ids

            for token_id, count in zip(top_ids.tolist(), top_vals.tolist()):
                token_str = self.id_to_token.get(token_id, f"<UNK:{token_id}>")
                decoded = self.tokenizer.convert_tokens_to_string([token_str])
                pct = 100 * count / total
                print(f"{decoded!r:20s} | count={count:5d} | {pct:5.2f}%")

        # ---- Print Real Experts ----
        for expert_id in range(num_real):
            layer_counts = expert_token_counts  # tensor
            token_counts = layer_counts[expert_id]
            print_single_expert(f"Real Expert {expert_id}", token_counts)

        # ---- Collapse NULL Experts - Last row is collapsed NULL
        if num_null > 0:
            null_tensor = expert_token_counts[self.num_experts]
            print_single_expert("NULL (collapsed)", null_tensor)

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
            # update global counts
            for layer_idx, routing_logits in enumerate(moe_router_logits):
                self._init_layer_counter(layer_idx)

                with torch.no_grad():
                    # routing_probs = torch.softmax(routing_logits, dim=-1)
                    # top1_experts = torch.topk(routing_probs, 1, dim=-1).indices.squeeze(-1)
                    top1_experts = torch.topk(routing_logits, 1, dim=-1).indices.squeeze(-1)

                    flat_tokens = input_ids.view(-1)
                    flat_experts = top1_experts.view(-1)

                    # ---- COLLAPSE NULL EXPERTS ----
                    is_null = flat_experts >= self.num_experts
                    flat_experts[is_null] = self.num_experts

                    vocab_size = self.tokenizer.vocab_size
                    # move to CPU for counting
                    combined = (flat_experts * vocab_size + flat_tokens).to("cpu", non_blocking=True)
                    self._count_buffer.zero_()
                    bincount = torch.bincount(
                        combined,
                        minlength=self._count_buffer.numel()
                    )
                    self._count_buffer[:bincount.numel()] += bincount
                    layer_counts = self._count_buffer.view(self.num_total_experts, vocab_size)
                    self.global_expert_token_counts[layer_idx] += layer_counts

            stats = self.router_health_analyzer.analyze_logits(
                input_ids, moe_router_logits
            )

            # update EMA stats
            self._set_global_stats(stats)

            if step % self.print_every == 0:
                for layer_idx in self.global_expert_token_counts:
                    print(f"\n==== STEP {step}: Layer {layer_idx} ====")
                    self.print_expert_tokens(
                        self.global_expert_token_counts[layer_idx], step, layer_idx
                    )
                json_str = json.dumps(self.global_stats_ema, indent=2)
                print(f"step: {step}")
                print(json_str)
                print("-" * 100)

            if step > 0 and step % self.print_every == 0:
                self.global_expert_token_counts = {}

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step > max_steps:
                break

    def _set_global_stats(self, stats):
        stats_dicts = [stat._asdict() for stat in stats]

        if self.global_stats_ema is None:
            self.global_stats_ema = [dict(layer) for layer in stats_dicts]
        else:
            for layer_idx in range(len(stats_dicts)):
                for key in stats_dicts[layer_idx]:
                    current = stats_dicts[layer_idx][key]
                    prev = self.global_stats_ema[layer_idx][key]

                    # only smooth numeric values
                    if isinstance(current, (float, int)):
                        self.global_stats_ema[layer_idx][key] = (
                                self.ema_decay * prev
                                + (1 - self.ema_decay) * current
                        )
                    else:
                        # lists like entropy_per_group
                        self.global_stats_ema[layer_idx][key] = [
                            self.ema_decay * p + (1 - self.ema_decay) * c
                            for p, c in zip(prev, current)
                        ]

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
        punctuation_set = set(string.punctuation)

        for token_str, token_id in tokenizer.get_vocab().items():

            decoded = tokenizer.convert_tokens_to_string([token_str]).strip()

            # 1️⃣ Empty after stripping → junk
            if decoded == "":
                mapping[token_id] = TokenGroups.junk
                continue

            # 2️⃣ Pure whitespace
            if decoded.isspace():
                mapping[token_id] = TokenGroups.junk
                continue

            # 3️⃣ Pure punctuation (e.g. ".", ",", "()", "{}", "--")
            if all(ch in punctuation_set for ch in decoded):
                mapping[token_id] = TokenGroups.junk
                continue

            # 4️⃣ Repeated structural symbol (====, ----, ^^^^, etc.)
            if len(set(decoded)) == 1 and not decoded.isalnum():
                mapping[token_id] = TokenGroups.junk
                continue

            # 5️⃣ Pure number token (optional but recommended for code-heavy data)
            if decoded.isdigit():
                mapping[token_id] = TokenGroups.junk
                continue

            # Otherwise → content
            mapping[token_id] = TokenGroups.content

        return mapping