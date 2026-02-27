from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Dict, Any

import torch
from torch.utils.data import DataLoader


class ProxyProvider(ABC):
    @abstractmethod
    def sample(self, device: torch.device, k: int, seq_len: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError


class RandomInDistributionProxyProvider(ProxyProvider):
    """
    Stage-A proxy provider using an independent in-distribution stream.
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iter: Iterator = iter(loader)
        self._seen = 0

    def _next_batch(self) -> Dict[str, torch.Tensor]:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            batch = next(self._iter)
        return batch

    def sample(self, device: torch.device, k: int, seq_len: int) -> torch.Tensor:
        chunks = []
        while sum(x.size(0) for x in chunks) < k:
            batch = self._next_batch()
            x = batch["input_ids"]
            assert x.size(1) >= seq_len, (
                f"Proxy stream returned sequence length {x.size(1)} < expected {seq_len}. "
                f"All sequences must be exactly {seq_len} tokens — no padding."
            )
            x = x[:, :seq_len]
            chunks.append(x)
        out = torch.cat(chunks, dim=0)[:k]
        self._seen += int(k)
        return out.to(device, non_blocking=True)

    def state_dict(self) -> Dict[str, Any]:
        return {"seen": self._seen}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._seen = int(state.get("seen", 0))


class BenchProxyProvider(ProxyProvider):
    """
    Stage-B proxy provider backed by a benchmark-retrieved token shard.

    Expected format is a torch tensor file with shape [N, L] of tokenized sequences.
    """

    def __init__(self, token_tensor_path: str):
        self.token_tensor_path = token_tensor_path
        self.tokens = torch.load(token_tensor_path, map_location="cpu")
        if not torch.is_tensor(self.tokens) or self.tokens.dim() != 2:
            raise ValueError("BenchProxyProvider expects a [N, L] token tensor")
        self._seen = 0

    def sample(self, device: torch.device, k: int, seq_len: int) -> torch.Tensor:
        if self.tokens.size(0) == 0:
            raise RuntimeError("Bench proxy shard is empty")
        assert self.tokens.size(1) >= seq_len, (
            f"Bench proxy shard has sequence length {self.tokens.size(1)} < expected {seq_len}. "
            f"All sequences must be exactly {seq_len} tokens — no padding."
        )
        idxs = torch.randint(0, self.tokens.size(0), (k,))
        rows = self.tokens[idxs][:, :seq_len]
        self._seen += k
        return rows.to(device, non_blocking=True)

    def state_dict(self) -> Dict[str, Any]:
        return {"seen": self._seen, "path": self.token_tensor_path}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._seen = int(state.get("seen", 0))
