from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

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
            if x.size(1) > seq_len:
                x = x[:, :seq_len]
            elif x.size(1) < seq_len:
                pad = torch.zeros(x.size(0), seq_len - x.size(1), dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
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
        self.cursor = 0

    def sample(self, device: torch.device, k: int, seq_len: int) -> torch.Tensor:
        if self.tokens.size(0) == 0:
            raise RuntimeError("Bench proxy shard is empty")
        out = []
        for _ in range(k):
            row = self.tokens[self.cursor % self.tokens.size(0)]
            self.cursor += 1
            if row.size(0) > seq_len:
                row = row[:seq_len]
            elif row.size(0) < seq_len:
                pad = torch.zeros(seq_len - row.size(0), dtype=row.dtype)
                row = torch.cat([row, pad], dim=0)
            out.append(row)
        return torch.stack(out, dim=0).to(device, non_blocking=True)

    def state_dict(self) -> Dict[str, Any]:
        return {"cursor": self.cursor, "path": self.token_tensor_path}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.cursor = int(state.get("cursor", 0))
