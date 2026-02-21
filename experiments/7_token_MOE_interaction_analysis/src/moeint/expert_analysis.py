from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class ModalityDistribution:
    """
    Holds the token distribution across modalities for a dataset.

    Attributes:
        distribution:      Float tensor of shape [num_modalities, vocab_size].
                           Row i is the token count vector for modality i.
        index_to_modality: Maps row index i -> modality label string.
        source_files:      Parquet files that were processed to produce this.
    """

    distribution: Tensor  # [num_modalities, vocab_size]
    index_to_modality: dict[int, str]
    source_files: list[str]

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "distribution": self.distribution,
                "index_to_modality": self.index_to_modality,
                "source_files": self.source_files,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModalityDistribution":
        data = torch.load(path, weights_only=False)
        return cls(
            distribution=data["distribution"],
            index_to_modality=data["index_to_modality"],
            source_files=data["source_files"],
        )

    def __repr__(self) -> str:
        modalities = list(self.index_to_modality.values())
        return (
            f"ModalityDistribution("
            f"modalities={modalities}, "
            f"vocab_size={self.distribution.shape[1]}, "
            f"source_files={len(self.source_files)} files)"
        )
