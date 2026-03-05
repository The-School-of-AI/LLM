from typing import Any


class Metrics:
    def __init__(self):
        self._values: dict[str, Any] = dict()
        self._pbar_values: set[str] = set()

    def add(self, name: str, value: Any, pbar=False):
        self._values[name] = value
        if pbar:
            self._pbar_values.add(name)

    def get_pbar_values(self) -> dict[str, Any]:
        return {k: self._values[k] for k in self._pbar_values}

    def get_values(self) -> dict[str, Any]:
        return self._values

    def __or__(self, other: "Metrics") -> "Metrics":
        merged = Metrics()
        merged._values = {**self._values, **other._values}
        merged._pbar_values = self._pbar_values | other._pbar_values
        return merged
