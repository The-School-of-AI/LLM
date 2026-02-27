from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class LayerCapture:
    name: str
    module: nn.Module
    weight: torch.nn.Parameter
    activations: torch.Tensor
    grad_outputs: torch.Tensor


@dataclass
class MoERoutedCapture:
    name: str
    module: nn.Module
    w_gate: torch.nn.Parameter
    w_up: torch.nn.Parameter
    w_down: torch.nn.Parameter
    activations: torch.Tensor
    grad_outputs: torch.Tensor
    topk_idx: torch.Tensor
    topk_weight: torch.Tensor
    is_null: torch.Tensor


class GhostCollector:
    """
    Collects per-sample activations and output gradients for linear layers.

    Intended for OPUS scoring pass where a single backward captures candidate and
    proxy ghost factors without per-sample backward loops.
    """

    def __init__(
        self,
        model: nn.Module,
        include_embeddings: bool = False,
        include_lm_head: bool = False,
    ):
        self.model = model
        self.include_embeddings = include_embeddings
        self.include_lm_head = include_lm_head
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._selected: Dict[str, nn.Module] = {}
        self._selected_moe: Dict[str, nn.Module] = {}
        self._activations: Dict[str, torch.Tensor] = {}
        self._grad_outputs: Dict[str, torch.Tensor] = {}
        self._moe_activations: Dict[str, torch.Tensor] = {}
        self._moe_grad_outputs: Dict[str, torch.Tensor] = {}
        self._moe_routing: Dict[str, Dict[str, torch.Tensor]] = {}

    def _should_track(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, nn.Linear):
            # Default: all linears inside transformer blocks.
            in_blocks = ("layers." in name) or ("blocks." in name)
            if in_blocks:
                return True
            if self.include_lm_head and name.endswith("lm_head"):
                return True
            return False
        if self.include_embeddings and isinstance(module, nn.Embedding):
            return True
        return False

    @staticmethod
    def _is_moe_module(module: nn.Module) -> bool:
        w_gate = getattr(module, "W_gate", None)
        w_up = getattr(module, "W_up", None)
        w_down = getattr(module, "W_down", None)
        if not isinstance(w_gate, torch.nn.Parameter):
            return False
        if not isinstance(w_up, torch.nn.Parameter):
            return False
        if not isinstance(w_down, torch.nn.Parameter):
            return False
        return (w_gate.dim() == 3) and (w_up.dim() == 3) and (w_down.dim() == 3)

    def _should_track_moe(self, name: str, module: nn.Module) -> bool:
        in_blocks = ("layers." in name) or ("blocks." in name)
        return in_blocks and self._is_moe_module(module)

    def discover_layers(self) -> Dict[str, nn.Module]:
        layers: Dict[str, nn.Module] = {}
        for name, module in self.model.named_modules():
            if self._should_track(name, module):
                if getattr(module, "weight", None) is not None and module.weight.dim() == 2:
                    layers[name] = module
        if not layers:
            raise RuntimeError("GhostCollector found zero scoreable layers")
        self._selected = layers
        return layers

    def discover_moe_layers(self) -> Dict[str, nn.Module]:
        layers: Dict[str, nn.Module] = {}
        for name, module in self.model.named_modules():
            if self._should_track_moe(name, module):
                layers[name] = module
        self._selected_moe = layers
        return layers

    def clear(self) -> None:
        self._activations.clear()
        self._grad_outputs.clear()
        self._moe_activations.clear()
        self._moe_grad_outputs.clear()
        self._moe_routing.clear()

    def register(self) -> None:
        if not self._selected:
            self.discover_layers()
        self.discover_moe_layers()
        self.clear()

        def _make_fwd(name: str):
            def _fwd(module: nn.Module, args: Tuple[torch.Tensor, ...], output: torch.Tensor):
                if not args:
                    return
                x = args[0]
                if not torch.is_tensor(x):
                    return
                self._activations[name] = x.detach()

            return _fwd

        def _make_bwd(name: str):
            def _bwd(module: nn.Module, grad_input, grad_output):
                if not grad_output:
                    return
                gout = grad_output[0]
                if not torch.is_tensor(gout):
                    return
                self._grad_outputs[name] = gout.detach()

            return _bwd

        def _make_moe_fwd(name: str):
            def _fwd(module: nn.Module, args: Tuple[torch.Tensor, ...], output):
                if not args:
                    return
                x = args[0]
                if not torch.is_tensor(x):
                    return
                routing = getattr(module, "last_routing", None)
                if routing is None:
                    return
                topk_idx = routing.get("topk_idx")
                topk_weight = routing.get("topk_weight")
                is_null = routing.get("is_null")
                if not (torch.is_tensor(topk_idx) and torch.is_tensor(topk_weight) and torch.is_tensor(is_null)):
                    return
                self._moe_activations[name] = x.detach()
                self._moe_routing[name] = {
                    "topk_idx": topk_idx.detach(),
                    "topk_weight": topk_weight.detach(),
                    "is_null": is_null.detach(),
                }

            return _fwd

        def _make_moe_bwd(name: str):
            def _bwd(module: nn.Module, grad_input, grad_output):
                if not grad_output:
                    return
                gout = grad_output[0]
                if not torch.is_tensor(gout):
                    return
                self._moe_grad_outputs[name] = gout.detach()

            return _bwd

        for name, module in self._selected.items():
            self._handles.append(module.register_forward_hook(_make_fwd(name)))
            self._handles.append(module.register_full_backward_hook(_make_bwd(name)))
        for name, module in self._selected_moe.items():
            self._handles.append(module.register_forward_hook(_make_moe_fwd(name)))
            self._handles.append(module.register_full_backward_hook(_make_moe_bwd(name)))

    def unregister(self) -> None:
        while self._handles:
            h = self._handles.pop()
            h.remove()

    def captures(self) -> Dict[str, LayerCapture]:
        out: Dict[str, LayerCapture] = {}
        for name, module in self._selected.items():
            if name not in self._activations or name not in self._grad_outputs:
                continue
            act = self._activations.pop(name)
            gout = self._grad_outputs.pop(name)
            out[name] = LayerCapture(
                name=name,
                module=module,
                weight=module.weight,
                activations=act,
                grad_outputs=gout,
            )
        if not out:
            raise RuntimeError("GhostCollector has no captured activations/gradients")
        return out

    def moe_captures(self) -> Dict[str, MoERoutedCapture]:
        out: Dict[str, MoERoutedCapture] = {}
        for name, module in self._selected_moe.items():
            if name not in self._moe_activations or name not in self._moe_grad_outputs or name not in self._moe_routing:
                continue
            act = self._moe_activations.pop(name)
            gout = self._moe_grad_outputs.pop(name)
            routing = self._moe_routing.pop(name)
            out[name] = MoERoutedCapture(
                name=name,
                module=module,
                w_gate=module.W_gate,
                w_up=module.W_up,
                w_down=module.W_down,
                activations=act,
                grad_outputs=gout,
                topk_idx=routing["topk_idx"],
                topk_weight=routing["topk_weight"],
                is_null=routing["is_null"],
            )
        return out

    def __enter__(self) -> "GhostCollector":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unregister()
        self.clear()
