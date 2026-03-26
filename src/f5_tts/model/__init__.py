"""Public exports for ``f5_tts.model``.

Keep imports lazy so lightweight use-cases (for example unit tests importing a
single backbone) do not eagerly import heavy training/runtime dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["CFM", "UNetT", "DiT", "MMDiT", "Mamba3Backbone", "Trainer"]

_EXPORTS = {
    "CFM": ("f5_tts.model.cfm", "CFM"),
    "UNetT": ("f5_tts.model.backbones.unett", "UNetT"),
    "DiT": ("f5_tts.model.backbones.dit", "DiT"),
    "MMDiT": ("f5_tts.model.backbones.mmdit", "MMDiT"),
    "Mamba3Backbone": ("f5_tts.model.backbones.mamba3", "Mamba3Backbone"),
    "Trainer": ("f5_tts.model.trainer", "Trainer"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
