from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np


class BasePolicyAdapter(ABC):
    """Model-agnostic adapter interface for benchmark evaluation."""

    def reset(self) -> None:
        """Reset any per-episode internal state if needed."""

    @abstractmethod
    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        """Return an action chunk with shape [horizon, action_dim]."""


class AdapterLoadError(RuntimeError):
    pass


def _load_module_from_path(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise AdapterLoadError(f"Cannot load adapter module from path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_factory_spec(factory_spec: str) -> tuple[str, str]:
    if ":" in factory_spec:
        module_spec, factory_name = factory_spec.split(":", 1)
        return module_spec, factory_name
    return factory_spec, "build_adapter"


def load_policy_adapter(factory_spec: str, **factory_kwargs: Any) -> BasePolicyAdapter:
    if not factory_spec:
        raise AdapterLoadError("Missing adapter spec. Expected 'module.path:build_adapter' or '/abs/path.py:build_adapter'.")

    module_spec, factory_name = _split_factory_spec(factory_spec)
    module_path = Path(module_spec)
    if module_path.suffix == ".py" and module_path.exists():
        module = _load_module_from_path(module_path)
    else:
        module = importlib.import_module(module_spec)

    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise AdapterLoadError(f"Factory '{factory_name}' not found in adapter module '{module_spec}'.")

    adapter = factory(**factory_kwargs)
    if not isinstance(adapter, BasePolicyAdapter):
        raise AdapterLoadError(
            f"Adapter factory '{factory_name}' returned {type(adapter)!r}, expected BasePolicyAdapter subclass."
        )
    return adapter


def ensure_action_chunk(actions: Any) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Policy adapter must return shape [horizon, action_dim], got {arr.shape}.")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"Policy adapter returned invalid empty action chunk: {arr.shape}.")
    return arr
