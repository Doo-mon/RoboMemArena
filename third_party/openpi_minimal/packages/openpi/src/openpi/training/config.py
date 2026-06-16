"""Minimal OpenPI config registry for RoboMemArena evaluation runtime.

This module intentionally exposes only the small subset needed by:
- scripts/serve_policy.py
- policy_config.create_trained_policy()

It avoids bundling large experiment registries or machine-specific training paths.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import difflib
import pathlib
from typing import Any

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.libero_policy as libero_policy
import openpi.transforms as _transforms


ModelType = _model.ModelType


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Minimal data config consumed by policy runtime."""

    # Directory inside checkpoint assets containing norm stats.
    asset_id: str | None = None
    # Runtime transforms used by policy inference.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Keep parity with OpenPI behavior: PI0 uses z-score, PI0.5 uses quantile norm by default.
    use_quantile_norm: bool = False


class DataConfigFactory:
    """Factory interface used by policy runtime."""

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        raise NotImplementedError


def _model_transforms(model_config: _model.BaseModelConfig) -> _transforms.Group:
    match model_config.model_type:
        case _model.ModelType.PI0:
            return _transforms.Group(
                inputs=[
                    _transforms.InjectDefaultPrompt(None),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len)),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ]
            )
        case _model.ModelType.PI05:
            assert isinstance(model_config, pi0_config.Pi0Config)
            return _transforms.Group(
                inputs=[
                    _transforms.InjectDefaultPrompt(None),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=model_config.discrete_state_input,
                    ),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ]
            )
        case _:
            raise ValueError(f"Unsupported model type for minimal runtime: {model_config.model_type}")


@dataclasses.dataclass(frozen=True)
class MinimalLiberoDataConfig(DataConfigFactory):
    """Minimal Libero data adapter used by RoboMemArena runtime."""

    asset_id: str = "robomemarena_assets"
    action_sequence_keys: Sequence[str] = ("actions",)

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        return DataConfig(
            asset_id=self.asset_id,
            data_transforms=data_transforms,
            model_transforms=_model_transforms(model_config),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Minimal train config shape required by policy runtime."""

    name: str
    model: _model.BaseModelConfig
    data: DataConfigFactory
    assets_dirs: pathlib.Path = pathlib.Path(".")
    policy_metadata: dict[str, Any] | None = None


_PI05_ROBOMEMARENA_TRAINING_DETAILS = {
    "model": "Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)",
    "data": {
        "base_config": "DataConfig(prompt_from_task=True)",
        "extra_delta_transform": False,
    },
    "initialization": "pi05_base/params",
    "batch_size": 128,
    "num_workers": 32,
    "optimizer": "AdamW",
    "clip_gradient_norm": 1.0,
    "lr_schedule": {
        "type": "CosineDecaySchedule",
        "warmup_steps": 10_000,
        "peak_lr": 5e-5,
        "decay_steps": 1_000_000,
        "decay_lr": 5e-5,
    },
    "ema_decay": 0.999,
    "num_train_steps": 40_000,
}


_DEFAULT_PI05_LIBERO = TrainConfig(
    name="pi05_robomemarena",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=MinimalLiberoDataConfig(asset_id="robomemarena_assets"),
    policy_metadata={"baseline_training_details": _PI05_ROBOMEMARENA_TRAINING_DETAILS},
)

_CONFIGS: dict[str, TrainConfig] = {
    "pi05_robomemarena": _DEFAULT_PI05_LIBERO,
    "pi05_libero": _DEFAULT_PI05_LIBERO,
}


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS:
        close = difflib.get_close_matches(config_name, _CONFIGS.keys(), n=1, cutoff=0.5)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(f"Config '{config_name}' not found.{hint}")
    return _CONFIGS[config_name]
