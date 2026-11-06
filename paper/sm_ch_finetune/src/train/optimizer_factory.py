"""Optax optimizers for heads-only, LoRA, full-model, and encoder-only fine-tuning."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import optax
from jaxtyping import PyTree

from alphagenome_ft.lora import get_lora_parameter_paths
from alphagenome_ft.optimizer_utils import (
    _build_adam_or_adamw,
    is_trainable_head_path,
    parameter_path_to_str,
)


def _label_params(params: PyTree, strategy: str, trainable_head_names: Sequence[str]) -> PyTree:
    head_set = {str(n) for n in trainable_head_names}
    lora_paths = set(get_lora_parameter_paths(params))

    def label_fn(path, _value):
        ps = parameter_path_to_str(path)
        leaf = ps.rsplit("/", 1)[-1]
        in_head = is_trainable_head_path(ps, head_set)
        if strategy == "full":
            return "train"
        if strategy in ("heads_only", "encoder_only"):
            return "train" if in_head else "frozen"
        if strategy == "lora":
            if ps in lora_paths:
                return "train"
            if in_head and leaf not in ("w",):
                return "train"
            return "frozen"
        raise ValueError(f"Unknown strategy {strategy!r}")

    return jax.tree_util.tree_map_with_path(label_fn, params)


def create_optimizer_for_strategy(
    params: PyTree,
    *,
    strategy: str,
    trainable_head_names: Sequence[str],
    learning_rate: Any,
    weight_decay: float | None = None,
    optimizer_type: str = "adamw",
    gradient_clip_global_norm: float | None = None,
) -> optax.GradientTransformation:
    train_opt = _build_adam_or_adamw(
        learning_rate, optimizer_type=optimizer_type, weight_decay=weight_decay,
    )
    if strategy == "full":
        inner = train_opt
    else:
        labels = _label_params(params, strategy, trainable_head_names)
        inner = optax.multi_transform(
            {"train": train_opt, "frozen": optax.set_to_zero()},
            labels,
        )
    if gradient_clip_global_norm is not None:
        return optax.chain(optax.clip_by_global_norm(gradient_clip_global_norm), inner)
    return inner
