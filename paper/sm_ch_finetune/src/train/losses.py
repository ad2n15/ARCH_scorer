"""Auxiliary training losses for variant heads."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def supervised_contrastive_loss(
    features: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    temperature: float = 0.1,
) -> jnp.ndarray:
    """Supervised contrastive loss (Khosla et al.) on L2-normalized features."""
    labels = labels.reshape(-1).astype(jnp.int32)
    b = features.shape[0]
    if b <= 1:
        return jnp.array(0.0, dtype=features.dtype)

    z = features / jnp.maximum(jnp.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    logits = (z @ z.T) / temperature
    logits = logits - jnp.max(logits, axis=1, keepdims=True)

    label_eq = labels[:, None] == labels[None, :]
    self_mask = jnp.eye(b, dtype=bool)
    pos_mask = label_eq & ~self_mask
    n_pos = jnp.sum(pos_mask, axis=1)

    exp_logits = jnp.exp(logits) * (~self_mask)
    log_prob = logits - jnp.log(jnp.maximum(jnp.sum(exp_logits, axis=1, keepdims=True), 1e-8))

    pos_log_prob = jnp.where(pos_mask, log_prob, 0.0)
    mean_log_prob_pos = jnp.sum(pos_log_prob, axis=1) / jnp.maximum(n_pos, 1)
    valid = n_pos > 0
    if jnp.sum(valid) == 0:
        return jnp.array(0.0, dtype=features.dtype)
    return -jnp.sum(mean_log_prob_pos * valid) / jnp.sum(valid)
