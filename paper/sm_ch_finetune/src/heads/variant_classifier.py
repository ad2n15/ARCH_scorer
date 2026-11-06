"""Custom binary classification head: CDS-masked mean pooling over 1bp embeddings."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from alphagenome_ft import CustomHead
from alphagenome_ft.lora import LoRAConfig, LoRALinear
from alphagenome_research.model import layers

from train.losses import supervised_contrastive_loss

# Passed into Haiku forward via context (alphagenome_ft head API has no cds_mask arg).
_CDS_MASK: list[jnp.ndarray | None] = [None]


@contextmanager
def cds_mask_context(mask: jnp.ndarray | None) -> Iterator[None]:
    _CDS_MASK[0] = mask
    try:
        yield
    finally:
        _CDS_MASK[0] = None


def pool_sequence_embedding(
    x: jnp.ndarray,
    *,
    cds_mask: jnp.ndarray | None = None,
    variant_index: jnp.ndarray | None = None,
    pooling: str = "variant",
) -> jnp.ndarray:
    """Pool [B, L, D] sequence embeddings to [B, D] (same logic as the head)."""
    if cds_mask is not None:
        w = cds_mask.astype(jnp.float32)
        w = w / jnp.maximum(jnp.sum(w, axis=-1, keepdims=True), 1.0)
        return jnp.sum(x * w[..., None], axis=1)
    if pooling == "window_mean":
        return jnp.mean(x, axis=1)
    if variant_index is None:
        variant_index = jnp.full((x.shape[0],), x.shape[1] // 2, dtype=jnp.int32)
    return x[jnp.arange(x.shape[0]), variant_index]


def head_logit(predictions):
    """Scalar logit from head output (scalar or dict)."""
    if isinstance(predictions, dict):
        return predictions["logit"]
    return predictions


def head_age_pred(predictions):
    """Scalar age prediction from head output (scalar or dict)."""
    if isinstance(predictions, dict):
        return predictions["age_pred"]
    return predictions


def _bce_metrics(logits, labels):
    labels = labels.astype(jnp.float32)
    bce = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = jnp.mean(((jax.nn.sigmoid(logits) >= 0.5).astype(jnp.float32) == labels))
    return bce, acc


def _pool_and_hidden(
    embeddings,
    *,
    hidden_dim: int,
    pooling: str,
    variant_index=None,
    cds_mask=None,
):
    x = embeddings.get_sequence_embeddings(resolution=1)
    if cds_mask is None:
        cds_mask = _CDS_MASK[0]
    x = pool_sequence_embedding(
        x, cds_mask=cds_mask, variant_index=variant_index, pooling=pooling,
    )
    hidden = jax.nn.relu(hk.Linear(hidden_dim, name="hidden")(x))
    logit = hk.Linear(1, name="output")(hidden).squeeze(-1)
    return hidden, logit


class VariantClassifierHead(CustomHead):
    def __init__(self, *, name, metadata, **kwargs):
        super().__init__(name=name, metadata=metadata, **kwargs)
        meta = dict(metadata or {})
        self._hidden_dim = int(meta.get("hidden_dim", 256))
        self._pooling = str(meta.get("pooling", "variant"))

    def predict(self, embeddings, organism_index, *, variant_index=None, cds_mask=None, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
        if cds_mask is None:
            cds_mask = _CDS_MASK[0]
        x = pool_sequence_embedding(
            x, cds_mask=cds_mask, variant_index=variant_index, pooling=self._pooling,
        )
        x = jax.nn.relu(hk.Linear(self._hidden_dim, name="hidden")(x))
        return hk.Linear(1, name="output")(x).squeeze(-1)

    def loss(self, predictions, batch):
        labels = batch.get("labels")
        if labels is None:
            return {"loss": jnp.array(0.0)}
        bce, acc = _bce_metrics(head_logit(predictions), labels)
        return {"loss": bce, "accuracy": acc}


class VariantClassifierAgeAuxHead(CustomHead):
    """BCE + auxiliary regression on normalized gnomAD mean carrier age."""

    def __init__(self, *, name, metadata, **kwargs):
        super().__init__(name=name, metadata=metadata, **kwargs)
        meta = dict(metadata or {})
        self._hidden_dim = int(meta.get("hidden_dim", 256))
        self._pooling = str(meta.get("pooling", "variant"))
        self._age_aux_lambda = float(meta.get("age_aux_lambda", 0.2))

    def predict(self, embeddings, organism_index, *, variant_index=None, cds_mask=None, **kwargs):
        hidden, logit = _pool_and_hidden(
            embeddings,
            hidden_dim=self._hidden_dim,
            pooling=self._pooling,
            variant_index=variant_index,
            cds_mask=cds_mask,
        )
        age_pred = hk.Linear(1, name="age_head")(hidden).squeeze(-1)
        return {"logit": logit, "age_pred": age_pred}

    def loss(self, predictions, batch):
        labels = batch.get("labels")
        if labels is None:
            return {"loss": jnp.array(0.0)}
        logit = head_logit(predictions)
        bce, acc = _bce_metrics(logit, labels)
        age_pred = predictions["age_pred"]
        age_target = batch.get("age_mean_norm")
        age_valid = batch.get("age_valid")
        if age_target is None or age_valid is None:
            return {"loss": bce, "accuracy": acc, "age_mse": jnp.array(0.0)}
        err = (age_pred - age_target) ** 2
        age_mse = jnp.sum(err * age_valid) / jnp.maximum(jnp.sum(age_valid), 1.0)
        loss = bce + self._age_aux_lambda * age_mse
        return {"loss": loss, "accuracy": acc, "bce": bce, "age_mse": age_mse}


class VariantClassifierSupConHead(CustomHead):
    """BCE + supervised contrastive loss on hidden embeddings."""

    def __init__(self, *, name, metadata, **kwargs):
        super().__init__(name=name, metadata=metadata, **kwargs)
        meta = dict(metadata or {})
        self._hidden_dim = int(meta.get("hidden_dim", 256))
        self._pooling = str(meta.get("pooling", "variant"))
        self._supcon_lambda = float(meta.get("supcon_lambda", 0.1))
        self._supcon_temperature = float(meta.get("supcon_temperature", 0.1))

    def predict(self, embeddings, organism_index, *, variant_index=None, cds_mask=None, **kwargs):
        hidden, logit = _pool_and_hidden(
            embeddings,
            hidden_dim=self._hidden_dim,
            pooling=self._pooling,
            variant_index=variant_index,
            cds_mask=cds_mask,
        )
        return {"logit": logit, "hidden": hidden}

    def loss(self, predictions, batch):
        labels = batch.get("labels")
        if labels is None:
            return {"loss": jnp.array(0.0)}
        logit = head_logit(predictions)
        bce, acc = _bce_metrics(logit, labels)
        supcon = supervised_contrastive_loss(
            predictions["hidden"],
            labels,
            temperature=self._supcon_temperature,
        )
        loss = bce + self._supcon_lambda * supcon
        return {"loss": loss, "accuracy": acc, "bce": bce, "supcon": supcon}


class VariantChAgeRegressorHead(CustomHead):
    """CH-only training: regress normalized gnomAD carrier age (median or mean)."""

    def __init__(self, *, name, metadata, **kwargs):
        super().__init__(name=name, metadata=metadata, **kwargs)
        meta = dict(metadata or {})
        self._hidden_dim = int(meta.get("hidden_dim", 256))
        self._pooling = str(meta.get("pooling", "variant"))

    def predict(self, embeddings, organism_index, *, variant_index=None, cds_mask=None, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
        if cds_mask is None:
            cds_mask = _CDS_MASK[0]
        x = pool_sequence_embedding(
            x, cds_mask=cds_mask, variant_index=variant_index, pooling=self._pooling,
        )
        hidden = jax.nn.relu(hk.Linear(self._hidden_dim, name="hidden")(x))
        age_pred = hk.Linear(1, name="age_output")(hidden).squeeze(-1)
        return {"age_pred": age_pred}

    def loss(self, predictions, batch):
        age_pred = head_age_pred(predictions)
        age_target = batch.get("age_target")
        if age_target is None:
            age_target = batch.get("age_mean_norm")
        age_valid = batch.get("age_valid")
        if age_target is None or age_valid is None:
            return {"loss": jnp.array(0.0), "age_mse": jnp.array(0.0)}
        err = (age_pred - age_target) ** 2
        age_mse = jnp.sum(err * age_valid) / jnp.maximum(jnp.sum(age_valid), 1.0)
        return {"loss": age_mse, "age_mse": age_mse}


def _classifier_mlp(x: jnp.ndarray, *, hidden_dim: int, use_lora: bool, lora_config: LoRAConfig | None):
    if use_lora:
        cfg = lora_config or LoRAConfig()
        x = jax.nn.relu(LoRALinear(hidden_dim, config=cfg, name="hidden_lora")(x))
        return LoRALinear(1, config=cfg, name="output_lora")(x).squeeze(-1)
    x = jax.nn.relu(hk.Linear(hidden_dim, name="hidden")(x))
    return hk.Linear(1, name="output")(x).squeeze(-1)


class VariantClassifierLoRAHead(CustomHead):
    """Head-only LoRA: train low-rank adapters in the MLP; base ``w`` weights stay fixed."""

    def __init__(self, *, name, metadata, **kwargs):
        super().__init__(name=name, metadata=metadata, **kwargs)
        meta = dict(metadata or {})
        self._hidden_dim = int(meta.get("hidden_dim", 256))
        self._lora_config = LoRAConfig(
            rank=int(meta.get("lora_rank", 8)),
            alpha=float(meta.get("lora_alpha", 1.0)),
        )
        self._pooling = str(meta.get("pooling", "variant"))

    def predict(self, embeddings, organism_index, *, variant_index=None, cds_mask=None, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
        if cds_mask is None:
            cds_mask = _CDS_MASK[0]
        x = pool_sequence_embedding(
            x, cds_mask=cds_mask, variant_index=variant_index, pooling=self._pooling,
        )
        return _classifier_mlp(
            x, hidden_dim=self._hidden_dim, use_lora=True, lora_config=self._lora_config,
        )

    def loss(self, predictions, batch):
        labels = batch.get("labels")
        if labels is None:
            return {"loss": jnp.array(0.0)}
        bce, acc = _bce_metrics(head_logit(predictions), labels)
        return {"loss": bce, "accuracy": acc}


class VariantClassifierEncoderHead(CustomHead):
    """Encoder-only path: pool 128 bp CNN features (requires ``use_encoder_output=True``)."""

    def __init__(self, *, name, metadata, **kwargs):
        super().__init__(name=name, metadata=metadata, **kwargs)
        meta = dict(metadata or {})
        self._hidden_dim = int(meta.get("hidden_dim", 256))

    def predict(self, embeddings, organism_index, *, variant_index=None, cds_mask=None, **kwargs):
        if not hasattr(embeddings, "encoder_output") or embeddings.encoder_output is None:
            raise ValueError(
                "VariantClassifierEncoderHead requires encoder_output. "
                "Create model with use_encoder_output=True."
            )
        x = embeddings.encoder_output
        x = layers.LayerNorm(name="norm")(x)
        if variant_index is None:
            variant_index = jnp.full((x.shape[0],), x.shape[1] // 2, dtype=jnp.int32)
        x = x[jnp.arange(x.shape[0]), variant_index]
        return _classifier_mlp(x, hidden_dim=self._hidden_dim, use_lora=False, lora_config=None)

    def loss(self, predictions, batch):
        labels = batch.get("labels")
        if labels is None:
            return {"loss": jnp.array(0.0)}
        bce, acc = _bce_metrics(head_logit(predictions), labels)
        return {"loss": bce, "accuracy": acc}
