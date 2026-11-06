"""Head-only training loop for variant classification."""
from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from heads.variant_classifier import cds_mask_context
from eval.checkpoint_loader import patch_encoder_checkpoint_config
from train.optimizer_factory import create_optimizer_for_strategy


def tokens_to_onehot(tokens: np.ndarray) -> np.ndarray:
    """Map int tokens 0=A,1=C,2=G,3=T,4=N -> float32 one-hot."""
    out = np.zeros((*tokens.shape, 4), dtype=np.float32)
    for i in range(4):
        out[..., i] = (tokens == i).astype(np.float32)
    return out


def _prepare_batch(variant_batch, organism_index: int) -> dict:
    seqs = tokens_to_onehot(variant_batch.sequences)
    b = seqs.shape[0]
    batch = {
        "sequences": jnp.asarray(seqs),
        "organism_index": jnp.full((b,), organism_index, dtype=jnp.int32),
        "negative_strand_mask": jnp.zeros((b,), dtype=jnp.bool_),
        "labels": jnp.asarray(variant_batch.labels, dtype=jnp.float32),
    }
    if variant_batch.cds_mask is not None:
        batch["cds_mask"] = jnp.asarray(variant_batch.cds_mask)
    if variant_batch.age_mean_norm is not None:
        batch["age_mean_norm"] = jnp.asarray(variant_batch.age_mean_norm, dtype=jnp.float32)
        batch["age_valid"] = jnp.asarray(variant_batch.age_valid, dtype=jnp.bool_)
    if variant_batch.age_target is not None:
        batch["age_target"] = jnp.asarray(variant_batch.age_target, dtype=jnp.float32)
    return batch


def _loss_batch(head_batch: dict, variant_batch) -> dict:
    out = {"labels": head_batch["labels"]}
    if variant_batch.age_mean_norm is not None:
        out["age_mean_norm"] = head_batch.get("age_mean_norm")
        out["age_valid"] = head_batch.get("age_valid")
    if variant_batch.age_target is not None:
        out["age_target"] = head_batch.get("age_target")
    return out


def train_variant_classifier(
    model,
    train_dm,
    valid_dm,
    head_id: str,
    *,
    organism: str,
    finetune_strategy: str = "heads_only",
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    max_train_steps: int | None,
    early_stopping_patience: int,
    checkpoint_dir: Path,
    seed: int = 42,
) -> None:
    organism_enum = ag_dna_model.Organism[organism]
    org_idx = int(research_dna_model.convert_to_organism_index(organism_enum))
    strand_reindexing = jax.device_put(model._metadata[organism_enum].strand_reindexing)

    loss_fn = model.create_loss_fn_for_head(head_id)
    save_full_model = finetune_strategy == "full"
    optimizer = create_optimizer_for_strategy(
        model._params,
        strategy=finetune_strategy,
        trainable_head_names=[head_id],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    opt_state = optimizer.init(model._params)
    params = model._params
    state = model._state
    rng = jax.random.PRNGKey(seed)

    steps_per_epoch = len(train_dm)
    planned = steps_per_epoch * num_epochs
    total_steps = min(planned, max_train_steps) if max_train_steps else planned
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training: strategy={finetune_strategy}, {steps_per_epoch} steps/epoch, "
        f"{num_epochs} epoch(s), cap {total_steps} total step(s)"
    )

    best_valid = None
    stale_epochs = 0
    global_step = 0

    with model._device_context:
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            train_losses = []
            for batch_np in train_dm.iter_batches(epoch=epoch):
                if global_step >= total_steps:
                    break
                batch = _prepare_batch(batch_np, org_idx)
                rng, step_rng = jax.random.split(rng)

                def loss_for_params(p):
                    with cds_mask_context(batch.get("cds_mask")):
                        preds = model._predict(
                            p,
                            state,
                            batch["sequences"],
                            batch["organism_index"],
                            requested_outputs=frozenset(),
                            negative_strand_mask=batch["negative_strand_mask"],
                            strand_reindexing=strand_reindexing,
                            rng=step_rng,
                        )
                    return loss_fn(preds[head_id], _loss_batch(batch, batch_np))["loss"]

                loss, grads = jax.value_and_grad(loss_for_params)(params)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                train_losses.append(float(loss))
                global_step += 1
                if global_step % 10 == 0 or global_step == 1:
                    print(f"  step {global_step}/{total_steps} loss={float(loss):.4f}")

            train_avg = float(np.mean(train_losses)) if train_losses else float("nan")
            print(f"  train loss: {train_avg:.4f}")

            valid_losses = []
            for batch_np in valid_dm.iter_batches(epoch=0):
                batch = _prepare_batch(batch_np, org_idx)
                with cds_mask_context(batch.get("cds_mask")):
                    preds = model._predict(
                        params,
                        state,
                        batch["sequences"],
                        batch["organism_index"],
                        requested_outputs=frozenset(),
                        negative_strand_mask=batch["negative_strand_mask"],
                        strand_reindexing=strand_reindexing,
                        rng=None,
                    )
                valid_losses.append(
                    float(loss_fn(preds[head_id], _loss_batch(batch, batch_np))["loss"])
                )
                if len(valid_losses) >= 200:
                    break
            valid_avg = float(np.mean(valid_losses)) if valid_losses else float("nan")
            print(f"  valid loss (subset): {valid_avg:.4f}")

            model._params = params
            model.save_checkpoint(checkpoint_dir / "last", save_full_model=save_full_model)
            if finetune_strategy == "encoder_only":
                patch_encoder_checkpoint_config(checkpoint_dir / "last")

            if math.isfinite(valid_avg) and (best_valid is None or valid_avg < best_valid):
                best_valid = valid_avg
                stale_epochs = 0
                model.save_checkpoint(checkpoint_dir / "best", save_full_model=save_full_model)
                if finetune_strategy == "encoder_only":
                    patch_encoder_checkpoint_config(checkpoint_dir / "best")
                print(f"  saved best (valid={valid_avg:.4f})")
            else:
                stale_epochs += 1

            if early_stopping_patience and stale_epochs >= early_stopping_patience:
                print(f"Early stop after {stale_epochs} epoch(s) without improvement")
                break
            if global_step >= total_steps:
                break

    model._params = params
    print("Training complete.")
