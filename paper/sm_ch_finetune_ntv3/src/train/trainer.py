"""NTv3-CH training loop (heads-only)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    prob = 1.0 / (1.0 + np.exp(-logits))
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, prob)),
        "auprc": float(average_precision_score(labels, prob)),
    }


@torch.no_grad()
def _run_epoch(model, loader, device, *, train: bool, optimizer=None) -> dict:
    model.train(train)
    losses, labels_all, logits_all = [], [], []
    for batch in loader:
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        var_idx = batch.variant_index.to(device)
        logits = model(input_ids, var_idx)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        labels_all.append(labels.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
    y = np.concatenate(labels_all)
    z = np.concatenate(logits_all)
    out = _metrics(y, z)
    out["loss"] = float(np.mean(losses)) if losses else float("nan")
    return out


def train_ntv3_ch(
    model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    checkpoint_dir: Path,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    early_stopping_patience: int = 0,
    max_train_steps: int | None = None,
    grad_accum_steps: int = 1,
    log_every_steps: int = 10,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    params = list(model.trainable_parameters())
    opt = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auroc = -1.0
    best_path = checkpoint_dir / "best_head.pt"
    stale = 0
    history: list[dict] = []
    global_step = 0
    n_train = len(train_loader)
    print(
        f"Training: {n_train} batches/epoch, grad_accum={grad_accum_steps}, "
        f"log_every={log_every_steps}",
        flush=True,
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        losses, labels_all, logits_all = [], [], []
        opt.zero_grad(set_to_none=True)
        epoch_t0 = time.monotonic()
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch.input_ids.to(device)
            labels = batch.labels.to(device)
            var_idx = batch.variant_index.to(device)
            logits = model(input_ids, var_idx)
            loss = F.binary_cross_entropy_with_logits(logits, labels) / grad_accum_steps
            loss.backward()
            if step % grad_accum_steps == 0 or step == n_train:
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if max_train_steps and global_step >= max_train_steps:
                    break
            step_loss = float(loss.item()) * grad_accum_steps
            losses.append(step_loss)
            labels_all.append(labels.detach().cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())

            if step == 1 or step % log_every_steps == 0 or step == n_train:
                elapsed = time.monotonic() - epoch_t0
                rate = step / elapsed if elapsed > 0 else 0.0
                eta_s = (n_train - step) / rate if rate > 0 else float("nan")
                eta_m = eta_s / 60.0 if eta_s == eta_s else float("nan")
                print(
                    f"epoch {epoch} step {step}/{n_train} "
                    f"loss={step_loss:.4f} global_step={global_step} "
                    f"({rate:.3f} batch/s, eta {eta_m:.1f} min)",
                    flush=True,
                )

            if max_train_steps and global_step >= max_train_steps:
                break

        y_tr = np.concatenate(labels_all) if labels_all else np.array([])
        z_tr = np.concatenate(logits_all) if logits_all else np.array([])
        tr = _metrics(y_tr, z_tr) if len(y_tr) else {"auroc": float("nan"), "auprc": float("nan")}
        tr["loss"] = float(np.mean(losses)) if losses else float("nan")
        va = _run_epoch(model, valid_loader, device, train=False)
        row = {"epoch": epoch, "train": tr, "valid": va, "global_step": global_step}
        history.append(row)
        print(
            f"epoch {epoch}: train loss={tr['loss']:.4f} auroc={tr['auroc']:.4f} | "
            f"valid loss={va['loss']:.4f} auroc={va['auroc']:.4f}",
            flush=True,
        )

        auroc = va["auroc"]
        if not np.isnan(auroc) and auroc >= best_auroc:
            best_auroc = auroc
            stale = 0
            torch.save({"head": model.head.state_dict()}, best_path)
        else:
            stale += 1
        if early_stopping_patience and stale >= early_stopping_patience:
            print(f"Early stop at epoch {epoch} (patience={early_stopping_patience})")
            break
        if max_train_steps and global_step >= max_train_steps:
            break

    meta = {
        "best_valid_auroc": best_auroc,
        "checkpoint": str(best_path),
        "history": history,
    }
    with open(checkpoint_dir / "train_metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        model.head.load_state_dict(ckpt["head"])
    return meta
