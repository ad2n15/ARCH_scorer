#!/usr/bin/env python3
"""Pooled emb+baseline MLP on GPU (PyTorch) — companion to scripts/109 (CPU LR/RF/XGB).

Same cohort / CV / features as 109: fold_1 emb DIFF (full, no PCA) + CADD + SBS6
+ Func.refGene (+ optional gene onehot). Topology matches sklearn MLPClassifier used
in 109: Linear(hidden) → ReLU → Linear(1), Adam, early stopping.

Example:
  sbatch run_binemo_l40.sh sm_ch_finetune/scripts/113_cv5_pooled_emb_baseline_mlp_gpu.py \\
      --genes ASXL1,...,TP53 \\
      --cv-mode stratify_gene_func --gene-feature onehot \\
      --out-dir results/pooled_arch21_cv5_fold1_emb_baseline_mlp_gpu_...
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_table import mutation_type_feature
from eval.classification_metrics import (
    attach_cosmic_haem_flags,
    haem_ge3_membership,
    prevalence_matched_table,
    ranking_metrics,
    source_sensitivity_pm,
)
from paths import project_subpath

_s102 = spec_from_file_location("cv5_emb", PROJECT / "scripts" / "102_cv5_fold1_embeddings.py")
_m102 = module_from_spec(_s102)
assert _s102.loader is not None
_s102.loader.exec_module(_m102)
_load_X = _m102._load_X

_s106 = spec_from_file_location("cv5_pooled", PROJECT / "scripts" / "106_cv5_pooled_arch_genes.py")
_m106 = module_from_spec(_s106)
assert _s106.loader is not None
_s106.loader.exec_module(_m106)
_load_annovar = _m106._load_annovar
_resolve_cadd = _m106._resolve_cadd
_resolve_annovar = _m106._resolve_annovar
_make_ohe = _m106._make_ohe

_s109 = spec_from_file_location("cv5_emb_base", PROJECT / "scripts" / "109_cv5_pooled_lof_emb_baseline.py")
_m109 = module_from_spec(_s109)
assert _s109.loader is not None
_s109.loader.exec_module(_m109)
_make_splits = _m109._make_splits
LOF9 = _m109.LOF9


class EmbBaselineMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    sw: np.ndarray,
    *,
    hidden_dim: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> EmbBaselineMLP:
    torch.manual_seed(seed)
    n = len(y_tr)
    # hold out 10% for early stopping (same idea as sklearn validation_fraction)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_t = torch.tensor(X_tr[tr_idx], dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr[tr_idx], dtype=torch.float32, device=device)
    w_t = torch.tensor(sw[tr_idx], dtype=torch.float32, device=device)
    X_v = torch.tensor(X_tr[val_idx], dtype=torch.float32, device=device)
    y_v = torch.tensor(y_tr[val_idx], dtype=torch.float32, device=device)
    w_v = torch.tensor(sw[val_idx], dtype=torch.float32, device=device)

    model = EmbBaselineMLP(X_tr.shape[1], hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_state = None
    best_val = float("inf")
    stall = 0
    bs = min(batch_size, max(32, len(tr_idx)))

    for _epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(tr_idx), device=device)
        for start in range(0, len(tr_idx), bs):
            idx = order[start : start + bs]
            logits = model(X_t[idx])
            loss = (bce(logits, y_t[idx]) * w_t[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = (bce(model(X_v), y_v) * w_v).mean().item()
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genes", default=",".join(LOF9))
    ap.add_argument("--feature-mode", default="diff", choices=("alt", "ref", "diff", "alt_diff", "concat"))
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument(
        "--cv-mode",
        default="stratify_gene_func",
        choices=(
            "stratify_label",
            "stratify_gene",
            "stratify_func",
            "stratify_sbs6",
            "stratify_gene_func",
            "stratify_func_sbs6",
            "stratify_gene_func_sbs6",
        ),
    )
    ap.add_argument("--gene-feature", choices=("none", "onehot"), default="onehot")
    ap.add_argument("--mutation-scheme", choices=("raw", "sbs6"), default="sbs6")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device if args.device != "cuda" else "cuda")

    genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
    rows = []
    feat_tag = None
    emb_dim = None

    for gene in genes:
        emb_dir = project_subpath(PROJECT, f"results/fold1_embeddings_pooled_sm/{gene}")
        if not (emb_dir / "variants_meta.parquet").is_file():
            raise SystemExit(f"Missing embeddings: {emb_dir}")
        meta, X_emb, tag = _load_X(emb_dir, args.feature_mode)
        feat_tag = tag
        emb_dim = X_emb.shape[1]
        meta = meta.copy()
        meta["variant_id"] = meta["variant_id"].astype(str)
        meta["gene"] = gene
        if "y" not in meta.columns:
            meta["y"] = (meta["label"].astype(str).str.lower() == "somatic").astype(int)
        if "source" not in meta.columns:
            meta["source"] = np.where(meta["y"] == 1, "cosmic_ncv", "gnomad")

        cadd_p = _resolve_cadd(gene)
        if cadd_p is None:
            raise SystemExit(f"Missing CADD for {gene}")
        cadd = pd.read_csv(cadd_p)
        cadd["variant_id"] = cadd["variant_id"].astype(str)
        cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")
        ann = _load_annovar(_resolve_annovar(gene))
        df = meta.merge(cadd[["variant_id", "cadd_phred"]], on="variant_id", how="left")
        df = df.merge(ann, on="variant_id", how="left")
        df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)
        df["mutation_type"] = [
            mutation_type_feature(str(r), str(a), args.mutation_scheme)
            for r, a in zip(df["ref"], df["alt"])
        ]
        emb_block = pd.DataFrame(
            X_emb, columns=[f"e{i}" for i in range(X_emb.shape[1])], index=df.index
        )
        df = pd.concat([df, emb_block], axis=1)
        df = df.loc[df["cadd_phred"].notna()].drop_duplicates("variant_id")
        rows.append(df)
        print(f"  loaded {gene}: n={len(df)} somatic={int(df.y.sum())}", flush=True)

    df = pd.concat(rows, ignore_index=True)
    df = attach_cosmic_haem_flags(df)
    emb_cols = [f"e{i}" for i in range(emb_dim)]
    y = df["y"].astype(int).values
    print(
        f"Pooled emb+baseline MLP-GPU n={len(df)} genes={df['gene'].nunique()} "
        f"somatic={int(y.sum())} mode={feat_tag} device={device} "
        f"cv={args.cv_mode} gene_feature={args.gene_feature}",
        flush=True,
    )

    splits = _make_splits(
        args.cv_mode,
        y,
        genes=df["gene"].astype(str).values,
        funcs=df["Func.refGene"].astype(str).values,
        muts=df["mutation_type"].astype(str).values,
        n_splits=int(args.n_splits),
        seed=int(args.seed),
    )
    print(f"n_folds={len(splits)}", flush=True)

    use_cols = emb_cols + ["cadd_phred", "mutation_type", "Func.refGene"]
    if args.gene_feature == "onehot":
        use_cols = use_cols + ["gene"]
    X_df = df[use_cols].copy()

    base_cats = ["mutation_type", "Func.refGene"]
    if args.gene_feature == "onehot":
        base_cats = base_cats + ["gene"]
    base_ct = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), ["cadd_phred"]),
            ("cat", _make_ohe(), base_cats),
        ],
        remainder="drop",
    )

    name = "emb_base_mlp"
    oof = np.full(len(df), np.nan)
    fold_metrics: list[dict] = []
    n_pos_all = max(int(y.sum()), 1)
    n_neg_all = max(int(len(y) - n_pos_all), 1)
    w_pos = 0.5 * len(y) / n_pos_all
    w_neg = 0.5 * len(y) / n_neg_all

    for fold_i, (tr, te) in enumerate(splits, 1):
        y_tr, y_te = y[tr], y[te]
        X_tr, X_te = X_df.iloc[tr], X_df.iloc[te]
        print(
            f"fold {fold_i}/{len(splits)} n_train={len(tr)} n_test={len(te)} "
            f"pos_te={int(y_te.sum())}",
            flush=True,
        )
        emb_pipe = Pipeline(
            [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
        )
        emb_tr = emb_pipe.fit_transform(np.asarray(X_tr[emb_cols], dtype=np.float64))
        emb_te = emb_pipe.transform(np.asarray(X_te[emb_cols], dtype=np.float64))
        b_tr = base_ct.fit_transform(X_tr)
        b_te = base_ct.transform(X_te)
        Z_tr = np.hstack([np.asarray(emb_tr), np.asarray(b_tr, dtype=float)])
        Z_te = np.hstack([np.asarray(emb_te), np.asarray(b_te, dtype=float)])
        sw = np.where(y_tr == 1, w_pos, w_neg).astype(np.float64)

        model = _train_mlp(
            Z_tr,
            y_tr.astype(np.float64),
            sw,
            hidden_dim=int(args.hidden_dim),
            max_epochs=int(args.max_epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            patience=int(args.patience),
            seed=int(args.seed) + fold_i,
            device=device,
        )
        with torch.no_grad():
            logits = model(torch.tensor(Z_te, dtype=torch.float32, device=device))
            prob = torch.sigmoid(logits).detach().cpu().numpy()
        oof[te] = prob
        fold_metrics.append(ranking_metrics(y_te, prob))
        print(
            f"  fold AUROC={fold_metrics[-1]['auroc']:.4f}",
            flush=True,
        )

    out_dir = project_subpath(
        PROJECT,
        args.out_dir
        or (
            f"results/pooled_lof{len(genes)}_cv5_fold1_emb_baseline_mlp_gpu_"
            f"{feat_tag}_{args.cv_mode}_genefeat_{args.gene_feature}"
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = df[
        [
            c
            for c in [
                "variant_id",
                "gene",
                "label",
                "source",
                "y",
                "n_haem",
                "haem_ge3",
                "Func.refGene",
                "mutation_type",
                "cadd_phred",
            ]
            if c in df.columns
        ]
    ].copy()
    pred[f"p_{name}"] = oof
    m = ranking_metrics(y, oof)
    pm = prevalence_matched_table(y, oof)
    print(
        f"\n{name} OOF: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
        f"PM_sens={pm.get('sensitivity')}",
        flush=True,
    )

    summary = [
        {
            "model": name,
            **m,
            **{f"pm_{k}": v for k, v in pm.items()},
            "cv_mode": args.cv_mode,
            "gene_feature": args.gene_feature,
            "fold_auroc_mean": float(np.mean([x["auroc"] for x in fold_metrics])),
            "fold_auroc_sd": float(np.std([x["auroc"] for x in fold_metrics])),
        }
    ]
    src_rows = [
        {"model": name, **r}
        for r in source_sensitivity_pm(
            y, oof, df["source"].values, extra_membership=haem_ge3_membership(df)
        )
    ]
    per_gene_rows = []
    for g in sorted(df["gene"].unique()):
        mask = df["gene"].values == g
        if y[mask].sum() == 0 or (y[mask] == 0).sum() == 0:
            continue
        gm = ranking_metrics(y[mask], oof[mask])
        gpm = prevalence_matched_table(y[mask], oof[mask])
        per_gene_rows.append(
            {
                "model": name,
                "gene": g,
                "n": int(mask.sum()),
                "n_somatic": int(y[mask].sum()),
                **gm,
                "pm_sensitivity": gpm.get("sensitivity"),
            }
        )

    pd.DataFrame(summary).to_csv(out_dir / "oof_summary.csv", index=False)
    pd.DataFrame(src_rows).to_csv(out_dir / "source_sensitivity_pm.csv", index=False)
    pd.DataFrame(per_gene_rows).to_csv(out_dir / "per_gene_oof.csv", index=False)
    pred.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "genes": genes,
                "n": len(df),
                "feature_mode": feat_tag,
                "emb_dim": emb_dim,
                "cv_mode": args.cv_mode,
                "gene_feature": args.gene_feature,
                "classifiers": ["mlp"],
                "backend": "pytorch",
                "device": str(device),
                "hidden_dim": args.hidden_dim,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
