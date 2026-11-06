#!/usr/bin/env python3
"""Maintainers: export the nine fine-tuned heads (3 backbones x 3 tasks) into models/finetuned/.

For each backbone and task this writes:
  head.safetensors      classifier head (PyTorch layout; AlphaGenome's Haiku weights are transposed)
  config.json           backbone, window, pooling, head shape, task
  fusion.json           late-fusion logistic regression (head probability + annotations), refit on the
                        training split with the paper's recipe
  score_quantiles.json  0..100th percentiles of held-out test scores (for the percentile column)
  test_metrics.json     held-out AUROC / AUPRC of the head alone and fused

Fusion recipes:
  arch21_noncoding, arch21_cnc  sm_ch_finetune/scripts/136 fit_meta: P + CADD-PHRED + SBS6 + Func.refGene + gene.
      CADD is taken from both the *_api and *_local CADD v1.7 files so it is never missing; the paper's
      script read only *_api files (CADD median-imputed for most variants), e.g. NTv3 NC fused AUROC 0.815 -> 0.818.
      The online tool always fetches CADD, so this matches how the models are used.
  genomewide  sm_ch_finetune/scripts/08: P + CADD (as in the Colin-Leitzinger table) + exonic function
      + top-100 genes + REF>ALT class (12 levels).

Held-out variants with their stored scores are written to checks/ (git-ignored: they include COSMIC-derived
labels) for scripts/verify_finetuned.py.

Run inside the training environment (needs torch, orbax/jax, the training projects):
  ALPHAGENOME_ROOT=/path/to/alphagenome python3 scripts/export_finetuned_models.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ALPHAGENOME_ROOT", HERE.parent))  # folder holding sm_ch_finetune*/ projects
SM, NT, BZ = ROOT / "sm_ch_finetune", ROOT / "sm_ch_finetune_ntv3", ROOT / "sm_ch_finetune_borzoi"
sys.path.insert(0, str(HERE / "src"))
from arch_scorer.finetuned.fusion import FusionLR  # noqa: E402

OUT = HERE / "models" / "finetuned"
CHECKS = HERE / "checks"
C, MAX_ITER = 0.5, 2000
ARCH21 = ["ASXL1", "ASXL2", "CBL", "DNMT3A", "ETNK1", "GNAS", "GNB1", "IDH1", "IDH2", "JAK2", "KIT",
          "KRAS", "MPL", "NRAS", "PPM1D", "RUNX1", "SETBP1", "SF3B1", "SRSF2", "TET2", "TP53"]

TASKS = {
    "arch21_noncoding": {"task": "Somatic vs germline, 21 ARCH genes (trained on non-coding variants)",
                         "positive_class": "somatic", "negative_class": "germline", "genes": ARCH21,
                         "pilot": "pilot_arch21_mane_noncoding_gvs_finetune", "pred": "arch21_nc_gvs",
                         "ckpt": "arch21_mane_noncoding_gvs"},
    "arch21_cnc": {"task": "Somatic vs germline, 21 ARCH genes (trained on coding + non-coding variants)",
                   "positive_class": "somatic", "negative_class": "germline", "genes": ARCH21,
                   "pilot": "pilot_arch21_mane_cnc_gvs_finetune", "pred": "arch21_cnc_gvs",
                   "ckpt": "arch21_mane_cnc_gvs"},
    "genomewide": {"task": "ARCH vs other somatic, genome-wide (Colin-Leitzinger age-filtered cohort)",
                   "positive_class": "ARCH", "negative_class": "other somatic", "pred": "full",
                   "ckpt": "full_age_filtered"},
}
BACKBONES = {
    "ntv3": {"backbone": "InstaDeepAI/NTv3_650M_post", "backbone_revision": "ad622051abbe9376bdea3f2727863fa559957e9f",
             "pooling": "final hidden state at the ALT base", "layer_norm": True,
             "pred": lambda k, s: NT / f"results/ntv3_{k}_{s}_predictions.parquet",
             "ckpt": lambda k: NT / f"checkpoints/{k}_64kb_bs8_5ep/best_head.pt"},
    "borzoi": {"backbone": "Genentech/borzoi-model human_rep0 (gReLU), crop disabled",
               "pooling": "32 bp trunk bin containing the ALT base", "layer_norm": False,
               "pred": lambda k, s: BZ / f"results/borzoi_{k}_{s}_predictions.parquet",
               "ckpt": lambda k: BZ / f"checkpoints/{k}_64kb_bs8_5ep/best_head.pt"},
    "alphagenome": {"backbone": "AlphaGenome fold_1 (JAX, Kaggle google/alphagenome)",
                    "pooling": "1 bp sequence embedding at the ALT base (bfloat16 compute)", "layer_norm": False,
                    "pred": lambda k, s: SM / f"results/finetune_{k}_{s}_predictions.parquet",
                    "ckpt": lambda k: SM / f"checkpoints/{k}_nocds_64kb_bs8_5ep/best/checkpoint"},
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def head_tensors(backbone: str, ckpt: Path) -> dict:
    if backbone == "alphagenome":
        import jax
        import orbax.checkpoint as ocp

        ck = ocp.PyTreeCheckpointer()
        meta = ck.metadata(ckpt)
        meta = meta.item_metadata.tree if hasattr(meta, "item_metadata") else meta
        sh = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
        tree = ck.restore(ckpt, restore_args=jax.tree_util.tree_map(lambda m: ocp.ArrayRestoreArgs(sharding=sh), meta))
        p = tree[0] if isinstance(tree, list) else tree
        hid, out = p["head/sm_ch_classifier/~predict/hidden"], p["head/sm_ch_classifier/~predict/output"]
        f = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        return {"fc1.weight": f(hid["w"]).T.copy(), "fc1.bias": f(hid["b"]),
                "fc2.weight": f(out["w"]).T.copy(), "fc2.bias": f(out["b"])}
    import torch

    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["head"]
    return {k: v.float().numpy() for k, v in sd.items()}


def export_head(backbone: str, task: str, out: Path) -> None:
    from safetensors.numpy import save_file

    b, t = BACKBONES[backbone], TASKS[task]
    w = head_tensors(backbone, b["ckpt"](t["ckpt"]))
    save_file({k: np.ascontiguousarray(v) for k, v in w.items()}, out / "head.safetensors")
    cfg = {"backbone": b["backbone"], **({"backbone_revision": b["backbone_revision"]} if "backbone_revision" in b else {}),
           "genome": "GRCh38", "window_size": 65536, "variant_index": 32768, "pooling": b["pooling"],
           "layer_norm": b["layer_norm"], "embed_dim": int(w["fc1.weight"].shape[1]),
           "hidden_dim": int(w["fc1.weight"].shape[0]),
           **{k: v for k, v in t.items() if k not in ("pilot", "pred", "ckpt")}}
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def fit(X: pd.DataFrame, y: np.ndarray, spec: list[dict]) -> FusionLR:
    fusion = FusionLR.from_training(X, spec)
    lr = LogisticRegression(max_iter=MAX_ITER, C=C, solver="lbfgs").fit(fusion.design(X), y)
    fusion.coef, fusion.intercept = lr.coef_[0].tolist(), float(lr.intercept_[0])
    return fusion


def _cadd_all() -> pd.DataFrame:
    parts = []
    for kind in ("noncoding", "coding"):
        for f in sorted((SM / f"data/arch_mane_{kind}_pooled_sm/cadd_by_gene").glob("*_cadd_phred_*.csv")):
            if "paper_mw" not in f.name:
                parts.append(pd.read_csv(f, usecols=["variant_id", "cadd_phred"]))
    c = pd.concat(parts)
    c["variant_id"] = c["variant_id"].astype(str)
    c["cadd_phred"] = pd.to_numeric(c["cadd_phred"], errors="coerce")
    return c.dropna().drop_duplicates("variant_id")


def arch21(backbone: str, task: str, cadd: pd.DataFrame) -> dict:
    s136 = _load(SM / "scripts/136_arch21_ft_vs_ft_baseline_lr.py", "s136")
    t = TASKS[task]
    pilot = SM / "data" / t["pilot"]
    tr, te = (s136.load_split(pilot, BACKBONES[backbone]["pred"](t["pred"], s), cadd, s)
              .rename(columns={"prob_ch": "finetune_prob"}) for s in ("train", "test"))
    spec = [{"name": "finetune_prob", "kind": "numeric"}, {"name": "cadd_phred", "kind": "numeric"},
            {"name": "mutation_type", "kind": "categorical", "encoding": "sbs6"},
            {"name": "Func.refGene", "kind": "categorical"}, {"name": "gene", "kind": "categorical"}]
    fusion = fit(tr, tr["y"].values, spec)
    p = fusion.predict(te)
    ref = s136.fit_meta(tr, te)
    assert np.abs(p - ref).max() < 1e-6, "numpy fusion differs from the paper's sklearn pipeline"
    return {"fusion": fusion, "y": te["y"].values, "p": te["finetune_prob"].values, "fused": p,
            "check": te[["variant_id", "label", "finetune_prob"]].assign(fused=p)}


def genomewide(backbone: str) -> dict:
    s08 = _load(SM / "scripts/08_evaluate_finetune_covariates.py", "s08")
    data = SM / "data/processed_age_filtered"
    cols = ["cadd", "exonic_func", "gene", "mutation_type", "ref", "alt", "func", "chrom", "pos"]
    frames = {}
    for split in ("train", "test"):
        meta = pd.read_parquet(data / f"{split}.parquet")
        pr = pd.read_parquet(BACKBONES[backbone]["pred"]("full", split))
        pr["variant_id"] = pr["variant_id"].astype(str)
        df = pr.rename(columns={"prob_ch": "finetune_prob"})[["variant_id", "finetune_prob"]]
        df = df.merge(meta[["variant_id", "label"]], on="variant_id")
        df = s08._attach_meta(df, meta, cols)
        frames[split] = s08._ensure_mutation_type(df, "mutation_type")
    tr, te = frames["train"], frames["test"]
    spec = [{"name": "finetune_prob", "kind": "numeric"}, {"name": "cadd", "kind": "numeric"},
            {"name": "exonic_func", "kind": "categorical"}, {"name": "gene", "kind": "categorical", "top_k": 100},
            {"name": "mutation_type", "kind": "categorical", "encoding": "raw"}]
    fusion = fit(tr, tr["label"].values, spec)
    p = fusion.predict(te)
    return {"fusion": fusion, "y": te["label"].values, "p": te["finetune_prob"].values, "fused": p,
            "check": te[["variant_id", "label", "finetune_prob"]].assign(fused=p)}


def quantiles(p) -> list[float]:
    return np.quantile(np.asarray(p), np.linspace(0, 1, 101)).round(6).tolist()


def main() -> None:
    CHECKS.mkdir(exist_ok=True)
    cadd = _cadd_all()
    for backbone in BACKBONES:
        for task in TASKS:
            out = OUT / backbone / task
            out.mkdir(parents=True, exist_ok=True)
            export_head(backbone, task, out)
            r = arch21(backbone, task, cadd) if task.startswith("arch21") else genomewide(backbone)
            r["fusion"].save(out / "fusion.json")
            y = r["y"]
            m = {"n": int(len(y)), "n_pos": int(y.sum()),
                 "auroc_finetuned": round(float(roc_auc_score(y, r["p"])), 4),
                 "auroc_fused": round(float(roc_auc_score(y, r["fused"])), 4),
                 "auprc_finetuned": round(float(average_precision_score(y, r["p"])), 4),
                 "auprc_fused": round(float(average_precision_score(y, r["fused"])), 4)}
            (out / "test_metrics.json").write_text(json.dumps(m, indent=2) + "\n")
            (out / "score_quantiles.json").write_text(json.dumps({"finetuned": quantiles(r["p"]),
                                                                  "fused": quantiles(r["fused"])}) + "\n")
            r["check"].to_csv(CHECKS / f"test_{backbone}_{task}.csv.gz", index=False)
            print(backbone, task, m, flush=True)


if __name__ == "__main__":
    main()
