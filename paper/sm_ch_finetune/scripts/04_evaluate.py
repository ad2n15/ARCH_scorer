#!/usr/bin/env python3
"""Evaluate fine-tuned model and baselines on test set."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score,
)

META_COLS = {"variant_id", "label", "gene", "func", "exonic_func"}


def _same_variant_split(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if len(a) != len(b):
        return False
    return set(a["variant_id"].astype(str)) == set(b["variant_id"].astype(str))


def _stratify_metrics(
    test: pd.DataFrame, prob: np.ndarray, *, threshold: float, min_n: int = 20,
) -> list[dict]:
    test = test.reset_index(drop=True)
    rows = []
    for func, g in test.groupby("func", dropna=False):
        name = str(func) if pd.notna(func) else "."
        n = len(g)
        if n < min_n:
            rows.append({"func": name, "n": n, "skipped": True, "reason": f"n < {min_n}"})
            continue
        y = g["label"].values.astype(int)
        p = prob[g.index.values]
        if len(np.unique(y)) < 2:
            rows.append({"func": name, "n": n, "skipped": True, "reason": "single class"})
            continue
        pred = (p >= threshold).astype(int)
        rows.append({
            "func": name, "n": n, "skipped": False,
            "n_ch": int(y.sum()), "n_sm": int(len(y) - y.sum()),
            "ch_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred)),
        })
    return rows

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from paths import find_kaggle_checkpoint, project_subpath, resolve_config, resolve_pilot_dir, resolve_split_dir
from data.variant_data_module import VariantDataModule
from train.variant_trainer import _prepare_batch
from eval.classification_metrics import (
    contingency_table,
    prevalence_matched_table,
    print_contingency,
    ranking_metrics,
    score_summary,
    tune_threshold_max_f1,
    tune_threshold_youden,
)
from eval.embedding_extract import PooledEmbeddingExtractor
from heads.variant_classifier import cds_mask_context, head_logit
from viz.embedding_plots import generate_embedding_plots


def _score_variants(model, head_id, dm, org_idx, strand_reindexing, embed_extractor, use_cds_mask: bool):
    probs, labels, embeddings, variant_ids = [], [], [], []
    want_emb = embed_extractor is not None
    with model._device_context:
        for i, batch_np in enumerate(dm.iter_batches()):
            batch = _prepare_batch(batch_np, org_idx)
            with cds_mask_context(batch.get("cds_mask")):
                preds = model._predict(
                    model._params, model._state,
                    batch["sequences"], batch["organism_index"],
                    requested_outputs=frozenset(),
                    negative_strand_mask=batch["negative_strand_mask"],
                    strand_reindexing=strand_reindexing,
                    rng=None,
                )
            if want_emb:
                pooled = embed_extractor(
                    batch["sequences"],
                    batch["organism_index"],
                    batch.get("cds_mask"),
                    use_cds_mask=use_cds_mask,
                )
                embeddings.append(np.asarray(pooled[0], dtype=np.float32))
            logit = float(np.asarray(head_logit(preds[head_id])).reshape(-1)[0])
            probs.append(1.0 / (1.0 + np.exp(-logit)))
            labels.append(float(batch["labels"][0]))
            if want_emb:
                variant_ids.append(str(batch_np.variant_ids[0]))
            if (i + 1) % 100 == 0:
                print(f"  scored {i + 1}/{len(dm)}")
    emb_out = np.stack(embeddings) if embeddings else None
    return np.array(probs), np.array(labels), emb_out, variant_ids


def _predictions_filename(export_prefix: str, split: str) -> str:
    if export_prefix:
        return f"finetune_{export_prefix}_{split}_predictions.parquet"
    return f"finetune_{split}_predictions.parquet"


def _default_pilot_export_prefix(pilot_dir: Path | str) -> str:
    """Derive a stable prefix from pilot_dir so pilots never clobber full-cohort files."""
    name = Path(pilot_dir).name
    for prefix in ("pilot_", "pilot"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("-", "_")


def _assert_safe_prediction_write(
    out_path: Path,
    n_new: int,
    *,
    force: bool = False,
) -> None:
    """Refuse to shrink-overwrite an existing predictions parquet (unless --force-overwrite)."""
    if force or not out_path.exists():
        return
    try:
        n_old = len(pd.read_parquet(out_path, columns=["variant_id"]))
    except Exception:
        return
    if n_old > n_new:
        raise SystemExit(
            f"Refusing to overwrite {out_path} ({n_old} rows) with {n_new} rows. "
            "Use a distinct --export-prefix / config export_prefix, or pass "
            "--force-overwrite if intentional."
        )


def _save_split_predictions(
    split_df: pd.DataFrame,
    prob: np.ndarray,
    out_path: Path,
    *,
    force_overwrite: bool = False,
) -> None:
    pred_df = pd.DataFrame({
        "variant_id": split_df["variant_id"].astype(str).values,
        "label": split_df["label"].astype(int).values,
        "prob_ch": prob,
    })
    for col in ("func", "exonic_func"):
        if col in split_df.columns:
            pred_df[col] = split_df[col].values
    _assert_safe_prediction_write(out_path, len(pred_df), force=force_overwrite)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_path, index=False)
    print(f"Saved {out_path} ({len(pred_df)} variants)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument("--checkpoint", default=None, help="default: checkpoints/pilot/best")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--embedding-plots", action="store_true", default=False,
                        help="Save PCA/t-SNE/UMAP of pooled 1536-d embeddings")
    parser.add_argument("--embedding-split", choices=("test", "train", "all"), default="test",
                        help="Which variants to use for embedding plots")
    parser.add_argument("--plot-dir", default=None, help="default: results/embedding_plots")
    parser.add_argument("--plots-from-npz", default=None,
                        help="Generate plots from saved .npz (skip model inference)")
    parser.add_argument("--skip-valid", action="store_true",
                        help="Skip valid-set scoring; use prevalence-matched top-k only")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed classification threshold (overrides valid tuning)")
    parser.add_argument(
        "--export-splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=None,
        help="Save prob_ch parquets as results/finetune_{split}_predictions.parquet",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only run --export-splits (skip test metrics)",
    )
    parser.add_argument(
        "--export-prefix",
        default="",
        help="Insert prefix: finetune_{prefix}_{split}_predictions.parquet",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Allow overwriting an existing predictions parquet with fewer rows",
    )
    args = parser.parse_args()

    if args.plots_from_npz:
        npz = np.load(args.plots_from_npz, allow_pickle=True)
        plot_dir = Path(args.plot_dir) if args.plot_dir else project_subpath(PROJECT, "results/embedding_plots")
        paths = generate_embedding_plots(npz["embeddings"], npz["labels"], plot_dir)
        for p in paths:
            print(f"Saved -> {p}")
        return
    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    # Fall back to config's export_prefix so callers don't need --export-prefix explicitly
    if not args.export_prefix and cfg.get("export_prefix"):
        args.export_prefix = str(cfg["export_prefix"])

    proc = resolve_split_dir(PROJECT, cfg)
    pilot_path = resolve_pilot_dir(PROJECT, cfg)
    if pilot_path:
        proc = pilot_path
        # Pilots must never write unprefixed finetune_{split}_predictions.parquet
        # (that path is reserved for the full age-filtered cohort).
        if not args.export_prefix:
            args.export_prefix = _default_pilot_export_prefix(pilot_path)
            print(
                f"Auto export_prefix={args.export_prefix!r} from pilot_dir "
                f"(avoids overwriting full-cohort predictions)"
            )

    # Prefixed *pilot* configs: export train for CV; test is written in the eval path below.
    # Full-cohort (no pilot_dir) must not auto-export 335k train on every eval.
    if args.export_prefix and args.export_splits is None and pilot_path:
        args.export_splits = ["train"]
    test_path = proc / "test.parquet"
    if not test_path.exists():
        raise SystemExit(f"Missing {test_path}")
    test = pd.read_parquet(test_path)
    use_cds_mask = cfg["data"].get("use_cds_mask", cfg["data"].get("use_cds_filter", False))
    gtf = project_subpath(PROJECT, cfg["data"]["gtf_path"]) if use_cds_mask else None
    results_dir = project_subpath(PROJECT, cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.max_samples:
        test = test.head(args.max_samples)
    print(f"Test set: {len(test)} variants, CH rate={test['label'].mean():.3f}")

    baseline_path = results_dir / "baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)
        print("\nBaseline metrics (full test, from 02_zero_shot_baseline.py):")
        for r in baselines:
            print(f"  {r['model']}: AUROC={r['auroc']:.4f} AUPRC={r['auprc']:.4f}")

    ckpt_dir = project_subpath(PROJECT, args.checkpoint or cfg["checkpoint_dir"] + "/best")
    if not (ckpt_dir / "config.json").exists():
        print(f"\nNo checkpoint at {ckpt_dir} — run training first.")
        return

    from alphagenome.models import dna_model as ag_dna_model, dna_output
    from alphagenome_research.model import dna_model as research_dna_model
    from eval.checkpoint_loader import load_variant_checkpoint
    from heads.registry import register_variant_head

    strategy = register_variant_head(cfg)
    head_id = cfg["model"]["head_id"]

    fasta = project_subpath(PROJECT, cfg["data"]["fasta_path"])
    workspace = PROJECT.parent
    base_ckpt = cfg["model"].get("checkpoint_path")
    if base_ckpt:
        base_ckpt = project_subpath(PROJECT, base_ckpt)
    else:
        base_ckpt = find_kaggle_checkpoint(workspace, cfg["model"]["model_version"])

    organism = ag_dna_model.Organism[cfg["model"]["organism"]]
    organism_settings = {
        organism: research_dna_model.OrganismSettings(fasta_path=str(fasta)),
    }
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    device = gpus[0] if gpus else jax.devices("cpu")[0]

    model = load_variant_checkpoint(
        ckpt_dir,
        cfg,
        base_model_version=cfg["model"]["model_version"],
        organism_settings=organism_settings,
        device=device,
        base_checkpoint_path=base_ckpt,
        init_seq_len=cfg["model"]["window_size"],
    )

    embed_extractor = None
    if args.embedding_plots:
        embed_extractor = PooledEmbeddingExtractor(model._params, model._state)

    import tempfile

    test_tmp: Path | None = None
    if args.max_samples:
        fd, tmp_name = tempfile.mkstemp(suffix=".parquet", dir="/tmp")
        import os
        os.close(fd)
        test_tmp = Path(tmp_name)
        test.to_parquet(test_tmp)
        dm_path = test_tmp
    else:
        dm_path = test_path

    test_dm = VariantDataModule(
        dm_path, fasta, window_size=cfg["model"]["window_size"], batch_size=1,
        gtf_path=gtf, use_cds_mask=use_cds_mask,
    )
    org_idx = int(research_dna_model.convert_to_organism_index(organism))
    strand_reindexing = jax.device_put(model._metadata[organism].strand_reindexing)

    if args.export_splits:
        import os
        import tempfile
        for split in args.export_splits:
            split_path = proc / f"{split}.parquet"
            if not split_path.exists():
                raise SystemExit(f"Missing {split_path}")
            split_df = pd.read_parquet(split_path)
            dm_path: Path | str = split_path
            split_tmp: Path | None = None
            if args.max_samples:
                split_df = split_df.head(args.max_samples)
                fd, tmp_name = tempfile.mkstemp(suffix=".parquet", dir="/tmp")
                os.close(fd)
                split_tmp = Path(tmp_name)
                split_df.to_parquet(split_tmp)
                dm_path = split_tmp
            print(f"\nExporting {split}: {split_path} ({len(split_df)} variants)")
            split_dm = VariantDataModule(
                dm_path, fasta, window_size=cfg["model"]["window_size"], batch_size=1,
                gtf_path=gtf, use_cds_mask=use_cds_mask,
            )
            p_split, _, _, _ = _score_variants(
                model, head_id, split_dm, org_idx, strand_reindexing, None, use_cds_mask,
            )
            out_name = _predictions_filename(args.export_prefix, split)
            _save_split_predictions(
                split_df,
                p_split,
                results_dir / out_name,
                force_overwrite=args.force_overwrite,
            )
            if split_tmp is not None:
                split_tmp.unlink(missing_ok=True)
        if args.export_only:
            return

    p_valid, y_valid = None, None
    valid_path = proc / "valid.parquet"
    score_valid = (
        not args.skip_valid
        and valid_path.exists()
        and not _same_variant_split(test, pd.read_parquet(valid_path, columns=["variant_id"]))
    )
    if not args.skip_valid and valid_path.exists() and not score_valid:
        print(
            "\nSkipping valid scoring: pilot valid.parquet == test.parquet "
            "(same holdout). Use prevalence-matched threshold or --threshold."
        )
    elif score_valid:
        print(f"\nScoring validation set for threshold tuning: {valid_path}")
        valid_dm = VariantDataModule(
            valid_path, fasta, window_size=cfg["model"]["window_size"], batch_size=1,
            gtf_path=gtf, use_cds_mask=use_cds_mask,
        )
        p_valid, y_valid, _, _ = _score_variants(
            model, head_id, valid_dm, org_idx, strand_reindexing, None, use_cds_mask,
        )

    print(f"\nScoring test set: {test_path} ({len(test)} variants)")
    p, y, test_embeddings, test_variant_ids = _score_variants(
        model, head_id, test_dm, org_idx, strand_reindexing, embed_extractor, use_cds_mask,
    )
    if test_tmp is not None:
        test_tmp.unlink(missing_ok=True)

    rank = ranking_metrics(y, p)
    scores = score_summary(p, y)
    if args.threshold is not None:
        tuned_thr = float(args.threshold)
        thr_source = "fixed"
    elif p_valid is not None and y_valid is not None:
        tuned_thr = tune_threshold_youden(y_valid, p_valid)
        thr_source = "valid_youden"
    else:
        tuned_thr = None
        thr_source = "none"

    cm_default = contingency_table(y, p, threshold=0.5)
    cm_prevalence = prevalence_matched_table(y, p)
    cm_tuned = contingency_table(y, p, threshold=tuned_thr) if tuned_thr is not None else None

    primary_thr = tuned_thr if tuned_thr is not None else cm_prevalence["threshold"]
    yhat = (p >= primary_thr).astype(int)
    metrics = {
        **rank,
        "checkpoint": str(ckpt_dir),
        "model": "sm_ch_classifier",
        "finetune_strategy": strategy,
        "score_summary": scores,
        "classification_threshold": {
            "primary": float(primary_thr),
            "source": thr_source if tuned_thr is not None else "prevalence_matched",
            "valid_youden": float(tuned_thr) if tuned_thr is not None else None,
            "valid_max_f1": float(tune_threshold_max_f1(y_valid, p_valid))
            if p_valid is not None and y_valid is not None else None,
        },
        "confusion_matrix_at_0_5": {
            "labels": {"negative": "SM (0)", "positive": "CH (1)"},
            **{k: cm_default[k] for k in ("threshold", "tn", "fp", "fn", "tp", "f1", "balanced_accuracy")},
            "note": "Reference only; misleading when all prob_ch < 0.5",
        },
        "confusion_matrix_prevalence_matched": {
            "labels": {"negative": "SM (0)", "positive": "CH (1)"},
            **{k: cm_prevalence[k] for k in (
                "method", "k", "threshold", "tn", "fp", "fn", "tp", "f1", "balanced_accuracy",
                "sensitivity", "specificity",
            )},
        },
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
    }
    if cm_tuned is not None:
        metrics["confusion_matrix_valid_tuned"] = {
            "labels": {"negative": "SM (0)", "positive": "CH (1)"},
            **{k: cm_tuned[k] for k in (
                "threshold", "tn", "fp", "fn", "tp", "f1", "balanced_accuracy",
                "sensitivity", "specificity",
            )},
        }
    if "func" in test.columns:
        metrics["by_func"] = _stratify_metrics(test, p, threshold=primary_thr)
    pred_df = pd.DataFrame({
        "variant_id": test["variant_id"].astype(str).values,
        "label": y.astype(int),
        "prob_ch": p,
        "pred_ch": yhat.astype(int),
        "pred_ch_at_0_5": (p >= 0.5).astype(int),
        "pred_ch_prevalence_matched": (
            np.zeros(len(p), dtype=int)
            if int(y.sum()) == 0
            else (np.argsort(np.argsort(-p)) < int(y.sum())).astype(int)
        ),
    })
    if "func" in test.columns:
        pred_df["func"] = test["func"].values
    if "exonic_func" in test.columns:
        pred_df["exonic_func"] = test["exonic_func"].values
    pred_path = results_dir / _predictions_filename(args.export_prefix, "test")
    _assert_safe_prediction_write(
        pred_path, len(pred_df), force=args.force_overwrite,
    )
    pred_df.to_parquet(pred_path, index=False)
    print(f"Saved {pred_path} ({len(pred_df)} variants)")

    primary_cm = cm_tuned if cm_tuned is not None else cm_prevalence
    cm_payload = {
        "primary_threshold": float(primary_thr),
        "threshold_source": metrics["classification_threshold"]["source"],
        "labels": {"true_negative": "SM (0)", "true_positive": "CH (1)"},
        "rows": "true label",
        "columns": "predicted label",
        "at_0_5": cm_default,
        "prevalence_matched": cm_prevalence,
        "valid_tuned": cm_tuned,
        "table_primary": [
            {"true": "SM", "pred_sm": primary_cm["tn"], "pred_ch": primary_cm["fp"]},
            {"true": "CH", "pred_sm": primary_cm["fn"], "pred_ch": primary_cm["tp"]},
        ],
    }
    cm_path = results_dir / "finetune_confusion_matrix.json"
    with open(cm_path, "w") as f:
        json.dump(cm_payload, f, indent=2)

    out = results_dir / cfg.get("metrics_file", "finetune_metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nFine-tuned model (test) — ranking metrics (threshold-free):")
    print(f"  AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f}")
    ss = scores["prob_ch_all"]
    print(
        f"  prob_ch: min={ss['min']:.3f} max={ss['max']:.3f} mean={ss['mean']:.3f} "
        f"n>=0.5={ss['n_ge_0_5']}"
    )
    if scores["prob_ch_true_ch"]["mean"] is not None:
        print(
            f"  mean prob_ch: true SM={scores['prob_ch_true_sm']['mean']:.3f} "
            f"true CH={scores['prob_ch_true_ch']['mean']:.3f}"
        )
    if cm_tuned is not None:
        print_contingency("Contingency (valid-tuned threshold, applied to test)", cm_tuned)
    print_contingency("Contingency (prevalence-matched top-k)", cm_prevalence)
    if ss["n_ge_0_5"] == 0:
        print("\n  Note: no prob_ch >= 0.5 — contingency at 0.5 is not meaningful; see tables above.")
    else:
        print_contingency("Contingency (reference threshold=0.5)", cm_default)
    if metrics.get("by_func"):
        print(f"\nBy Func.refGene (min n=20):")
        for r in metrics["by_func"]:
            if r.get("skipped"):
                print(f"  {r['func']:10s} n={r['n']:3d}  skipped ({r.get('reason','')})")
            else:
                print(f"  {r['func']:10s} n={r['n']:3d}  AUROC={r['auroc']:.3f}  AUPRC={r['auprc']:.3f}  F1={r['f1']:.3f}")
    print(f"  Saved -> {out}")
    print(f"  Predictions -> {pred_path}")
    print(f"  Confusion matrix -> {cm_path}")

    embed_dir = Path(args.plot_dir) if args.plot_dir else results_dir / "embedding_plots"
    embed_prefix = f"{Path(cfg['checkpoint_dir']).name}_"
    if not args.embedding_plots:
        return

    if test_embeddings is None:
        raise SystemExit("Embedding extraction failed (test_embeddings is None)")

    npz_path = results_dir / f"finetune_embeddings_{Path(cfg['checkpoint_dir']).name}.npz"
    plot_embeddings = test_embeddings
    plot_labels = y.astype(int)
    plot_ids = np.array(test_variant_ids)
    plot_split = "test"

    if args.embedding_split in ("train", "all"):
        train_df = pd.read_parquet(proc / "train.parquet")
        if args.max_samples:
            train_df = train_df.head(args.max_samples)
        train_tmp = proc / "_eval_train_subset.parquet"
        train_df.to_parquet(train_tmp)
        train_dm = VariantDataModule(
            train_tmp, fasta, window_size=cfg["model"]["window_size"], batch_size=1,
            gtf_path=gtf, use_cds_mask=use_cds_mask,
        )
        print(f"Scoring train embeddings: {len(train_dm)} variants")
        _, y_tr, emb_tr, ids_tr = _score_variants(
            model, head_id, train_dm, org_idx, strand_reindexing, embed_extractor, use_cds_mask,
        )
        train_tmp.unlink(missing_ok=True)
        if args.embedding_split == "all":
            plot_embeddings = np.vstack([emb_tr, plot_embeddings])
            plot_labels = np.concatenate([y_tr.astype(int), plot_labels])
            plot_ids = np.concatenate([np.array(ids_tr), plot_ids])
            plot_split = "all"
        else:
            plot_embeddings, plot_labels, plot_ids = emb_tr, y_tr.astype(int), np.array(ids_tr)
            plot_split = "train"

    np.savez(
        npz_path,
        embeddings=plot_embeddings,
        labels=plot_labels,
        variant_id=plot_ids,
        split=plot_split,
        embedding_dim=plot_embeddings.shape[1],
    )
    print(f"  Embeddings -> {npz_path} ({plot_embeddings.shape[0]} x {plot_embeddings.shape[1]})")

    if plot_embeddings.shape[0] < 5:
        print("Skipping embedding plots: need at least 5 variants")
    else:
        plot_paths = generate_embedding_plots(
            plot_embeddings, plot_labels, embed_dir, prefix=f"{embed_prefix}_",
        )
        for pp in plot_paths:
            print(f"  Plot -> {pp}")

if __name__ == "__main__":
    main()
