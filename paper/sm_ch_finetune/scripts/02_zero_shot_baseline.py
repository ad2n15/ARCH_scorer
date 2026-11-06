#!/usr/bin/env python3
"""Feature baselines and optional AlphaGenome API zero-shot scoring."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

def metrics(y, prob, pred):
    return {"auroc": float(roc_auc_score(y,prob)), "auprc": float(average_precision_score(y,prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y,pred)), "f1": float(f1_score(y,pred)), "n": len(y)}

def eval_split(name, model, X, y):
    prob = model.predict_proba(X)[:,1]; pred = (prob>=0.5).astype(int)
    m = metrics(y, prob, pred); m["model"] = name; return m

def build_gem_features(df):
    feat = pd.DataFrame(index=df.index)
    feat["cadd"] = df["cadd"].fillna(df["cadd"].median())
    feat["is_exonic"] = (df["func"]=="exonic").astype(int)
    feat["is_intronic"] = (df["func"]=="intronic").astype(int)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    mt = enc.fit_transform(df[["mutation_type"]])
    mt_df = pd.DataFrame(mt, columns=[f"mt_{c}" for c in enc.categories_[0]], index=df.index)
    enc2 = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ef = enc2.fit_transform(df[["exonic_func"]].fillna("."))
    ef_df = pd.DataFrame(ef, columns=[f"ef_{c}" for c in enc2.categories_[0]], index=df.index)
    out = pd.concat([feat, mt_df, ef_df], axis=1)
    return out, list(out.columns)

def run_baselines(train, test):
    results, y_tr, y_te = [], train["label"].values, test["label"].values
    Xtr = train[["cadd"]].fillna(train["cadd"].median()).values
    Xte = test[["cadd"]].fillna(train["cadd"].median()).values
    m = LogisticRegression(max_iter=1000); m.fit(Xtr, y_tr); results.append(eval_split("cadd_only", m, Xte, y_te))
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    m2 = LogisticRegression(max_iter=1000)
    m2.fit(enc.fit_transform(train[["exonic_func"]].fillna(".")), y_tr)
    results.append(eval_split("vep_only", m2, enc.transform(test[["exonic_func"]].fillna(".")), y_te))
    Xtr_df, cols = build_gem_features(train); Xte_df, _ = build_gem_features(test)
    Xte_df = Xte_df.reindex(columns=cols, fill_value=0)
    m3 = LogisticRegression(max_iter=2000, C=0.5); m3.fit(Xtr_df.values, y_tr)
    results.append(eval_split("gem_style_lr", m3, Xte_df.values, y_te))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument("--run-api", action="store_true")
    args = parser.parse_args()
    import sys
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir / "src"))
    from paths import project_subpath, resolve_config, resolve_split_dir
    with open(resolve_config(args.config, project_dir)) as f: cfg = yaml.safe_load(f)
    proc = resolve_split_dir(project_dir, cfg)
    if not (proc/"train.parquet").exists():
        raise SystemExit("Run 01_prepare_pilot_data.py first")
    train, test = pd.read_parquet(proc/"train.parquet"), pd.read_parquet(proc/"test.parquet")
    results = run_baselines(train, test)
    results_dir = project_subpath(project_dir, cfg["results_dir"]); results_dir.mkdir(parents=True, exist_ok=True)
    if args.run_api:
        raise SystemExit(
            "API scoring moved to scripts/05_score_variant_tracks.py (config/track_scoring.yaml)"
        )
    out = results_dir / "baseline_metrics.json"
    with open(out,"w") as f: json.dump(results, f, indent=2)
    for r in results: print(f"{r['model']}: AUROC={r['auroc']:.4f} AUPRC={r['auprc']:.4f}")
    print(f"Saved {out}")

if __name__ == "__main__": main()
