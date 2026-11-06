#!/usr/bin/env python3
"""Check that live inference reproduces the stored held-out scores of each released head.

Embeds n held-out test variants per task with the frozen backbone, applies the exported head, and compares
with the probabilities written at the end of training (checks/test_<backbone>_<task>.csv.gz, made by
export_finetuned_models.py). Uses a local FASTA (ARCH_FASTA), so it runs on offline GPU nodes.

  ARCH_FASTA=/path/hg38.fa python3 scripts/verify_finetuned.py --backbone ntv3 -n 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from arch_scorer.finetuned import annotate as A  # noqa: E402
from arch_scorer.finetuned.backbones import BACKBONE_CLASSES, WINDOW  # noqa: E402
from arch_scorer.finetuned.heads import load_head  # noqa: E402
from arch_scorer.finetuned.predict import MODELS, TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", choices=list(BACKBONE_CLASSES), required=True)
ap.add_argument("-n", type=int, default=20)
ap.add_argument("--model-path", default=None)
ap.add_argument("--device", default=None, help="torch device for NTv3/Borzoi, e.g. cpu")
a = ap.parse_args()

t0 = time.time()
bb = BACKBONE_CLASSES[a.backbone](a.model_path, a.device)
print(f"{a.backbone} loaded on {bb.device} in {time.time() - t0:.0f}s", flush=True)
worst = 0.0
for task in TASKS:
    ref = pd.read_csv(HERE / f"checks/test_{a.backbone}_{task}.csv.gz").sample(a.n, random_state=1)
    head, _ = load_head(MODELS / a.backbone / task)
    vs = [A.parse_variant(v) for v in ref["variant_id"]]
    wins = [A.variant_window(v, WINDOW) for v in vs]
    emb = torch.cat([bb.embed([w[0] for w in wins[i:i + bb.batch_size]], [w[1] for w in wins[i:i + bb.batch_size]])
                     for i in range(0, len(wins), bb.batch_size)])
    with torch.no_grad():
        p = torch.sigmoid(head(emb)).numpy()
    d = np.abs(p - ref["finetune_prob"].to_numpy())
    worst = max(worst, d.max())
    print(f"{a.backbone} {task}: n={a.n} max |live - stored| = {d.max():.2e}, median {np.median(d):.2e}; "
          f"REF matches genome {np.mean([w[2] == v.ref for w, v in zip(wins, vs)]):.0%}", flush=True)
print(f"WORST {a.backbone} {worst:.2e}")
