#!/usr/bin/env python3
"""Copy AG heads-only fine-tune Orbax checkpoint into ARCH_scorer/artifacts.

Default: ARCH21 NC (published test AUROC ~0.761).

  ./run_binemo_AMD.sh ARCH_scorer/scripts/export_ag_finetuned_head.py
  ./run_binemo_AMD.sh ARCH_scorer/scripts/export_ag_finetuned_head.py --cohort cnc
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SM = ROOT / "sm_ch_finetune"
OUT = ROOT / "ARCH_scorer" / "artifacts"

COHORTS = {
    "nc": {
        "src": SM
        / "checkpoints"
        / "arch21_mane_noncoding_gvs_nocds_64kb_bs8_5ep"
        / "best",
        "name": "ag_finetuned_arch21_nc_heads_only",
        "metrics": SM / "results" / "finetune_metrics_arch21_mane_noncoding_gvs.json",
    },
    "cnc": {
        "src": SM
        / "checkpoints"
        / "arch21_mane_cnc_gvs_nocds_64kb_bs8_5ep"
        / "best",
        "name": "ag_finetuned_arch21_cnc_heads_only",
        "metrics": SM / "results" / "finetune_metrics_arch21_mane_cnc_gvs.json",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=sorted(COHORTS), default="nc")
    args = ap.parse_args()
    spec = COHORTS[args.cohort]
    src: Path = spec["src"]
    if not (src / "config.json").is_file() or not (src / "checkpoint").is_dir():
        raise SystemExit(f"Missing AG FT head at {src}")

    dst = OUT / spec["name"]
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copy2(src / "config.json", dst / "config.json")
    shutil.copytree(src / "checkpoint", dst / "checkpoint")

    meta = {
        "name": spec["name"],
        "cohort": args.cohort,
        "source_checkpoint": str(src),
        "finetune_strategy": "heads_only",
        "head": "sm_ch_classifier",
        "format": "orbax",
        "backbone": "fold_1 (not bundled)",
    }
    mpath = spec["metrics"]
    if mpath.is_file():
        meta["held_out_test"] = json.loads(mpath.read_text())
    meta_path = OUT / f"{spec['name']}.meta.json"
    # Keep hand-curated meta if present for nc; refresh compact sidecar always
    if args.cohort == "cnc" or not meta_path.is_file():
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {dst} ({sum(p.stat().st_size for p in dst.rglob('*') if p.is_file())} bytes)")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
