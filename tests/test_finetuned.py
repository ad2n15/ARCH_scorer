"""Offline checks of the released fine-tuned model files (no backbone download, no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from arch_scorer.finetuned import annotate as A
from arch_scorer.finetuned.fusion import FusionLR
from arch_scorer.finetuned.heads import load_head
from arch_scorer.finetuned.predict import BACKBONES, MODELS, TASKS, fusion_frame

EMBED = {"ntv3": 1536, "alphagenome": 1536, "borzoi": 1920}
PAPER_AUROC = {("ntv3", "arch21_noncoding"): 0.789, ("alphagenome", "arch21_noncoding"): 0.761,
               ("borzoi", "arch21_noncoding"): 0.746, ("ntv3", "genomewide"): 0.579,
               ("alphagenome", "genomewide"): 0.569, ("borzoi", "genomewide"): 0.559}


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("task", TASKS)
def test_model_files(backbone, task):
    d = MODELS / backbone / task
    head, cfg = load_head(d)
    assert cfg["embed_dim"] == EMBED[backbone] and cfg["window_size"] == 65536
    torch.manual_seed(0)
    p = torch.sigmoid(head(0.01 * torch.randn(3, cfg["embed_dim"]))).detach().numpy()
    assert p.shape == (3,) and np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    q = json.loads((d / "score_quantiles.json").read_text())
    assert len(q["finetuned"]) == 101 and np.all(np.diff(q["finetuned"]) >= 0)
    m = json.loads((d / "test_metrics.json").read_text())
    if (backbone, task) in PAPER_AUROC:
        assert round(m["auroc_finetuned"], 3) == PAPER_AUROC[(backbone, task)]


@pytest.mark.parametrize("task", TASKS)
def test_fusion_inputs(task):
    fus = FusionLR.load(MODELS / "ntv3" / task / "fusion.json")
    ann = pd.DataFrame([{"gene": "ASXL1", "overlapping_genes": ["ASXL1"], "func": "UTR3", "exonic_func": ".",
                         "cadd_phred": 5.0, "mutation_type_sbs6": "T>G", "mutation_type_raw": "T>G"}])
    cfg = json.loads((MODELS / "ntv3" / task / "config.json").read_text())
    X = fusion_frame(task, ann, np.array([0.3]), cfg.get("genes"))
    p = fus.predict(X)
    assert p.shape == (1,) and 0 < p[0] < 1


def test_parse_variant():
    for text in ("20-32437597-T-G", "chr20:32437597 T>G", "20 32437597 T G"):
        v = A.parse_variant(text)
        assert (v.chrom, v.pos, v.ref, v.alt) == ("20", 32437597, "T", "G")
    assert A.sbs6(v) == "T>G" and A.sbs6(A.parse_variant("1-100-A-G")) == "T>C"
    with pytest.raises(ValueError):
        A.parse_variant("20-32437597-TA-G")
