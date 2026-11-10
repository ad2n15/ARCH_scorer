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


class _Head(torch.nn.Module):
    """Stand-in head that returns a fixed logit per row."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, emb):
        return self.logits[: len(emb)]


def _fake_scorer(monkeypatch, logits):
    """A Scorer with the released ntv3 heads' metadata but no backbone and no network."""
    from arch_scorer.finetuned import predict as P

    s = P.Scorer.__new__(P.Scorer)
    s.name = "ntv3"
    s.tasks, s.meta = {}, {}
    real = P.Scorer.__init__

    class _BB:
        def __init__(self, *a, **k):
            self.batch_size, self.device = 4, "cpu"

    monkeypatch.setitem(P.BACKBONE_CLASSES, "ntv3", _BB)
    real(s, "ntv3")
    for t in s.tasks:
        s.tasks[t]["head"] = _Head(logits)
    ann = {"20-32437607-T-G": ("ASXL1", ["ASXL1"], "intronic", 12.0),
           "1-1000000-C-T": ("HES4", ["HES4"], "intergenic", 2.0),
           "2-25234373-C-T": ("DNMT3A", ["DNMT3A"], "exonic", None)}

    def fake_annotate(v):
        g, ov, func, cadd = ann[v.id]
        return {"variant_id": v.id, "chrom": v.chrom, "pos": v.pos, "ref": v.ref, "alt": v.alt,
                "mutation_type_raw": A.mutation_type(v), "mutation_type_sbs6": A.sbs6(v), "notes": [],
                "func": func, "gene": g, "overlapping_genes": ov, "cadd_phred": cadd,
                "exonic_func": "nonsynonymous SNV" if func == "exonic" else "."}

    monkeypatch.setattr(P.A, "annotate", fake_annotate)
    rec = {"genome": {"ac": 18, "an": 1000, "homozygote_count": 0,
                      "age_distribution": {"het": {"bin_freq": [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                                                   "n_smaller": 0, "n_larger": 2}},
                      "quality_metrics": {"allele_balance": {"alt": {"bin_freq": [0] * 4 + [7, 7, 1, 2, 1] + [0] * 11}}}}}
    monkeypatch.setattr(P.G, "fetch", lambda ids, progress=None: [
        {c: P.G.summarise(rec if k == 0 else None).get(c, np.nan) for c in P.G.COLUMNS} for k in range(len(ids))])
    monkeypatch.setattr(P.Scorer, "embed", lambda self, vs, progress=None: (
        torch.zeros(len(vs), 1536), [v.ref for v in vs]))
    return s


def test_thresholds_from_quantiles(monkeypatch):
    s = _fake_scorer(monkeypatch, [0.0, 0.0, 0.0])
    for t, m in s.meta.items():
        q = json.loads((MODELS / "ntv3" / t / "score_quantiles.json").read_text())
        for k, thr in m["threshold"].items():
            assert q[k][0] <= thr <= q[k][-1]
        assert 0 < m["prevalence"] < 0.5


def test_flags_summary_and_html(monkeypatch, tmp_path):
    from arch_scorer.finetuned.report import summary_table, write_html

    s = _fake_scorer(monkeypatch, [6.0, -6.0, 6.0])
    df = s.score(["20-32437607-T-G", "1-1000000-C-T", "2-25234373-C-T", "not-a-variant"])
    for t in TASKS:
        for _, r in df.iterrows():
            score = r[f"{t}_score"]
            if pd.isna(score):
                assert r[f"{t}_flag"] in ("", None) or pd.isna(r[f"{t}_flag"])
                continue
            kind = r[f"{t}_score_type"]
            assert (r[f"{t}_flag"] == "high") == (score >= s.meta[t]["threshold"][kind])
            assert kind == ("fused" if not pd.isna(r[f"{t}_ntv3_plus_annotations"]) else "finetuned")
    # ASXL1 variant: very high logit -> flagged; HES4 is outside the 21 genes -> no 21-gene score
    asxl1 = df[df.variant_id == "20-32437607-T-G"].iloc[0]
    assert asxl1["arch21_noncoding_flag"] == "high"
    hes4 = df[df.variant_id == "1-1000000-C-T"].iloc[0]
    assert pd.isna(hes4["arch21_cnc_score"]) and hes4["genomewide_flag"] == "-"
    # missing CADD -> the fine-tuned score is the headline score
    assert df[df.variant_id == "2-25234373-C-T"].iloc[0]["arch21_cnc_score_type"] == "finetuned"

    text = summary_table(df, TASKS, s.meta)
    assert "non-coding model" in text and "coding + non-coding model" in text and "HIGH" in text
    assert "not scored" in text and "(18)" in text and "NC" not in text.split()
    assert max(len(line) for line in text.splitlines()[:7]) <= 130

    df.loc[0, "notes"] = "<script>x</script>"
    out = write_html(df, TASKS, s.meta, "ntv3", tmp_path / "r.html")
    page = out.read_text()
    assert page.count('class="card"') == len(df)
    assert "<script>x</script>" not in page and "&lt;script&gt;" in page
    assert "Allele balance: median <b>" in page and "not in gnomAD v4.1" in page


def test_gnomad_summaries():
    from arch_scorer.finetuned import gnomad as G

    assert G.ab_median([0, 0, 0, 0, 7, 7, 1, 2, 1] + [0] * 11) == pytest.approx(0.275)  # 18 carriers: median (9.5th) falls in 0.25-0.30
    assert G.ab_median([0] * 20) != G.ab_median([0] * 20)  # nan
    # 2 carriers > 80 and three in 35-40, 45-50, 70-75 -> median carrier is in 70-75
    assert G.age_median([0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 0, 2) == 72.5
    two = {"ac": 1, "an": 10, "homozygote_count": 0}
    r = G.summarise({"exome": two, "genome": two})
    assert r["gnomad_ac"] == 2 and r["gnomad_an"] == 20 and r["gnomad_af"] == 0.1
    assert G.summarise({"exome": {"ac": 0, "an": 5}})["gnomad_status"].startswith("not observed")
