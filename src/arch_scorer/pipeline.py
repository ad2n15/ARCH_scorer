"""Score variants with full-fit emb+baseline or api+baseline XGB."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .ag_tracks import score_api_tracks
from .annovar import annotate_func_refgene
from .cadd_api import fetch_cadd_phred
from .genes import map_variant_to_gene
from .sbs6 import to_sbs6

ARTIFACTS = {
    "emb_baseline": "emb_baseline_xgb_gene_onehot.joblib",
    "emb_baseline_xgb": "emb_baseline_xgb_gene_onehot.joblib",
    "api_baseline": "api_baseline_xgb_gene_onehot.joblib",
    "api_baseline_xgb": "api_baseline_xgb_gene_onehot.joblib",
}


def _artifacts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts"


def resolve_artifact(model: str, artifact_path: str | Path | None = None) -> Path:
    if artifact_path:
        return Path(artifact_path)
    key = model.strip().lower().replace("-", "_").replace("+", "_")
    # normalize aliases
    aliases = {
        "emb": "emb_baseline",
        "emb_base": "emb_baseline",
        "api": "api_baseline",
        "api_base": "api_baseline",
    }
    key = aliases.get(key, key)
    if key not in ARTIFACTS:
        raise ValueError(f"Unknown model {model!r}; choose emb_baseline or api_baseline")
    path = _artifacts_dir() / ARTIFACTS[key]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/export_models_from_sm_ch_finetune.py"
        )
    return path


@dataclass
class ScoreResult:
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str
    cadd_phred: float
    mutation_type: str
    func_refgene: str
    p_somatic: float
    call_somatic_pm: bool
    pm_threshold: float
    model_name: str
    api_tracks: dict[str, float] = field(default_factory=dict)


class ArchScorer:
    """Full-fit production scorer (emb+baseline or api+baseline)."""

    def __init__(
        self,
        model: str = "emb_baseline",
        artifact_path: str | Path | None = None,
        *,
        annovar_dir: str | Path | None = None,
        fetch_cadd: bool = True,
        fetch_annovar: bool = True,
        api_key: str | None = None,
    ):
        path = resolve_artifact(model, artifact_path)
        self.artifact: dict[str, Any] = joblib.load(path)
        self.annovar_dir = annovar_dir
        self.fetch_cadd = fetch_cadd
        self.fetch_annovar = fetch_annovar
        self.api_key = api_key
        self.pm_threshold = float(self.artifact["pm_threshold"])
        self.model_name = str(self.artifact["model_name"])
        self.feature_set = str(self.artifact.get("feature_set") or "")
        self.emb_dim = self.artifact.get("emb_dim")
        self.api_feats: list[str] | None = self.artifact.get("api_feats")

    def _baseline_block(
        self, *, gene: str, cadd_phred: float, mutation_type: str, func_refgene: str
    ) -> np.ndarray:
        base = pd.DataFrame(
            [
                {
                    "cadd_phred": float(cadd_phred),
                    "mutation_type": mutation_type,
                    "Func.refGene": func_refgene,
                    "gene": gene,
                }
            ]
        )
        return np.asarray(self.artifact["base_ct"].transform(base), dtype=np.float64)

    def score(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        *,
        emb_diff: np.ndarray | None = None,
        api_tracks: dict[str, float] | None = None,
        gene: str | None = None,
        cadd_phred: float | None = None,
        func_refgene: str | None = None,
        fetch_api: bool = True,
    ) -> ScoreResult:
        g = map_variant_to_gene(chrom, pos, gene=gene)
        mut = to_sbs6(ref, alt)
        if cadd_phred is None:
            if not self.fetch_cadd:
                raise ValueError("cadd_phred required when fetch_cadd=False")
            cadd_phred = fetch_cadd_phred(chrom, pos, ref, alt)
        if func_refgene is None:
            if not self.fetch_annovar:
                raise ValueError("func_refgene required when fetch_annovar=False")
            func_refgene = annotate_func_refgene(
                chrom, pos, ref, alt, annovar_dir=self.annovar_dir
            )

        tracks_out: dict[str, float] = {}
        if self.model_name.startswith("emb") or self.feature_set.startswith("emb"):
            if emb_diff is None:
                raise ValueError(
                    "emb_baseline requires emb_diff (npy) or run extract_fold1_emb_diff.py"
                )
            emb = np.asarray(emb_diff, dtype=np.float64).reshape(1, -1)
            if self.emb_dim and emb.shape[1] != int(self.emb_dim):
                raise ValueError(f"emb_diff dim={emb.shape[1]} != {self.emb_dim}")
            z_front = self.artifact["emb_pca"].transform(emb)
        else:
            feats = self.api_feats or []
            if api_tracks is None:
                if not fetch_api:
                    raise ValueError("api_tracks required when fetch_api=False")
                api_tracks = score_api_tracks(
                    chrom, pos, ref, alt, g, api_key=self.api_key, feature_names=feats
                )
            tracks_out = dict(api_tracks)
            row = pd.DataFrame([{c: float(api_tracks[c]) for c in feats}])
            z_front = np.asarray(self.artifact["api_imp"].transform(row), dtype=np.float64)

        z_base = self._baseline_block(
            gene=g,
            cadd_phred=float(cadd_phred),
            mutation_type=mut,
            func_refgene=str(func_refgene),
        )
        z = np.hstack([z_front, z_base])
        p = float(self.artifact["clf"].predict_proba(z)[0, 1])
        return ScoreResult(
            chrom=str(chrom).replace("chr", ""),
            pos=int(pos),
            ref=str(ref).upper(),
            alt=str(alt).upper(),
            gene=g,
            cadd_phred=float(cadd_phred),
            mutation_type=mut,
            func_refgene=str(func_refgene),
            p_somatic=p,
            call_somatic_pm=p >= self.pm_threshold,
            pm_threshold=self.pm_threshold,
            model_name=self.model_name,
            api_tracks=tracks_out,
        )
