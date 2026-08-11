"""ARCH21 noncoding somatic vs germline scorer (AlphaGenome companion)."""

from .pipeline import ArchScorer, ScoreResult

__all__ = ["ArchScorer", "ScoreResult"]
__version__ = "0.1.0"

# Production model: emb DIFF + CADD + SBS6 + Func.refGene + gene → XGB.
# API RF track scores are for interpretability only (not shipped as default).
