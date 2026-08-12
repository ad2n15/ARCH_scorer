"""ARCH21 noncoding somatic vs germline scorer (AlphaGenome companion)."""

from .pipeline import ArchScorer, ScoreResult

__all__ = ["ArchScorer", "ScoreResult"]
__version__ = "0.1.0"

# Production: emb_baseline (best) · api_baseline (no local emb).
# Interpretability: api_only RF + --explain → top-10 SHAP ranks of CD34+ tracks.
