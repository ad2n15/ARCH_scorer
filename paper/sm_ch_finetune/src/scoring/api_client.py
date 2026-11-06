"""AlphaGenome API client for variant track scoring (not used for fine-tuning)."""
from __future__ import annotations

import os

from alphagenome.models import dna_client, dna_model


def load_api_client(
    *,
    api_key: str | None = None,
    key_env: str = "ALPHA_GENOME_API_KEY",
    model_version: str | None = None,
) -> dna_client.DnaClient:
    """Create DnaClient using ALPHA_GENOME_API_KEY from the environment."""
    key = api_key or os.environ.get(key_env)
    if not key:
        raise EnvironmentError(
            f"{key_env} not set. Add it to alphagenome/.env (login node with internet required)."
        )
    version = None
    if model_version and model_version.lower() not in ("default", "null", "none"):
        version = dna_model.ModelVersion[model_version.upper()]
    print(f"AlphaGenome API client (model_version={version or 'server default'})")
    return dna_client.create(key, model_version=version)
