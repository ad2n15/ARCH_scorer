"""Extract pooled 1536-d embeddings from a loaded alphagenome-ft model."""
from __future__ import annotations

from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import model as model_lib
from alphagenome_research.model.metadata import metadata as metadata_lib

from heads.variant_classifier import pool_sequence_embedding


def _alphagenome_metadata() -> dict:
    return {org: metadata_lib.load(org) for org in ag_dna_model.Organism}


def build_pooled_embedding_transform():
    metadata = _alphagenome_metadata()
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    hk.mixed_precision.set_policy(model_lib.AlphaGenome, policy)

    @hk.transform_with_state
    def forward(dna_sequence, organism_index, cds_mask, use_cds_mask: bool):
        alphagenome = model_lib.AlphaGenome(metadata)
        _, embeddings = alphagenome(dna_sequence, organism_index)
        x = embeddings.get_sequence_embeddings(resolution=1)
        mask = cds_mask if use_cds_mask else None
        return pool_sequence_embedding(x, cds_mask=mask)

    return forward


class PooledEmbeddingExtractor:
    """JIT helper: AlphaGenome backbone -> pooled [B, 1536] vector."""

    def __init__(self, params, state):
        self._transform = build_pooled_embedding_transform()
        self._params = params
        self._state = state

        @partial(jax.jit, static_argnames=("use_cds_mask",))
        def _run(sequences, organism_index, cds_mask, *, use_cds_mask: bool):
            pooled, _ = self._transform.apply(
                self._params,
                self._state,
                None,
                sequences,
                organism_index,
                cds_mask,
                use_cds_mask,
            )
            return pooled

        self._run = _run

    def __call__(
        self,
        sequences: jnp.ndarray,
        organism_index: jnp.ndarray,
        cds_mask: jnp.ndarray | None,
        *,
        use_cds_mask: bool,
    ) -> jnp.ndarray:
        mask = cds_mask if cds_mask is not None else jnp.zeros((sequences.shape[0], sequences.shape[1]), dtype=jnp.bool_)
        return self._run(sequences, organism_index, mask, use_cds_mask=use_cds_mask)
