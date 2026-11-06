"""Frozen backbones, each returning the embedding the trained head was fitted on.

All three use the same input: a 65,536 bp GRCh38 window with the ALT allele at index 32,768
(annotate.variant_window), as in training.

  ntv3         InstaDeepAI/NTv3_650M_post; final hidden state at the variant base (1,536-d).
               Mirrors sm_ch_finetune_ntv3/src/model/ntv3_ch.py.
  borzoi       Genentech/borzoi-model human_rep0 trunk via gReLU, crop disabled for 64 kb input;
               the 32 bp bin containing the variant (1,920-d). Mirrors sm_ch_finetune_borzoi/src/model/borzoi_ch.py.
  alphagenome  AlphaGenome fold_1 (JAX), 1 bp sequence embedding at the variant base (1,536-d), bfloat16
               compute as in training. Mirrors sm_ch_finetune/src/heads/variant_classifier.py.

Weights are downloaded by the user under each model's own licence; none are redistributed here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

WINDOW = 65536


class NTv3Backbone:
    name = "ntv3"
    REPO = "InstaDeepAI/NTv3_650M_post"
    REVISION = "ad622051abbe9376bdea3f2727863fa559957e9f"  # weights snapshot used in training
    CODE_REVISION = "0ecff3637f0d3ba5b686d1095083218157c2ca34"  # InstaDeepAI/ntv3_base_model code used in training

    def __init__(self, model_path: str | None = None, device: str | None = None):
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel

        src = model_path or os.environ.get("ARCH_NTV3_PATH") or self.REPO
        kw = {"trust_remote_code": True, "code_revision": self.CODE_REVISION}
        if src == self.REPO:
            kw["revision"] = self.REVISION
        try:
            config = AutoConfig.from_pretrained(src, **kw)
            full = AutoModel.from_pretrained(src, config=config, **kw)
        except OSError as e:
            if "gated" in str(e).lower() or "401" in str(e):
                raise SystemExit(
                    "NTv3 is a gated model on Hugging Face. Once per user:\n"
                    "  1. open https://huggingface.co/InstaDeepAI/NTv3_650M_post and accept the licence\n"
                    "     (also https://huggingface.co/InstaDeepAI/ntv3_base_model, which holds its code)\n"
                    "  2. create a read token at https://huggingface.co/settings/tokens\n"
                    "  3. run `hf auth login` (or export HF_TOKEN=...) and try again\n"
                    "Or point ARCH_NTV3_PATH / --model-path at a local copy of the weights.") from None
            raise
        core = type(full.core).__bases__[0](config)  # the core without NTv3's own task heads
        missing = core.load_state_dict(full.core.state_dict(), strict=False).missing_keys
        if missing:
            raise RuntimeError(f"NTv3 core weights incomplete: {missing[:5]}")
        del full
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.core = core.to(self.device).eval()
        self.species = torch.LongTensor([config.species_to_token_id.get("human", 2)])
        vocab = Path(src) / "vocab.json" if Path(src).is_dir() else Path(
            hf_hub_download(self.REPO, "vocab.json", revision=self.REVISION))
        self.vocab = json.loads(vocab.read_text())
        self.unk = self.vocab["<unk>"]
        self.batch_size = 4 if self.device.type == "cuda" else 1

    @torch.no_grad()
    def embed(self, seqs: list[str], centers: list[int]) -> torch.Tensor:
        ids = torch.stack([torch.LongTensor([self.vocab.get(c, self.unk) for c in s]) for s in seqs]).to(self.device)
        species = self.species.to(self.device).repeat(len(seqs))
        hidden = self.core(ids, [species], output_hidden_states=True)["hidden_states"][-1]
        idx = torch.tensor(centers, device=self.device)
        return hidden[torch.arange(len(seqs), device=self.device), idx].float().cpu()


class BorzoiBackbone:
    name = "borzoi"
    REPO, FILE = "Genentech/borzoi-model", "human_rep0.ckpt"
    BIN = 32

    def __init__(self, model_path: str | None = None, device: str | None = None):
        from grelu.lightning import LightningModel
        from grelu.model.layers import Crop

        ckpt = model_path or os.environ.get("ARCH_BORZOI_CKPT")
        if not ckpt:
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download(self.REPO, self.FILE)
        pl_model = LightningModel.load_from_checkpoint(str(ckpt), map_location="cpu", weights_only=False)
        self.trunk = pl_model.model.embedding
        if hasattr(self.trunk, "crop"):
            self.trunk.crop = Crop(crop_len=0)  # the native crop is longer than a 64 kb window
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.trunk = self.trunk.to(self.device).eval()
        self.batch_size = 4 if self.device.type == "cuda" else 1

    @torch.no_grad()
    def embed(self, seqs: list[str], centers: list[int]) -> torch.Tensor:
        from grelu.sequence.format import strings_to_one_hot

        x = torch.stack([strings_to_one_hot(s, add_batch_axis=False) for s in seqs]).to(self.device)
        h = self.trunk(x)  # [B, C, L / 32]
        idx = torch.tensor([c // self.BIN for c in centers], device=self.device).clamp(0, h.shape[2] - 1)
        return h[torch.arange(len(seqs), device=self.device), :, idx].float().cpu()


class AlphaGenomeBackbone:
    name = "alphagenome"

    def __init__(self, model_path: str | None = None, device: str | None = None):
        import jax
        import jax.numpy as jnp
        from alphagenome.models import dna_model as ag_dna_model
        from alphagenome_research.model import dna_model as research_dna_model

        from ..extract_emb import VariantSiteEmbedder
        from ..fold1 import find_fold1_checkpoint

        ckpt = find_fold1_checkpoint(explicit=model_path)
        organism = ag_dna_model.Organism.HOMO_SAPIENS
        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        dev = gpus[0] if gpus else jax.devices("cpu")[0]
        # sequences are supplied directly, so the organism FASTA setting is never read
        model = research_dna_model.create(str(ckpt), organism_settings={
            organism: research_dna_model.OrganismSettings(fasta_path=os.environ.get("ARCH_FASTA", ""))}, device=dev)
        self._org = int(research_dna_model.convert_to_organism_index(organism))
        self._embedder = VariantSiteEmbedder(model._params, model._state)
        self._jnp = jnp
        self.device = dev
        self.batch_size = 1

    def embed(self, seqs: list[str], centers: list[int]) -> torch.Tensor:
        from ..extract_emb import _enc, tokens_to_onehot

        jnp = self._jnp
        out = []
        for s, c in zip(seqs, centers):
            tok = np.array([_enc(b) for b in s], dtype=np.int32)[None]
            e = self._embedder(jnp.asarray(tokens_to_onehot(tok)), jnp.asarray([self._org], dtype=jnp.int32),
                               jnp.asarray([c], dtype=jnp.int32))
            out.append(np.asarray(e[0], dtype=np.float32))
        return torch.from_numpy(np.stack(out))


BACKBONE_CLASSES = {"ntv3": NTv3Backbone, "borzoi": BorzoiBackbone, "alphagenome": AlphaGenomeBackbone}
