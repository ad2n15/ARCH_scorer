"""Register variant classifier heads for each alphagenome-ft fine-tune strategy."""
from __future__ import annotations

from alphagenome.models import dna_output
from alphagenome_ft import CustomHeadConfig, CustomHeadType, register_custom_head

from heads.variant_classifier import (
    VariantClassifierEncoderHead,
    VariantClassifierHead,
    VariantClassifierLoRAHead,
    VariantClassifierAgeAuxHead,
    VariantClassifierSupConHead,
    VariantChAgeRegressorHead,
)

HEAD_CLASSES = {
    "sm_ch_classifier": VariantClassifierHead,
    "sm_ch_age_aux": VariantClassifierAgeAuxHead,
    "sm_ch_supcon": VariantClassifierSupConHead,
    "sm_ch_age_regressor": VariantChAgeRegressorHead,
}

STRATEGIES = ("heads_only", "lora", "full", "encoder_only")


def resolve_strategy(cfg: dict) -> str:
    model = cfg.get("model", {})
    strategy = model.get("finetune_strategy")
    if strategy:
        strategy = str(strategy).lower()
    elif model.get("heads_only", True):
        strategy = "heads_only"
    else:
        strategy = "full"
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown finetune_strategy {strategy!r}; choose from {STRATEGIES}")
    return strategy


def register_variant_head(cfg: dict) -> str:
    """Register the head class for ``cfg['model']``; return strategy name."""
    model = cfg["model"]
    head_id = model["head_id"]
    strategy = resolve_strategy(cfg)
    hidden_dim = int(model.get("hidden_dim", 256))
    pooling = str(model.get("pooling", "variant"))

    if strategy == "lora":
        cls = VariantClassifierLoRAHead
        meta_extra = {
            "lora_rank": int(model.get("lora_rank", 8)),
            "lora_alpha": float(model.get("lora_alpha", 1.0)),
        }
    elif strategy == "encoder_only":
        cls = VariantClassifierEncoderHead
        meta_extra = {}
    else:
        cls = HEAD_CLASSES.get(head_id, VariantClassifierHead)
        meta_extra = {}
        if cls is VariantClassifierAgeAuxHead:
            meta_extra["age_aux_lambda"] = float(model.get("age_aux_lambda", 0.2))
        if cls is VariantClassifierSupConHead:
            meta_extra["supcon_lambda"] = float(model.get("supcon_lambda", 0.1))
            meta_extra["supcon_temperature"] = float(model.get("supcon_temperature", 0.1))

    register_custom_head(
        head_id,
        cls,
        CustomHeadConfig(
            type=CustomHeadType.CUSTOM,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata={"hidden_dim": hidden_dim, "pooling": pooling, **meta_extra},
        ),
    )
    return strategy
