"""Fine-tuned AlphaGenome, NTv3 and Borzoi heads for ARCH variant classification.

    from arch_scorer.finetuned import Scorer
    Scorer("ntv3").score(["20-32437597-T-G"])

Tasks (see models/finetuned/<backbone>/<task>/config.json):
  arch21_noncoding  somatic vs germline, 21 ARCH genes, trained on non-coding variants (paper Fig. 4b)
  arch21_cnc        same, trained on coding + non-coding variants (Fig. 4c)
  genomewide        ARCH vs other somatic, genome-wide (Fig. 5)
"""
__all__ = ["BACKBONES", "TASKS", "Scorer"]


def __getattr__(name):  # import torch-dependent code only when it is used
    if name in __all__:
        from . import predict
        return getattr(predict, name)
    raise AttributeError(name)
