# Model licences

The code in this repository is released under the Apache-2.0 licence (`LICENSE`). The trained model files
are derived from third-party models and carry those models' terms. No backbone weights are included: each
user downloads them from the original provider and accepts that provider's terms.

| Files | Derived from | Terms |
|---|---|---|
| `models/finetuned/ntv3/` | NTv3 650M post-trained (InstaDeep, `InstaDeepAI/NTv3_650M_post`) | InstaDeep Open Model Licence v1.0 (`models/finetuned/ntv3/LICENSE-NTv3.md`). Non-commercial use only; offering outputs as a service requires InstaDeep's permission. |
| `models/finetuned/alphagenome/`, `artifacts/*.joblib` | AlphaGenome fold_1 weights and AlphaGenome API outputs (Google DeepMind) | AlphaGenome model terms of use (https://www.kaggle.com/models/google/alphagenome) and API terms. Non-commercial research use. |
| `models/finetuned/borzoi/` | Borzoi human_rep0 via gReLU (`Genentech/borzoi-model`) | MIT |

Annotation sources used at run time (UCSC Genome Browser API, CADD API v1.7, optionally ANNOVAR) have their
own terms; CADD is free for non-commercial use.

**Research use only.** These models are not validated for clinical use and must not be used to make
decisions about individual patients.
