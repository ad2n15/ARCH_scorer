"""Late fusion: logistic regression on the fine-tuned probability plus annotations.

A plain-numpy copy of the paper's sklearn pipelines (median imputation for numeric columns,
one-hot with unknown categories ignored, optional top-k bucketing to "__other__"), stored as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OTHER = "__other__"


class FusionLR:
    def __init__(self, spec: list[dict], coef: list[float] | None = None, intercept: float = 0.0):
        self.spec, self.coef, self.intercept = spec, coef, intercept

    @classmethod
    def from_training(cls, df: pd.DataFrame, spec: list[dict]) -> "FusionLR":
        spec = [dict(s) for s in spec]
        for s in spec:
            col = df[s["name"]]
            if s["kind"] == "numeric":
                s["median"] = float(pd.to_numeric(col, errors="coerce").median())
            else:
                v = col.fillna("unknown").astype(str)
                if s.get("top_k"):
                    s["keep"] = v.value_counts().head(s["top_k"]).index.tolist()
                    v = v.where(v.isin(s["keep"]), OTHER)
                s["categories"] = sorted(v.unique().tolist())  # sklearn OneHotEncoder order
        return cls(spec)

    def design(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for s in self.spec:
            col = df[s["name"]]
            if s["kind"] == "numeric":
                blocks.append(pd.to_numeric(col, errors="coerce").fillna(s["median"]).to_numpy(float)[:, None])
            else:
                v = col.fillna("unknown").astype(str)
                if "keep" in s:
                    v = v.where(v.isin(s["keep"]), OTHER)
                blocks.append((v.to_numpy()[:, None] == np.array(s["categories"])[None, :]).astype(float))
        return np.hstack(blocks)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        z = self.design(df) @ np.asarray(self.coef) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def categories(self, name: str) -> list[str]:
        return next(s for s in self.spec if s["name"] == name).get("categories", [])

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"spec": self.spec, "coef": self.coef, "intercept": self.intercept}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "FusionLR":
        d = json.loads(Path(path).read_text())
        return cls(d["spec"], d["coef"], d["intercept"])
