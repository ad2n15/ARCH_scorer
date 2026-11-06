"""Path helpers: run from alphagenome/ or sm_ch_finetune/."""
from __future__ import annotations
from pathlib import Path


def resolve_config(config_arg: str, project: Path) -> Path:
    """Find config whether cwd is alphagenome/ or sm_ch_finetune/."""
    p = Path(config_arg)
    candidates = [p, project / p, Path.cwd() / p]
    if p.is_absolute():
        candidates.insert(0, p)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Config not found: {config_arg}\nTried: {tried}")


def project_subpath(project: Path, rel: str | Path) -> Path:
    """Resolve data/checkpoint paths relative to sm_ch_finetune project root.

    Also accepts paths prefixed with ``sm_ch_finetune/`` when cwd is alphagenome/,
    so callers do not double the project directory.
    """
    rel = Path(rel)
    if rel.is_absolute():
        return rel.resolve()
    parts = rel.parts
    if parts and parts[0] in ("sm_ch_finetune", project.name):
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return (project / rel).resolve()


def use_age_filter(cfg: dict) -> bool:
    """Default True: exclude SM variants lacking gnomAD age data (paper protocol)."""
    return bool(cfg.get("data", {}).get("use_age_filter", True))


def resolve_raw_csv(project: Path, cfg: dict) -> Path:
    """CSV for 01_prepare: age-filtered output when use_age_filter is enabled."""
    data = cfg["data"]
    if use_age_filter(cfg):
        p = project_subpath(project, data["age_filtered_dir"]) / "sm_ch_age_filtered.csv"
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}\nRun scripts/00_filter_no_age_sm.py first (use_age_filter defaults to true)."
            )
        return p
    return project_subpath(project, data["raw_csv"])


def resolve_split_dir(project: Path, cfg: dict, *, prefer_mb: bool = False) -> Path:
    """Parquet split directory: 1 Mb CDS > 32 kb CDS > age-filtered > processed."""
    data = cfg["data"]
    if prefer_mb and data.get("mb_filtered_dir"):
        p = project_subpath(project, data["mb_filtered_dir"])
        if any((p / f"{s}.parquet").exists() for s in ("train", "valid", "test", "pilot_balanced_snv")):
            return p
    if data.get("use_cds_filter", True):
        return project_subpath(project, data["cds_filtered_dir"])
    if use_age_filter(cfg):
        age_dir = project_subpath(project, data["age_filtered_dir"])
        if (age_dir / "train.parquet").exists():
            return age_dir
    return project_subpath(project, data["processed_dir"])


def resolve_pilot_dir(project: Path, cfg: dict) -> Path | None:
    rel = cfg.get("data", {}).get("pilot500_dir") or cfg.get("data", {}).get("pilot_dir")
    return project_subpath(project, rel) if rel else None


def resolve_data_split_dir(project: Path, cfg: dict, *, prefer_mb: bool = False) -> Path:
    """Pilot parquets if configured, else resolve_split_dir."""
    pilot = resolve_pilot_dir(project, cfg)
    if pilot and any((pilot / f"{s}.parquet").exists() for s in ("train", "valid", "test")):
        return pilot
    return resolve_split_dir(project, cfg, prefer_mb=prefer_mb)


def find_kaggle_checkpoint(workspace: Path, model_version: str) -> Path | None:
    """Return cached Kaggle weights path (alphagenome/.cache/...) if present."""
    cache = workspace / ".cache/kagglehub/models/google/alphagenome/jax" / model_version.lower()
    if not cache.is_dir():
        return None
    versions = sorted(p for p in cache.iterdir() if p.is_dir())
    return versions[-1] if versions else None
