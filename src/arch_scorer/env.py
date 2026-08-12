"""Load repo/cwd `.env` into os.environ (no external dependency)."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """Parse KEY=VALUE lines into os.environ if not already set.

    Search order when path is None:
      1) $ARCH_SCORER_ENV
      2) ./ .env (cwd)
      3) ARCH_scorer repo root /.env
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        env_path = os.environ.get("ARCH_SCORER_ENV")
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path.cwd() / ".env")
        candidates.append(Path(__file__).resolve().parents[2] / ".env")

    for p in candidates:
        p = p.expanduser()
        if not p.is_file():
            continue
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val
        return p.resolve()
    return None
