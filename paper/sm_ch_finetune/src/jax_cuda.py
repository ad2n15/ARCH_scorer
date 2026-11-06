"""JAX/CUDA env tweaks for AlphaGenome fine-tuning on Iridis."""
from __future__ import annotations

import os
import site
from pathlib import Path


def _pip_nvidia_cu12_lib_dirs() -> list[str]:
    """Collect cu12 pip wheel lib dirs; exclude nvidia/cu13 (causes cudnn mismatch)."""
    roots = []
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    roots.extend(Path(p) for p in site.getsitepackages())
    lib_dirs: list[str] = []
    for root in roots:
        nv = root / "nvidia"
        if not nv.is_dir():
            continue
        for sub in sorted(nv.iterdir()):
            if sub.name == "cu13":
                continue
            lib = sub / "lib"
            if lib.is_dir():
                lib_dirs.append(str(lib.resolve()))
    lib_dirs.sort(key=lambda p: (0 if p.endswith("/cudnn/lib") else 1, p))
    return lib_dirs


def _strip_bad_ld_paths(ld_path: str) -> list[str]:
    """Drop cluster cuda module + pip cu13 paths; keep apptainer driver libs."""
    out: list[str] = []
    for part in ld_path.split(":"):
        if not part:
            continue
        if "/ixsoftware/cuda/" in part:
            continue
        if "/nvidia/cu13/" in part:
            continue
        out.append(part)
    return out


def configure_jax_cuda() -> None:
    """Prefer pip cu12 NVIDIA wheels; keep GPU driver libs from apptainer --nv."""
    pip_dirs = _pip_nvidia_cu12_lib_dirs()
    kept = _strip_bad_ld_paths(os.environ.get("LD_LIBRARY_PATH", ""))
    merged: list[str] = []
    for d in pip_dirs + kept:
        if d not in merged:
            merged.append(d)
    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
