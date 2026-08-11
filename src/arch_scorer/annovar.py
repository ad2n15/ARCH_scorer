"""ANNOVAR wrapper (hg38 refGene → Func.refGene)."""
from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from pathlib import Path


def annotate_func_refgene(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    *,
    annovar_dir: str | Path | None = None,
    table_annovar: str | Path | None = None,
    humandb: str | Path | None = None,
) -> str:
    """Return ANNOVAR Func.refGene for one SNV (hg38)."""
    root = Path(annovar_dir) if annovar_dir else Path(
        os.environ.get("ANNOVAR_DIR", "")
    )
    pl = Path(table_annovar) if table_annovar else (
        root / "table_annovar.pl" if root.as_posix() else Path()
    )
    if not pl.is_file():
        env_pl = os.environ.get("ANNOVAR_PL")
        pl = Path(env_pl) if env_pl else pl
    db = Path(humandb) if humandb else (
        Path(os.environ["ANNOVAR_HUMANDB"])
        if os.environ.get("ANNOVAR_HUMANDB")
        else (root / "humandb" if root.as_posix() else Path())
    )
    if not pl.is_file():
        raise FileNotFoundError(
            "table_annovar.pl not found. Set ANNOVAR_DIR or pass annovar_dir= "
            "(see scripts/install_annovar.sh)."
        )
    if not db.is_dir():
        raise FileNotFoundError(f"ANNOVAR humandb missing: {db}")

    chrom_n = str(chrom).replace("chr", "")
    ref, alt = str(ref).upper(), str(alt).upper()
    with tempfile.TemporaryDirectory(prefix="arch_annovar_") as td:
        td_p = Path(td)
        avi = td_p / "var.avinput"
        avi.write_text(f"{chrom_n}\t{int(pos)}\t{int(pos)}\t{ref}\t{alt}\n")
        out_prefix = td_p / "var_annovar"
        cmd = [
            "perl",
            str(pl),
            str(avi),
            str(db) + ("/" if not str(db).endswith("/") else ""),
            "-buildver",
            "hg38",
            "-out",
            str(out_prefix),
            "-remove",
            "-protocol",
            "refGene",
            "-operation",
            "g",
            "-nastring",
            ".",
            "-csvout",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        csv_path = Path(str(out_prefix) + ".hg38_multianno.csv")
        if not csv_path.is_file():
            raise RuntimeError(f"ANNOVAR did not write {csv_path}")
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return "unknown"
        func = rows[0].get("Func.refGene") or rows[0].get("Func.refgene") or "."
        return "unknown" if func in (".", "", None) else str(func)
