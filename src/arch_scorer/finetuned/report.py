"""Human-readable views of arch-score-ft output: a terminal summary and a self-contained HTML report.

Both take the DataFrame returned by Scorer.score and the per-task metadata in Scorer.meta
({task: {"prevalence", "n", "n_pos", "auroc_finetuned", "auroc_fused", "threshold"}}).
"""
from __future__ import annotations

import html
import math
from datetime import date
from pathlib import Path

import pandas as pd

# "\n" splits a label over two header lines in the terminal; the HTML shows it as one line
LABELS = {"arch21_noncoding": "21 ARCH genes\nnon-coding model",
          "arch21_cnc": "21 ARCH genes\ncoding + non-coding model",
          "genomewide": "Genome-wide\nmodel"}


def label(task: str) -> str:
    return LABELS.get(task, task).replace("\n", ", " if task.startswith("arch21") else " ")
TITLES = {
    "arch21_noncoding": "21 ARCH genes, somatic vs germline (trained on non-coding variants)",
    "arch21_cnc": "21 ARCH genes, somatic vs germline (trained on coding + non-coding variants)",
    "genomewide": "Genome-wide, ARCH vs other somatic",
}


def _num(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _text(x) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def flag_cutoff(meta: dict) -> float:
    """Percentile at which HIGH starts: the score at the (1 - prevalence) quantile of the test set."""
    return 100 * (1 - meta["prevalence"])


def _cell(row, task: str) -> str:
    s, pct = _num(row.get(f"{task}_score")), _num(row.get(f"{task}_percentile"))
    if s is None:
        return "n/a"
    out = f"{s:.3f} p{pct:.0f}" if pct is not None else f"{s:.3f}"
    return out + (" HIGH" if row.get(f"{task}_flag") == "high" else "")


def _variant(row) -> str:
    return _text(row.get("variant_id"))


def summary_rows(df: pd.DataFrame, tasks) -> tuple[list[str], list[list[str]], list[tuple[str, str]]]:
    """Header, table rows and (variant, note) pairs; rows that could not be parsed go to the notes only."""
    header = ["variant", "gene", "region", "CADD", "gnomAD AB\nmedian (n)", "gnomAD age\nmedian (n)"] + \
        [LABELS.get(t, t) for t in tasks]
    rows, notes = [], []
    for _, r in df.iterrows():
        note = _text(r.get("notes"))
        if pd.isna(r.get("pos")):  # bad input line: only variant_id and notes are set
            notes.append((_variant(r), f"not scored: {note}"))
            continue
        cadd, ab, age = _num(r.get("cadd_phred")), _num(r.get("gnomad_ab_median")), _num(r.get("gnomad_age_median"))
        rows.append([_variant(r), _text(r.get("gene")), _text(r.get("func")),
                     "" if cadd is None else f"{cadd:.1f}",
                     "" if ab is None else f"{ab:.2f} ({int(r['gnomad_ab_n'])})",
                     "" if age is None else f"{age:.1f} ({int(r['gnomad_age_n'])})"]
                    + [_cell(r, t) for t in tasks])
        if note:
            notes.append((_variant(r), note))
    return header, rows, notes


def summary_table(df: pd.DataFrame, tasks, meta: dict) -> str:
    header, rows, notes = summary_rows(df, tasks)
    top = [h.split("\n")[0] if "\n" in h else "" for h in header]
    bottom = [h.split("\n")[-1] for h in header]
    w = [max(len(top[i]), len(bottom[i]), *(len(r[i]) for r in rows)) if rows else max(len(top[i]), len(bottom[i]))
         for i in range(len(header))]
    line = lambda cells: "  ".join(c.ljust(w[i]) for i, c in enumerate(cells)).rstrip()  # noqa: E731
    out = [line(top), line(bottom), line(["-" * x for x in w])] + [line(r) for r in rows]
    if notes:
        out += ["", "Notes:"] + [f"  {v}: {n}" for v, n in notes]
    cut = [f"         {label(t)}: p{flag_cutoff(meta[t]):.0f} and above" for t in tasks]
    out += [
        "",
        "score  = probability from the fine-tuned head fused with annotations (fine-tuned only if annotations are missing)",
        "pNN    = percentile of that score among held-out test variants (mostly germline / other somatic)",
        "HIGH   = above the prevalence-matched threshold used in the paper:",
        *cut,
        "gnomAD = v4.1 exomes + genomes: median allele balance (AB) of carriers, and median age of heterozygous",
        "         carriers with age data. Germline: AB ~0.5. Clonal blood variants: AB well below 0.5, older carriers.",
        "The genome-wide model is weakly discriminative (AUROC ~0.58): use it to rank variants. Research use only.",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------- HTML

_CSS = """
:root{--bg:#fff;--fg:#1d2433;--muted:#5d6678;--line:#dfe3ea;--card:#f7f8fb;--hi:#b42318;--hibg:#fdecea;
--bar:#e4e8ef;--acc:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2c323d;
--card:#1b1f27;--hi:#ff8a7a;--hibg:#3a1d1a;--bar:#2c323d;--acc:#7da2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:32px 0 8px}.muted{color:var(--muted)}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{cursor:pointer;user-select:none;font-weight:600}th:after{content:" \\2195";color:var(--muted);font-size:11px}
tr.hi td{background:var(--hibg)}.flag{color:var(--hi);font-weight:700}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:12px 0}
.card h3{margin:0 0 2px;font-size:15px;font-family:ui-monospace,Menlo,monospace}
.task{display:grid;grid-template-columns:230px 1fr 190px;gap:10px;align-items:center;margin-top:8px}
.bar{position:relative;height:10px;background:var(--bar);border-radius:5px}
.cut{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--hi);opacity:.6}
.dot{position:absolute;top:-3px;width:16px;height:16px;margin-left:-8px;border-radius:50%;
background:var(--acc);border:2px solid var(--bg)}.dot.hi{background:var(--hi)}
.note{margin-top:8px;color:var(--muted);font-size:13px}
.gn{margin-top:12px;padding-top:10px;border-top:1px solid var(--line)}
.gngrid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:6px}
.hist{width:100%;max-width:300px;height:auto;display:block;margin-top:4px}.hist rect{fill:var(--acc);opacity:.8}
.hist .ref{stroke:var(--hi);stroke-width:1.5;stroke-dasharray:3 2}.hist text{font-size:10px;fill:var(--muted)}
@media (max-width:600px){.task,.gngrid{grid-template-columns:1fr}}
"""

_JS = """
document.querySelectorAll('table.sortable th').forEach((th,i)=>th.addEventListener('click',()=>{
 const tb=th.closest('table').tBodies[0],rows=[...tb.rows],asc=th.dataset.asc!=='1';
 const key=r=>{const t=r.cells[i].innerText.trim(),n=parseFloat(t);return isNaN(n)?t:n};
 rows.sort((a,b)=>{const x=key(a),y=key(b);return (x>y?1:x<y?-1:0)*(asc?1:-1)});
 th.dataset.asc=asc?'1':'0';rows.forEach(r=>tb.appendChild(r));}));
"""


def _e(x) -> str:
    return html.escape(_text(x))


def _task_row(r, task: str, m: dict) -> str:
    s, pct = _num(r.get(f"{task}_score")), _num(r.get(f"{task}_percentile"))
    label_ = f"<b>{_e(label(task))}</b>"
    if s is None:
        return f'<div class="task">{label_}<div class="muted">not applicable</div><div></div></div>'
    hi = r.get(f"{task}_flag") == "high"
    ft, fu = _num(r.get(f"{task}_{r.get('_backbone')}")), _num(r.get(f"{task}_{r.get('_backbone')}_plus_annotations"))
    detail = f"fine-tuned {ft:.3f}" + (f" · fused {fu:.3f}" if fu is not None else "") if ft is not None else ""
    dot = f'<div class="dot{" hi" if hi else ""}" style="left:{min(max(pct or 0, 0), 100):.1f}%"></div>'
    cut = f'<div class="cut" style="left:{flag_cutoff(m):.1f}%" title="HIGH threshold"></div>'
    flag = ' <span class="flag">HIGH</span>' if hi else ""
    return (f'<div class="task">{label_}<div class="bar" title="percentile among held-out test variants">{cut}{dot}'
            f'</div><div>{s:.3f} · p{pct:.0f}{flag}<br><span class="muted">{_e(detail)}</span></div></div>')


def _hist_svg(counts: list[float], labels: list[str], mark: float | None = None, title: str = "") -> str:
    """Small inline bar chart; mark = x position (0-1) of a reference line."""
    n = len(counts)
    top = max(counts) or 1
    w, h, gap = 260, 60, 2
    bw = (w - gap * (n - 1)) / n
    bars = "".join(
        f'<rect x="{i * (bw + gap):.1f}" y="{h - h * c / top:.1f}" width="{bw:.1f}" height="{h * c / top:.1f}" '
        f'rx="1"><title>{_e(labels[i])}: {int(c)}</title></rect>' for i, c in enumerate(counts))
    line = (f'<line class="ref" x1="{mark * w:.1f}" x2="{mark * w:.1f}" y1="0" y2="{h}"/>' if mark is not None else "")
    ticks = (f'<text x="0" y="{h + 12}">{_e(labels[0].split("-")[0])}</text>'
             f'<text x="{w}" y="{h + 12}" text-anchor="end">{_e(labels[-1].split("-")[-1])}</text>')
    return (f'<svg class="hist" viewBox="0 0 {w} {h + 14}" role="img" aria-label="{_e(title)}">'
            f'{bars}{line}{ticks}</svg>')


def _gnomad_block(r) -> str:
    status = _text(r.get("gnomad_status"))
    if not status:
        return ""
    if status != "found":
        return f'<div class="gn"><b>gnomAD v4.1</b> <span class="muted">{_e(status)}</span></div>'
    af, ac, hom = _num(r.get("gnomad_af")), int(_num(r.get("gnomad_ac")) or 0), int(_num(r.get("gnomad_hom")) or 0)
    head = (f'<b>gnomAD v4.1</b> <span class="muted">AF {af:.2e} · AC {ac:,} · {hom:,} homozygous</span>'
            if af is not None else "<b>gnomAD v4.1</b>")
    parts = []
    abh = [float(x) for x in _text(r.get("gnomad_ab_hist")).split("|") if x]
    abm, abn = _num(r.get("gnomad_ab_median")), int(_num(r.get("gnomad_ab_n")) or 0)
    if abh and abn:
        lab = [f"{i * 0.05:.2f}-{(i + 1) * 0.05:.2f}" for i in range(len(abh))]
        parts.append(f'<div><div>Allele balance: median <b>{abm:.2f}</b> <span class="muted">({abn:,} carriers; '
                     f'line = 0.5, germline)</span></div>{_hist_svg(abh, lab, 0.5, "allele balance histogram")}</div>')
    ageh = [float(x) for x in _text(r.get("gnomad_age_hist")).split("|") if x]
    agem, agen = _num(r.get("gnomad_age_median")), int(_num(r.get("gnomad_age_n")) or 0)
    if ageh and agen:
        lab = ["<30"] + [f"{a}-{a + 5}" for a in range(30, 80, 5)] + [">80"]
        parts.append(f'<div><div>Carrier age: median <b>{agem:.1f}</b> <span class="muted">({agen:,} heterozygous '
                     f'carriers with age; years)</span></div>{_hist_svg(ageh, lab, None, "carrier age histogram")}</div>')
    return f'<div class="gn">{head}<div class="gngrid">{"".join(parts)}</div></div>'


def write_html(df: pd.DataFrame, tasks, meta: dict, backbone: str, path: str | Path, version: str = "") -> Path:
    tasks = list(tasks)
    header, rows, _ = summary_rows(df, tasks)
    body = []
    for r in rows:
        hi = any(c.endswith("HIGH") for c in r)
        cells = "".join(f'<td>{_e(c)}</td>' if not c.endswith("HIGH")
                        else f'<td>{_e(c[:-5])} <span class="flag">HIGH</span></td>' for c in r)
        body.append(f'<tr{" class=hi" if hi else ""}>{cells}</tr>')
    table = ('<div class="wrap"><table class="sortable"><thead><tr>' + "".join(f"<th>{_e(h).replace(chr(10), '<br>')}</th>" for h in header)
             + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")

    cards = []
    for _, r in df.iterrows():
        r = r.copy()
        r["_backbone"] = backbone
        note = _text(r.get("notes"))
        if pd.isna(r.get("pos")):
            cards.append(f'<div class="card"><h3>{_e(r.get("variant_id"))}</h3>'
                         f'<div class="note">Not scored: {_e(note)}</div></div>')
            continue
        cadd = _num(r.get("cadd_phred"))
        facts = " · ".join(x for x in [
            f"gene {_e(r.get('gene'))}" if _text(r.get("gene")) else "",
            f"region {_e(r.get('func'))}" if _text(r.get("func")) else "",
            f"CADD-PHRED {cadd:.1f}" if cadd is not None else "",
            f"SBS6 {_e(r.get('mutation_type_sbs6'))}" if _text(r.get("mutation_type_sbs6")) else ""] if x)
        cards.append(f'<div class="card"><h3>{_e(r.get("variant_id"))}</h3><div class="muted">{facts}</div>'
                     + "".join(_task_row(r, t, meta[t]) for t in tasks) + _gnomad_block(r)
                     + (f'<div class="note">Notes: {_e(note)}</div>' if note else "") + "</div>")

    how = "".join(
        f"<li><b>{_e(label(t))}</b>: {_e(TITLES.get(t, t))}. Held-out test set n = {meta[t]['n']:,} "
        f"({meta[t]['n_pos']:,} positive); AUROC {meta[t]['auroc_finetuned']:.3f} fine-tuned, "
        f"{meta[t]['auroc_fused']:.3f} with annotations. HIGH from p{flag_cutoff(meta[t]):.0f}.</li>" for t in tasks)
    n_ok = len(rows)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ARCH_scorer report</title>
<style>{_CSS}</style></head><body><main>
<h1>ARCH_scorer report</h1>
<div class="muted">{_e(backbone)} backbone · {n_ok} of {len(df)} variants scored · {date.today().isoformat()}
{(' · arch_scorer ' + _e(version)) if version else ''} · <b>Research use only</b></div>
<h2>Summary</h2>
<p class="muted">Each task cell shows the score, its percentile (pNN) and HIGH when it is above the threshold.
Click a column to sort.</p>
{table}
<h2>Variants</h2>
<p class="muted">The bar is the percentile among held-out test variants; the red line marks the HIGH threshold.</p>
{''.join(cards)}
<h2>How to read this</h2>
<ul>{how}</ul>
<p><b>Score</b>: probability from the fine-tuned head fused with CADD-PHRED, functional class, substitution type and
gene by logistic regression; the fine-tuned score alone is used when an annotation is missing.<br>
<b>Percentile</b>: rank of the score among held-out test variants, most of which are negatives (germline for the
21-gene tasks, other somatic for genome-wide).<br>
<b>HIGH</b>: score above the prevalence-matched threshold used in the paper, i.e. the top share of test variants equal
to the share of positives. It is a screening flag, not a diagnosis.<br>
<b>gnomAD v4.1</b> (exomes + genomes, public API): allele frequency, the allele balance (AB, share of reads
carrying ALT) across carriers, and the ages of heterozygous carriers. Germline heterozygous variants have AB near 0.5;
clonal blood variants are carried by a minority of cells, so AB is well below 0.5, and carriers tend to be older. The
paper used these gnomAD signals to label variants as somatic, so treat them as context rather than an independent
check of the score. Age is known only for a subset of carriers.<br>
<b>Genome-wide</b> scores separate ARCH from other somatic variants only weakly (AUROC about 0.58); use them to rank
variants, not to classify them.</p>
</main><script>{_JS}</script></body></html>"""
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path
