#!/usr/bin/env python
"""Main and supplementary figures of the paper, drawn only from the summary files in
paper/sm_ch_finetune/results (no individual-level data needed).

  Fig 1  study design and cohorts            Fig 4  fine-tuning, 21 ARCH genes (held-out test)
  Fig 2  covariate-adjusted ORs (Table 1)    Fig 5  genome-wide ARCH vs other somatic
  Fig 3  a scores vs embeddings, b feature importance, c per-gene AUROC
  Supplementary Fig 3  per-gene AUROC, nine models
  Supplementary Fig 4  ASXL1 ROC and score distributions

Run:  python paper/figures/make_figures.py      (writes paper/figures/output/)
"""
import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", category=FutureWarning)
import fitz  # PyMuPDF  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[1] / "sm_ch_finetune"
RES = ROOT / "results"
PLOTS = RES / "plots"
PUB = PLOTS / "arch21_nc_publish"

MM = 1 / 25.4
W2, W1 = 180 * MM, 88 * MM  # Nature Communications double / single column
DPI = 1200  # Nature line-art standard

# same palette as the poster: one colour per model, annotations always grey
MODEL = {"AlphaGenome": "#E4572E", "NTv3": "#005C85", "Borzoi": "#6A4C93"}
ANNOT = "#8A9BA8"
GERM = "#C9D2D8"
INK = "#2B3A44"
SOFT = "#5B6B76"
FEATURE_NAMES = {"SPLICE_JUNC": "Splice junction", "SPLICE_USAGE": "Splice-site usage", "H3K27me3": "H3K27me3",
                 "H3K36me3": "H3K36me3", "H3K9me3": "H3K9me3", "H3K4me3": "H3K4me3", "DNASE": "DNase",
                 "H3K27ac": "H3K27ac", "H3K4me1": "H3K4me1", "RNA_SEQ": "RNA-seq"}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "legend.frameon": False,
    "mathtext.fontset": "custom",  # P-values etc. in the same sans as the text
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
}
plt.rcParams.update(STYLE)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0], ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plt.rcdefaults()
    plt.rcParams.update(STYLE)  # imported plotting scripts may change rcParams on import
    return mod


def label(ax, letter, dx=-0.13, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=DPI)
    plt.close(fig)
    to_tiff(stem)
    print(stem)


def to_tiff(stem):
    Image.open(OUT / f"{stem}.png").convert("RGB").save(OUT / f"{stem}.tiff", compression="tiff_lzw",
                                                         dpi=(DPI, DPI))


def box(ax, x, y, w, h, text, fc, ec=None, fs=6, bold=False, tc="k", r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                ec=ec or fc, lw=0.7, transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", transform=ax.transAxes, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=SOFT):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                                 transform=ax.transAxes))


# --------------------------------------------------------------------------- Fig 1
def _counts():
    """Cohort sizes: 21-gene arms from the Table 1 rebuild (CADD-PHRED rows), genome-wide from the split summary."""
    t1 = RES / "arch_mane_table1_covariate_adjusted"
    nc = pd.read_csv(t1 / "table1b_covariate_adjusted_noncoding_all.csv").set_index("feature").loc["cadd_phred"]
    co = pd.read_csv(t1 / "table1b_covariate_adjusted_coding.csv").set_index("feature").loc["cadd_phred"]
    sp = json.loads((ROOT / "data/processed_age_filtered/split_stats.json").read_text())["splits"]
    c = {"nc": int(nc.n), "nc_som": int(nc.n_somatic), "co": int(co.n), "co_som": int(co.n_somatic)}
    c.update(nc_germ=c["nc"] - c["nc_som"], co_germ=c["co"] - c["co_som"], all21=c["nc"] + c["co"])
    c["gw"] = sum(s["n"] for s in sp.values())
    c["gw_ch"] = sum(s["n_ch"] for s in sp.values())
    c["gw_sm"] = c["gw"] - c["gw_ch"]
    c["split"] = {k: (v["n"], v["n_ch"]) for k, v in sp.items()}
    return c


def fig1():
    """a: study logic; b: 21-gene cohort; c: genome-wide cohort and split."""
    c = _counts()
    ha, hb, hc = 42 * MM, 52 * MM, 50 * MM
    H = ha + hb + hc
    fig = plt.figure(figsize=(W2, H))

    # a: what is compared, with which approach, and where the results are
    ax = fig.add_axes([0, (hb + hc) / H, 1, ha / H])
    ax.axis("off")
    ax.text(0.005, 0.99, "a", fontsize=8, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.03, 0.99, "Study design: two comparisons, four analyses", fontsize=6.5, fontweight="bold", va="top",
            transform=ax.transAxes)
    tw, tx = 0.21, [0.335, 0.555, 0.775]  # task box width and x positions of the three columns
    rows = [(0.50, "Germline vs somatic\n21 ARCH genes", tx[0] - 0.01), (0.06, "ARCH vs other somatic\ngenome-wide", tx[2] - 0.01)]
    for y, txt, x_end in rows:
        box(ax, 0.03, y, 0.25, 0.34, txt, INK, fs=6.5, bold=True, tc="white")
        arrow(ax, 0.29, y + 0.17, x_end, y + 0.17)
    tasks = [(tx[0], 0.50, "(i) Regulatory effects", "AlphaGenome CD34$^+$ CMP\ntrack scores vs somatic status"),
             (tx[1], 0.50, "(ii) Scores vs embeddings", "AlphaGenome regulatory scores\nand variant embeddings, non-coding"),
             (tx[2], 0.50, "(iii) Heads-only fine-tuning", "AlphaGenome · NTv3 · Borzoi\nfrozen backbones, 64 kb windows"),
             (tx[2], 0.06, "(iv) Heads-only fine-tuning", "AlphaGenome · NTv3 · Borzoi\nvs annotation-based approaches")]
    for x, y, title, body in tasks:
        box(ax, x, y, tw, 0.34, "", "#EEF1F3", ANNOT)
        ax.text(x + 0.012, y + 0.24, title, fontsize=6.3, fontweight="bold", va="center", transform=ax.transAxes)
        ax.text(x + 0.012, y + 0.12, body, fontsize=5.8, va="center", linespacing=1.2, transform=ax.transAxes)

    grey, edge = "#EEF1F3", ANNOT

    def panel(bottom, height, letter, header):
        bx = fig.add_axes([0, bottom / H, 1, height / H])
        bx.axis("off")
        bx.text(0.005, 0.99, letter, fontsize=8, fontweight="bold", va="top", transform=bx.transAxes)
        bx.text(0.03, 0.99, header, fontsize=6.5, fontweight="bold", va="top", transform=bx.transAxes)
        return bx

    def node(bx, cx, y, w, h, title, detail, fc, ec, tc="k"):
        bx.add_patch(FancyBboxPatch((cx - w / 2, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc,
                                    ec=ec, lw=0.7, transform=bx.transAxes))
        bx.text(cx, y + h / 2, title, ha="center", va="bottom", fontsize=6.3, fontweight="bold", color=tc,
                transform=bx.transAxes)
        bx.text(cx, y + h / 2 - 0.015, detail, ha="center", va="top", fontsize=5.8, color=tc,
                transform=bx.transAxes, linespacing=1.2)

    def tree(bx, parents, py, kids, ky, ymid):
        kw = dict(color=INK, lw=0.7, transform=bx.transAxes, solid_capstyle="butt")
        for px in parents:
            bx.plot([px, px], [py, ymid], **kw)
        bx.plot([min(parents + kids), max(parents + kids)], [ymid, ymid], **kw)
        for k in kids:
            bx.plot([k, k], [ymid, ky], **kw)

    # b: 21 ARCH genes, germline vs somatic
    bx = panel(hc, hb, "b", "21 ARCH genes: germline vs somatic")
    node(bx, 0.5, 0.72, 0.34, 0.18, "Pooled variants from 21 ARCH-associated genes", f"n = {c['all21']:,}", grey, edge)
    for cx, title, n in [(0.25, "Non-coding", f"n = {c['nc']:,}"), (0.75, "Coding", f"n = {c['co']:,}")]:
        node(bx, cx, 0.40, 0.26, 0.16, title, n, grey, edge)
    tree(bx, [0.5], 0.72, [0.25, 0.75], 0.56, 0.64)
    som = "(gnomAD v4 / COSMIC / IARC-TP53)"
    for cx, title, det in [(0.135, "Germline", f"n = {c['nc_germ']:,}\n(all gnomAD v4)"), (0.365, "Somatic", f"n = {c['nc_som']:,}\n{som}"),
                           (0.635, "Germline", f"n = {c['co_germ']:,}\n(all gnomAD v4)"), (0.865, "Somatic", f"n = {c['co_som']:,}\n{som}")]:
        dark = title == "Somatic"
        node(bx, cx, 0.06, 0.21, 0.24, title, det, INK if dark else "white", INK if dark else edge,
             tc="white" if dark else "k")
    tree(bx, [0.25], 0.40, [0.135, 0.365], 0.30, 0.35)
    tree(bx, [0.75], 0.40, [0.635, 0.865], 0.30, 0.35)

    # c: genome-wide, ARCH vs other somatic
    cx_ = panel(0, hc, "c", "Genome-wide: ARCH vs other somatic")
    node(cx_, 0.5, 0.72, 0.44, 0.18, "Somatic variants (Colin-Leitzinger et al. 2025)",
         f"age-filtered; n = {c['gw']:,}", grey, edge)
    node(cx_, 0.32, 0.40, 0.24, 0.16, "ARCH (age-skewed)", f"n = {c['gw_ch']:,}", INK, INK, tc="white")
    node(cx_, 0.68, 0.40, 0.24, 0.16, "Other somatic", f"n = {c['gw_sm']:,}", "white", edge)
    tree(cx_, [0.5], 0.72, [0.32, 0.68], 0.56, 0.64)
    splits = [(x, title, "n = {:,}\n{:,} ARCH".format(*c["split"][key])) for x, title, key in
              [(0.2, "Train (90%)", "train"), (0.5, "Validation (5%)", "valid"), (0.8, "Test (5%)", "test")]]
    for x, title, det in splits:
        node(cx_, x, 0.03, 0.24, 0.24, title, det, grey, edge)
    tree(cx_, [0.32, 0.68], 0.40, [s[0] for s in splits], 0.27, 0.335)
    save(fig, "Fig1_study_design_cohort")


# --------------------------------------------------------------------------- Fig 2
T1_ORDER = ["SPLICE_JUNC", "SPLICE_USAGE", "H3K27me3", "H3K36me3", "H3K9me3", "H3K4me3", "DNASE", "H3K27ac",
            "H3K4me1", "RNA_SEQ"]


def _table1():
    """(feature, then OR, lo, hi, BH q for: non-coding all | non-coding without splice sites | coding).
    From 119_table1_covariate_adjusted_or.py; ORs rounded to 2 dp as printed in the figure."""
    w = pd.read_csv(RES / "arch_mane_table1_covariate_adjusted/table1b_covariate_adjusted_wide.csv").set_index("feature")
    rows = []
    for f in T1_ORDER:
        r = [f]
        for pre in ("nc_all", "nc_nosplice", "coding"):
            r += [round(w.loc[f, f"{pre}_or_per_sd_adj"], 2), round(w.loc[f, f"{pre}_or_ci_lo_adj"], 2),
                  round(w.loc[f, f"{pre}_or_ci_hi_adj"], 2), float(w.loc[f, f"{pre}_q_value_bh_adj"])]
        rows.append(tuple(r))
    return rows


TABLE1 = _table1()
# (offset, marker, face, edge, text colour, fade if not significant) per block
T1_BLOCKS = [(0.25, "o", INK, INK, "k", True), (0.0, "o", "white", INK, "k", True),
             (-0.25, "s", ANNOT, ANNOT, "k", False)]
T1_COLS = [0.495, 0.60, 0.672, 0.777, 0.849, 0.954]  # OR, P for each block


def fmt_p(p):
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def fig2():
    fig = plt.figure(figsize=(W2, 78 * MM))
    ax = fig.add_axes([0.13, 0.12, 0.33, 0.72])
    n = len(TABLE1)
    ax.set_xscale("log")
    ax.set_xlim(0.86, 1.37)
    ax.set_ylim(-1.15, n - 0.4)
    for i, row in enumerate(TABLE1):
        y = n - 1 - i
        ty = fig.transFigure.inverted().transform(ax.transData.transform((1, y)))[1]
        for k, (dy, mk, face, edge, tc, fade) in enumerate(T1_BLOCKS):
            o, lo, hi, p = row[1 + 4 * k: 5 + 4 * k]
            sig = p < 0.05
            a = 1 if (sig or not fade) else 0.35
            ax.plot([lo, hi], [y + dy] * 2, color=edge, lw=1.0, alpha=a)
            ax.plot(o, y + dy, mk, ms=3.4 if mk == "o" else 3.0, mfc=face, mec=edge, mew=0.8, alpha=a)
            for x, txt in [(T1_COLS[2 * k], f"{o:.2f} ({lo:.2f}–{hi:.2f})"), (T1_COLS[2 * k + 1], fmt_p(p))]:
                fig.text(x, ty, txt, fontsize=5.5, color=tc, va="center", ha="left",
                         fontweight="bold" if (sig and fade) else "normal")
    top = fig.transFigure.inverted().transform(ax.transData.transform((1, n - 0.4)))[1]
    for k, grp in enumerate(["Non-coding, all", "Non-coding, excl. splice sites", "Coding"]):
        x0, x1 = T1_COLS[2 * k], min(T1_COLS[2 * k + 1] + 0.06, 0.995)
        fig.text(x0, top + 0.055, grp, fontsize=6, fontweight="bold", va="bottom", ha="left")
        fig.add_artist(plt.Line2D([x0, x1], [top + 0.05] * 2, color="k", lw=0.5, transform=fig.transFigure))
        fig.text(x0, top + 0.005, "OR (95% CI)", fontsize=5.5, va="bottom", ha="left")
        fig.text(T1_COLS[2 * k + 1], top + 0.005, "P$_{FDR}$", fontsize=5.5, va="bottom", ha="left")
    ax.axvline(1, color="k", lw=0.6, ls="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels([FEATURE_NAMES[r[0]] for r in TABLE1][::-1])
    ax.set_xticks([0.9, 1.0, 1.1, 1.2, 1.3])
    ax.set_xticklabels(["0.9", "1.0", "1.1", "1.2", "1.3"])
    ax.minorticks_off()
    ax.set_xlabel("OR per 1-SD increase (somatic vs germline)")
    ax.text(0.865, -1.05, "← lower in somatic", fontsize=5.5, color="k", va="bottom")
    ax.text(1.015, -1.05, "higher →", fontsize=5.5, color="k", va="bottom")
    labels = ["Non-coding, all (n = 71,544; 1,914 somatic)",
              "Non-coding, excl. splice sites (n = 71,433; 1,820 somatic)", "Coding (n = 6,037; 1,913 somatic)"]
    for (dy, mk, face, edge, tc, fade), lab in zip(T1_BLOCKS, labels):
        ax.plot([], [], mk, ms=3.4, mfc=face, mec=edge, mew=0.8, label=lab)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, handletextpad=0.3, labelspacing=0.25)
    save(fig, "Fig2_alphagenome_feature_forest")


# --------------------------------------------------------------------------- Fig 3
T2_FEATURE = {"CADD_PHRED": "CADD-PHRED", "AlphaGenome scores (n=10)": "AlphaGenome scores (10)",
              "64 PCs compressed embeddings": "AlphaGenome embedding (64 PCs)"}
T2_TRAIN = {"non-coding": "Non-coding", "Coding and non-coding": "Coding + non-coding"}


def _table2():
    """(train, eval, features, AUROC fold mean, fold SD), from 146_table2_cadd_api_emb_lr_fold_sd.py."""
    d = pd.read_csv(RES / "table2_cadd_api_emb_lr_train_x_eval.csv")
    return [(T2_TRAIN[r.Train], r.Eval, T2_FEATURE[r.Feature], round(r.auroc_fold_mean, 3), round(r.auroc_fold_sd, 3))
            for r in d.itertuples()]


TABLE2 = _table2()
FEAT_STYLE = {  # same encoding as Fig 2: (line/edge colour, marker, face)
    "CADD-PHRED": (ANNOT, "s", ANNOT), "AlphaGenome scores (10)": (INK, "o", "white"),
    "AlphaGenome embedding (64 PCs)": (INK, "o", INK)}
HEAT_CMAP = "RdYlBu"  # Fig 3c: red = low, blue = high AUROC; colour-blind safe
HEAT_COLS = [("baseline_lr", "LR"), ("baseline_rf", "RF"), ("baseline_xgb", "XGB"),
             ("api_baseline_lr", "LR"), ("api_baseline_rf", "RF"), ("api_baseline_xgb", "XGB"),
             ("emb_base_lr", "LR"), ("emb_base_rf", "RF"), ("emb_base_xgb", "XGB")]


def fig3():
    """a: scores vs embeddings (Table 2); b: pooled feature importance (Table 3); c: per-gene AUROC."""
    fig = plt.figure(figsize=(W2, 150 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.30, 1], width_ratios=[1, 1.05], hspace=0.32, wspace=0.42,
                          left=0.15, right=0.92, top=0.95, bottom=0.07)

    # a: Table 2 as a dot plot
    a = fig.add_subplot(gs[0, :])
    groups = [("Non-coding", "non-coding"), ("Coding + non-coding", "non-coding"),
              ("Coding + non-coding", "coding")]
    for gi, (tr, ev) in enumerate(groups):
        rows = [r for r in TABLE2 if r[0] == tr and r[1] == ev]
        for k, (_, _, feat, m, sd) in enumerate(rows):
            c, mk, face = FEAT_STYLE[feat]
            y = gi + (k - 1) * 0.24
            a.errorbar(m, y, xerr=sd, fmt=mk, ms=3.4 if mk == "o" else 3.0, mfc=face, mec=c, mew=0.8,
                       ecolor=c, elinewidth=0.9, capsize=1.5)
            a.text(m + sd + 0.005, y, f"{m:.3f}", va="center", fontsize=5.5, color="k")
    a.set_yticks(range(3))
    a.set_yticklabels([f"train {tr.lower()}\n→ test {ev}" for tr, ev in groups], fontsize=5.6)
    a.invert_yaxis()
    a.set_xlim(0.58, 0.9)
    a.set_xlabel("AUROC, somatic vs germline (logistic regression, 5-fold CV, mean ± SD)")
    for feat, (c, mk, face) in FEAT_STYLE.items():
        a.plot([], [], mk, ms=3.4, mfc=face, mec=c, mew=0.8, label=feat)
    a.legend(loc="upper right", ncol=1, handletextpad=0.3, borderaxespad=0.2)
    a.grid(axis="x", color="#eee", lw=0.5)
    a.set_axisbelow(True)
    label(a, "a", dx=-0.16)

    # b + c share one gene order (by embedding+XGB AUROC)
    df = pd.read_csv(PUB / "per_gene_auroc_ordered_by_emb_benefit.csv")
    df = df.sort_values("emb_base_xgb").reset_index(drop=True)
    b = fig.add_subplot(gs[1, 1])
    ag = INK
    for i, r in df.iterrows():
        up = r.emb_base_xgb > r.best_baseline
        b.plot([r.best_baseline, r.emb_base_xgb], [i, i], color=ag if up else ANNOT, lw=1.4, alpha=0.45, zorder=1)
        b.scatter(r.best_baseline, i, s=10, marker="s", color=ANNOT, zorder=2, lw=0)
        b.scatter(r.emb_base_xgb, i, s=14, color=ag, zorder=3, lw=0)
    b.set_yticks(range(len(df)))
    b.set_yticklabels([f"{g} ({int(s)})" for g, s in zip(df.gene, df.n_somatic)], style="italic")
    b.set_ylim(-0.6, len(df) - 0.4)
    b.set_xlim(0.55, 0.9)
    b.set_xlabel("AUROC (non-coding, 5-fold CV)")
    b.grid(axis="x", color="#eee", lw=0.5)
    b.set_axisbelow(True)
    b.scatter([], [], s=10, marker="s", color=ANNOT, label="Best annotation-only model")
    b.scatter([], [], s=14, color=ag, label="Annotations + AlphaGenome\nembedding (XGB)")
    b.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    n_up = int((df.emb_base_xgb > df.best_baseline).sum())
    b.set_title(f"{n_up}/{len(df)} genes improve (somatic n in brackets)", loc="left", fontsize=6, color="k")
    label(b, "c", dx=-0.32)

    # b: pooled feature importance (Table 3)
    c = fig.add_subplot(gs[1, 0])
    importance_bars(c, "Share of total importance (%)")
    c.set_title("Random forest on 10 AlphaGenome CD34$^+$ CMP features", loc="left", fontsize=6, color="k")
    label(c, "b", dx=-0.3)
    save(fig, "Fig3_representations_per_gene")


# --------------------------------------------------------------------------- Fig 4
def _table3():
    """(feature, impurity %, mean |SHAP| %), from 119b_pooled_api_rf_importance.py, ordered by SHAP share."""
    d = pd.read_csv(PUB / "pooled_api_rf_impurity_vs_shap.csv").sort_values("shap_pct", ascending=False)
    return [(r.feature, round(r.impurity_pct, 1), round(r.shap_pct, 1)) for r in d.itertuples()]


TABLE3 = _table3()


def importance_bars(dd, xlabel):
    """Fig 4d: pooled feature importance (manuscript Table 3), SHAP and impurity shares."""
    yy = np.arange(len(TABLE3))[::-1]
    dd.barh(yy + 0.19, [r[2] for r in TABLE3], height=0.36, color=INK, label="Mean |SHAP|")
    dd.barh(yy - 0.19, [r[1] for r in TABLE3], height=0.36, color=GERM, label="Impurity importance")
    for yi, r in zip(yy, TABLE3):
        dd.text(r[2] + 0.5, yi + 0.19, f"{r[2]:.1f}", va="center", fontsize=5.2)
    dd.set_yticks(yy)
    dd.set_yticklabels([FEATURE_NAMES[r[0]] for r in TABLE3])
    dd.set_xlabel(xlabel)
    dd.legend(loc="lower right")
    dd.set_xlim(0, 41)


def supp_fig4():
    """Supplementary Fig 4: ASXL1 out-of-fold ROC (a) and score distributions (b, c)."""
    m127 = load_script("127_plot_emb_baseline_xgb_discrimination.py")  # prevalence-matched threshold helper
    d = pd.read_csv(PUB / "asxl1_oof_scores_deidentified.csv")  # gene, y, out-of-fold scores; no variant IDs
    y = d.y.to_numpy(int)

    fig = plt.figure(figsize=(W2, 56 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.11, right=0.98, top=0.82, bottom=0.18)
    a = fig.add_subplot(gs[0, 0])
    keys = [("p_baseline_xgb", ANNOT, "Annotations"), ("p_emb_base_xgb", INK, "+ AlphaGenome embedding")]
    for col, colour, lab in keys:
        fpr, tpr, _ = roc_curve(y, d[col])
        a.plot(fpr, tpr, color=colour, lw=1.3,
               label=f"{lab.replace('AlphaGenome', 'AG')}  {roc_auc_score(y, d[col]):.2f}")
    a.plot([0, 1], [0, 1], ls="--", color="#bbb", lw=0.6)
    a.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False-positive rate", ylabel="True-positive rate")
    a.set_aspect("equal")
    a.legend(loc="lower right", title="AUROC", title_fontsize=5.5, fontsize=5.5, alignment="left", handlelength=1,
             borderaxespad=0.1)
    a.set_title(f"ASXL1 non-coding, out-of-fold\nn = {len(d):,}; {int(y.sum())} somatic", loc="left",
                fontsize=5.8, color="k")
    label(a, "a", dx=-0.38)

    bins = np.linspace(0, 1, 41)
    for k, (col, colour, lab) in enumerate(keys):
        ax = fig.add_subplot(gs[0, 1 + k])
        s = d[col].to_numpy(float)
        ax.hist(s[y == 0], bins=bins, density=True, color=GERM, label="Germline")
        ax.hist(s[y == 1], bins=bins, density=True, color=INK, alpha=0.7, label="Somatic")
        ax.axvline(m127._local_pm_thr(s, y), color=INK, ls="--", lw=0.7)
        ax.set(xlim=(0, 1), xlabel="Out-of-fold P(somatic)", ylabel="Density")
        ax.set_title(lab, loc="left", fontsize=6.5, fontweight="bold", color="k")
        ax.legend(loc="upper left" if k == 0 else "upper right", handlelength=1, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=0.9)
        label(ax, "bc"[k], dx=-0.28)

    save(fig, "SuppFig4_asxl1_interpretation")


# --------------------------------------------------------------------------- Fig 5
def fig5():
    fig = plt.figure(figsize=(W2, 118 * MM))
    a = fig.add_axes([0, 0.56, 1, 0.44])
    a.axis("off")
    a.text(0.005, 0.97, "a", fontsize=8, fontweight="bold", va="top")
    box(a, 0.03, 0.40, 0.14, 0.32, "64 kb sequence\ncentred on\nALT allele", "#F6F7F8", "#D5DBDF")
    arrow(a, 0.175, 0.56, 0.205, 0.56)
    specs = [("AlphaGenome", "fold_1 · 1 bp × 1,536 d"), ("NTv3", "650M post-trained · 1 bp × 1,536 d"),
             ("Borzoi", "trunk · 32 bp bin × 1,920 d")]
    for i, (name, sp) in enumerate(specs):
        yb = 0.70 - i * 0.26
        box(a, 0.21, yb, 0.25, 0.21, f"{name}\n{sp}", "white", INK, fs=5.8)
        a.text(0.21 + 0.25 / 2, yb + 0.168, "", fontsize=1)
    a.text(0.335, 0.955, "Frozen backbone (no gradient)", ha="center", fontsize=6, fontweight="bold", color="k")
    arrow(a, 0.465, 0.56, 0.50, 0.56)
    box(a, 0.505, 0.38, 0.15, 0.36, "Variant-site\nembedding\n→ MLP head\n(256 → 1, BCE)", "#DDE3E7",
        INK)
    a.text(0.58, 0.80, "Only trainable part", ha="center", fontsize=5.6, color="k", style="italic")
    arrow(a, 0.66, 0.56, 0.70, 0.56)
    box(a, 0.705, 0.42, 0.11, 0.28, "Fine-tuned\nscore", "white", INK)
    box(a, 0.705, 0.04, 0.11, 0.26, "Annotations\nLR: CADD, function,\ngene, SBS", "#EEF1F3", ANNOT, fs=5.2)
    arrow(a, 0.82, 0.56, 0.86, 0.45)
    arrow(a, 0.82, 0.17, 0.86, 0.36)
    box(a, 0.865, 0.27, 0.12, 0.26, "Late fusion\nP(somatic)", INK, fs=6, bold=True, tc="white")

    # b: held-out AUROC with 95% bootstrap CIs, two training regimes (both scored on non-coding test)
    s5 = pd.read_csv(RES / "arch21_finetune_table_s5_bootstrap.csv")
    base = pd.read_csv(RES / "arch21_baseline_only_lr_summary.csv")
    boot = pd.read_csv(RES / "arch21_nc_finetune_test_bootstrap" / "finetune_test_auroc_bootstrap.csv")
    boot = boot[boot.model == "baseline_LR"].iloc[0]
    regimes = [("NC", "overall", "Trained on non-coding", boot.auroc_ci_lo, boot.auroc_ci_hi),
               ("CNC", "noncoding", "Trained on coding + non-coding", None, None)]
    order = [("Borzoi", "Borzoi"), ("AlphaGenome", "AG"), ("NTv3", "NTv3")]
    for k, (cohort, region, title, blo, bhi) in enumerate(regimes):
        ax = fig.add_axes([0.11 + k * 0.46, 0.09, 0.38, 0.36])
        br = base[(base.cohort == cohort) & (base.region == region)].iloc[0]
        if blo is not None:
            ax.axvspan(blo, bhi, color=ANNOT, alpha=0.18, lw=0)
        ax.axvline(br.auroc, color=ANNOT, lw=1.2)
        ax.text(br.auroc + 0.003, 2.6, f"annotations only {br.auroc:.3f}", color="k", fontsize=5.5, ha="left",
                va="bottom")
        for i, (name, key) in enumerate(order):
            col = INK
            for setting, dy, filled in [("finetune_only", 0.15, False), ("finetune_plus_baseline_lr", -0.15, True)]:
                r = s5[(s5.model == key) & (s5.cohort == cohort) & (s5.setting == setting)
                       & (s5.region == region)].iloc[0]
                ax.plot([r.auroc_ci_lo, r.auroc_ci_hi], [i + dy] * 2, color=col, lw=1.0)
                ax.scatter(r.auroc, i + dy, s=16, color=col if filled else "white", edgecolor=col, lw=1.0, zorder=3)
                ax.text(r.auroc_ci_hi + 0.003, i + dy, f"{r.auroc:.3f}", va="center", fontsize=5.3)
        ax.set_yticks(range(3))
        ax.set_yticklabels([o[0] for o in order] if k == 0 else [])
        ax.set_xlim(0.66, 0.875)
        ax.set_ylim(-0.5, 3.05)
        n, npos = int(r.n), int(r.n_pos)
        ax.set_xlabel(f"AUROC, held-out non-coding test (n = {n:,}; {npos} somatic)")
        ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold")
        label(ax, "bc"[k], dx=-0.2 if k == 0 else -0.06, dy=1.05)
    ax.scatter([], [], s=16, color="white", edgecolor=INK, lw=1.0, label="Fine-tuned model only")
    ax.scatter([], [], s=16, color=INK, edgecolor=INK, lw=1.0, label="+ annotations (late fusion)")
    ax.plot([], [], color=ANNOT, lw=1.2, label="Annotations only")
    fig.legend(*ax.get_legend_handles_labels(), loc="center", bbox_to_anchor=(0.5, 0.505), ncol=3)
    save(fig, "Fig4_finetuning_arch21")


# --------------------------------------------------------------------------- Fig 6
GW_JSON = {"AlphaGenome": "finetune_covariates_full_age_filtered.json", "NTv3": "ntv3_full_finetune_covariates.json",
           "Borzoi": "borzoi_full_finetune_covariates.json"}
GW_FUNCS = ["splicing", "ncRNA_exonic", "ncRNA_intronic", "exonic", "intronic", "UTR5", "UTR3"]


def _genomewide():
    """Held-out AUROCs for the genome-wide task (same values as the paper's Fig. 5)."""
    from types import SimpleNamespace
    js = {k: json.loads((RES / f).read_text()) for k, f in GW_JSON.items()}

    def metric(d, prefix):
        return next(m["auroc"] for m in d["metrics"] if m["model"].startswith(prefix))

    def by_func(d):
        key = next(k for k in d["by_func"] if k.startswith("ensemble"))
        return {r["func"]: (r["n"], r["auroc"]) for r in d["by_func"][key]["rows"] if not r.get("skipped")}

    names = list(GW_JSON)
    series = [(lab, [round(metric(js[b], pre), 3) for b in names]) for lab, pre in
              [("Covariates only", "covariates_only"), ("Finetune only", "finetune"), ("Ensemble", "ensemble")]]
    bf = {b: by_func(js[b]) for b in names}
    rows = [(f, bf["AlphaGenome"][f][0], *[round(bf[b][f][1], 10) for b in names]) for f in GW_FUNCS]
    return SimpleNamespace(SERIES_A=series, BACKBONES=names, ROWS_B=rows,
                           BACKBONES_B=[(b, 2 + i, None) for i, b in enumerate(names)])


SHAPE = {"AlphaGenome": ("o", 14), "NTv3": ("^", 16), "Borzoi": ("D", 10)}  # Fig 6b marker per model


def fig6():
    f5 = _genomewide()
    f43 = load_script("43_plot_full_cohort_ensemble_delta_vs_covariates_by_func.py")
    cov = f43.load_by_func(RES / "finetune_covariates_full_age_filtered.json", "covariates_only")
    agree = json.loads((RES / "full_model_agreement.json").read_text())

    fig = plt.figure(figsize=(W2, 96 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.15, 0.72], wspace=0.6, left=0.11, right=0.93, top=0.91,
                          bottom=0.31)
    a = fig.add_subplot(gs[0])
    cov_all, ft, ens = (s[1] for s in f5.SERIES_A)
    for i, name in enumerate(f5.BACKBONES):
        a.plot([min(cov_all[i], ft[i], ens[i]), max(cov_all[i], ft[i], ens[i])], [i, i], color="#ddd", lw=1)
        a.scatter(cov_all[i], i, s=14, marker="s", color=ANNOT, zorder=3, lw=0)
        a.scatter(ft[i], i, s=18, color="white", edgecolor=INK, lw=1.0, zorder=3)
        a.scatter(ens[i], i, s=18, color=INK, zorder=3, lw=0)
        a.text(ens[i] + 0.0015, i, f"{ens[i]:.3f}", va="center", fontsize=5.5)
    a.set_yticks(range(3))
    a.set_yticklabels(f5.BACKBONES)
    a.invert_yaxis()
    a.set_ylim(2.5, -0.5)
    a.set_xlim(0.553, 0.597)
    a.set_xticks([0.56, 0.57, 0.58, 0.59])
    a.set_xlabel(f"AUROC, ARCH vs other somatic\n(held-out n = {agree['n_test']:,}; {agree['n_ch']:,} ARCH)")
    a.set_title("Overall", loc="left", fontsize=6.5, fontweight="bold")
    label(a, "a", dx=-0.42, dy=1.04)

    names = {"splicing": "Splicing", "ncRNA_exonic": "ncRNA exonic", "ncRNA_intronic": "ncRNA intronic",
             "exonic": "Exonic", "intronic": "Intronic", "UTR5": "5′UTR", "UTR3": "3′UTR"}
    rows = f5.ROWS_B
    off = {"annot": -0.27, "AlphaGenome": -0.09, "NTv3": 0.09, "Borzoi": 0.27}
    b = fig.add_subplot(gs[1])
    for i, r in enumerate(rows):
        b.axhspan(i - 0.42, i + 0.42, color="#f4f6f8" if i % 2 == 0 else "white", zorder=0, lw=0)
        b.scatter(cov[r[0]]["auroc"], i + off["annot"], s=11, marker="s", color=ANNOT, zorder=2, lw=0)
    for name, idx, _ in f5.BACKBONES_B:
        b.scatter([r[idx] for r in rows], [i + off[name] for i in range(len(rows))], s=SHAPE[name][1], marker=SHAPE[name][0],
                  color=INK, zorder=3, lw=0)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([f"{names[r[0]]} ({r[1]:,})" for r in rows])
    b.invert_yaxis()
    b.axvline(0.5, color="#999", lw=0.6, ls=":")
    b.set_xlim(0.48, 0.79)
    b.set_xlabel("AUROC, fine-tuned model + annotations")
    b.set_title("By variant class (n in test)", loc="left", fontsize=6.5, fontweight="bold")
    label(b, "b", dx=-0.42, dy=1.04)

    c = fig.add_subplot(gs[2])
    order = ["covariates", "alphagenome", "ntv3", "borzoi"]
    labs = ["Annotations", "AlphaGenome", "NTv3", "Borzoi"]
    full = np.full((4, 4), np.nan)
    for key, v in agree["spearman"].items():
        p, q = key.split("__")
        i, j = order.index(p), order.index(q)
        full[max(i, j), min(i, j)] = v
    m = full[1:, :3]  # rows AlphaGenome..Borzoi, columns Annotations..NTv3
    im = c.pcolormesh(np.arange(4) - 0.5, np.arange(4) - 0.5, np.ma.masked_invalid(m), cmap="Greys", vmin=0, vmax=1)
    c.set_xlim(-0.5, 2.5)
    c.set_ylim(2.5, -0.5)
    c.set_aspect("equal")
    for (i, j), v in np.ndenumerate(m):
        if not np.isnan(v):
            c.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else "k")
    c.set_xticks(range(3))
    c.set_xticklabels(labs[:3], rotation=45, ha="right")
    c.set_yticks(range(3))
    c.set_yticklabels(labs[1:])
    for s in c.spines.values():
        s.set_visible(False)
    c.tick_params(length=0)
    c.set_title("Score concordance\n(Spearman ρ)", loc="left", fontsize=6.5, fontweight="bold")
    label(c, "c", dx=-0.52, dy=1.2)

    # one key per panel: in a the model is the row, in b the marker shape identifies the model
    ha_ = [a.scatter([], [], s=14, marker="s", color=ANNOT, lw=0, label="Annotations only"),
           a.scatter([], [], s=18, color="white", edgecolor=INK, lw=1.0, label="Fine-tuned model only"),
           a.scatter([], [], s=18, color=INK, lw=0, label="Fine-tuned + annotations")]
    a.legend(handles=ha_, loc="upper left", bbox_to_anchor=(-0.05, -0.2), handletextpad=0.2)
    hb_ = [b.scatter([], [], s=11, marker="s", color=ANNOT, lw=0, label="Annotations only")]
    hb_ += [b.scatter([], [], s=SHAPE[n][1], marker=SHAPE[n][0], color=INK, lw=0, label=f"{n} + annotations")
            for n in ("AlphaGenome", "NTv3", "Borzoi")]
    b.legend(handles=hb_, loc="upper left", bbox_to_anchor=(-0.05, -0.2), ncol=2, handletextpad=0.2,
             columnspacing=1.0)
    save(fig, "Fig5_genomewide_arch")
    print("  annotation-only by class:", {k: round(v["auroc"], 3) for k, v in cov.items()})


def supp_heatmap():
    """Supplementary Fig. 3 (v5 final set): the former Fig 3c 9-model per-gene AUROC heatmap, standalone."""
    df = pd.read_csv(PUB / "per_gene_auroc_ordered_by_emb_benefit.csv")
    df = df.sort_values("emb_base_xgb").reset_index(drop=True)
    fig = plt.figure(figsize=(120 * MM, 110 * MM))
    c = fig.add_axes([0.2, 0.05, 0.64, 0.86])
    mat = df[[k for k, _ in HEAT_COLS]].to_numpy()
    im = c.pcolormesh(np.arange(mat.shape[1] + 1) - 0.5, np.arange(mat.shape[0] + 1) - 0.5, mat,
                      cmap=HEAT_CMAP, vmin=0.5, vmax=0.9, edgecolors="face", lw=0.1)
    for (i, j), v in np.ndenumerate(mat):
        c.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.8, color="k")
    c.set_xticks(range(len(HEAT_COLS)))
    c.set_xticklabels([lab for _, lab in HEAT_COLS])
    c.set_yticks(range(len(df)))
    c.set_yticklabels([f"{g} ({int(s)})" for g, s in zip(df.gene, df.n_somatic)], style="italic")
    c.tick_params(length=0)
    for s in c.spines.values():
        s.set_visible(False)
    for x in (2.5, 5.5):
        c.axvline(x, color="white", lw=1.5)
    for j, grp in enumerate(["Annotations", "+ AG scores", "+ AG embedding"]):
        c.text(1 + 3 * j, len(df) - 0.1, grp, ha="center", va="bottom", fontsize=6, fontweight="bold", color="k")
    cb = fig.colorbar(im, cax=fig.add_axes([0.87, 0.25, 0.025, 0.45]))
    cb.set_label("AUROC (non-coding, 5-fold CV)")
    cb.solids.set_rasterized(False)
    cb.outline.set_linewidth(0.4)
    save(fig, "SuppFig3_per_gene_9model_heatmap")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig5()  # -> Fig4_finetuning_arch21
    fig6()  # -> Fig5_genomewide_arch
    supp_heatmap()
    supp_fig4()
