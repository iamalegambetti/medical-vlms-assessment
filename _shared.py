#!/usr/bin/env python3
"""
_shared.py — Shared utilities for CIKM experiment scripts.

Imported by alignment_profile.py, imbalance_signature.py,
retrieval_predictivity.py, and eigenspectrum_concentration.py.
Handles path setup, feature loading, metric computation,
retrieval evaluation, statistics, and table formatting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np

try:
    from scipy.stats import spearmanr, mannwhitneyu, ttest_1samp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[WARNING] scipy not found — statistical tests will be skipped.")

from config import MODEL_CARDS
from utils import get_data
from retrieval import evaluate_retrieval
from metrics import (
    linear_cka,
    coral_distance,
    mmd                                as _mmd,
    svcca                              as _svcca,
    relative_modality_gap              as _rmg,
    spectral_alignment_score           as _sas,
    symmetric_spectral_alignment_score as _sym_sas,
    centroid_distances,
)

# ── registry ───────────────────────────────────────────────────────────────────
NATURAL  = ["MSCOCO2014", "Flickr30k"]
MEDICAL  = ["ROCO", "MIMIC-CXR"]
ALL_DS   = NATURAL + MEDICAL
DOMAIN   = {**{d: "natural" for d in NATURAL}, **{d: "medical" for d in MEDICAL}}

MODELS = [m for m in MODEL_CARDS
          if m not in ("bedrock", "azure", "vertex", "plip", "pubmed-clip-base")]

MEDICAL_BENCHMARKS   = ["plip", "pubmed-clip-base"]
MEDICAL_BENCHMARK_DS = MEDICAL

SAS_Q = 0.1
SPLIT = "test"

HIGHER_BETTER = {"CKA", "SVCCA", "SAS(I→T)", "SAS(T→I)", "SAS(sym)", "Δcos"}
LOWER_BETTER  = {"CORAL", "MMD", "RMG"}

# ── feature loading ────────────────────────────────────────────────────────────
def tensors(data: list) -> tuple:
    """Stack (N, D) image and first-caption text float tensors from a feature list."""
    imgs = torch.cat([d["image"]    for d in data], dim=0).float()
    txts = torch.cat([d["text"][:1] for d in data], dim=0).float()
    return imgs, txts


def load_XY(dataset: str, model: str, split: str = SPLIT):
    """
    Load pre-extracted features for (dataset, model).
    Returns (imgs, txts, raw_data) or (None, None, None) if the file is missing.
    """
    try:
        data = get_data(dataset, model, split)
        imgs, txts = tensors(data)
        return imgs, txts, data
    except Exception:
        return None, None, None


# ── metric computation ─────────────────────────────────────────────────────────
def _safe(fn, *args, **kwargs) -> float:
    try:
        v = fn(*args, **kwargs)
        return float(v.item() if hasattr(v, "item") else v)
    except Exception:
        return float("nan")


def cosine_gap(imgs: torch.Tensor, txts: torch.Tensor) -> float:
    """Δcos: mean diagonal cosine similarity minus mean off-diagonal."""
    i = F.normalize(imgs, dim=-1)
    t = F.normalize(txts, dim=-1)
    sim = i @ t.T
    n   = sim.shape[0]
    on  = sim.diagonal().mean().item()
    off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)].mean().item()
    return on - off


def compute_metrics(
    imgs: torch.Tensor,
    txts: torch.Tensor,
    skip_slow: bool = False,
    topq: float = SAS_Q,
) -> dict:
    """
    Compute all alignment metrics for one (imgs, txts) pair.
    skip_slow=True omits MMD and SVCCA.
    Returns a dict of metric_name -> float.
    """
    d: dict = {}
    d["CKA"]       = _safe(linear_cka, imgs, txts)
    d["CORAL"]     = _safe(coral_distance, imgs, txts)
    d["RMG"]       = _safe(_rmg, imgs, txts)
    d["Δcos"]      = _safe(cosine_gap, imgs, txts)
    d["SAS(I→T)"]  = _safe(_sas, imgs, txts, topq)
    d["SAS(T→I)"]  = _safe(_sas, txts, imgs, topq)
    d["SAS(sym)"]  = _safe(_sym_sas, imgs, txts, topq)
    d["SAS_imbal"] = (
        d["SAS(I→T)"] - d["SAS(T→I)"]
        if not (np.isnan(d["SAS(I→T)"]) or np.isnan(d["SAS(T→I)"]))
        else float("nan")
    )
    if not skip_slow:
        d["MMD"]   = _safe(_mmd, imgs, txts)
        d["SVCCA"] = _safe(_svcca, imgs, txts)
    return d


def retrieval_stats(data: list) -> dict:
    """
    Compute I→T and T→I retrieval metrics.
    Returns rSum, rSum_IT, rSum_TI, R@1_IT, R@1_TI.
    """
    try:
        r  = evaluate_retrieval(data, ks=(1, 5, 10))
        it = sum(r.get(f"I→T R@{k}", 0.0) for k in (1, 5, 10)) * 100
        ti = sum(r.get(f"T→I R@{k}", 0.0) for k in (1, 5, 10)) * 100
        return {
            "rSum":    it + ti,
            "rSum_IT": it,
            "rSum_TI": ti,
            "R@1_IT":  r.get("I→T R@1", float("nan")),
            "R@1_TI":  r.get("T→I R@1", float("nan")),
        }
    except Exception:
        return {k: float("nan") for k in ("rSum", "rSum_IT", "rSum_TI", "R@1_IT", "R@1_TI")}


# ── statistics ─────────────────────────────────────────────────────────────────
def spearman(xs, ys):
    """Spearman ρ and p-value. Returns (nan, nan) for n < 3 or missing scipy."""
    if not _HAS_SCIPY:
        return float("nan"), float("nan")
    pairs = [(x, y) for x, y in zip(xs, ys)
             if not (np.isnan(float(x)) or np.isnan(float(y)))]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    a, b = zip(*pairs)
    r, p = spearmanr(a, b)
    return float(r), float(p)


def mwu(xs, ys, alt="two-sided") -> float:
    """Mann-Whitney U p-value."""
    if not _HAS_SCIPY:
        return float("nan")
    xs = [x for x in xs if not np.isnan(x)]
    ys = [y for y in ys if not np.isnan(y)]
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    _, p = mannwhitneyu(xs, ys, alternative=alt)
    return float(p)


def mwu_full(xs, ys, alt="two-sided"):
    """Mann-Whitney U statistic and p-value."""
    if not _HAS_SCIPY:
        return float("nan"), float("nan")
    xs = [x for x in xs if not np.isnan(x)]
    ys = [y for y in ys if not np.isnan(y)]
    if len(xs) < 2 or len(ys) < 2:
        return float("nan"), float("nan")
    u, p = mannwhitneyu(xs, ys, alternative=alt)
    return float(u), float(p)


def t1samp(xs, mu: float = 0.0) -> float:
    """One-sample t-test p-value against mu."""
    if not _HAS_SCIPY:
        return float("nan")
    xs = [x for x in xs if not np.isnan(x)]
    if len(xs) < 3:
        return float("nan")
    _, p = ttest_1samp(xs, mu)
    return float(p)


def t1samp_full(xs, mu: float = 0.0):
    """One-sample t-test statistic and p-value against mu."""
    if not _HAS_SCIPY:
        return float("nan"), float("nan")
    xs = [x for x in xs if not np.isnan(x)]
    if len(xs) < 3:
        return float("nan"), float("nan")
    t, p = ttest_1samp(xs, mu)
    return float(t), float(p)


# ── formatting ─────────────────────────────────────────────────────────────────
def _fmt(v, dec: int = 4) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(v)


def ptable(headers: list, rows: list, dec: int = 4, title: str = ""):
    """Print a plain-text fixed-width table."""
    if title:
        print(f"\n  {title}\n")
    if not rows:
        print("  (no data)")
        return
    str_rows = [[_fmt(c, dec) for c in r] for r in rows]
    hdrs     = [str(h) for h in headers]
    col_w    = [
        max(len(h), max(len(r[i]) for r in str_rows))
        for i, h in enumerate(hdrs)
    ]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(hdrs, col_w)))
    print("  " + "  ".join("─" * w for w in col_w))
    for row in str_rows:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, col_w)))


def hdr(title: str, width: int = 76):
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)


def sig_str(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "


def fmt_rho(r: float, p: float, w: int = 10) -> str:
    if np.isnan(r):
        return "—".rjust(w)
    return f"{r:+.3f}{sig_str(p)}".rjust(w)
