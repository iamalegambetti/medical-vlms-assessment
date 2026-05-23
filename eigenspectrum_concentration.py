#!/usr/bin/env python3
"""
eigenspectrum_concentration.py — Eigenspectrum Concentration
═════════════════════════════════════════════════════════════

Question
  How does the shape of the SAS score-vs-topq curve differ between natural
  and medical datasets?

Hypothesis
  Medical datasets produce a steeper drop in SAS(I→T) as topq increases.
  The alignment signal in medical data is concentrated in the few top
  principal components shared between images and reports, while the bulk
  of the spectrum is mismatched.
  Natural datasets produce a flatter curve because image-text alignment is
  distributed more uniformly across the eigenspectrum.

Method
  For topq ∈ {0.1, 0.2, …, 1.0} and every available (model × dataset):
    compute SAS(I→T) and SAS(T→I).
  Average per domain per q.
  Report the q that maximises |mean_natural − mean_medical|.

Run:
    python eigenspectrum_concentration.py
"""

import argparse
import numpy as np

from _shared import (
    load_XY, tensors, hdr, ptable,
    ALL_DS, DOMAIN, MODELS, NATURAL, MEDICAL,
    MEDICAL_BENCHMARKS, MEDICAL_BENCHMARK_DS,
)

from congruence.metrics import spectral_alignment_score as _sas

parser = argparse.ArgumentParser()
parser.parse_args()

Q_VALUES = [round(i * 0.1, 1) for i in range(1, 11)]   # 0.1 … 1.0

# ── collect ────────────────────────────────────────────────────────────────────
feature_store = {}   # (model, dataset) -> (imgs, txts)

for model in MODELS:
    for dataset in ALL_DS:
        imgs, txts, _ = load_XY(dataset, model)
        if imgs is None:
            continue
        feature_store[(model, dataset)] = (imgs, txts)

if not feature_store:
    print("No feature files found.")
    raise SystemExit(0)

# ── sweep topq ─────────────────────────────────────────────────────────────────
def _safe_sas(imgs, txts, q):
    try:
        v = _sas(imgs, txts, q)
        return float(v.item() if hasattr(v, "item") else v)
    except Exception:
        return float("nan")

sas_it_scores: dict = {q: {} for q in Q_VALUES}
sas_ti_scores: dict = {q: {} for q in Q_VALUES}

for (model, dataset), (imgs, txts) in feature_store.items():
    for q in Q_VALUES:
        sas_it_scores[q][(model, dataset)] = _safe_sas(imgs, txts, q)
        sas_ti_scores[q][(model, dataset)] = _safe_sas(txts, imgs, q)

# ── aggregate by domain ────────────────────────────────────────────────────────
def domain_mean(score_dict, domain):
    vals = [v for (_, d), v in score_dict.items()
            if DOMAIN[d] == domain and not np.isnan(v)]
    return np.mean(vals) if vals else float("nan")

def domain_std(score_dict, domain):
    vals = [v for (_, d), v in score_dict.items()
            if DOMAIN[d] == domain and not np.isnan(v)]
    return np.std(vals, ddof=1) if len(vals) > 1 else float("nan")

def domain_imbalance_stats(it_dict, ti_dict, domain):
    vals = []
    for (m, d) in it_dict:
        if DOMAIN[d] != domain:
            continue
        it_v = it_dict[(m, d)]
        ti_v = ti_dict.get((m, d), float("nan"))
        if not (np.isnan(it_v) or np.isnan(ti_v)):
            vals.append(it_v - ti_v)
    mean = np.mean(vals) if vals else float("nan")
    std  = np.std(vals, ddof=1) if len(vals) > 1 else float("nan")
    return mean, std

it_nat     = {q: domain_mean(sas_it_scores[q], "natural") for q in Q_VALUES}
it_med     = {q: domain_mean(sas_it_scores[q], "medical") for q in Q_VALUES}
ti_nat     = {q: domain_mean(sas_ti_scores[q], "natural") for q in Q_VALUES}
ti_med     = {q: domain_mean(sas_ti_scores[q], "medical") for q in Q_VALUES}
it_nat_std = {q: domain_std(sas_it_scores[q], "natural") for q in Q_VALUES}
it_med_std = {q: domain_std(sas_it_scores[q], "medical") for q in Q_VALUES}
ti_nat_std = {q: domain_std(sas_ti_scores[q], "natural") for q in Q_VALUES}
ti_med_std = {q: domain_std(sas_ti_scores[q], "medical") for q in Q_VALUES}

it_diff = {q: it_nat[q] - it_med[q] for q in Q_VALUES}
ti_diff = {q: ti_nat[q] - ti_med[q] for q in Q_VALUES}
best_q_it = max(Q_VALUES, key=lambda q: abs(it_diff[q]) if not np.isnan(it_diff[q]) else 0.0)
best_q_ti = max(Q_VALUES, key=lambda q: abs(ti_diff[q]) if not np.isnan(ti_diff[q]) else 0.0)

# ── RESULTS ────────────────────────────────────────────────────────────────────
hdr("EIGENSPECTRUM CONCENTRATION")
print(f"\n  Observations: {len(feature_store)} (model × dataset) pairs\n"
      f"  topq range: {Q_VALUES[0]} – {Q_VALUES[-1]}\n")

# Table 1: SAS(I→T) by topq and domain
it_rows = []
for q in Q_VALUES:
    diff_s = ("—" if np.isnan(it_diff[q])
              else f"{it_diff[q]:+.4f}" + (" ◀ max" if q == best_q_it else ""))
    nat_s = ("—" if np.isnan(it_nat[q]) else
              f"{it_nat[q]:.4f}" + (f" ± {it_nat_std[q]:.4f}" if not np.isnan(it_nat_std[q]) else ""))
    med_s = ("—" if np.isnan(it_med[q]) else
              f"{it_med[q]:.4f}" + (f" ± {it_med_std[q]:.4f}" if not np.isnan(it_med_std[q]) else ""))
    it_rows.append([str(q), nat_s, med_s, diff_s])
ptable(["topq", "SAS(I→T) natural", "SAS(I→T) medical", "Δ (nat−med)"],
       it_rows, title="SAS(I→T) vs topq by domain  [◀ marks most discriminative q]")

# Table 2: SAS(T→I) by topq and domain
ti_rows = []
for q in Q_VALUES:
    diff_s = ("—" if np.isnan(ti_diff[q])
              else f"{ti_diff[q]:+.4f}" + (" ◀ max" if q == best_q_ti else ""))
    nat_s = ("—" if np.isnan(ti_nat[q]) else
              f"{ti_nat[q]:.4f}" + (f" ± {ti_nat_std[q]:.4f}" if not np.isnan(ti_nat_std[q]) else ""))
    med_s = ("—" if np.isnan(ti_med[q]) else
              f"{ti_med[q]:.4f}" + (f" ± {ti_med_std[q]:.4f}" if not np.isnan(ti_med_std[q]) else ""))
    ti_rows.append([str(q), nat_s, med_s, diff_s])
ptable(["topq", "SAS(T→I) natural", "SAS(T→I) medical", "Δ (nat−med)"],
       ti_rows, title="SAS(T→I) vs topq by domain  [◀ marks most discriminative q]")

# Table 3: imbalance per domain per q
imbal_rows = []
for q in Q_VALUES:
    imbal_nat_mean, imbal_nat_std = domain_imbalance_stats(sas_it_scores[q], sas_ti_scores[q], "natural")
    imbal_med_mean, imbal_med_std = domain_imbalance_stats(sas_it_scores[q], sas_ti_scores[q], "medical")
    nat_s = ("—" if np.isnan(imbal_nat_mean) else
              f"{imbal_nat_mean:+.4f}" + (f" ± {imbal_nat_std:.4f}" if not np.isnan(imbal_nat_std) else ""))
    med_s = ("—" if np.isnan(imbal_med_mean) else
              f"{imbal_med_mean:+.4f}" + (f" ± {imbal_med_std:.4f}" if not np.isnan(imbal_med_std) else ""))
    imbal_rows.append([str(q), nat_s, med_s])
ptable(["topq", "Imbalance natural", "Imbalance medical"],
       imbal_rows, title="SAS Imbalance = SAS(I→T) − SAS(T→I) per topq and domain")

# Table 4: per-model mean SAS(I→T) at default q=0.1 and best q
SHOW_Q = sorted({0.1, best_q_it})
models_seen = sorted(set(m for m, _ in feature_store))
dsets_seen  = sorted(set(d for _, d in feature_store))
per_model_rows = []
for model in models_seen:
    row = [model]
    for q in SHOW_Q:
        vals = [sas_it_scores[q].get((model, d), float("nan")) for d in dsets_seen]
        mean = np.nanmean(vals) if any(not np.isnan(v) for v in vals) else float("nan")
        row.append("—" if np.isnan(mean) else f"{mean:.4f}")
    per_model_rows.append(row)
ptable(["Model"] + [f"SAS(I→T) q={q}" for q in SHOW_Q],
       per_model_rows,
       title="Per-model mean SAS(I→T) at q=0.1 and most-discriminative q")

print(f"\n  Most discriminative topq — SAS(I→T): q={best_q_it}   SAS(T→I): q={best_q_ti}\n")

# ── medical-domain benchmark reference ────────────────────────────────────────
hdr("MEDICAL DOMAIN BENCHMARK REFERENCE")

bm_feature_store = {}
for _bm in MEDICAL_BENCHMARKS:
    for _dataset in MEDICAL_BENCHMARK_DS:
        _imgs, _txts, _ = load_XY(_dataset, _bm)
        if _imgs is None:
            continue
        bm_feature_store[(_bm, _dataset)] = (_imgs, _txts)

if bm_feature_store:
    bm_it_scores = {q: {} for q in Q_VALUES}
    bm_ti_scores = {q: {} for q in Q_VALUES}
    for (_bm, _dataset), (_imgs, _txts) in bm_feature_store.items():
        for _q in Q_VALUES:
            bm_it_scores[_q][(_bm, _dataset)] = _safe_sas(_imgs, _txts, _q)
            bm_ti_scores[_q][(_bm, _dataset)] = _safe_sas(_txts, _imgs, _q)

    # SAS(I→T) by topq — benchmarks vs GP medical mean
    _bm_rows = []
    for (_bm, _dataset), _ in bm_feature_store.items():
        _row = [_bm, _dataset]
        for _q in Q_VALUES:
            _it = bm_it_scores[_q].get((_bm, _dataset), float("nan"))
            _row.append("—" if np.isnan(_it) else f"{_it:.4f}")
        _bm_rows.append(_row)

    _gp_row = ["── GP med mean", "ROCO+MIMIC"]
    for _q in Q_VALUES:
        _gp_vals = [v for (_, _d), v in sas_it_scores[_q].items()
                    if DOMAIN[_d] == "medical" and not np.isnan(v)]
        _gp_row.append(f"{np.mean(_gp_vals):.4f}" if _gp_vals else "—")
    _bm_rows.append(_gp_row)

    ptable(["Model", "Dataset"] + [f"q={q}" for q in Q_VALUES],
           _bm_rows, title="SAS(I→T) by topq — benchmarks vs GP medical mean")

    # Imbalance by topq — benchmarks vs GP medical mean
    _imb_rows = []
    for (_bm, _dataset), _ in bm_feature_store.items():
        _row = [_bm, _dataset]
        for _q in Q_VALUES:
            _it = bm_it_scores[_q].get((_bm, _dataset), float("nan"))
            _ti = bm_ti_scores[_q].get((_bm, _dataset), float("nan"))
            _ib = _it - _ti if not (np.isnan(_it) or np.isnan(_ti)) else float("nan")
            _row.append("—" if np.isnan(_ib) else f"{_ib:+.4f}")
        _imb_rows.append(_row)

    _gp_imb_row = ["── GP med mean", "ROCO+MIMIC"]
    for _q in Q_VALUES:
        _gp_it = [v for (_, _d), v in sas_it_scores[_q].items()
                  if DOMAIN[_d] == "medical" and not np.isnan(v)]
        _gp_ti = [v for (_, _d), v in sas_ti_scores[_q].items()
                  if DOMAIN[_d] == "medical" and not np.isnan(v)]
        if _gp_it and _gp_ti and len(_gp_it) == len(_gp_ti):
            _gp_imb_row.append(f"{np.mean(_gp_it) - np.mean(_gp_ti):+.4f}")
        else:
            _gp_imb_row.append("—")
    _imb_rows.append(_gp_imb_row)

    ptable(["Model", "Dataset"] + [f"q={q}" for q in Q_VALUES],
           _imb_rows, title="SAS Imbalance by topq — benchmarks vs GP medical mean")
else:
    print("  No benchmark feature files found.")
