#!/usr/bin/env python3
"""
retrieval_predictivity.py — Retrieval Predictivity
═══════════════════════════════════════════════════

Question
  Which metric is the best zero-label proxy for retrieval performance?
  Does the directional form of SAS predict the directionally corresponding
  retrieval task better than any symmetric metric does?

Hypothesis
  SAS(I→T) should correlate more strongly with image-to-text retrieval
  (I→T rSum) than with text-to-image retrieval (T→I rSum), and vice versa
  for SAS(T→I). Symmetric metrics (CKA, CORAL, RMG, Δcos) correlate equally
  with I→T and T→I rSum because they cannot distinguish direction.

Method
  1. For each (model × dataset), compute all metrics AND retrieval rSum
     (both I→T and T→I components, summed over R@1, R@5, R@10 × 100).
  2. Spearman ρ(metric, rSum) computed across models per dataset.
  3. Directional test: |ρ(SAS(I→T), rSum_IT)| > |ρ(SAS(I→T), rSum_TI)|?

Run:
    python retrieval_predictivity.py
    python retrieval_predictivity.py --skip_slow
"""

import argparse
import re as _re
import numpy as np

from _shared import (
    load_XY, compute_metrics, retrieval_stats, ptable, hdr, spearman, fmt_rho,
    ALL_DS, DOMAIN, MODELS,
    MEDICAL_BENCHMARKS, MEDICAL_BENCHMARK_DS, MEDICAL,
)

parser = argparse.ArgumentParser()
parser.add_argument("--skip_slow", action="store_true", help="Skip MMD and SVCCA")
args = parser.parse_args()

# ── collect ────────────────────────────────────────────────────────────────────
rows_data = []   # list of dicts per (model, dataset)

for model in MODELS:
    for dataset in ALL_DS:
        imgs, txts, data = load_XY(dataset, model)
        if imgs is None:
            continue
        m   = compute_metrics(imgs, txts, skip_slow=args.skip_slow)
        ret = retrieval_stats(data)
        rows_data.append({
            "model": model, "dataset": dataset, "domain": DOMAIN[dataset],
            **m, **ret,
        })

if not rows_data:
    print("No feature files found.")
    raise SystemExit(0)

metric_names  = [k for k in rows_data[0]
                 if k not in ("model", "dataset", "domain",
                              "rSum", "rSum_IT", "rSum_TI", "R@1_IT", "R@1_TI")]
datasets_seen = sorted(set(r["dataset"] for r in rows_data))

# ── RESULTS ────────────────────────────────────────────────────────────────────
hdr("RETRIEVAL PREDICTIVITY")
n_obs = len(rows_data)
print(f"\n  Observations: {n_obs}  "
      f"({len(set(r['model'] for r in rows_data))} models × {len(datasets_seen)} datasets)\n")

# Table 1: ρ(metric, rSum) per dataset — three rSum targets
RSUM_TARGETS = [
    ("rSum",    "rSum (full)"),
    ("rSum_IT", "rSum I→T"),
    ("rSum_TI", "rSum T→I"),
]

def _rho_sort_key(row):
    m = _re.search(r"[+-]?\d+\.\d+", row[-1])
    return -abs(float(m.group())) if m else 0.0

for rsum_key, rsum_label in RSUM_TARGETS:
    col_headers = ["Metric"] + datasets_seen + ["pooled"]
    rho_rows = []
    for m in metric_names:
        row = [m]
        all_m, all_r = [], []
        for dataset in datasets_seen:
            ds_rows = [r for r in rows_data if r["dataset"] == dataset]
            m_vals  = [r[m]        for r in ds_rows]
            r_vals  = [r[rsum_key] for r in ds_rows]
            rho_val, p_val = spearman(m_vals, r_vals)
            row.append(fmt_rho(rho_val, p_val, w=9))
            all_m.extend(m_vals)
            all_r.extend(r_vals)
        rho_pool, p_pool = spearman(all_m, all_r)
        row.append(fmt_rho(rho_pool, p_pool, w=9))
        rho_rows.append(row)
    rho_rows.sort(key=_rho_sort_key)
    ptable(col_headers, rho_rows, dec=3,
           title=f"Spearman ρ(metric, {rsum_label}) — one column per dataset")

print("  Significance: *** p<.001  ** p<.01  * p<.05\n")

# Table 2: directional asymmetry test
directional_rows = []
for sas_key, expected_match in [("SAS(I→T)", "rSum_IT"), ("SAS(T→I)", "rSum_TI")]:
    expected_nonmatch = "rSum_TI" if expected_match == "rSum_IT" else "rSum_IT"
    m_vals = [r[sas_key]          for r in rows_data]
    ri_vals = [r[expected_match]    for r in rows_data]
    rj_vals = [r[expected_nonmatch] for r in rows_data]
    rho_match,    p_match    = spearman(m_vals, ri_vals)
    rho_nonmatch, p_nonmatch = spearman(m_vals, rj_vals)
    directional_rows.append([
        sas_key,
        expected_match.replace("rSum_", ""),
        fmt_rho(rho_match,    p_match,    w=9),
        fmt_rho(rho_nonmatch, p_nonmatch, w=9),
        "✓" if (not np.isnan(rho_match) and not np.isnan(rho_nonmatch)
                and abs(rho_match) > abs(rho_nonmatch)) else "✗",
    ])

ptable(
    ["Metric", "Expected match", "ρ matched dir.", "ρ opposite dir.", "Hypothesis holds?"],
    directional_rows,
    title="Directional asymmetry test  [|ρ matched| > |ρ opposite|?]"
)

# Table 3: dataset-level means ± std
sanity_rows = []
for dataset in datasets_seen:
    ds_rows = [r for r in rows_data if r["dataset"] == dataset]
    sanity_rows.append([
        dataset, DOMAIN[dataset],
        f"{np.nanmean([r['R@1_IT']   for r in ds_rows]):.3f}±{np.nanstd([r['R@1_IT']   for r in ds_rows], ddof=1):.3f}",
        f"{np.nanmean([r['R@1_TI']   for r in ds_rows]):.3f}±{np.nanstd([r['R@1_TI']   for r in ds_rows], ddof=1):.3f}",
        f"{np.nanmean([r['rSum']      for r in ds_rows]):.1f}±{np.nanstd([r['rSum']      for r in ds_rows], ddof=1):.1f}",
        f"{np.nanmean([r['SAS(I→T)'] for r in ds_rows]):.4f}±{np.nanstd([r['SAS(I→T)'] for r in ds_rows], ddof=1):.4f}",
        f"{np.nanmean([r['SAS(T→I)'] for r in ds_rows]):.4f}±{np.nanstd([r['SAS(T→I)'] for r in ds_rows], ddof=1):.4f}",
    ])
ptable(
    ["Dataset", "Domain", "R@1 I→T", "R@1 T→I", "rSum", "SAS(I→T)", "SAS(T→I)"],
    sanity_rows,
    title="Dataset-level means ± std (averaged over models)"
)

# Table 4: domain-stratified ρ
_nat4 = [r for r in rows_data if r["domain"] == "natural"]
_med4 = [r for r in rows_data if r["domain"] == "medical"]

print()
print("  Spearman ρ(metric, rSum) stratified by domain:\n")
print(f"  {'Metric':<12}  {'ρ natural':>15}  {'ρ medical':>15}  {'ρ pooled':>15}")
print(f"  {'──────':<12}  {'─────────':>15}  {'─────────':>15}  {'────────':>15}")
for _met in ["Δcos", "SAS(sym)", "SAS(I→T)", "SAS(T→I)", "CKA", "CORAL", "SVCCA", "MMD", "RMG", "SAS_imbal"]:
    if _met not in metric_names:
        continue
    _rn, _pn = spearman([r[_met] for r in _nat4], [r["rSum"] for r in _nat4])
    _rm, _pm = spearman([r[_met] for r in _med4], [r["rSum"] for r in _med4])
    _ra, _pa = spearman([r[_met] for r in rows_data], [r["rSum"] for r in rows_data])
    print(f"  {_met:<12}  {fmt_rho(_rn,_pn,13):>15}  {fmt_rho(_rm,_pm,13):>15}  {fmt_rho(_ra,_pa,13):>15}")
print()

# ── medical-domain benchmark reference ────────────────────────────────────────
hdr("MEDICAL DOMAIN BENCHMARK REFERENCE")

bm_results = []
for _bm in MEDICAL_BENCHMARKS:
    for _dataset in MEDICAL_BENCHMARK_DS:
        _imgs, _txts, _data = load_XY(_dataset, _bm)
        if _imgs is None:
            continue
        _m_bm = compute_metrics(_imgs, _txts, skip_slow=args.skip_slow)
        _ret  = retrieval_stats(_data)
        bm_results.append({"model": _bm, "dataset": _dataset, **_m_bm, **_ret})

if bm_results:
    _bm_rows = []
    for _r in bm_results:
        _bm_rows.append([
            _r["model"], _r["dataset"],
            f"{_r.get('R@1_IT', float('nan')):.3f}",
            f"{_r.get('R@1_TI', float('nan')):.3f}",
            f"{_r.get('rSum',   float('nan')):.1f}",
            f"{_r.get('SAS(I→T)', float('nan')):.4f}",
            f"{_r.get('SAS(T→I)', float('nan')):.4f}",
            f"{_r.get('Δcos',   float('nan')):.4f}",
        ])
    for _dataset in MEDICAL:
        _gp = [r for r in rows_data if r["dataset"] == _dataset]
        if not _gp:
            continue
        _bm_rows.append([
            "── GP med mean", _dataset,
            f"{np.nanmean([r['R@1_IT']   for r in _gp]):.3f}",
            f"{np.nanmean([r['R@1_TI']   for r in _gp]):.3f}",
            f"{np.nanmean([r['rSum']      for r in _gp]):.1f}",
            f"{np.nanmean([r['SAS(I→T)'] for r in _gp]):.4f}",
            f"{np.nanmean([r['SAS(T→I)'] for r in _gp]):.4f}",
            f"{np.nanmean([r['Δcos']     for r in _gp]):.4f}",
        ])
    ptable(["Model", "Dataset", "R@1 I→T", "R@1 T→I", "rSum", "SAS(I→T)", "SAS(T→I)", "Δcos"],
           _bm_rows, title="Retrieval + alignment — benchmarks vs GP medical mean")

    # Out-of-sample generalization check (rank consistency, ROCO only)
    _roco_gp  = [r for r in rows_data if r["dataset"] == "ROCO"]
    _roco_dc  = sorted(r["Δcos"]     for r in _roco_gp)
    _roco_sas = sorted(r["SAS(sym)"] for r in _roco_gp)
    _roco_rs  = sorted(r["rSum"]     for r in _roco_gp)
    _n_gp = len(_roco_gp)

    print("\n  Out-of-sample rank consistency (ROCO only):\n")
    print(f"  {'Model':<22}  {'Δcos':>8}  {'rank (Δcos)':>12}  "
          f"{'rSum':>8}  {'rank (rSum)':>12}  {'gap':>5}  {'status':>14}")
    print(f"  {'─────':<22}  {'────':>8}  {'──────────':>12}  "
          f"{'────':>8}  {'──────────':>12}  {'───':>5}  {'──────':>14}")
    for _bm_r in bm_results:
        if _bm_r["dataset"] != "ROCO":
            continue
        _bm_dc = _bm_r.get("Δcos", float("nan"))
        _bm_rs = _bm_r.get("rSum", float("nan"))
        _above_dc = sum(1 for v in _roco_dc if v < _bm_dc) if not np.isnan(_bm_dc) else -1
        _above_rs = sum(1 for v in _roco_rs if v < _bm_rs) if not np.isnan(_bm_rs) else -1
        _gap      = abs(_above_dc - _above_rs)
        _status   = "✓ consistent" if _gap <= 2 else "✗ inconsistent"
        print(f"  {_bm_r['model']:<22}  {_bm_dc:>8.4f}  "
              f"  {_above_dc:2d}/{_n_gp} GP   {_bm_rs:>8.1f}  "
              f"  {_above_rs:2d}/{_n_gp} GP   {_gap:>5d}  {_status:>14}")
    print()
else:
    print("  No benchmark feature files found.")
