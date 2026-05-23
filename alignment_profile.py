#!/usr/bin/env python3
"""
alignment_profile.py — Cross-Domain Alignment Profile
══════════════════════════════════════════════════════

Question
  Does the full metric suite consistently detect alignment degradation
  when moving from natural (MSCOCO2014, Flickr30k) to medical (ROCO,
  MIMIC-CXR) datasets?

Hypothesis
  Every metric should score lower alignment on medical data for general-
  purpose VLMs. The magnitude of the drop differs across metrics, which
  is itself evidence that the suite is non-redundant.

Method
  For every available (model × dataset) pair compute: CKA, CORAL, RMG,
  Δcos, SAS(sym), SAS(I→T), SAS(T→I), and optionally MMD + SVCCA.
  Report mean ± std per domain and Mann-Whitney U significance (two-sided).

Run:
    python alignment_profile.py
    python alignment_profile.py --skip_slow
"""

import argparse
import numpy as np

from _shared import (
    load_XY, compute_metrics, ptable, hdr, mwu, mwu_full,
    ALL_DS, DOMAIN, MODELS, NATURAL, MEDICAL,
    HIGHER_BETTER, LOWER_BETTER,
    MEDICAL_BENCHMARKS, MEDICAL_BENCHMARK_DS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--skip_slow", action="store_true",
                    help="Skip MMD and SVCCA (slow for large datasets)")
args = parser.parse_args()

# ── collect ────────────────────────────────────────────────────────────────────
results = {}   # (model, dataset) -> {metric: float}

for model in MODELS:
    for dataset in ALL_DS:
        imgs, txts, _ = load_XY(dataset, model)
        if imgs is None:
            continue
        results[(model, dataset)] = compute_metrics(imgs, txts, skip_slow=args.skip_slow)

if not results:
    print("No feature files found. Run extract_features.py first.")
    raise SystemExit(0)

metric_names = list(next(iter(results.values())).keys())

# ── aggregate by domain ────────────────────────────────────────────────────────
dom: dict = {
    "natural": {m: [] for m in metric_names},
    "medical": {m: [] for m in metric_names},
}
for (_, dataset), vals in results.items():
    for k, v in vals.items():
        if not np.isnan(v):
            dom[DOMAIN[dataset]][k].append(v)

# ── RESULTS ────────────────────────────────────────────────────────────────────
hdr("CROSS-DOMAIN ALIGNMENT PROFILE")

n_models  = len(set(m for m, _ in results))
n_natural = sum(1 for (_, d) in results if DOMAIN[d] == "natural")
n_medical = sum(1 for (_, d) in results if DOMAIN[d] == "medical")
print(f"\n  Models: {n_models}   Natural pairs: {n_natural}   Medical pairs: {n_medical}\n")

# Table 1: domain summary with statistical test
rows = []
for m in metric_names:
    nat = dom["natural"][m]
    med = dom["medical"][m]
    dir_s = "↑" if m in HIGHER_BETTER else ("↓" if m in LOWER_BETTER else "±")
    nat_s = f"{np.mean(nat):.4f} ± {np.std(nat):.4f}" if nat else "—"
    med_s = f"{np.mean(med):.4f} ± {np.std(med):.4f}" if med else "—"
    u, p  = mwu_full(nat, med)
    u_s   = f"{u:.0f}" if not np.isnan(u) else "n/a"
    p_s   = (f"{p:.3f}" + (" *" if p < 0.05 else "  ")) if not np.isnan(p) else "n/a"
    rows.append([m, dir_s, nat_s, med_s, u_s, p_s])

ptable(
    ["Metric", "Dir", "Natural (mean ± std)", "Medical (mean ± std)", "MWU U", "MWU p"],
    rows,
    title="Metric values by domain (averaged across models)"
)

# Table 2: SAS(sym) per model × dataset
models_seen = sorted(set(m for m, _ in results))
dsets_seen  = [d for d in ALL_DS if any((mdl, d) in results for mdl in models_seen)]

detail_rows = []
for model in models_seen:
    row = [model]
    for d in dsets_seen:
        v = results.get((model, d), {}).get("SAS(sym)", float("nan"))
        row.append("—" if np.isnan(v) else f"{v:.4f}")
    detail_rows.append(row)

footer = ["── mean"]
for d in dsets_seen:
    vals  = [results.get((mdl, d), {}).get("SAS(sym)", float("nan")) for mdl in models_seen]
    valid = [v for v in vals if not np.isnan(v)]
    footer.append(f"{np.mean(valid):.4f}" if valid else "—")
detail_rows.append(footer)

ptable(["Model"] + dsets_seen, detail_rows, dec=4,
       title="SAS(sym) per model × dataset  [higher = more aligned]")

# Table 3: per-dataset mean for every metric
metric_ds_rows = []
for m in metric_names:
    row = [m]
    for d in dsets_seen:
        vals  = [results.get((mdl, d), {}).get(m, float("nan")) for mdl in models_seen]
        valid = [v for v in vals if not np.isnan(v)]
        row.append(f"{np.mean(valid):.4f}" if valid else "—")
    metric_ds_rows.append(row)

ptable(["Metric"] + dsets_seen, metric_ds_rows,
       title="All metric means per dataset (averaged over models)")

# ── medical-domain benchmark reference ────────────────────────────────────────
hdr("MEDICAL DOMAIN BENCHMARK REFERENCE")

bm_results = {}
for bm in MEDICAL_BENCHMARKS:
    for dataset in MEDICAL_BENCHMARK_DS:
        imgs, txts, _ = load_XY(dataset, bm)
        if imgs is None:
            continue
        bm_results[(bm, dataset)] = compute_metrics(imgs, txts, skip_slow=args.skip_slow)

if bm_results:
    KEY_METRICS = ["CKA", "Δcos", "SAS(sym)", "SAS_imbal", "MMD", "SVCCA"]
    key_mets = [m for m in KEY_METRICS if m in metric_names]

    bm_rows = []
    for bm in MEDICAL_BENCHMARKS:
        for d in MEDICAL:
            v = bm_results.get((bm, d), {})
            if not v:
                continue
            row = [bm, d]
            for met in key_mets:
                val = v.get(met, float("nan"))
                row.append("—" if np.isnan(val)
                            else f"{val:+.4f}" if met == "SAS_imbal" else f"{val:.4f}")
            bm_rows.append(row)

    for label, domain_key in [("── GP nat mean", "natural"), ("── GP med mean", "medical")]:
        row = [label, "all"]
        for met in key_mets:
            vals  = dom[domain_key][met]
            row.append(f"{np.mean(vals):.4f}" if vals else "—")
        bm_rows.append(row)

    ptable(["Model", "Dataset"] + key_mets, bm_rows,
           title="Key metrics — benchmarks + GP domain means")
else:
    print("  No benchmark feature files found.")
