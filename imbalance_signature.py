#!/usr/bin/env python3
"""
imbalance_signature.py — SAS Imbalance Signature on Medical Data
════════════════════════════════════════════════════════════════

Question
  Is SAS(I→T) − SAS(T→I) near zero on natural datasets but significantly
  positive on medical datasets?

Hypothesis
  On MSCOCO / Flickr30k, images and text are mutually expressive: the
  dominant eigenmodes of the image space are well-matched by the text space
  and vice versa → imbalance ≈ 0.
  On ROCO / MIMIC-CXR, radiology images contain rich anatomical structure
  concentrated in their top principal components, but free-text reports do
  not fully capture this → SAS(I→T) > SAS(T→I) → imbalance > 0.
  This asymmetry is invisible to all symmetric metrics (CKA, CORAL, MMD, RMG).

Method
  1. Compute SAS(I→T) and SAS(T→I) for every available (model × dataset) pair.
  2. One-sample t-test (H0: μ = 0) on the imbalance values per domain.
  3. One-sided Mann-Whitney U: medical imbalance > natural imbalance.

Run:
    python imbalance_signature.py
"""

import argparse
import numpy as np

from _shared import (
    load_XY, compute_metrics, ptable, hdr, t1samp, t1samp_full, mwu,
    ALL_DS, DOMAIN, MODELS, NATURAL, MEDICAL, SAS_Q,
    MEDICAL_BENCHMARKS, MEDICAL_BENCHMARK_DS,
)

parser = argparse.ArgumentParser()
parser.parse_args()

# ── collect ────────────────────────────────────────────────────────────────────
results = {}   # (model, dataset) -> metric dict

for model in MODELS:
    for dataset in ALL_DS:
        imgs, txts, _ = load_XY(dataset, model)
        if imgs is None:
            continue
        results[(model, dataset)] = compute_metrics(imgs, txts, skip_slow=True)

if not results:
    print("No feature files found. Run extract_features.py first.")
    raise SystemExit(0)

# ── aggregate ──────────────────────────────────────────────────────────────────
nat_imbal = [v["SAS_imbal"] for (_, d), v in results.items()
             if DOMAIN[d] == "natural" and not np.isnan(v["SAS_imbal"])]
med_imbal = [v["SAS_imbal"] for (_, d), v in results.items()
             if DOMAIN[d] == "medical" and not np.isnan(v["SAS_imbal"])]

models_seen = sorted(set(m for m, _ in results))
dsets_seen  = [d for d in ALL_DS if any((mdl, d) in results for mdl in models_seen)]

# ── RESULTS ────────────────────────────────────────────────────────────────────
hdr("SAS IMBALANCE SIGNATURE")
print(f"\n  SAS topq = {SAS_Q}  (top {int(SAS_Q*100)}% of eigenspectrum)\n")

# Table 1: SAS(I→T) per model × dataset
for sas_key in ["SAS(I→T)", "SAS(T→I)"]:
    rows = []
    for model in models_seen:
        row = [model]
        for d in dsets_seen:
            v = results.get((model, d), {}).get(sas_key, float("nan"))
            row.append("—" if np.isnan(v) else f"{v:.4f}")
        rows.append(row)
    footer = ["── mean"]
    for d in dsets_seen:
        vals  = [results.get((mdl, d), {}).get(sas_key, float("nan")) for mdl in models_seen]
        valid = [v for v in vals if not np.isnan(v)]
        footer.append(f"{np.mean(valid):.4f}" if valid else "—")
    rows.append(footer)
    ptable(["Model"] + dsets_seen, rows,
           title=f"{sas_key} per model × dataset  [higher = more aligned]")

# Table 2: imbalance per model × dataset
imbal_rows = []
for model in models_seen:
    row = [model]
    for d in dsets_seen:
        v = results.get((model, d), {}).get("SAS_imbal", float("nan"))
        row.append("—" if np.isnan(v) else f"{v:+.4f}")
    imbal_rows.append(row)

footer = ["── mean"]
for d in dsets_seen:
    vals  = [results.get((mdl, d), {}).get("SAS_imbal", float("nan")) for mdl in models_seen]
    valid = [v for v in vals if not np.isnan(v)]
    footer.append(f"{np.mean(valid):+.4f}" if valid else "—")
imbal_rows.append(footer)

ptable(["Model"] + dsets_seen, imbal_rows,
       title="SAS Imbalance = SAS(I→T) − SAS(T→I)  [positive = image space more structured]")

# Table 3: domain-level summary + statistical tests
def _ms(xs):
    xs = [x for x in xs if not np.isnan(x)]
    return (f"{np.mean(xs):+.4f} ± {np.std(xs):.4f}" if xs else "—"), len(xs)

def _ts(t, p):
    if np.isnan(t):
        return "n/a", "n/a"
    return f"{t:.3f}", f"{p:.3f}" + (" *" if p < 0.05 else "")

t_nat, p_nat = t1samp_full(nat_imbal, mu=0.0)
t_med, p_med = t1samp_full(med_imbal, mu=0.0)
p_mwu        = mwu(med_imbal, nat_imbal, alt="greater")

nat_s, n_nat = _ms(nat_imbal)
med_s, n_med = _ms(med_imbal)
nat_t_s, nat_p_s = _ts(t_nat, p_nat)
med_t_s, med_p_s = _ts(t_med, p_med)
p_mwu_s = (f"{p_mwu:.3f}" + (" *" if p_mwu < 0.05 else "")) if not np.isnan(p_mwu) else "n/a"

ptable(
    ["Domain", "Imbalance (mean ± std)", "N", "t", "t-test p (μ=0)", "Prediction"],
    [
        ["Natural", nat_s, str(n_nat), nat_t_s, nat_p_s, "≈ 0"],
        ["Medical", med_s, str(n_med), med_t_s, med_p_s, "> 0"],
    ],
    title="Domain-level summary"
)
print(f"\n  Mann-Whitney U (one-sided, H1: medical > natural):  p = {p_mwu_s}")
print("  Significance: * p<.05   ** p<.01   *** p<.001\n")

# Table 4: per-dataset breakdown
print("  SAS Imbalance per dataset (all models):\n")
print(f"  {'Dataset':<12}  {'Domain':<8}  {'Mean':>8}  {'Std':>8}  {'All positive?':>14}")
print(f"  {'───────':<12}  {'──────':<8}  {'────':>8}  {'───':>8}  {'─────────────':>14}")
for ds, dom_label in [("MSCOCO2014", "natural"), ("Flickr30k", "natural"),
                      ("ROCO", "medical"), ("MIMIC-CXR", "medical")]:
    imbs = [v["SAS_imbal"] for (_, d), v in results.items()
            if d == ds and not np.isnan(v["SAS_imbal"])]
    if not imbs:
        continue
    all_pos = "yes" if all(v > 0 for v in imbs) else "no"
    print(f"  {ds:<12}  {dom_label:<8}  {np.mean(imbs):>+8.4f}  {np.std(imbs):>8.4f}  {all_pos:>14}")
print()

# ── medical-domain benchmark reference ────────────────────────────────────────
hdr("MEDICAL DOMAIN BENCHMARK REFERENCE")

bm_results = {}
for bm in MEDICAL_BENCHMARKS:
    for dataset in MEDICAL_BENCHMARK_DS:
        imgs, txts, _ = load_XY(dataset, bm)
        if imgs is None:
            continue
        bm_results[(bm, dataset)] = compute_metrics(imgs, txts, skip_slow=True)

if bm_results:
    bm_rows = []
    for bm in MEDICAL_BENCHMARKS:
        for d in MEDICAL:
            v = bm_results.get((bm, d), {})
            if not v:
                continue
            it  = v.get("SAS(I→T)", float("nan"))
            ti  = v.get("SAS(T→I)", float("nan"))
            ib  = v.get("SAS_imbal", float("nan"))
            bm_rows.append([
                bm, d,
                "—" if np.isnan(it) else f"{it:.4f}",
                "—" if np.isnan(ti) else f"{ti:.4f}",
                "—" if np.isnan(ib) else f"{ib:+.4f}",
            ])

    for d in MEDICAL:
        gp_it  = [results.get((m, d), {}).get("SAS(I→T)",  float("nan")) for m in models_seen]
        gp_ti  = [results.get((m, d), {}).get("SAS(T→I)",  float("nan")) for m in models_seen]
        gp_ib  = [results.get((m, d), {}).get("SAS_imbal", float("nan")) for m in models_seen]
        bm_rows.append([
            "── GP med mean", d,
            f"{np.nanmean(gp_it):.4f}",
            f"{np.nanmean(gp_ti):.4f}",
            f"{np.nanmean(gp_ib):+.4f}",
        ])

    ptable(["Model", "Dataset", "SAS(I→T)", "SAS(T→I)", "Imbalance"],
           bm_rows, title="SAS imbalance — benchmarks vs GP medical mean")
else:
    print("  No benchmark feature files found.")
