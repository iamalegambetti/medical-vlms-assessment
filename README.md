# Cross-Modal Alignment Evaluation

Code accompanying the paper on cross-modal alignment metrics for vision-language models (VLMs), with a focus on natural vs. medical imaging domains.

## Overview

We evaluate a suite of alignment metrics — CKA, CORAL, MMD, RMG, Δcos, and the directional **Spectral Alignment Score (SAS)** — across four datasets and multiple general-purpose VLMs, benchmarking against domain-specialist medical models (PLIP, PubMed-CLIP).

**Datasets**
| Dataset | Domain |
|---|---|
| MSCOCO2014 | Natural |
| Flickr30k | Natural |
| ROCO | Medical |
| MIMIC-CXR | Medical |

## Repository Structure

```
repository/
├── extract_features.py             # Step 1: extract and save image/text embeddings
├── datasets.py                     # Dataset classes (MSCOCO2014, Flickr30k, ROCO, MIMIC-CXR)
├── config.py                       # Model registry (MODEL_CARDS) and device helpers
├── utils.py                        # Model loading, collation, and feature loading
├── extractors.py                   # Image and text feature extraction
├── metrics.py                      # Alignment metrics (CKA, CORAL, MMD, SAS, …)
├── retrieval.py                    # Retrieval evaluation (R@k, rSum)
├── _shared.py                      # Shared utilities (metrics, stats, formatting)
├── alignment_profile.py            # Exp: cross-domain alignment profile
├── imbalance_signature.py          # Exp: SAS directional imbalance signature
├── retrieval_predictivity.py       # Exp: metric correlation with retrieval performance
├── eigenspectrum_concentration.py  # Exp: SAS vs topq eigenspectrum sweep
├── requirements.txt
└── README.md
```

The repository is fully self-contained — no external packages beyond those listed in `requirements.txt`.

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `transformers`, `Pillow`, `tqdm`, `scipy`, `numpy`, `pandas`, `scikit-learn`.

## Usage

All scripts are run from within the `repository/` directory (or via absolute path from anywhere).

### Step 1 — Set dataset path

Point `DATA_PATH` to the directory containing raw dataset folders:

```bash
export DATA_PATH=/path/to/your/data   # must contain MSCOCO2014/, Flickr30k/, ROCO/, MIMIC-CXR/
```

### Step 2 — Extract Features

Extract and save image/text embeddings for each model and dataset:

```bash
cd repository/

python extract_features.py --dataset MSCOCO2014 --model clip-large --split test
python extract_features.py --dataset Flickr30k   --model clip-large --split test
python extract_features.py --dataset ROCO        --model clip-large --split test
python extract_features.py --dataset MIMIC-CXR   --model clip-large --split test
```

Repeat for each model in `MODEL_CARDS` (see `config.py`). Features are saved to `data/<model>/`.

The default `--output_dir` is `data/` next to the script, which is also where the experiment scripts load from. Override with `--output_dir /custom/path` and set `DATA_DIR=/custom/path` accordingly.

### Step 3 — Run Experiments

Run each experiment script from the `repository/` directory:

**Cross-domain alignment profile**
```bash
python alignment_profile.py
python alignment_profile.py --skip_slow   # skip MMD and SVCCA (much faster)
```

**SAS imbalance signature**
```bash
python imbalance_signature.py
```

**Retrieval predictivity**
```bash
python retrieval_predictivity.py
python retrieval_predictivity.py --skip_slow
```

**Eigenspectrum concentration**
```bash
python eigenspectrum_concentration.py
```

All scripts auto-discover available feature files and print structured tables to stdout. Missing model/dataset combinations are silently skipped.

## Experiments

### Alignment Profile (`alignment_profile.py`)

Measures whether the full metric suite consistently detects alignment degradation when moving from natural to medical datasets. Reports mean ± std per domain for each metric and Mann-Whitney U significance.

### Imbalance Signature (`imbalance_signature.py`)

Tests whether `SAS(I→T) − SAS(T→I)` is near zero on natural datasets but significantly positive on medical datasets. Uses one-sample t-tests (H₀: μ = 0) per domain and a one-sided Mann-Whitney U test. This directional asymmetry is invisible to all symmetric metrics (CKA, CORAL, MMD, RMG).

### Retrieval Predictivity (`retrieval_predictivity.py`)

Evaluates which metric is the strongest zero-label proxy for retrieval performance (rSum = Σ R@{1,5,10} × 100 for I→T and T→I). Includes a directional test: `|ρ(SAS(I→T), rSum_IT)| > |ρ(SAS(I→T), rSum_TI)|`?

### Eigenspectrum Concentration (`eigenspectrum_concentration.py`)

Sweeps `topq ∈ {0.1, …, 1.0}` and plots the domain-level SAS curves. Identifies the topq that maximally discriminates natural from medical alignment, empirically justifying the default `topq = 0.1`.

## Metrics

| Metric | Type | Direction | Description |
|---|---|---|---|
| CKA | similarity | ↑ | Centered kernel alignment |
| CORAL | distance | ↓ | Covariance distribution distance |
| MMD | distance | ↓ | Maximum mean discrepancy |
| RMG | distance | ↓ | Relative modality gap (centroid distance) |
| Δcos | similarity | ↑ | Mean matched − mean unmatched cosine similarity |
| SAS(I→T) | similarity | ↑ | Spectral alignment score, image → text direction |
| SAS(T→I) | similarity | ↑ | Spectral alignment score, text → image direction |
| SAS(sym) | similarity | ↑ | Symmetric SAS |
| SAS_imbal | directional | — | SAS(I→T) − SAS(T→I); positive = image space more structured |

## Medical Benchmark Models

PLIP and PubMed-CLIP are domain-specialist models included as reference points. They are **excluded from all pooled statistical tests** and reported separately in each experiment under "Medical Domain Benchmark Reference".

