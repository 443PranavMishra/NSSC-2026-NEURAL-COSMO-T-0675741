# Unsupervised Anomaly Detection in Mars HiRISE Orbital Imagery

**NSSC 2026 — National Students' Space Challenge**
**Indian Institute of Technology, Kharagpur | Data Analytics Track**

**Team:** NEURAL COSMO (Team ID: T-0675741)
**Members:** Pranav Mishra (26-846766), Arhan Ansari (26-838121)

---

## Overview

This project implements a complete, end-to-end **unsupervised anomaly detection pipeline** for identifying "Genesis Outlier" contamination in a dataset of 10,422 unlabeled Mars HiRISE surface crops. No labels, pretrained weights, or transfer learning were used at any stage — the pipeline learns what normal Martian terrain looks like from the data alone, and flags deviations from that learned representation as statistically anomalous.

The disclosed contamination spans three categories: **semantic splicing** (spliced terrain), **domain-shifted terrestrial content** (non-Martian material), and **synthetic sensor artifacts** (simulated hardware glitches) — deliberately mixed so that no single naive heuristic could solve the task.

## Pipeline

```
10,422 Mars Images
       ↓
  Autoencoder (from scratch, PyTorch)
       ↓
  256-Dimensional Latent Vector
       ↓
  Isolation Forest (scikit-learn)
       ↓
  Novelty Score (higher = more anomalous)
       ↓
  Statistical Threshold (Median + 3×MAD, not arbitrary)
       ↓
  46 Anomalous Images (0.44%)
       ↓
  Reconstruction Error Heatmaps (top-5)
       ↓
  Physical / Geological Explanation
```

## Repository Contents

| File | Description |
|---|---|
| `NSSC2026_HiRISE_Anomaly_Detection.ipynb` | Full, documented Jupyter notebook — all code, markdown headers per question/sub-question, and executed outputs |
| `NEURAL_COSMO_REPORT.pdf` | Complete PDF report — architecture, statistical justification, plots, tables, heatmaps, geological hypotheses, and the engineering changelog (see **pages 21–24**) |
| `Novelty_Scores.csv` | Novelty score and flagged status for all 10,422 crops (`filename`, `novelty_score`, `flagged`) |
| `README.md` | This file |

## Key Results

| Metric | Value |
|---|---|
| Total crops analyzed | 10,422 |
| Crops flagged as anomalous | **46 (0.44%)** |
| Statistical threshold | 0.5256 (Median + 3×MAD) |
| Threshold vs. 75th percentile gap | +0.082 |
| D'Agostino normality test | p < 0.000001 (distribution is non-Gaussian, right-skewed) |
| Autoencoder parameters | ~5.1 million |
| Final model checkpoint size | ~61.5 MB |
| Training duration | ~90 epochs (early-stopped, patience = 15) |

## Model Architecture

- **Encoder:** 5 stride-2 convolutions, 227×227 → 7×7 spatial resolution, followed by a 1×1 channel-squeeze (256→64 channels)
- **Bottleneck:** fully connected layer to a 256-dimensional latent vector, with dropout (p=0.3)
- **Decoder:** mirrors the encoder exactly in reverse, with precisely tuned `output_padding` to reconstruct an exact 227×227 output
- **Loss:** combined MSE + SSIM (`loss = 0.6·MSE + 0.4·(1 − SSIM)`)
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)
- **Novelty detection:** Isolation Forest (300 estimators) on the latent vectors, with the sklearn score convention explicitly corrected so that higher `novelty_score` = more anomalous

## Engineering Changelog Summary

The model went through four documented iterations (symptom → diagnosis → fix), exceeding the required minimum of three:

1. **v1 → v2 — Checkpoint size failure:** ~26M-parameter bottleneck produced checkpoints exceeding GitHub's 100MB limit → added a downsampling layer + channel squeeze → reduced to ~4.3M parameters (~52MB checkpoints)
2. **v2 → v3 — Underfitting:** overly aggressive 32-channel squeeze caused generic, content-independent blurry reconstructions → widened to 64 channels → reconstructions became input-specific
3. **v3 → v4 — Overfitting:** wider bottleneck caused validation loss to diverge from training loss around epoch 12 → added dropout, weight decay, and early stopping → losses now track closely throughout training

Full details, evidence, and outcomes for each iteration are documented in the PDF report.

## Known Limitation

Manual testing against known candidate foreign-content images showed the pipeline reliably detects **high-contrast** anomalies (pixel std. dev. ≥ 0.22) but misses **low-contrast** variants of the same content type (pixel std. dev. ≤ 0.06). This indicates the novelty signal is partly correlated with image contrast magnitude rather than semantic content alone — documented transparently in the report rather than concealed.

## Tech Stack

- **Language:** Python 3
- **Deep learning:** PyTorch
- **Novelty detection:** scikit-learn (Isolation Forest)
- **Loss function:** pytorch-msssim (SSIM)
- **Dimensionality reduction (diagnostic):** UMAP
- **Statistics:** SciPy
- **Data handling:** pandas, NumPy
- **Environment:** Google Colab (NVIDIA T4 GPU), Google Drive (dataset persistence), private GitHub repo (automated per-epoch checkpointing for crash resilience)

## Compliance

- No pretrained weights, pretrained feature extractors, or transfer learning were used at any stage.
- All statistical thresholds were derived from the observed data distribution — no arbitrary fixed-count cutoffs or assumed contamination fractions.
- All models were initialized randomly and trained exclusively on the provided HiRISE dataset.

## How to Reproduce

1. Upload `images.zip`, `crop_metadata_index.csv`, and `source_image_metadata.csv` to a Google Drive folder (one-time step).
2. Open `NSSC2026_HiRISE_Anomaly_Detection.ipynb` in Google Colab.
3. Add a `GITHUB_TOKEN` secret in Colab (🔑 panel) if you want per-epoch checkpointing to a GitHub repo.
4. Run all cells top to bottom. Training automatically resumes from the last checkpoint if the runtime disconnects.

---

*Submitted for NSSC 2026, Data Analytics Track — IIT Kharagpur.*
