# io-heterogeneity

Machine learning–guided evaluation of immunotherapy treatment-effect heterogeneity in advanced solid tumors using real-world data.

## Overview

Immune checkpoint inhibitors (ICIs) have transformed the treatment of advanced solid tumors, but their benefit is not uniform across patients. A consistent pattern in landmark immunotherapy trials is that survival curves cross within the first several months. The immunotherapy arm performs worse early, with superior outcomes emerging only among patients who survive this initial window. This suggests that patients with high baseline mortality risk may not survive long enough to realize the delayed benefits of immunotherapy.

This project develops and applies a four-step framework to evaluate how absolute immunotherapy benefit varies as a function of short-term baseline prognosis across multiple tumor types and first-line treatment comparisons. At the core is a machine learning–derived estimate of each patient's probability of surviving 6 months from treatment initiation, which is used as a continuous measure of baseline mortality risk. We then model how absolute long-term treatment benefit — measured using restricted mean survival time (RMST) — varies across that risk spectrum, and translate findings into clinically interpretable risk-stratified survival analyses.

All analyses use real-world electronic health record data from the Flatiron Health Research Database.

## Cohorts

Five first-line treatment comparisons were analyzed:

| Tumor Type | Comparison |
|---|---|
| Advanced NSCLC (PD-L1 TPS ≥50%) | Pembrolizumab + chemotherapy vs. pembrolizumab monotherapy |
| Recurrent/metastatic HNSCC | Pembrolizumab + chemotherapy vs. pembrolizumab monotherapy |
| Advanced urothelial carcinoma | Pembrolizumab vs. carboplatin-based chemotherapy |
| Metastatic colorectal cancer (dMMR/MSI-H) | Pemborlizuamb vs. chemotherapy |
| Advanced melanoma (BRAF-mutant) | Ipilimumab + nivolumab vs. BRAF/MEK inhibitor combination |
| Metastatic clear cell RCC | Ipilimumab + nivolumab vs. single-agent antiangiogenic therapy |

## Repository Structure

```
io-heterogeneity/
├── advHeadNeck/
│   └── notebooks/
├── aNSCLC/
│   └── notebooks/
├── aUC/
│   └── notebooks/
├── mCRC/
│   └── notebooks/
├── advMelanoma/
│   └── notebooks/
└── mRCC/
    └── notebooks/
```

Data and model outputs are excluded from this repository and are not publicly available due to data governance restrictions governing the Flatiron Health Research Database.

## Dependencies

Analysis was performed in Python 3.13. Key packages include:

- `scikit-survival` — gradient-boosted survival modeling
- `scikit-learn` — preprocessing and cross-validation
- `statsmodels` — weighted least-squares regression
- `flatiron-cleaner` — data preprocessing for Flatiron Health EHR data
- `iptw-survival` — IPTW and overlap weighting