# io-heterogeneity

Analysis code for Baseline Predicted Survival and Treatment Benefit from First-Line Immune Checkpoint Inhibitor Therapy: A Multicohort Target Trial Emulation Study.

## Overview

This project evaluates whether baseline predicted survival modifies the absolute benefit of first-line immune checkpoint inhibitor (ICI) treatment strategies across advanced solid tumors. ICI monotherapy can be associated with an early survival disadvantage relative to chemotherapy or targeted therapy, raising the question of whether short-term prognosis can complement tumor biomarkers when selecting treatment.

Using deidentified EHR data from the Flatiron Health database, we conducted target trial emulations of five first-line treatment comparisons involving 8,689 patients who initiated treatment between 2011 and 2023. A secondary renal cell carcinoma (RCC) analysis evaluated whether the framework could recover previously established treatment-effect heterogeneity.

## Cohorts

| Cohort | First-line treatment comparison |
|---|---|
| Advanced NSCLC (PD-L1 TPS ≥50%) | Pembrolizumab plus platinum-based chemotherapy vs. pembrolizumab monotherapy |
| Recurrent/metastatic HNSCC (PD-L1 CPS ≥1 or Unknown) | Pembrolizumab plus platinum-based chemotherapy vs. pembrolizumab monotherapy |
| Advanced urothelial carcinoma | Carboplatin plus gemcitabine vs. pembrolizumab |
| Metastatic colorectal cancer (dMMR/MSI-H) | Pembrolizumab vs. fluoropyrimidine-based combination chemotherapy, with or without biologic therapy |
| Advanced melanoma (BRAF-mutant) | Ipilimumab plus nivolumab vs. BRAF/MEK inhibitor combination therapy |
| Metastatic clear cell RCC (concordance analysis) | Ipilimumab plus nivolumab vs. single-agent VEGF-targeted therapyy |

## Analytic framework

1. Estimate baseline prognosis. Cancer-specific gradient-boosted survival models estimate the probability of surviving 6 months after first-line treatment initiation. Cross-validation generates out-of-sample predictions, followed by cross-validated isotonic calibration. Treatment assignment is excluded from the prognostic feature set; the score reflects prognosis under observed treatment patterns rather than a fixed reference treatment.

2. Estimate continuous treatment-effect heterogeneity. An overlap-weighted regression of RMST pseudo-observations includes treatment, calibrated predicted survival, and their interaction. The primary outcome is the between-treatment difference in 2-year RMST.

3. Summarize risk-stratified outcomes. Where the fitted treatment-effect function reaches a prespecified 30-day RMST benefit threshold, the corresponding predicted survival probability (r*) defines higher- and lower-risk strata. Stabilized IPTW were used to estimate survival curves and RMST differences in the full cohort and within risk strata. 

4. Assess robustness. Analyses include randomized-trial benchmarking where applicable, alternative prognostic models and model-development populations, spline interaction models, shorter RMST horizons, alternative benefit thresholds, and stricter biomarker timing. 

## Repository structure

Analysis notebooks are organized by cancer cohort:

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

## Software and reproducibility

Analysis was performed in Python 3.13. Key packages include:

- `scikit-survival` — gradient-boosted survival modeling
- `scikit-learn` — preprocessing and cross-validation
- `statsmodels` — weighted least-squares regression
- `flatiron-cleaner` — data preprocessing for Flatiron Health EHR data
- `iptw-survival` — IPTW and overlap weighting

Data and model outputs are excluded from this repository; the repository cannot reproduce the study results using public data alone.

## Target trial protocol 

This observational target trial emulation was not prospectively registered. The study protocol is available on Zenodo: https://doi.org/10.5281/zenodo.23113983.