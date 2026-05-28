# support

Support and diagnostic scripts for matrix comparison, AUROC/correlation analysis, and UMI troubleshooting.

## Purpose
- This directory contains ad-hoc analysis utilities used to validate and inspect outputs from the main workflows.
- Scripts are generally configured inside each file (paths, sample IDs, thresholds) rather than via full CLI arguments.

## Script index
- `auroc_calc.py`: computes AUROC and cell-cell correlation summaries (for MT-RNR genes) across conditions and writes summary CSV + violin/heatmap plots.
- `auroc_plot.py`: plotting helpers used by `auroc_calc.py` (violin plots and correlation heatmaps).
- `umi_issue_create_count_mt_bam.py`: rebuilds gene count matrices directly from BAM reads for debugging UMI/count discrepancies.
- `umi_issue_compare_matrix.py`: compares BAM-derived and expression-derived matrices with element-wise scatter plots.
- `umi_issue_plot_corr.py`: extended matrix comparison with log scatter and per-cell correlation histograms.
- `umi_issue_count_read_length.py`: links read-length behavior with per-barcode read counts for selected gene panels.
- `umi_issue_select_topk_genes.py`: ranks genes (high mean + low variance) across combined/clustered matrices and exports overlap summaries.

## Dependencies
- Python packages used across scripts include: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `pysam`.
- Local helper modules are expected to be importable from the repository context (for example `common`, `count_read_length`, `auroc_plot`/`plot_auroc`).

## Typical usage
```bash
# Run from repository root after editing hard-coded paths and sample IDs in each script.
python3 support/auroc_calc.py
python3 support/umi_issue_create_count_mt_bam.py
python3 support/umi_issue_compare_matrix.py
python3 support/umi_issue_plot_corr.py
python3 support/umi_issue_count_read_length.py
python3 support/umi_issue_select_topk_genes.py
```

## Output patterns
- AUROC analysis: `results*/plots/auroc*/cov*/...` (summary CSV + violin/heatmap plots).
- Matrix comparison: `results*/plots/compare_matrix/...` (scatter, log-scatter, correlation plots).
- Gene ranking: `results*/select_topk_genes/...` (z-score ranking CSV + diagnostic figures).

## Notes
- These scripts are not the main production pipeline and are best treated as exploratory utilities.
- Most scripts currently assume an HPC-style absolute path layout (`/home/users/...`) and specific sample naming.
- If imports fail, run from the project root and verify helper module names/paths match the current filenames.
