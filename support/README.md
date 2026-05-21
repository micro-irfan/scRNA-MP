# support

Support scripts for downstream evaluation and visualization.

## Scripts
- `auroc_calc.py`: computes AUROC/correlation summaries from reactivity matrices and produces plots.
- `auroc_plot.py`: plotting helpers used by `auroc_calc.py`.
- `umi_issue_compare_matrix.py`: compares BAM-derived count matrix vs UMI-tools expression matrix.
- `umi_issue_count_read_length.py`: analyzes read-length/read-count behavior for selected genes.
- `umi_issue_create_count_mt_bam.py`: builds count matrices from BAM for UMI troubleshooting.
- `umi_issue_select_topk_genes.py`: ranks genes by mean/variance behavior for diagnostic panels.

## Notes
- These are analysis/support utilities, not the main production pipeline.
- Most scripts use hard-coded project paths and sample IDs; edit those values before running.
- Several scripts depend on local helper imports (for example `common`, `count_read_length`, `plot_auroc`) and are best run from the repository root.

## Typical run style
```bash
# Most support scripts are configured internally; run after editing paths/IDs
python3 support/auroc_calc.py
python3 support/umi_issue_compare_matrix.py
python3 support/umi_issue_select_topk_genes.py
```
