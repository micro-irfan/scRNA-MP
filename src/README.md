# src

Core Python scripts for mutation-rate matrix construction, QC, normalization, and clustering.

## Main groups of scripts
- Matrix construction: `create_matrix.py`, `create_windows.py`, `create_single_base_fixed_window.py`
- Normalization: `normalization.py`, `normalization_gene_level.py`, `normalization_utils.py`
- QC: `qc_create_count_matrix.py`, `qc_plot_counts.py`, `qc_cells_distribution.py`
- Clustering/visualization: `cluster_cells.py`, `cluster_cells_pca.py`, `cluster_cells_plot.py`
- Shared helpers: `common.py`, `create_utils.py`, `cluster_cells_utils.py`

## Usage pattern
These scripts are mostly called by workflow files under `workflows/`, but can also be run manually with CLI arguments for custom analyses.
