# src

Core Python scripts for mutation-rate matrix generation, normalization, QC, clustering, and pseudobulk.

## Main workflow scripts
- `create_windows.py`: generate per-barcode windows from pileups, then create mutrate/coverage/mutant matrices.
- `create_windows_from_metadata.py`: aggregate barcodes by metadata groups and write outputs per sample.
- `create_rbp_cell_metadata.py`: build `cellbarcodes/sample_RBP/sample` metadata CSV from `selected_rbp_transcriptome_window_raw.pkl`.
- `create_matrix.py`: build matrices from existing window CSV files.
- `create_pseudobulk.py`: build pseudobulk profiles across coverage thresholds from matrix outputs.
- `normalization.py`: apply reactivity normalization workflow.
- `cluster_cells.py`: cluster cells from normalized matrices and write cluster outputs.

## Supporting scripts
- Window/feature helpers: `create_utils.py`, `create_single_base_fixed_window.py`
- QC utilities: `qc_create_count_matrix.py`, `qc_plot_counts.py`, `qc_cells_distribution.py`
- Clustering utilities: `cluster_cells_pca.py`, `cluster_cells_plot.py`, `cluster_cells_utils.py`
- Shared utilities: `common.py`, `normalization_utils.py`

## Typical usage
```bash
# 1) Build windows + matrices from sample pileups
python3 src/create_windows.py \
  --sample_id SAMPLE_A \
  --workdir /path/to/pileups \
  --output_path /path/to/results \
  --barcode /path/to/SAMPLE_A_filter40_barcode.txt \
  --reference /path/to/reference.fa \
  --method single_base \
  --threads 8

# 2) Metadata-based aggregation (writes per-sample outputs)
python3 src/create_windows_from_metadata.py \
  --metadata-csv /path/to/metadata.csv \
  --pileup-root /path/to/pileup_root \
  --barcode-root /path/to/barcode_root \
  --reference /path/to/reference.fa \
  --output-path /path/to/results \
  --method single_base

# 3) Pseudobulk from generated matrices
python3 src/create_pseudobulk.py \
  --sample_id SAMPLE_A \
  --method single_base \
  --work_path /path/to/results

# 4) Cluster cells from normalized/matrix outputs
python3 src/cluster_cells.py \
  --sample_list SAMPLE_A,SAMPLE_B \
  --coverage 50 \
  --method single_base \
  --work_path /path/to/results \
  --barcode-root /path/to/barcodes \
  --batch_id BATCH_1 \
  --hvw compare_cluster

# 5) Cluster cells with an alternate sample template
# Example sample IDs: SNUCROP_D_notsoR_1, SNUCROP_N_notsoR_2
python3 src/cluster_cells.py \
  --sample_list SNUCROP_D_notsoR_1,SNUCROP_N_notsoR_2 \
  --coverage 50 \
  --method single_base \
  --work_path /path/to/results \
  --barcode-root /path/to/barcodes \
  --template "{}_{treatment}_{}_{}"
```

## Notes
- These scripts are commonly orchestrated by pipeline files under `workflows/`.
- Some legacy scripts still contain project-specific default paths; prefer passing explicit CLI arguments.
- `cluster_cells.py` expects a comma-separated `--sample_list` and reads per-sample barcode files from `--barcode-root` if provided, otherwise from `<work_path>/preprocessing/<sample>/mapping/`.
