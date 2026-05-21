# crop-seq

Utilities for CROP-seq guide assignment and expression-level QC.

## Quick start
1. Build per-sample gRNA matrices from BAM files.
2. Generate combined singlet summaries from all samples.
3. Run one-sample expression QC against `NEGCON`.

## Scripts
- `make_matrix.py`
  - Input: per-sample BAMs under `--input-dir` (for example `bowtie2/<sample>/<sample>.bam`)
  - Output: per-sample count/binary matrices + combined singlet summaries/plots
- `qc_grna_expression_scanpy.py`
  - Input: one sample ID + combined singlet CSV + expression tables
  - Output: target-vs-NEGCON summary CSV + QC expression panels
- `hamming_pairs_from_fasta.py`
  - Input: guide FASTA
  - Output: CSV of sequence pairs with `hamming_distance < threshold`
- `run_script.sh`
  - Batch example wiring `make_matrix.py` and `qc_grna_expression_scanpy.py`

## Typical commands
```bash
# 1) Build matrices for all discovered samples
python3 crop-seq/make_matrix.py \
  --input-dir bowtie2 \
  --output-dir matrix \
  --plot-dir plots \
  --threshold 3

# 2) Or run only selected samples
python3 crop-seq/make_matrix.py \
  --input-dir bowtie2 \
  --output-dir matrix \
  --plot-dir plots \
  --threshold 3 \
  SNUCROP_D_notsoR_1 SNUCROP_N_notsoR_1

# 3) QC for one sample
python3 crop-seq/qc_grna_expression_scanpy.py \
  --sample-id SNUCROP_D_notsoR_1 \
  --singlet-csv plots/combined_singlet_barcodes_by_gRNA_threshold_t3.csv \
  --expression-root expression \
  --output-dir qc/SNUCROP_D_notsoR_1 \
  --top-genes-to-plot 12

# 4) Optional guide-distance screen
python3 crop-seq/hamming_pairs_from_fasta.py \
  --fasta target_guides_rbp.fa \
  --max-distance 3 \
  --output-csv hamming_pairs_below_threshold.csv
```

## Key outputs
- Per sample:
  - `<sample>.gRNA_count_matrix.csv`
  - `<sample>.gRNA_binary_matrix_t<threshold>.csv`
- Combined across processed samples:
  - `combined_singlet_gRNA_summary_threshold_t<threshold>.csv`
  - `combined_singlet_barcodes_by_gRNA_threshold_t<threshold>.csv`
  - `combined_singlet_barcode_counts_by_gRNA_threshold_t<threshold>.csv`
  - `combined_singlet_barcode_counts_by_gene_threshold_t<threshold>.csv`

## Notes
- `run_script.sh` is project-specific; edit paths/sample IDs before use.
- `make_matrix.py` auto-discovers samples from directory structure if sample IDs are not provided.
- In current code, NM filtering is effectively off (`require_nm0=False`), so alignments are counted without requiring `NM==0`.
