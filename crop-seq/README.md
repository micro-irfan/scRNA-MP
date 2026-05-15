# crop-seq

Utilities for CROP-seq guide assignment and expression-level QC.

## What is here
- `make_matrix.py`: builds per-cell gRNA assignment/count matrices from mapping outputs.
- `qc_grna_expression_scanpy.py`: runs Scanpy-based QC and plotting for selected samples.
- `run_script.sh`: example batch script that runs matrix generation and QC for multiple samples.

## Typical usage
1. Generate matrix outputs with `make_matrix.py`.
2. Run `qc_grna_expression_scanpy.py` per sample (or via `run_script.sh`).
3. Review QC outputs in your chosen output directory.

## Notes
- Paths in `run_script.sh` are project-specific and usually need local edits before running.
