# gene_preprocessing

Gene-expression preprocessing workflows for liver-cell single-cell data, with both Scanpy (Python) and SCTransform (Seurat/R) paths.

## Files
- `preprocess_livercells.scanpy.py`: AnnData-based preprocessing flow (QC metrics, filtering, normalization, PCA, neighbors/UMAP, Leiden, `.h5ad` export).
- `preprocess_livercells.scTransform.R`: Seurat/SCTransform workflow (sparse matrix loading, SCT normalization, PCA/UMAP/clustering, `.h5ad` export via SeuratDisk).

## Workflow summary
1. Load per-sample count matrices (BAM-derived, basepair-derived, or expression tables).
2. Merge/align genes across samples.
3. Attach sample/treatment metadata.
4. Run normalization + dimensionality reduction + clustering.
5. Export processed objects and optional matrix/metadata artifacts.

## Dependencies
- Python workflow: `scanpy`, `anndata`, `pandas`, `matplotlib` (plus standard library modules).
- R workflow: `Matrix`, `Seurat`, `SeuratDisk`, `data.table` (and `cc.genes` availability for cell-cycle scoring).

## Typical usage
```bash
# Python / Scanpy
python3 gene_preprocessing/preprocess_livercells.scanpy.py

# R / Seurat SCTransform
Rscript gene_preprocessing/preprocess_livercells.scTransform.R
```

## Expected outputs
- Processed AnnData exports such as:
  - `anndata/adata_processed.<mode>.<dedup>.PC10.h5ad` (Scanpy script)
  - `fastp_filterNone/adata_processed.<analysis>.scT.RNA.h5ad` (R script)
  - `fastp_filterNone/adata_processed.<analysis>.scT.SCT.h5ad` (R script)
- Optional side products from the R script:
  - matrix market files (`counts.mtx`, `sct.mtx`, `SCT_corrected_counts.mtx`)
  - gene/barcode TSV files
  - PCA variance and embedding CSVs
  - cluster metadata CSVs

## Notes
- Both scripts contain project-specific absolute paths and sample lists; update these before running in a new environment.
- Run from repository root to keep relative output paths predictable.
- The scripts are currently dataset-specific and are best used as templates when adapting to new cohorts.
