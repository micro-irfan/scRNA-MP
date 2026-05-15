# workflows

Pipeline entrypoints and configuration for preprocessing and post-mapping analyses.

## Workflow files
- `pipseq_pipeline.snakemake`: preprocessing pipeline (FASTQ handling, QC, mapping, dedup, expression outputs).
- `normalization.snakemake`: post-mapping matrix generation, normalization, fixed-window conversion, and pseudobulk steps.
- `generate_pileup.nf`: Nextflow workflow to generate per-cell pileup files from BAM inputs.

## Config files
- `config.yaml`: sample- and run-level settings for workflow execution.
- `nextflow.config`: Nextflow runtime/executor configuration.

## Typical usage
1. Run preprocessing with `pipseq_pipeline.snakemake`.
2. Run pileup generation with `generate_pileup.nf` if needed.
3. Run post-mapping normalization with `normalization.snakemake`.

## Notes
- Update paths, container/image locations, and sample settings before execution.
