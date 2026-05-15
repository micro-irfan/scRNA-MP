# multiomics

Multi-omics analyses that integrate mutation/reactivity with splicing and factor models.

## Subdirectories
- `splicing/`: Nextflow pipeline for per-cell splicing quantification (FASTQ extraction from BAM, then Salmon quantification).
- `mofa2/`: MOFA2 training and post-analysis scripts for multi-view latent factor analysis.

## Key files
- `splicing/main.nf`: primary splicing workflow.
- `splicing/run_splicing.sh`: helper launcher for the splicing workflow.
- `splicing/salmon_2_mx.py`: converts Salmon outputs into matrix format for downstream analysis.
- `mofa2/mofa2_train.livercells.py`: MOFA2 model setup and training.
- `mofa2/post_mofa_analysis.py`: downstream interpretation/plotting steps.

## Notes
- Several scripts include absolute cluster paths; update these for your environment.
