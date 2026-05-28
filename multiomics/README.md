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
- Several scripts include absolute cluster paths; update these for the specific environment.
- Treatment lives at the cell level → Compare factor values across treatments
    - Method
        - Get factors and metadata
        - Plot factor distribution by treatment
        - Extract weights for that factor
        - Plot top features (colored by direction)
    - Interpreting
        For each factor:
        
        - the factor-by-treatment plot tells you whether the latent factor separates DM vs DMS
        - the summary csv tells you which treatment has higher factor values
        - the top positive/negative weights tell you which features drive that factor
        
        A practical interpretation is:
        
        - if Factor2 is higher in DM than DMS
        - and in Reactivity, windows A/B/C have strong positive weights
        - then those windows are more associated with the DM side of that factor
        
        while strong negative weights are associated with the opposite side.
        
- Weights = feature-level (genes / windows / PSI)
- Compare feature importance across treatments / cells
    - Feature contribution=Z×W**T
        - Z = factors (cells × factors) - how active each factor is in each cell
        - W = weights (features × factors) - what features define each factor
    - Method
        - get factors `Z`
        - get weights `W` per view
        - compute contribution matrix `Z @ W.T`
            - reconstructed signal → feature importance per cell
        - normalize contribution per cell (Z-Score)
            - removes global scale bias
        - run UMAP on contribution space
            - feature-driven embedding
            - clustering driven by features, not factors
        - plot UMAP by treatment
        - plot correlation heatmap
            - similarity between cells/features
        - plot **per-view bar plots** for treatment differences
    
    For each view, like `RNA`, `Reactivity`, or `Splicing`, you’ll get:
    
    - `umap_by_treatment.png`
    UMAP from the **contribution matrix** of that view
    - `correlation_heatmap_cells.png`
    similarity of cells in contribution space
    - `top_feature_diff_barplot.png`
    one bar per feature, where:
        - **positive bar** = higher mean contribution in group1
        - **negative bar** = higher mean contribution in group2
    - `top_feature_grouped_means.png`
    side-by-side bar view of the actual mean contribution in each treatment
    
