#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scanpy as sc
from mofapy2.run.entry_point import entry_point
from pathlib import Path
from typing import Optional


def scale_df_nan(df):
    mean = df.mean(axis=0, skipna=True)
    std = df.std(axis=0, skipna=True)
    std = std.replace(0, np.nan)
    return df.subtract(mean, axis=1).divide(std, axis=1)


def zscore_df(df, axis=1):
    mean = df.mean(axis=axis, skipna=True)
    std = df.std(axis=axis, skipna=True)

    if axis == 1:
        return df.sub(mean, axis=0).div(std + 1e-8, axis=0)
    else:
        return df.sub(mean, axis=1).div(std + 1e-8, axis=1)


def train_mofa(rna_df: pd.DataFrame, 
               react_df: pd.DataFrame, 
               psi_df: pd.DataFrame, 
               location: str,
               scale_df=True,
               add_gene=False, 
               to_impute=True):

    rna_features = rna_df.columns.to_list()
    react_features = react_df.columns.to_list()
    psi_features = psi_df.columns.to_list()

    if scale_df:
        rna_df = zscore_df(rna_df)
        react_df = zscore_df(react_df)
        psi_df = zscore_df(psi_df)

    react_features = [f'React_{x.replace("-","_")}' for x in react_features]

    assert (len(rna_features) == len(set(rna_features)))
    assert (len(react_features) == len(set(react_features)))

    print (set(react_features).intersection(set(rna_features)))

    group_name = "LiverCells"

    data = {
        "Reactivity" : {group_name: react_df},
        "Splicing"   : {group_name: psi_df}
    }   

    if add_gene:
        data['RNA'] = {group_name: rna_df}
        features_names = [react_features, psi_features, rna_features]
        likelihoods = ["gaussian", "gaussian", "gaussian"]
        views_names=["Reactivity", "Splicing", "RNA"]
        txt = ''
    
    else:
        features_names = [react_features, psi_features]
        likelihoods = ["gaussian", "gaussian"]
        views_names=["Reactivity", "Splicing"]
        txt = '.psi_reactivity'

    samples_names = [rna_df.index.to_list()]

    seed = 42
    ep = entry_point()

    # Set data
    ep.set_data_matrix(
        data=data,
        groups_names=[group_name],
        likelihoods=likelihoods,   
        views_names=views_names,
        samples_names = samples_names,
        features_names = features_names
    )

    # Model options
    ep.set_model_options(
        factors = 10, # number of latent factors to start with
        spikeslab_weights = True, 
        ard_weights = True
    )

    # Training options (sensible defaults)
    ep.set_train_options(
        iter=2000,
        convergence_mode = "medium", 
        dropR2 = 0.001, 
        seed=seed,
        verbose=True
    )

    # Optional: useful if you have missing values in Reactivity (NaNs).
    # In many setups, MOFA can handle missing values; if you pre-imputed, ignore this.
    if to_impute:
        ep.set_data_options(
            scale_views=False  # set True only if you want per-view scaling inside MOFA
        )

    # Build and run
    ep.build()
    ep.run()

    outfile = f'{location}/mofa_model{txt}.hdf5'
    ep.save(outfile=outfile)


def open_matrices(filename, sep=','):    
    print(f"Opening {filename.split('/')[-1]}!")
    df = pd.read_csv(filename, index_col=0, sep=sep)
    return df

def open_cluster_file(sample_id):
    workdir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline"
    bam_location = f"{workdir}/data_3/{sample_id}/mapping"
    barcode_file = f"{bam_location}/{sample_id}_filter40_barcode.txt"

    barcode_dict = {}
    with open(barcode_file, 'r') as f:
        for c, line in enumerate(f):
            barcode = line.strip('\n')
            barcode_dict[barcode] = f'bc{c+1}'

    return barcode_dict


def add_source_prefix(df, source, sep="__"):
    df2 = df.copy()
    df2.columns = [f"{source}{sep}{c}" for c in df2.columns]
    return df2


workdir = '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline'
cellline = 'results_3'

def open_gene_expression(batch_id, use_bam=False):

    sample_list = ['DMS_PIP_NAIN3_3in4', 'DMS_PIP_NAIN3_1in4', 'DM_PIP_NAIN3_3in4', 'DM_PIP_NAIN3_1in4']    
    if 'combined' in batch_id:
        assert len(sample_list) == 4
    else:
        tmp_id = '3in4' if '3in4' in batch_id else '1in4'
        sample_list = [s for s in sample_list if tmp_id in s]
        assert len(sample_list) == 2
    
    readcount_dict = {}
    for sample_id in sample_list:
        barcode_dict = open_cluster_file(sample_id) 

        if use_bam:
            path_to_gene_file = f'{workdir}/{cellline}/matrices/bam_gene_count'
            filename = f'{path_to_gene_file}/{sample_id}/gene_count.mx'
        else:
            path_to_gene_file = f'{workdir}/data_3'
            filename = f'{path_to_gene_file}/{sample_id}/expression/{sample_id}_filter40_exp_umi.tsv'

        tmp_df = open_matrices(filename, sep='\t')
        tmp_df = tmp_df.rename(columns=barcode_dict)
        readcount_dict[sample_id] = add_source_prefix(tmp_df, sample_id)

    expr = pd.concat(readcount_dict, axis=1, join="outer")
    expr.columns = [x[1] for x in expr.columns]
    readcount_df = preprocess_rna_for_mofa(expr, n_hvg=1000)

    return readcount_df


def preprocess_rna_for_mofa(
    expr: pd.DataFrame,
    *,
    method: str = "log",          # "log" or "vst"
    n_hvg: int = 2000,
    min_cells: int = 10,
    scale: bool = True,
    clip_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Preprocess RNA count matrix for MOFA2.

    Parameters
    ----------
    expr : pd.DataFrame
        Raw counts, shape = (cells, genes)
    method : {"log", "vst"}
        Normalization method
    n_hvg : int
        Number of highly variable genes to keep
    min_cells : int
        Filter genes expressed in fewer than this many cells
    scale : bool
        Z-score genes (recommended for MOFA)
    clip_value : float or None
        Optional value to clip scaled values (e.g. 5.0)

    Returns
    -------
    pd.DataFrame
        Processed RNA matrix (cells × genes), ready for MOFA
    """

    if method not in {"log", "vst"}:
        raise ValueError("method must be 'log' or 'vst'")

    # --- Build AnnData ---
    adata = sc.AnnData(expr.T)
    adata.var_names = expr.index.tolist()     # genes
    adata.obs_names = expr.columns.tolist()   # cells

    # --- Filter genes ---
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=n_hvg)

    print (adata)

    # --- Normalization ---
    if method == "log":
        print(adata.shape)
        print(adata.X.dtype)
        X = adata.X.A if hasattr(adata.X, "A") else np.asarray(adata.X)

        print("has_nan:", np.isnan(X).any())
        print("has_inf:", np.isinf(X).any())
        print("min/max:", np.nanmin(X), np.nanmax(X))
        print("all_integer_like:", np.allclose(X, np.round(X)))
        print("genes all zero:", np.sum(X.sum(axis=0) == 0))
        print("cells all zero:", np.sum(X.sum(axis=1) == 0))

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            flavor="seurat_v3",
            subset=True
        )

        print (adata)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)

    elif method == "vst":
        sc.experimental.pp.normalize_pearson_residuals(adata)
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            flavor="pearson_residuals",
            subset=True
        )

    print (adata)

    # --- Scaling ---
    if scale:
        sc.pp.scale(adata, zero_center=True)
        if clip_value is not None:
            adata.X = np.clip(adata.X, -clip_value, clip_value)

    # --- Return DataFrame ---
    return pd.DataFrame(
        adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )


def preprocess(batch_id, to_impute = True):
    
    use_bam = False
    coverage = 50
    cell_txt = "AllCells"   
    method = 'single_base'
    
    impute_txt = 'filtered' if to_impute else 'imputed_filtered'

    print ("Loading Reactivity Data...")
    output_location = f'{workdir}/{cellline}/clustering_combined/{method}/{batch_id}/skip_hvw/{coverage}cov'
    filename = f"{output_location}/{batch_id}.{impute_txt}.matrix{coverage}.{cell_txt}.csv"
    reactivity_df = open_matrices(filename) 

    print ("Loading Gene Expression Data...")
    gene_count_df = open_gene_expression(batch_id, use_bam=use_bam)

    print ("Loading Splicing Data...")
    impute_txt = 'filtered' if to_impute else 'imputed'
    output_location = f'{workdir}/{cellline}/clustering_splicing/{batch_id}'
    filename = f"{output_location}/{batch_id}.{impute_txt}.matrix.PSI.csv"
    psi_df = open_matrices(filename) 

    return gene_count_df, reactivity_df.T, psi_df.T


def pipeline():
    batch_id =  'combined_15000_removePoorCells' # '3in4_removePoorCells'
    to_impute = True
    gene_count_df, reactivity_df, psi_df = preprocess(batch_id, to_impute=to_impute)

    common_index = (
        reactivity_df.index
        .intersection(gene_count_df.index)
        .intersection(psi_df.index)
    )

    reactivity_df = reactivity_df.loc[common_index]
    gene_count_df = gene_count_df.loc[common_index]
    psi_df = psi_df.loc[common_index]

    if len(common_index) != len(reactivity_df.index):
        raise ValueError("Some reactivity cells missing in RNA")

    assert (gene_count_df.index.equals(reactivity_df.index)), "Cells must match and be in the same order"

    txt = "sb_psi7_1000g_Zscale"

    impute_txt = '_impute' if to_impute else ''
    location = f'{workdir}/{cellline}/mofa_results/{batch_id}{impute_txt}_{txt}'
    Path(location).mkdir(parents=True, exist_ok=True)

    print ("Training MOFA Model...")
    
    train_mofa(
        rna_df=gene_count_df,
        react_df=reactivity_df,
        psi_df=psi_df,
        location=location,
        scale_df=True,
        add_gene=True,
        to_impute=to_impute
    )


def main(): 
    pipeline()


if __name__ == "__main__":
    main()