# Core scverse libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import os
from matplotlib.pyplot import rc_context
from pathlib import Path
import scanpy.external as sce
import os


def create_df(sample_name, steptwo_path, rRNA_path="", sep="\t"):

    # -------- load steptwo (always required) --------
    adata_steptwo = sc.read_text(
        steptwo_path, delimiter=sep, first_column_names=True
    ).T

    print(f"{sample_name} steptwo: {adata_steptwo.shape}")

    # -------- load rRNA if present --------
    if rRNA_path:
        assert os.path.exists(rRNA_path)
        adata_rRNA = sc.read_text(
            rRNA_path, delimiter=sep, first_column_names=True
        ).T
        print(f"{sample_name} rRNA: {adata_rRNA.shape}")

        # align cells
        common_cells = adata_rRNA.obs_names.intersection(
            adata_steptwo.obs_names
        )
        print(f"{sample_name} common cells: {len(common_cells)}")

        adata_rRNA = adata_rRNA[common_cells]
        adata_steptwo = adata_steptwo[common_cells]

        df_rRNA = adata_rRNA.to_df()
    else:
        print(f"{sample_name} rRNA: MISSING → using zeros")

        # create zero rRNA matrix with steptwo cells
        df_rRNA = pd.DataFrame(
            0,
            index=adata_steptwo.obs_names,
            columns=[]
        )

    # -------- steptwo dataframe --------
    df_steptwo = adata_steptwo.to_df()

    # -------- union of genes (stable ordering) --------
    all_genes = sorted(
        set(df_rRNA.columns).union(df_steptwo.columns)
    )

    df_rRNA = df_rRNA.reindex(columns=all_genes, fill_value=0)
    df_steptwo = df_steptwo.reindex(columns=all_genes, fill_value=0)

    # -------- sum matrices --------
    df_combined = df_rRNA + df_steptwo

    # -------- create AnnData --------
    # Create combined AnnData object
    adata_combined = ad.AnnData(X=df_combined.values, 
                                obs=df_combined.index.to_frame(index=False, name="cell"),
                                var=pd.DataFrame(index=df_combined.columns))
    
    adata_combined.obs_names = df_combined.index
    adata_combined.var_names = df_combined.columns
    adata_combined.obs["sample"] = sample_name

    print(f"{sample_name} combined: {adata_combined.shape}\n")

    return adata_combined


batch_dict = {
    'DM_PIP_3in4' : ['RHK516','RHK518'],
    'DMS_PIP_3in4' : ['RHK517','RHK519'],
    'DM_PIP_1in4' : ['RHK520','RHK522'],
    'DMS_PIP_1in4' : ['RHK521','RHK523'],
}

treatment_dict = {
    0 : 'DMSO',
    1 : 'NAIN3'
}

batch_dict_reverse = {}
for k, v in batch_dict.items():
    for c,v1 in enumerate(v):
        batch_dict_reverse[v1] = f'{k}_{treatment_dict[c]}'


def preprocess(sample_list, process_bam=False, process_basepair=False, dedup='dedup'):
    adata = {}

    count_location = '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/data_2'

    for sample_id in sample_list:

        if process_bam:
            workdir = '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline'
            ## Bam Read Count
            
            location = f"{workdir}/results_2/matrices/bam_gene/{dedup}/{sample_id}"

            rRNA_path = f'{location}/gene_count.rRNA.mx'
            steptwo_path = f'{location}/gene_count.mx'
            sep = '\t'
        elif process_basepair:
            rRNA_path = ''
            count_location = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/results_2/matrices/basepair_gene"
            steptwo_path = f'{count_location}/{sample_id}/matrix_gene.mtx'
            sep = '\t'
        else:
            rRNA_path = f"{count_location}/{sample_id}/expression/{sample_id}_rRNA_mtRNA_filter40_exp.tsv"
            steptwo_path = f"{count_location}/{sample_id}/expression/{sample_id}_steptwo_filter40_exp.tsv"
            sep = ','
        
        # sep = '\t' if process_bam else ','

    # Concatenate all samples together
    adata = ad.concat(adata, label="sample") 
    adata.obs_names_make_unique()

    print("Final concatenated AnnData:") 
    print(adata) 
    print("\nSample distribution:") 
    print(adata.obs["sample"].value_counts())

    treatment = ['DMSO' if 'DMSO' in i else 'NAIN3' for i in adata.obs['sample']]
    steatosis = ['DM' if 'DM_' in i else 'DMS' for i in adata.obs['sample']]
    dilution = ['1in4' if '1in4' in i else '3in4' for i in adata.obs['sample']]

    adata.obs['dilution'] = dilution
    adata.obs['steatosis'] = steatosis
    adata.obs['treatment'] = treatment

    return adata


def process_adata(adata, filename, process_bam=True):
    
    ## Mark MT genes
    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )

    X = 5000  # minimum counts
    adata = adata[adata.obs["total_counts"] >= X].copy()

    ## Filter Cells and Genes
    sc.pp.filter_cells(adata, min_genes=100)

    # pct_count_cutoff = 10 if not process_bam else 20
    # adata = adata[adata.obs['pct_counts_mt'] < pct_count_cutoff, :].copy() 

    sc.pp.filter_genes(adata, min_cells=3)

    ## Doublet Dectection
    # npcs = min(30, adata.n_vars - 1, adata.n_obs - 1)
    # print (30, adata.n_vars - 1, adata.n_obs - 1)
    # sc.pp.scrublet(
    #     adata,
    #     batch_key="sample",
    #     n_prin_comps=npcs
    # )

    ## Normalization
    # Saving count data
    adata.layers["counts"] = adata.X.copy()
    # Normalizing to median total counts
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")

    sc.tl.pca(adata)

    # sc.pp.neighbors(adata)
    # sc.tl.umap(adata)

    # 30 PCs
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    adata.obsm["X_umap_pcs30"] = adata.obsm["X_umap"].copy()

    # 10 PCs
    sc.pp.neighbors(adata, n_pcs=10)
    sc.tl.umap(adata)
    adata.obsm["X_umap_pcs10"] = adata.obsm["X_umap"].copy()

    sc.tl.leiden(adata, resolution=0.5)

    adata.write(filename) 


def pipeline():
    process_bam = True
    select_dedup = True
    dedup = 'dedup' if select_dedup else 'non_dedup'

    process_basepair = True 
    process_bam = process_bam if not process_basepair else False
    dedup = 'dedup' if process_basepair else dedup

    threeInFour =  ['RHK516','RHK518', 'RHK517','RHK519']
    # threeInFour =  ['RHK516', 'RHK517']
    oneInFour = ['RHK520','RHK522', 'RHK521','RHK523']

    sample_list = threeInFour + oneInFour
    sample_list = threeInFour if process_basepair else sample_list
    
    adata = preprocess(sample_list, 
                       process_bam=process_bam,
                       process_basepair=process_basepair, 
                       dedup=dedup)

    txt = 'umi' if not process_bam else 'bam'
    txt = 'basepair' if process_basepair else txt

    filename = f"anndata/adata_processed.{txt}.{dedup}.PC10.h5ad"
    process_adata(adata, filename, process_bam=process_bam)


pipeline()