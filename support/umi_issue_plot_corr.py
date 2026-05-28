#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess


def read_matrix(path):
    """
    Read a matrix file with row names and column names.
    Supports CSV and TSV.
    """
    sep = '\t' # ',' if path.endswith('.mx') else '\t'
    df = pd.read_csv(path, sep=sep, index_col=0)
    return df

def subset_df1_to_df2_rows(df1, df2):
    """
    Keep only rows in df1 that are present in df2.
    Order will match df2.index exactly.
    """
    common_rows = df2.index.intersection(df1.index)

    # Reindex df1 to df2's row order
    df1_sub = df1.loc[common_rows]

    return df1_sub

def subset_df1_to_df2_columns(df1, df2):
    """
    Subset df1 to rows and columns present in df2.
    Order will exactly match df2 (rows and columns).
    """
    # Find common rows / columns
    common_cols = df2.columns.intersection(df1.columns)

    # Subset df1 to match df2 ordering
    df1_sub = df1.loc[:, common_cols]

    return df1_sub

def assert_same_shape_and_index(df1, df2):
    assert df1.shape == df2.shape, "Matrix shapes do not match"
    assert (df1.index == df2.index).all(), "Row indices do not match"
    assert (df1.columns == df2.columns).all(), "Column indices do not match"


def plot_matrix_scatter_log(
    df1,
    df2,
    *,
    alpha=0.4,
    s=8,
    title=None,
    filename="",
    xlabel="Bam Read Count",
    ylabel="UMI Tool Gene Count",
    log_base=10,
):
    """
    Scatter plot of matrix1 vs matrix2 (element-wise),
    log-transformed with R^2 annotation.
    """
    assert_same_shape_and_index(df1, df2)

    x = df1.values.ravel()
    y = df2.values.ravel()

    # Remove NaNs and (0,0)
    mask = np.isfinite(x) & np.isfinite(y) & ~((x == 0) & (y == 0))
    x = x[mask]
    y = y[mask]

    # ---- log transform (safe) ----
    if log_base == 10:
        x_log = np.log10(x + 1)
        y_log = np.log10(y + 1)
        log_label = "log10(x + 1)"
    elif log_base == 2:
        x_log = np.log2(x + 1)
        y_log = np.log2(y + 1)
        log_label = "log2(x + 1)"
    else:
        x_log = np.log1p(x)
        y_log = np.log1p(y)
        log_label = "log1p(x)"

    # ---- R^2 (Pearson correlation squared) ----
    r = np.corrcoef(x_log, y_log)[0, 1]
    r2 = r ** 2

    # ---- plot ----
    plt.figure(figsize=(6, 6))
    plt.scatter(x_log, y_log, s=s, alpha=alpha)
    plt.xlabel(f"{xlabel} ({log_label})")
    plt.ylabel(f"{ylabel} ({log_label})")

    # LOWESS on log scale
    smoothed = lowess(y_log, x_log, frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], color="red", lw=2)

    # R^2 annotation
    plt.text(
        0.05,
        0.95,
        f"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_matrix_scatter(df1, df2, *, alpha=0.4, s=8, title=None, filename="", xlabel="Bam Read Count", ylabel="UMI Tool Gene Count"):
    """
    Scatter plot of matrix1 vs matrix2 (element-wise).
    """
    assert_same_shape_and_index(df1, df2)

    x = df1.values.ravel()
    y = df2.values.ravel()

    # Remove NaNs
    mask = np.isfinite(x) & np.isfinite(y) & ~((x == 0) & (y == 0))
    x = x[mask]
    y = y[mask]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=s, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    smoothed = lowess(y, x, frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], color="red", lw=2)

    if title:
        plt.title(title)

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def lognorm_cell_correlation(
    mat1: pd.DataFrame,
    mat2: pd.DataFrame,
    out_png: str,
    log_base: float = 2,
    bins: int = 50,
):
    """
    Log-normalize two matrices, compute per-cell correlation, and save histogram.

    Assumes:
    - rows = genes
    - columns = cells
    - same shape, same index, same columns
    """

    # --- sanity checks ---
    assert mat1.shape == mat2.shape, "Matrix shapes differ"
    assert mat1.index.equals(mat2.index), "Gene order differs"
    assert mat1.columns.equals(mat2.columns), "Cell order differs"

    # --- log normalization ---
    if log_base == 2:
        m1 = np.log2(mat1 + 1)
        m2 = np.log2(mat2 + 1)
    elif log_base == 10:
        m1 = np.log10(mat1 + 1)
        m2 = np.log10(mat2 + 1)
    else:
        m1 = np.log(mat1 + 1)
        m2 = np.log(mat2 + 1)

    # --- per-cell correlation ---
    cell_corr = m1.corrwith(m2, axis=0)

    # --- plot ---
    plt.figure(figsize=(6, 4))
    cell_corr.plot(kind="hist", bins=bins)
    plt.xlabel("Per-cell correlation")
    plt.ylabel("Count")
    plt.title("Cell-wise correlation (log-normalized)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    workdir = '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline'

    threeInFour =  ['RHK516','RHK518', 'RHK517','RHK519']
    oneInFour = ['RHK520','RHK522', 'RHK521','RHK523']

    sample_list = threeInFour + oneInFour

    sample_list = ['RHK516']

    for sample_id in sample_list:
        print (f'Plotting For {sample_id}')

        ## Bam Read Count
        # dedup = 'non_dedup'
        # location = f"{workdir}/results_2/matrices/bam_gene/{dedup}/{sample_id}"

        # steptwo_path = f'{location}/gene_count.mx'
        # df1_gene = read_matrix(steptwo_path)

        # rRNA_path = f'{location}/gene_count.rRNA.mx'
        # df1_rRNA= read_matrix(rRNA_path)

        # df1 = pd.concat([df1_gene, df1_rRNA], axis=0, ignore_index=False)
        
        location = f"{workdir}/results_2/matrices/basepair_gene/{sample_id}"
        filename = f'{location}/matrix_gene.mtx'
        df1= read_matrix(filename)

        ## Umi Tool Read Count
        steptwo_path = f"{workdir}/data_2/{sample_id}/expression/{sample_id}_steptwo_filter40_exp.tsv"
        df2_gene = read_matrix(steptwo_path)

        rRNA_path = f"{workdir}/data_2/{sample_id}/expression/{sample_id}_rRNA_mtRNA_filter40_exp.tsv"
        df2_rRNA = read_matrix(rRNA_path)

        df2 = pd.concat([df2_gene, df2_rRNA], axis=0, ignore_index=False)

        # dedup = 'dedup'
        # location = f"{workdir}/results_2/matrices/bam_gene/{dedup}/{sample_id}"

        # steptwo_path = f'{location}/gene_count.mx'
        # df2_gene = read_matrix(steptwo_path)

        # rRNA_path = f'{location}/gene_count.rRNA.mx'
        # df2_rRNA= read_matrix(rRNA_path)

        # df2 = pd.concat([df2_gene, df2_rRNA], axis=0, ignore_index=False)

        df1 = subset_df1_to_df2_rows(df1, df2)
        df2 = subset_df1_to_df2_rows(df2, df1)
        df2 = subset_df1_to_df2_columns(df2, df1)
        print (df1.shape)
        print (df2.shape)

        location = f"{workdir}/results_2/plots/compare_matrix/{sample_id}"
        Path(location).mkdir(parents=True, exist_ok=True)

        xlabel = "Basepair"
        ylabel = "UMI Tools (In-House)"

        analysis = 'Dedup_nonDedup'
        filename = f"{location}/{analysis}_scatter_plot.png"
        plot_matrix_scatter(df1, 
                            df2, 
                            title=f"NonDedup v Dedup ({df1.shape[1]} Cells)", 
                            filename=filename,
                            xlabel=xlabel,
                            ylabel="Dedup Gene Count")
        
        filename = f"{location}/{analysis}_scatter_plot.log.png"
        plot_matrix_scatter_log(df1, 
                                df2, 
                                title=f"NonDedup v Dedup ({df1.shape[1]} Cells)", 
                                filename=filename,
                                xlabel=xlabel,
                                ylabel="Dedup Gene Count")

        filename = f"{location}/{analysis}_corr_plot.png"
        lognorm_cell_correlation(df1, df2, filename)

        if sample_id not in threeInFour: 
            print (f"Ignoring {sample_id} since no DRAGEN data")
            continue

        baseinc_location = f"{workdir}/results_2/matrices/basepair_gene"
        filename = f'{baseinc_location}/{sample_id}/matrix_gene.mtx'
        df3 = read_matrix(filename)
        print (df3.shape)

        df3 = subset_df1_to_df2_rows(df3, df1)
        df1 = subset_df1_to_df2_rows(df1, df3)
        df3 = subset_df1_to_df2_columns(df3, df1)
        df1 = subset_df1_to_df2_columns(df1, df3)
        print (df1.shape)
        print (df3.shape)

        analysis = 'NonDedup_BaseInc'
        filename = f"{location}/{analysis}_scatter_plot.png"
        plot_matrix_scatter(df1, 
                            df3, 
                            title=f"NonDedup v BaseInc ({df3.shape[1]} Cells)", 
                            filename=filename, 
                            xlabel="NonDedup Gene Count",
                            ylabel="BaseInc Gene Count")

        filename = f"{location}/{analysis}_corr_plot.png"
        lognorm_cell_correlation(df1, df3, filename)

        df2 = subset_df1_to_df2_rows(df2, df3)
        df2 = subset_df1_to_df2_columns(df2, df3)

        analysis = 'Dedup_BaseInc'
        filename = f"{location}/{analysis}_scatter_plot.png"
        plot_matrix_scatter(df2, 
                            df3, 
                            title=f"Dedup v BaseInc ({df3.shape[1]} Cells)", 
                            filename=filename, 
                            xlabel="Dedup Gene Count",
                            ylabel="BaseInc Gene Count")

        filename = f"{location}/{analysis}_corr_plot.png"
        lognorm_cell_correlation(df2, df3, filename)
    

main()