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
    sep = ',' if path.endswith('.mx') else '\t'
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

def plot_matrix_scatter(df1, df2, *, alpha=0.4, s=8, title=None, filename=""):
    """
    Scatter plot of matrix1 vs matrix2 (element-wise).
    """
    assert_same_shape_and_index(df1, df2)

    colors = create_colours(df1)

    x = df1.values.ravel()
    y = df2.values.ravel()

    # Remove NaNs
    mask = np.isfinite(x) & np.isfinite(y) & ~((x == 0) & (y == 0))
    x = x[mask]
    y = y[mask]
    colors = colors[mask]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=colors, s=s, alpha=alpha)
    plt.xlabel("Bam Read Count")
    plt.ylabel("UMI Tool Gene Count")

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='High Mean',
            markerfacecolor='tab:blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='High zscore',
            markerfacecolor='tab:orange', markersize=8),
    ]

    plt.legend(handles=legend_elements)

    smoothed = lowess(y, x, frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], color="red", lw=2)

    if title:
        plt.title(title)

    # # Identity line
    # lims = [
    #     min(x.min(), y.min()),
    #     max(x.max(), y.max())
    # ]
    # plt.plot(lims, lims)
    # plt.xlim(lims)
    # plt.ylim(lims)

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_colours(df1):
    import count_read_length as utils
    list1 = utils.genes_to_keep_mean
    list2 = utils.genes_to_keep_std

    gene_colors = np.where(
        df1.index.isin(list1),
        "tab:blue",
        np.where(df1.index.isin(list2), "tab:orange", "lightgray")
    )

    colors = np.repeat(gene_colors, df1.shape[1])

    return colors

def main():
    depth = 10
    workdir = '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline'

    ## Bam Read Count
    location = f"{workdir}/results_1/matrices/bam_gene_counts/PIP_2cell_NAIN3_1_{depth}"
    filename = f'{location}/gene_count.mx'
    df1 = read_matrix(filename)
    
    ## Umi Tool Read Count
    filename = f'PIP_2cell_NAIN3_1_{depth}'
    filename = f"{workdir}/data_1/{filename}/expression/{filename}_steptwo_filter40_exp.tsv"
    df2 = read_matrix(filename)

    df1 = subset_df1_to_df2_rows(df1, df2)
    df2 = subset_df1_to_df2_rows(df2, df1)
    df2 = subset_df1_to_df2_columns(df2, df1)
    print (df1.shape)
    print (df2.shape)

    location = f"{workdir}/results_1/plots/compare_matrix"
    Path(location).mkdir(parents=True, exist_ok=True)
    filename = f"{location}/compare_matrix.png"
    plot_matrix_scatter(df1, df2, title="RAW v UMI", filename=filename)

main()