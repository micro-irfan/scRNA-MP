#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from common import open_matrices, add_source_prefix, check_header

CELL_TXT = "AllCells"
MIN_READ_COUNT_BAM = 50000

def plot_qc_scatter_percent_mt(
    nCount,
    nFeature,
    percent_mt,
    *,
    title="percent.mt",
    xlabel="nCount_RNA",
    ylabel="nFeature_RNA",
    xlim=None,
    ylim=None,
    mt_lim=None,        # e.g. (0, 15) to cap color scale
    s=35,
    alpha=0.9,
    cmap="viridis",
    add_colorbar=True,
    ax=None,
):
    """
    Scatter QC plot: nCount vs nFeature colored by percent_mt.

    Parameters
    ----------
    nCount, nFeature, percent_mt : array-like (same length)
        Per-cell values.
    xlim, ylim : tuple or None
        Axis limits (min, max).
    mt_lim : tuple or None
        Color scale limits (vmin, vmax).
    ax : matplotlib axis or None
        If None, creates a new figure+axis.

    Returns
    -------
    fig, ax
    """
    nCount = np.asarray(nCount)
    nFeature = np.asarray(nFeature)
    percent_mt = np.asarray(percent_mt)

    if not (len(nCount) == len(nFeature) == len(percent_mt)):
        raise ValueError(
            f"Length mismatch: nCount={len(nCount)}, nFeature={len(nFeature)}, percent_mt={len(percent_mt)}"
        )

    # Drop non-finite points (safe for plotting)
    valid = np.isfinite(nCount) & np.isfinite(nFeature) & np.isfinite(percent_mt)
    x = nCount[valid]
    y = nFeature[valid]
    c = percent_mt[valid]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
    else:
        fig = ax.figure

    vmin = mt_lim[0] if mt_lim is not None else None
    vmax = mt_lim[1] if mt_lim is not None else None

    sc = ax.scatter(x, y, c=c, s=s, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if add_colorbar:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("percent.mt")

    return fig, ax

def plot_percent_cells_non_nan(
    matrix,
    filename,
    bins=50,
    min_percent=0,
    min_cells=None,
    title=None,
    figsize=(6, 4)
):
    """
    Plot histogram of % cells with non-NaN values per row.

    Parameters
    ----------
    matrix : np.ndarray (n_positions x n_cells)
    bins : int
        Number of histogram bins
    min_cells : float or None
        Optional: only keep rows with at least this many non-NaN cells
    return_values : bool
        If True, return the computed percentages
    """

    matrix = np.asarray(matrix)

    n_cells = matrix.shape[1]

    # Count non-NaN per row
    non_nan_counts = np.sum(~np.isnan(matrix), axis=1)

    if min_cells is not None:
        min_cells = min_cells*n_cells
        mask = non_nan_counts >= min_cells
        non_nan_counts = non_nan_counts[mask]

    percent_non_nan = (non_nan_counts / n_cells) * 100

    if min_percent > 0:
        for threshold in [70, 90]:
            n_rows_passing = np.sum(percent_non_nan > threshold)
            print (f'Number of Position > {threshold} % : {n_rows_passing}')
        percent_non_nan = percent_non_nan[percent_non_nan >= min_percent]

    plt.figure(figsize=figsize)
    plt.hist(percent_non_nan, bins=bins, alpha=0.7)
    plt.xlabel("% Cells with non-NaN values")
    plt.ylabel("Number of Positions")
    plt.title(title if title else "Distribution of Cell Coverage per Position")
    plt.axvline(70, linestyle="--")
    plt.axvline(90, linestyle="--")
    if min_percent > 0:
        plt.xlim(min_percent, 100)   # show only 50% to 100%
    plt.tight_layout()
    plt.savefig(f"{filename}", dpi=300, bbox_inches='tight')


def threshold_to_nan(matrix, threshold, copy=True):
    """
    Replace values < threshold with NaN.

    Parameters
    ----------
    matrix : np.ndarray (2D)
        Input matrix (positions x cells)
    threshold : float
        Values strictly below this will be set to NaN
    copy : bool
        If True, return a copy. If False, modify in place.

    Returns
    -------
    np.ndarray
    """

    if copy:
        matrix = matrix.copy()

    # Ensure float (required for NaN support)
    if not np.issubdtype(matrix.dtype, np.floating):
        matrix = matrix.astype(float)

    matrix[matrix < threshold] = np.nan

    return matrix


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


def percent_feature_set(counts, gene_names, pattern=r"^MT-"):
    """
    Compute percentage of counts per cell coming from genes
    matching a prefix pattern (e.g. 'MT-').

    Parameters
    ----------
    counts : np.ndarray
        Gene x cell count matrix (n_genes, n_cells)
    gene_names : list or array
        Gene names corresponding to rows
    pattern : str
        Prefix to match (default 'MT-')

    Returns
    -------
    percent : np.ndarray
        Percentage per cell (length n_cells)
    """
    
    gene_names = np.array(gene_names)

    # Identify matching genes
    # mask = np.char.startswith(gene_names.astype(str), pattern)
    mask = pd.Index(gene_names).str.contains(pattern)
    
    # Sum mitochondrial counts per cell
    mt_counts = counts[mask, :].sum(axis=0)

    # Total counts per cell
    total_counts = np.nansum(counts, axis=0)

    # Avoid division by zero
    percent = np.divide(
        mt_counts,
        total_counts,
        out=np.zeros_like(mt_counts, dtype=float),
        where=total_counts != 0
    ) * 100
    
    return percent


def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Plot Basic Statistics',
        description = "Plot Basic Statistics - Gene Count and Base Count that is above >X Coverage"
    )

    parser.add_argument('-b', '--batch_id', required = True,
                        help='Batch id for the current samples')

    parser.add_argument('-s', '--sample_id', required = True,
                        help='Comma Separated List of Sample IDs')
    
    parser.add_argument('-w', '--work_path', required = False, type=str, default='',
                        help='Path to store output, Assumed Path to Matrices is stored here')
    
    parser.add_argument('-m', '--method', required = False, type=str, default='fixed_single_base',
                        help='Rolling Window, Fixed Window, single_base or fixed_single_base')  
    
    parser.add_argument('-c', '--coverage', required = False, default = '50,80,100' , type=str,
                        help='Coverage to Analyze, (Default: 50,80,100)')  
    
    parser.add_argument('--min-reads', required = False, default = 0, type=int, dest='min_reads',
                        help='Minimum Number Of Reads Per Cell to Include (At Least 50000 for Bam File)')  
    
    parser.add_argument('--bam', action="store_false", dest='use_bam', 
						help='Use Bam Matrix instead of UMI (Default: UMI matrix is referred)')
    
    return parser.parse_args()


def plot_cells_non_nan(cov_matrix, coverage_to_plot, location):
    for cov in coverage_to_plot:
        if cov != 50: cov_matrix = threshold_to_nan(cov_matrix, cov)
        print (f'Analyzing Coverage at {cov}X')

        filename = f'{location}/Cell_Coverage_Per_Position_Distribution.{cov}cov.png'
        plot_percent_cells_non_nan(
            cov_matrix,
            filename,
            title = f"Distribution of Cell Coverage per Position For Coverage > {cov}"
        )

        filename = f'{location}/Cell_Coverage_Per_Position_Distribution.{cov}cov.50perc.png'
        plot_percent_cells_non_nan(
            cov_matrix,
            filename,
            title = f"Distribution of Cell Coverage per Position For Coverage > {cov}",
            min_percent=50,
        )


def plot_for_MT(readcount_mat, gene_names, path_to_save, batch_id):
    read_sums = np.nansum(readcount_mat, axis=0) 
    gene_count = np.sum(
        (np.isfinite(readcount_mat)) & (readcount_mat != 0),
        axis=0
    )

    pattern_dict = {
        'percentMT' : r"^MT-",
        'percentRB' : r"^(RPL|RPS)"
    }

    for label, pattern in pattern_dict.items():
        percent = percent_feature_set(readcount_mat, gene_names, pattern=pattern)
        fig, ax = plot_qc_scatter_percent_mt(
            read_sums, gene_count, percent,
        )

        filename = f'{path_to_save}/{label}.{batch_id}.png'
        plt.tight_layout()
        plt.savefig(f"{filename}", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    args = get_args()
    sample_list = args.sample_id.split(',')
    batch_id = args.batch_id
    method = args.method
    coverage_to_plot = [int(i) for i in args.coverage]
    coverage = min(coverage_to_plot)

    if not args.work_path:
        from common import workdir
        path_to_results = f"{workdir}/results_3/plot_distribution/{method}"
        path_to_matrix = f"{workdir}/results_3/single_base_to_fixed_absolute"

    else: 
        path_to_results = f'{args.work_path}/plot_distribution/{method}'
        path_to_matrix = f"{args.work_path}/single_base_to_fixed_absolute"

    read_threshold = args.min_reads
    to_filter_by_readcount = False
    if read_threshold > 0:
        batch_id += f'_{read_threshold}'
        to_filter_by_readcount = True

    use_bam = args.use_bam
    if use_bam:
        if read_threshold > 0: assert read_threshold > MIN_READ_COUNT_BAM
        batch_id += '_bam'
        

    location = f'{path_to_results}/{batch_id}'
    Path(f'{location}').mkdir(parents=True, exist_ok=True)

    coverage_df = {}
    readcount_df = {}
    for sample_id in sample_list:
        print (f"Analyzing Sample: {sample_id}")

        filename = f"{path_to_matrix}/{sample_id}.coverage.matrix{coverage}.{CELL_TXT}.csv"
        cov_df = open_matrices(filename)
        cov_df.index = cov_df.index.str.replace(r":.*$", "", regex=True)
        coverage_df[sample_id] = add_source_prefix(cov_df, sample_id)

        ##### TODO #####
        barcode_dict = open_cluster_file(sample_id) 
        if use_bam:
            path_to_gene_file = f'{workdir}/results_3/matrices/bam_gene_count'
            filename = f'{path_to_gene_file}/{sample_id}/gene_count.mx'
        else:
            path_to_gene_file = f'{workdir}/data_3'
            filename = f'{path_to_gene_file}/{sample_id}/expression/{sample_id}_filter40_exp_umi.tsv'
        ##### TODO #####
        
        tmp_df = open_matrices(filename, sep='\t')
        tmp_df = tmp_df.rename(columns=barcode_dict)
        readcount_df[sample_id] = add_source_prefix(tmp_df, sample_id)
        
    coverage_merged = pd.concat(coverage_df, axis=1, join="outer") 
    cov_matrix = coverage_merged.to_numpy(dtype=float)
    print ('Coverage Matrix Shape:', cov_matrix.shape)

    readcount_merged = pd.concat(readcount_df, axis=1, join="outer")
    readcount_merged = readcount_merged.reindex(columns=coverage_merged.columns)
    check_header(coverage_merged, readcount_merged)
    readcount_mat = readcount_merged.to_numpy(dtype=float)

    if to_filter_by_readcount:
        read_sums = np.nansum(readcount_mat, axis=0) 
        mask = read_sums > read_threshold
        cov_matrix = cov_matrix[:, mask]
        readcount_mat = readcount_mat[:, mask]
        print ('Coverage Matrix Shape Post Filter By ReadCount:', cov_matrix.shape)

    barcode_list = list(coverage_merged.columns)
    barcode_list = np.array(barcode_list)
    if to_filter_by_readcount: 
        barcode_list = barcode_list[mask]
    
    gene_names = readcount_merged.index.tolist() 
    plot_for_MT(readcount_mat, gene_names, location, batch_id)
    plot_cells_non_nan(cov_matrix, coverage_to_plot, location)


if __name__ == "__main__":
    main()
