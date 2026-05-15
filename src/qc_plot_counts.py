#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from common import open_file


def gene_cell_window_counts(coverage_matrix, reference_list, *, cov_threshold=10, ge=True):
    """
    coverage_matrix: np.ndarray, shape (n_windows, n_cells)
    reference_list: list[str], len n_windows, entries like 'GENE-123'
    cov_threshold: window is valid if coverage >= cov_threshold (or > if ge=False)
    ge: True => >= threshold, False => > threshold

    Returns: pd.DataFrame of shape (n_genes, n_cells) with counts of valid windows.
    """
    cov = np.asarray(coverage_matrix)

    # gene per window-row
    genes = np.array([r.split("-", 1)[0] for r in reference_list])

    # valid window mask: finite AND above threshold
    if ge:
        valid = np.isfinite(cov) & (cov >= cov_threshold)
    else:
        valid = np.isfinite(cov) & (cov > cov_threshold)

    df = pd.DataFrame(valid.astype(np.int32))
    df["gene"] = genes

    gene_cell_counts = df.groupby("gene", sort=False).sum()

    return gene_cell_counts.to_numpy()


def generate_genes_stats(matrix):
    # rows (genes/windows) that have at least one non-zero, non-NaN value
    nonzero_mask = np.any((matrix != 0) & np.isfinite(matrix), axis=1)
    total_genes = np.sum(nonzero_mask)

    # total number of windows (values) > 0 and non-NaN
    total_windows = np.sum((matrix > 0) & np.isfinite(matrix))

    return total_genes, total_windows


def generate_stats(cov_mat, ref_list, thresholds=[10,20,50]):
    output_genes = {}
    output_windows = {}
    for threshold in thresholds:
        gene_cell_counts = gene_cell_window_counts(cov_mat, ref_list, cov_threshold=threshold)
        total_genes, total_windows = generate_genes_stats(gene_cell_counts)

        output_genes[threshold] = total_genes
        output_windows[threshold] = total_windows

    return output_genes, output_windows


def write_nested_dict_to_csv(results_genes, results_windows, filename="results.csv"):
    """
    Write a nested dictionary to a CSV file.

    Parameters:
    - results: dict of {key: (val1, val2)}
    - filename: output CSV file name
    - headers: column headers for the CSV
    """
    with open(filename, mode='w') as write_file:
        write_file.write('Clusters,Threshold,Genes,Windows\n')
        for threshold, data in results_genes.items(): 
            for key, genes in data.items():
                windows = results_windows[threshold][key]
                write_file.write(f'{key},{threshold},{genes},{windows}\n')


def plot_grouped_barplot_log(results, filename, title="Gene Count by Threshold", log_scale=False,
                              ylabel="Number of Genes", xlabel="Count Threshold (x)",
                              figsize=(10,6)):
    """
    Plot grouped bar plot with log scale Y-axis from a nested dict.

    Parameters:
    - results: dict of dicts {x_label: {category: value}}
    """

    categories = list(next(iter(results.values())).keys())
    thresholds = list(results.keys())

    # Data matrix: rows = thresholds, cols = categories
    values = np.array([[results[thr].get(cat, 0) for cat in categories] for thr in thresholds])

    num_groups = len(thresholds)
    group_spacing=0.3

    num_bars = len(categories)

    # Expanded x positions to introduce spacing between groups
    x = np.arange(num_groups) * (1 + group_spacing)
    width = 0.25
    
    _, ax = plt.subplots(figsize=figsize)

    for i, cat in enumerate(categories):
        offset = (i - num_bars/2 + 0.5) * width
        bars = ax.bar(x + offset, values[:, i], width, label=cat)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    if log_scale:
        ax.set_yscale('log')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, rotation=0, fontweight='bold')
    ax.legend()
    ax.grid(True, which="both", axis='y', linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  


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
    
    parser.add_argument('-c', '--coverage', required = False, default = '10,20,50,100', type=str,
                        help='Coverage to Analyze, e.g., 10,20 - comma separated (Default: 10,20,50,100)')  
    
    args = parser.parse_args()
    return args


def main(): 
    args = get_args()
    sample_list = args.sample_id.split(',')
    batch_id = args.batch_id

    if not args.work_path:
        from common import workdir
        path_to_results = f"{workdir}/results_3/plot_count/{batch_id}"
        path_to_matrix = f"{workdir}/results_3/matrices/single_base"

    else: 
        path_to_results = f'{args.work_path}/plot_count/{batch_id}'
        path_to_matrix = f"{args.work_path}/matrices/single_base"

    Path(path_to_results).mkdir(parents=True, exist_ok=True)

    thresholds = [int(c) for c in args.coverage.split(',')]

    results_genes = {}
    results_windows = {}
    results_windows_per_cell = {}

    for sample_id in sample_list:
        filename = f"{path_to_matrix}/{sample_id}/{sample_id}.coverage.matrix10.AllCells.csv"
        _, reference_list, coverage_matrix = open_file(filename)
        output_genes, output_windows = generate_stats(coverage_matrix, reference_list, thresholds=thresholds)

        print (coverage_matrix.shape)

        for threshold in output_genes:  
            if threshold not in results_genes:
                results_genes[threshold] = {}
                results_windows[threshold] = {}
                results_windows_per_cell[threshold] = {}

            results_genes[threshold][sample_id] = output_genes[threshold]
            results_windows[threshold][sample_id] = output_windows[threshold]
            results_windows_per_cell[threshold][sample_id] = output_windows[threshold] / coverage_matrix.shape[1]

    xlabel = "Threshold (x)"
    plot_grouped_barplot_log(results_genes, 
                             f"{path_to_results}/{batch_id}.barplot_genes.png",
                             title=f"Gene Count by Threshold for {batch_id}",
                             xlabel=xlabel)

    plot_grouped_barplot_log(results_windows, 
                             f"{path_to_results}/{batch_id}.barplot_bases.png",
                             title=f"Base Count by Threshold for {batch_id}",
                             ylabel="Number of Bases", 
                             xlabel=xlabel,
                             log_scale=True)

    plot_grouped_barplot_log(results_windows_per_cell, 
                             f"{path_to_results}/{batch_id}.barplot_bases.avg.png",
                             title=f"Base Count / Cell by Threshold for {batch_id}",
                             ylabel="Mean Number of Bases / Cell", 
                             xlabel=xlabel,
                             log_scale=True)
    
    write_nested_dict_to_csv(results_genes, results_windows, filename=f'{path_to_results}/{batch_id}_gene_stats.csv')


if __name__ == "__main__":
    main()
        



