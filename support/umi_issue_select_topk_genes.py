#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import common as utils
from collections import defaultdict

def rank_genes_by_mean_std(matrix_selected, gene_names, plot=None):
    """
    Rank genes by high mean AND low standard deviation using:
        score = Z_mean - Z_std

    Parameters
    ----------
    mean_arr : 1D array
        Mean value per gene.
    std_arr : 1D array
        Standard deviation per gene.
    top_k : int or None
        If provided, returns only the indices of the top_k genes.

    Returns
    -------
    score : np.ndarray
        Combined Z-score for each gene.
    ranked_idx : np.ndarray
        Indices sorted from best to worst (descending score).
    """

    # Row-wise mean and std (axis=1)
    row_mean = np.nanmean(matrix_selected, axis=1)
    row_std  = np.nanstd(matrix_selected, axis=1, ddof=1)  # sample std
    row_median = np.nanmedian(matrix_selected, axis=1)

    mean_arr = np.asarray(row_mean, dtype=float)
    std_arr  = np.asarray(row_std, dtype=float)
    median_arr = np.asarray(row_median, dtype=float)

    # Compute Z-scores
    z_mean = (mean_arr - mean_arr.mean()) / (mean_arr.std() + 1e-12)
    z_std  = (std_arr - std_arr.mean()) / (std_arr.std() + 1e-12)

    # Low std is good ⇒ invert the sign
    score = z_mean - z_std
    score = np.asarray(score, dtype=float)

    # Sort descending (highest score = best gene)
    ranked_idx = np.argsort(score)[::-1].astype(int)

    if plot:
        plot_mean_std_with_labels(mean_arr, std_arr, score, ranked_idx, gene_names, filename=plot)

    return (gene_names, mean_arr, std_arr, median_arr, z_std, ranked_idx, score, z_mean)


def plot_mean_std_with_labels(mean_arr, std_arr, score, ranked_idx, gene_names, filename,  *,
                              top_k=20, figsize=(9,7), jitter=0.003):
    """
    Scatter plot of mean vs std, colored by combined Z-score.
    Labels the top_k genes with highest score.

    Parameters
    ----------
    gene_names : list/array of str
        Gene names, length N.
    top_k : int
        Number of top genes to label (highest scoring).
    figsize : tuple
        Figure size.
    jitter : float
        Small offset added to labels to prevent overlap.
    """
    gene_names = np.asarray(gene_names)

    # Plot core scatter
    plt.figure(figsize=figsize)
    sc = plt.scatter(mean_arr, std_arr, c=score, s=20)

    plt.xlabel("Mean gene count")
    plt.ylabel("Standard deviation")
    plt.title("Gene Selection: High Mean + Low STD (Labeled Top Genes)")
    plt.colorbar(sc, label="Combined Score (Z_mean − Z_std)")
    plt.grid(True, alpha=0.3)

    # --- Label the top_k genes ---
    top_idx = ranked_idx[:top_k]
    for idx in top_idx:
        plt.scatter(mean_arr[idx], std_arr[idx], 
                    edgecolor="black", facecolor="none", s=80)
        
        # Add a small jitter to avoid label overlap
        plt.text(mean_arr[idx] + jitter * mean_arr.max(),
                 std_arr[idx]  + jitter * std_arr.max(),
                 gene_names[idx],
                 fontsize=8,
                 weight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    

def classify(row):
    c = row["Combined_topk"]
    a = row["C0_topk"]
    b = row["C1_topk"]

    if c and a and b:
        return "Combined,C0,C1"

    if c and a:
        return "Combined,C0"

    if c and b:
        return "Combined,C1"

    if a and b:
        return "C0,C1"

    if c:
        return "Combined"

    if a:
        return "C0"

    if b:
        return "C1"

    return "Missing"


def merge_mean_std_arrays(
    arrays,
    top_k=25,
    csv_path=None
):
    """
    Merge mean-std ranking results stored as arrays instead of DataFrames.

    Each input must be of the form:
        (gene_names, mean_arr, std_arr, z_mean_arr, z_std_arr, score_arr, rank_arr)

    Output: A single merged DataFrame indexed by gene.
    """

    def build_df(arr, prefix):
        gene, mean, std, median, z_std, ranked_idx, score, z_mean = arr

        rank_arr = np.empty_like(ranked_idx)
        rank_arr[ranked_idx] = np.arange(len(ranked_idx)) + 1

        df = pd.DataFrame({
            "gene": gene,
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
            f"{prefix}_med": median,
            f"{prefix}_z_mean": z_mean,
            f"{prefix}_z_std": z_std,
            f"{prefix}_score": score,
            f"{prefix}_rank": rank_arr,
        })
        return df

    # Build individual DataFrames
    df = {}
    for cluster in ['Combined', 'C0', 'C1']:
        df[cluster] = build_df(arrays[cluster], cluster)

    # Merge all three on gene
    merged = df['Combined'].merge(df['C0'], on="gene", how="outer") \
                           .merge(df['C1'], on="gene", how="outer")

    for col in ["Combined_rank", "C0_rank", "C1_rank"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    # ---------------------------------------------------------
    # Add overlap flags (in top-k for each cluster)
    # ---------------------------------------------------------
    merged["Combined_topk"] = merged["Combined_rank"] <= top_k
    merged["C0_topk"]       = merged["C0_rank"] <= top_k
    merged["C1_topk"]       = merged["C1_rank"] <= top_k

    merged["overlap_label"] = merged.apply(classify, axis=1)
    print_topk_combined(merged, label='rank', top_k=top_k, print_list=False)
    

    merged = merged.sort_values(["Combined_rank", "C0_rank", "C1_rank"], ascending=True)

    # Save if requested
    merged.to_csv(csv_path, index=False)
    print(f"Saved merged CSV to {csv_path}")


def print_topk_combined(df, label='rank', top_k=50, print_list = False):
    ascending = True if label == 'rank' else False
    df_sorted = df.sort_values([f"Combined_{label}", f"C0_{label}", f"C1_{label}"], ascending=ascending)
    ## top = df[df[f"Combined_{label}"] <= top_k]
    top = df_sorted.head(top_k)
    label = 'mean'
    print(f"\nTop {top_k} Combined genes:")
    for _, row in top.iterrows():
        if print_list:
            print(f'"{row["gene"]}",')
        else:
            print(f'"{row["gene"]}", {row[f"Combined_{label}"]}, {row[f"C0_{label}"]}, {row[f"C1_{label}"]}')
    return top


def combine_results(cluster_list, matrix, gene_names, location):
    arr = {}
    print (cluster_list)
    for cluster in ['Combined', 'C0', 'C1']:
        if cluster != 'Combined':
            cols = [i for i, c in enumerate(cluster_list) if c == cluster]
            print (cols)
            matrix_selected = matrix[:, cols]
        else:
            matrix_selected = matrix

        print (matrix_selected.shape)

        filename = f"{location}/{cluster}.zscore.png"
        arr[cluster] = rank_genes_by_mean_std(matrix_selected, gene_names, filename)

    csv_path = f'{location}/zscore.csv'
    merge_mean_std_arrays(arr, csv_path=csv_path)


cluster_location = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/results_1/cluster_info"    

def open_read_count(depth, sample_id='NAIN3'):
    barcode_dict = {}
    cluster_dict = {}
    # file = f'{cluster_location}/{depth}/2cellline.cell_clusters.csv'   
    file = f'{cluster_location}/{sample_id}_cell_clusters_depth.{depth}.csv'
    with open(file, 'r') as f:
        next(f)
        for line in f:
            line = line.strip('\n').split(',')

            if line[0] != sample_id: continue

            bc_id = line[1].replace('bc','')
            bc = line[2]
            barcode_dict[bc] = int(bc_id)
            cluster_dict[line[1]] = f'C{line[3]}' 
            
    return barcode_dict, cluster_dict


def create_barcode_idx(line):
    barcode_dict = {}
    barcode_idx = line[1:]
    for c, bc in enumerate(list(barcode_idx)):
        barcode_dict[c] = bc

    return barcode_dict


def open_count_file(sample_id, reference_index, cluster_dict, depth=40):
    count_location = '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/data_1'
    filename = f'PIP_2cell_{sample_id}_1_{depth}'
    files = [
        f"{count_location}/{filename}/expression/{filename}_steptwo_filter40_exp.tsv",
        # f"{count_location}/{filename}/expression/{filename}_rRNA_mtRNA_filter40_exp.tsv",
    ]

    print (f'Opening {filename}_steptwo_filter40_exp.tsv')

    missing = set()
    barcode_index = {}
    matrix = np.zeros((len(reference_index), len(cluster_dict)))

    for file in files:
        opener, mode = utils.create_opener(file)
        with opener(file, mode) as f:
            for c, line in enumerate(f):
                line = line.strip('\n').split('\t')
                if c == 0: 
                    if not barcode_index:
                        barcode_index = create_barcode_idx(line)
                        print (f'Number of Barcode: {len(barcode_index)}')
                        
                    continue

                gene_index = reference_index[line[0]]
                for idx, count in enumerate(line[1:]):
                    bc = barcode_index[idx]
                    if bc not in cluster_dict.keys(): 
                        missing.add(bc)
                        continue

                    idx = cluster_dict[bc] - 1 ## Sort according to existing cluster file
                    matrix[gene_index, idx] = count            

    print(f'Missing Barcode: {len(missing)}')
    all_zero = (matrix == 0).all(axis=1)
    keep_mask = ~(all_zero)
    keep_indices = np.where(keep_mask)[0]
    cleaned_matrix = matrix[keep_mask]

    gene_names = []
    rev_reference = {v:k for k,v in reference_index.items()}
    for idx in keep_indices:
        gene_names.append(rev_reference[idx])

    return cleaned_matrix, gene_names

def main():

    depth = 10
    sample_id='NAIN3'

    barcode_dict, cluster_mapping = open_read_count(depth, sample_id=sample_id)

    rev_cluster_dict = defaultdict(list)
    for k, v in cluster_mapping.items():
        rev_cluster_dict[v].append(k)

    barcode_list = [f'bc{v}' for _,v in barcode_dict.items()]

    _, reference_index = utils.create_reference()
    gene_matrix, gene_names = open_count_file(sample_id, reference_index, barcode_dict, depth)

    cluster_label = utils.create_cluster_label(barcode_list, cluster_mapping)

    workdir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline"
    location = f"{workdir}/results_1/select_topk_genes/{depth}"

    Path(location).mkdir(parents=True, exist_ok=True)
    combine_results(cluster_label, gene_matrix, gene_names, location)


if __name__ == "__main__":
    main()
    