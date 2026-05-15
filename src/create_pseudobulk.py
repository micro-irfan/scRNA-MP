#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from common import open_file


def create_coverage_mask(matrix_raw, matrix_cov, min_cov=10):
    keep_mask = matrix_cov > min_cov
    matrix_raw = np.where(keep_mask, matrix_raw, np.nan)
    return matrix_raw


def generate_psuedobulk_by_cluster(matrix, cluster_dict=None, barcode_index=None):
    pseudobulk = {}
    pseudobulk['All'] = np.nanmean(matrix, axis=1)
    if not cluster_dict:
        return pseudobulk

    for i in range(len(cluster_dict.keys())):    
        bc_list = cluster_dict[f'C{i}']
        indices_to_keep = []
        for bc in bc_list:
            if bc not in barcode_index.keys(): continue
            indices_to_keep.append(barcode_index[bc])

        filtered_mat = matrix[:, indices_to_keep]
        row_mean = np.nanmean(filtered_mat, axis=1)
        pseudobulk[f'C{i}'] = row_mean

    return pseudobulk



def write_pseudobulk(data, filename, reference_list, coverages = [10, 20, 50]):
    cluster_label = list(data[coverages[0]].keys())

    frame = {
        (cluster, cov): pd.Series(data[cov][cluster], index=reference_list)
        for cov in coverages
        for cluster in cluster_label
    }

    df = pd.DataFrame(frame)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["cluster", "coverage"])
    df.index.name = "reference"
    df_cleaned = df[~(df.isna()).all(axis=1)]
    df_cleaned.to_csv(filename)


def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Calculate Psuedobulk For DMSO Files',
        description = ""
    )

    parser.add_argument('-s', '--sample_id', required = True,
                        help='Sample ID to Analyze')

    parser.add_argument('-m', '--method', required = False, type=str, default='single_base',
                        help='Rolling Window, Fixed Window Or Single Base')

    parser.add_argument('-w', '--work_path', required = True, type=str, default='',
                        help='Path to store output')          

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    method = args.method  # or 'fixed'
    cell_txt = "AllCells"
    sample_id = args.sample_id

    work_path = args.work_path
    path_to_pseudobulk = f"{work_path}/pseudobulk/{method}"
    Path(path_to_pseudobulk).mkdir(parents=True, exist_ok=True)

    path_to_matrices = f'{work_path}/matrices/{method}'

    filename = f"{path_to_matrices}/{sample_id}/{sample_id}.mutrate.matrix10.{cell_txt}.csv"
    mut_bc, window_label, mut_mat = open_file(filename)  

    filename = f"{path_to_matrices}/{sample_id}/{sample_id}.coverage.matrix10.{cell_txt}.csv"
    cov_bc, _, cov_mat = open_file(filename)

    assert mut_mat.shape == cov_mat.shape, (mut_mat.shape, cov_mat.shape)
    assert mut_bc == cov_bc

    coverages = [10,20,50,100,200]
    pseudobulk = {}
    for coverage in coverages:
        print (f"Calculating Pseudobulk for {coverage} cov!")
        raw_matrix_filtered = create_coverage_mask(mut_mat, cov_mat, min_cov=coverage)
        pseudobulk[coverage] = generate_psuedobulk_by_cluster(raw_matrix_filtered)

    filename = f'{path_to_pseudobulk}/{sample_id}.pseudobulk.filtered.byWindows.allCells.csv'
    write_pseudobulk(pseudobulk, filename, window_label, coverages=coverages)


if __name__ == "__main__":  
    main()
