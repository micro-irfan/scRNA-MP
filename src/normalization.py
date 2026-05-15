#!/usr/bin/env python3

from pathlib import Path
import normalization_utils as utils
import numpy as np
import argparse
from common import convert_to_df, open_file


def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Normalzation For Reactivities',
        description = ""
    )

    parser.add_argument('-s', '--sample_id', required = True,
                        help='Sample ID to Analyze')

    parser.add_argument('-c', '--coverage', required = False, default = '10,20', type=str,
                        help='Coverage to Analyze, e.g., 10,20 - comma separated')  

    parser.add_argument('-m', '--method', required = False, type=str, default='single_base',
                        help='Rolling Window, Fixed Window Or Single Base')  

    parser.add_argument('-w', '--work_path', required = True, type=str, default='',
                        help='Path to store output')    

    return parser.parse_args()


def main():
    args = get_args()
    sample_id = args.sample_id

    cell_txt = "AllCells" # Future Update to include a cut eg. top100 cells
    method = args.method
    coverage_to_normalize = [int(i) for i in args.coverage.split(',')]
    by_gene_level_normalization = method != 'fixed'

    work_path = args.work_path
    path_to_output = f"{work_path}/normalized_mtx/{method}/{sample_id}"
    Path(path_to_output).mkdir(parents=True, exist_ok=True)
    path_to_matrix = f"{work_path}/matrices/{method}"

    filename = f"{path_to_matrix}/{sample_id}/{sample_id}.mutrate.matrix10.{cell_txt}.csv"
    raw_barcode_list, _, raw_matrix = open_file(filename)

    filename = f"{path_to_matrix}/{sample_id}/{sample_id}.coverage.matrix10.{cell_txt}.csv"
    cov_barcode_list, reference_list, cov_matrix = open_file(filename)  

    assert raw_barcode_list == cov_barcode_list
    assert raw_matrix.shape == cov_matrix.shape, (raw_matrix.shape, cov_matrix.shape)

    for coverage in coverage_to_normalize: 
        print (f"Normalizing For for {coverage} cov!")

        print ("Creating Coverage Mask", raw_matrix.shape)
        raw_matrix_filtered = utils.create_coverage_masks(
            sample_id,
            raw_matrix,
            cov_matrix,
            reference_list,
            coverage,
            work_path,
            method=method,
        )

        print ("Filtering Nan", raw_matrix_filtered.shape)
        raw_matrix_filtered, row_labels_filtered = utils.filter_nan_rows(raw_matrix_filtered, reference_list)

        assert not np.array_equal(raw_matrix, raw_matrix_filtered, equal_nan=True)
        
        filename = f"{path_to_output}/{sample_id}.raw_reactivity.matrix{coverage}.{cell_txt}.csv"
        convert_to_df(raw_matrix_filtered, row_labels_filtered, raw_barcode_list, filename)
        print ("Shape of Matrix:", raw_matrix_filtered.shape)
        print(f'Saved Raw Matrix File To {filename}')

        if by_gene_level_normalization:
            normalized_matrix = utils.winsorized_normalization_by_gene(
                raw_matrix_filtered,
                row_labels_filtered,
                None,
                bottom_winsorize=0.05,
                top_winsorize=0.01,
                transcriptome_winsorize=False,
                gene_winsorize=True,
                sep="-" # adjust if your row labels use a different separator
            )   
            filename = f"{path_to_output}/{sample_id}.normalized_reactivity.matrix{coverage}.{cell_txt}.gene_level.csv"
        else:
            normalized_matrix = utils.winsorized_normalization(raw_matrix_filtered)
            filename = f"{path_to_output}/{sample_id}.normalized_reactivity.matrix{coverage}.{cell_txt}.winsorized.csv"
        
        assert not np.array_equal(normalized_matrix, raw_matrix_filtered, equal_nan=True)
        convert_to_df(normalized_matrix, row_labels_filtered, raw_barcode_list, filename)
        print(f'Saved Normalized Matrix File To {filename}')


if __name__ == "__main__":
    main()