#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cluster_cells_utils as utils
import re
import argparse
from pathlib import Path

cell_txt = "AllCells"

def file_exists(path):
    return Path(path).exists()


template = "{treatment}_{}_{}_{volume}" 


def generate_path(args, sample_id):
    work_path = args.work_path
    work_path_post_mapping = f'{work_path}/post_mapping'
    method = args.method
    coverage = args.coverage

    if method == 'fixed_single_base':
        matrix_location = f'{work_path_post_mapping}/single_base_to_fixed_absolute'
        c_file = f"{matrix_location}/{sample_id}.coverage.matrix{coverage}.{cell_txt}.csv"
        r_file = f'{matrix_location}/{sample_id}.fixed_single_base.matrix{coverage}.{cell_txt}.csv'
    elif method == 'single_base':
        matrix_location = f'{work_path_post_mapping}/matrices/{method}/{sample_id}'
        c_file = f"{matrix_location}/{sample_id}.coverage.matrix10.{cell_txt}.csv"
        matrix_location = f'{work_path_post_mapping}/normalized_mtx/{method}/{sample_id}'
        r_file = f'{matrix_location}/{sample_id}.normalized_reactivity.matrix{coverage}.{cell_txt}.gene_level.csv' 

    if args.use_bam:
        path_to_readcount = f'{work_path_post_mapping}/matrices/bam_gene_count' 
        rc_file = f'{path_to_readcount}/{sample_id}.gene_count.mx'
    else:
        path_to_readcount = f'{work_path}/preprocessing/{sample_id}/expression'
        rc_file = f'{path_to_readcount}/{sample_id}_filter40_exp_fastp.tsv'
        ## Need to transpose

    return (c_file, r_file, rc_file)


def check_arguments(args):
    if int(args.coverage) < utils.ACCEPTED_COVERAGE: 
        return True, "Script accepts only Coverage >= 10"
    
    for sample_id in args.sample_list.split(','):
        for file in generate_path(args, sample_id):
            if file_exists(file): continue
            return True, f"{file} does not exist..."
    
    if 'treatment' not in args.template:
        return True, f"Keyword treatment is missing in the template! Example: {template}"

    return False, None
        

def threshold_and_drop_all_nan_rows(path, X):
    """
    Convert values < X to NaN and drop rows that are all NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (numeric values).
    X : float
        Threshold.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = utils.open_matrices(path)

    # Convert values < X to NaN
    df[df < X] = np.nan

    # Drop rows where all columns are NaN
    df = df.dropna(axis=0, how="all")

    return df


def open_cluster_file(sample_id, work_path):
    bam_location = f"{work_path}/preprocessing/{sample_id}/mapping"
    barcode_file = f"{bam_location}/{sample_id}_filter40_barcode.txt"

    barcode_dict = {}
    with open(barcode_file, 'r') as f:
        for c, line in enumerate(f):
            barcode = line.strip('\n')
            barcode_dict[barcode] = f'bc{c+1}'

    return barcode_dict


class OpenMatrices:

    def __init__(self, args):
        self.method = args.method
        self.sample_list = args.sample_list
        coverage_df = {}
        reactivity_df = {}
        readcount_df = {}

        for sample_id in self.sample_list.split(','):
            c_file, r_file, rc_file = generate_path(args, sample_id)

            r_df = utils.open_matrices(r_file) 
            if self.method == 'fixed_single_base':
                c_df = utils.open_matrices(c_file)
                c_df.index = c_df.index.str.replace(r":.*$", "", regex=True)
                r_df.index = r_df.index.str.replace(r":.*$", "", regex=True)

            elif self.method == 'single_base':
                c_df = threshold_and_drop_all_nan_rows(c_file, int(args.coverage))

            coverage_df[sample_id] = utils.add_source_prefix(c_df, sample_id)
            reactivity_df[sample_id] = utils.add_source_prefix(r_df, sample_id)

            barcode_dict = open_cluster_file(sample_id, args.work_path) 
            sep = ',' if args.use_bam else '\t'
            tmp_df = utils.open_matrices(rc_file, sep=sep)
            tmp_df = tmp_df.rename(columns=barcode_dict)
            readcount_df[sample_id] = utils.add_source_prefix(tmp_df, sample_id)

        coverage_merged = pd.concat(coverage_df, axis=1, join="outer") 
        reactivity_merged = pd.concat(reactivity_df, axis=1, join="outer")

        if self.method == 'single_base':
            missing_rows = reactivity_merged.index.difference(coverage_merged.index)
            print("Missing in df2:", len(missing_rows))
            
            common_idx = reactivity_merged.index.intersection(coverage_merged.index)
            coverage_merged = coverage_merged.reindex(common_idx)
            reactivity_merged = reactivity_merged.loc[common_idx]

            print("Rows kept:", len(coverage_merged)) 

        print ('Concordant Matrix Shape ::', coverage_merged.shape)

        readcount_merged = pd.concat(readcount_df, axis=1, join="outer")
        readcount_merged = readcount_merged.reindex(columns=coverage_merged.columns)
        
        readcount_merged.columns = np.array([i[1] for i in readcount_merged.columns])
        coverage_merged.columns = np.array([i[1] for i in coverage_merged.columns])
        reactivity_merged.columns = np.array([i[1] for i in reactivity_merged.columns])
        
        self.r_df = reactivity_merged
        self.c_df = coverage_merged
        self.rc_df = readcount_merged
        

def get_args():
    parser = argparse.ArgumentParser(
        prog = 'CLuster Cells Post Normalization',
        description = ""
    )

    parser.add_argument('--batch_id', required = False, default='',
                        help='Batch ID to name output path')

    parser.add_argument('-s', '--sample_list', required = True,
                        help='Sample ID to Analyze')

    parser.add_argument('-c', '--coverage', required = False, default = '10,20', type=str,
                        help='Coverage to Analyze, e.g., 10,20 - comma separated')  

    parser.add_argument('-m', '--method', required = False, type=str, default='single_base',
                        help='Only single_base and fixed_single_base Supported, Possible to Try fixed (All Lower Caps)')  

    parser.add_argument('-w', '--work_path', required = True, type=str, default='',
                        help='Path to store output')   
    
    parser.add_argument('--template', required = False, type=str, default=template,
                        help=f'Treatment is required! Example of a template: {template}')  
    
    parser.add_argument('--use_bam', action="store_true",
						help="Use Bam Matrix instead of UMI")

    parser.add_argument('--hvw', dest='highly_variable_window', required = False, type=str, default=None,
                        help='Highly Variable Window Selection, Default: None. Options available: compare_cluster')  


    """
    python3 cluster_cells.py -s "DMS_PIP_NAIN3_1in4,DM_PIP_NAIN3_1in4,DMS_PIP_NAIN3_3in4,DM_PIP_NAIN3_3in4" \
                             -m fixed_single_base \
                             -w /scratch/users/astar/gis/muhdih/sgRNA_mutational_rate/cellline/results_3/fastp_filter \ 
                             -c 100
    """

    return parser.parse_args()


def main():
    from cluster_cells_utils import MatrixFiltering
    from cluster_cells_pca import PlotResultsClustering

    args = get_args()

    failed, error_message = check_arguments(args) 
    if failed: 
        print (error_message)
        return
    
    matrices = OpenMatrices(args)

    if args.batch_id:
        batch_id = f'{args.batch_id}_{args.method}'
    else:
        batch_id = args.method

    ## TODO ## - Make it flexible
    use_bam = args.use_bam
    filter_high_mt = False
    mt_threshold = 10
    to_filter_by_readcount = False
    read_threshold = 10000
    if use_bam and to_filter_by_readcount:
        assert read_threshold >= 80000
    
    remove_bad_cells = True

    if use_bam:
        batch_id += '_bam'
    if filter_high_mt:
        batch_id += f'_filterMT{mt_threshold}'
    if to_filter_by_readcount:
        batch_id += f'_{read_threshold}'
    if remove_bad_cells:
        batch_id += '_removePoorCells'
    if args.highly_variable_window:
        batch_id += f'_{args.highly_variable_window}' 

    normalisation = 'winsor'
    extension = "by_matrix" if normalisation == 'quantile' else 'by_gene_per_cell'
    batch_id += f'_{normalisation}_{extension}'
    
    # remove_low_bases = 5
    # batch_id += f'_min{remove_low_bases}bases'

    path_to_output = f"{args.work_path}/clustering_results/{batch_id}/{args.coverage}"
    path_to_plots = f'{path_to_output}/plots'
    Path(path_to_plots).mkdir(parents=True, exist_ok=True)
    
    ## Matrix Filtering!
    filtered_matrices = MatrixFiltering(args, 
                                        batch_id, 
                                        matrices, 
                                        path_to_output,
                                        highly_variable_window=args.highly_variable_window,
                                        second_normalization=normalisation)
    
    ## Plotting Functions! 
    plot_obj = PlotResultsClustering(args, batch_id, filtered_matrices, path_to_plots)
    plot_obj.plot_clustermap()
    plot_obj.generate_matrix_heatmap()
    plot_obj.initialize_pca()
    plot_obj.run_pca()

if __name__ == "__main__":
    main()