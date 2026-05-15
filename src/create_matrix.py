#!/usr/bin/env python3

import common as utils
import argparse
import numpy as np
import os
from collections import OrderedDict

from common import create_reference

def create_pileup_dict_by_transrcipt(reference):
    pileup = {gene:{} for gene in reference.keys()}
    return pileup    


def open_pileup(file, reference, filter_low_depth=10, min_count=1):
    '''
    Returns a dict of genes(key) mapped to positions found
    '''
    output = create_pileup_dict_by_transrcipt(reference)
    to_include = set()
    with open(file, 'r') as f:
        for c, line in enumerate(f):
            if c == 0: 
                header = line.strip('\n').split(',')
                continue

            line = line.strip('\n').split(',')
            data = utils.struct(dict(zip(header, line)))
            gene = data.tx_id
            pos = int(data.pos)
            cov = float(data.cov)

            if cov < filter_low_depth: continue
            if min_count > int(data.bases): continue

            output[gene][pos] = data
            to_include.add(gene)

    tmp = {}
    for gene, gene_data in output.items():
        if gene in to_include: 
            tmp[gene] = gene_data            

    return tmp


def make_matrix_per_cell(gene, seq_len, pileup, filter_cov=10):
    cell_count = len(pileup)
    mat_cov = np.full((seq_len, cell_count), np.nan)
    mat_raw = np.full((seq_len, cell_count), np.nan)
    mat_mut = np.full((seq_len, cell_count), np.nan)
    for c, (bc, gene_data) in enumerate(pileup.items()):
        if gene not in gene_data.keys():
            continue
        
        print (gene, bc)
        for p, mut in gene_data[gene].items():
            p -= 1 
            if float(mut.cov) < filter_cov: continue

            mat_raw[p, c] = float(mut.mutrate)    
            mat_cov[p, c] = float(mut.cov)
            mat_mut[p, c] = float(mut.mut)          

    ## 0-Based
    indices = np.nonzero(~np.all(np.isnan(mat_raw), axis=1))[0]
    
    mat_raw = mat_raw[indices]
    mat_cov = mat_cov[indices]
    mat_mut = mat_mut[indices]

    return mat_raw, mat_cov, mat_mut, indices


def generate_matrix(pileup, reference):
    mat_raw_list = []
    mat_cov_list = []
    mat_mut_list = []
    position_list = []
    for gene, seq_len in reference.items():
        mat_raw, mat_cov, mat_mut, pos = make_matrix_per_cell(gene, seq_len, pileup)
        mat_raw_list.append(mat_raw)
        mat_cov_list.append(mat_cov)
        mat_mut_list.append(mat_mut)
        position_list += [f'{gene}-{i}' for i in pos]

    stacked_raw_matrix = np.concatenate(mat_raw_list, axis=0)
    stacked_cov_matrix = np.concatenate(mat_cov_list, axis=0)
    stacked_mut_matrix = np.concatenate(mat_mut_list, axis=0)
    return stacked_raw_matrix, stacked_cov_matrix, stacked_mut_matrix, position_list


def generate_barcode_idx(pileup):
    barcode_idx = []
    for _, (bc, _) in enumerate(pileup.items()):
        barcode_idx.append(bc)

    return {bc_id:c for c, bc_id in enumerate(barcode_idx)}


def filter_nan_rows(matrix_raw, matrix_cov, mutrix_mut, reference_list):

    assert matrix_raw.shape == matrix_cov.shape, (matrix_raw.shape, matrix_cov.shape, mutrix_mut.shape)

    keep_mask = ~np.all(np.isnan(matrix_raw), axis=1)
    matrix_raw = matrix_raw[keep_mask]
    matrix_cov = matrix_cov[keep_mask]
    mutrix_mut = mutrix_mut[keep_mask]
    
    keep_indices = np.nonzero(keep_mask)[0]
    reference_list = [reference_list[i] for i in keep_indices]

    assert matrix_raw.shape == matrix_cov.shape, (matrix_raw.shape, matrix_cov.shape, mutrix_mut.shape)
    assert matrix_raw.shape == mutrix_mut.shape, (matrix_raw.shape, matrix_cov.shape, mutrix_mut.shape)
    
    return matrix_raw, matrix_cov, mutrix_mut, reference_list


def create_coverage_mask(matrix_raw, matrix_cov, mutrix_mut, min_cov=10):
    keep_mask = matrix_cov > min_cov

    matrix_raw = np.where(keep_mask, matrix_raw, np.nan)
    matrix_cov = np.where(keep_mask, matrix_cov, np.nan)
    mutrix_mut = np.where(keep_mask, mutrix_mut, np.nan)
    
    return matrix_raw, matrix_cov, mutrix_mut


def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Create Matrices For Coverage / Mutrate / Mutation',
        description = ""
    )
    
    parser.add_argument('-s', '--sample_id', required = True,
                        help='Sample ID of Run')    

    parser.add_argument('-c', '--coverage', required = False, default = 10, type=int,
                        help='Coverage As Threshold')  
    
    parser.add_argument('-m', '--method', required = False, type=str, default='single_base',
                        help='Rolling Window, Fixed Window Or Single Base')      

    parser.add_argument('--cells', required = False, default = 0, type=int,
                        help='Number of Top Cells By Read Count To Include, 0 will include all reads with >10000 reads')  

    args = parser.parse_args()
    return args


def main():
    from pathlib import Path
    from common import convert_to_df, workdir, fasta_file

    args = get_args()
    sample_id = args.sample_id
    method = args.method
    
    RESULTS_DIRECTORY = f"{workdir}/matrices/{method}"
    location = f"{RESULTS_DIRECTORY}/{sample_id}"

    Path(location).mkdir(parents=True, exist_ok=True)

    cell_count = 0
    cell_txt = "AllCells" if not cell_count else f'{cell_count}Cells'
    
    reference, _ = create_reference(fasta_file, keep_seq = True)

    pileup_location = f"/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/results_3/make_windows/{method}"
    tmp_location = f"{pileup_location}/{sample_id}"

    files = [f for f in os.listdir(tmp_location) if f.endswith('.csv') and sample_id in f] 
    files.sort()
    
    min_bases = 1 if method == 'single_base' else 6

    count = 0
    results = OrderedDict()
    for f in files:
        count += 1
        print (f'Opening File {count}: {f}')

        bc_id = f.split('.')[0].split('-')[1]
        f = f'{tmp_location}/{f}'
        results[bc_id] = open_pileup(f, reference, min_count=min_bases)
        
    barcode_idx = generate_barcode_idx(results)
    mutrate_matrix, cov_matrix, mut_matrix, row_labels = generate_matrix(results, reference)

    for coverage in [10]:
        print (f'Generating Matrices For Coverage: {coverage}')
        print (mutrate_matrix.shape, cov_matrix.shape, mut_matrix.shape)
        if coverage > 10:
            print (mutrate_matrix.shape)
            mutrate_matrix, cov_matrix, mut_matrix = create_coverage_mask(mutrate_matrix, 
                                                                          cov_matrix, 
                                                                          mut_matrix, 
                                                                          min_cov = coverage)

            print (mutrate_matrix.shape)
        
        mutrate_matrix_filtered, cov_matrix_filtered, mut_matrix_filtered, row_labels_filtered = filter_nan_rows(mutrate_matrix, cov_matrix, mut_matrix, row_labels)
        print (mutrate_matrix_filtered.shape)
        print (cov_matrix_filtered.shape)

        filename = f"{location}/{sample_id}.mutrate.matrix{coverage}.{cell_txt}.csv"
        convert_to_df(mutrate_matrix_filtered, row_labels_filtered, barcode_idx, filename)

        filename = f"{location}/{sample_id}.coverage.matrix{coverage}.{cell_txt}.csv"
        convert_to_df(cov_matrix_filtered, row_labels_filtered, barcode_idx, filename)

        filename = f"{location}/{sample_id}.mutant.matrix{coverage}.{cell_txt}.csv"
        convert_to_df(mut_matrix_filtered, row_labels_filtered, barcode_idx, filename)


if __name__ == "__main__":
    main()