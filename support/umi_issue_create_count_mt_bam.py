#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import count_read_length as utils
import common
import pysam

def convert_to_df(matrix, row_labels, barcode_list, filename):
    # Convert to DataFrame
    df = pd.DataFrame(matrix, index=row_labels, columns=barcode_list)
    df.index.name = "gene" 

    # Save to CSV
    df_cleaned = df[~(df.isna()).all(axis=1)]
    df_cleaned.to_csv(filename, sep='\t')

def open_bam(in_bam, barcode_dict, read_count_matrix, filename, reference_index):
    results = {}
    rev_barcode_dict = {v:k for k,v in barcode_dict.items()}
    with pysam.AlignmentFile(in_bam, "rb") as infile:
        for c, read in enumerate(infile):
            if (c+1) % 1000000 == 0:
                print(f"Processed {c+1} reads...")
            
            # @LH00504:187:22YLC5LT3:8:1101:9627:1240_AAGCTCCTCATCTAGTAATCGCCAACAT_AGGACG
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            read_id = read.query_name
            barcode = read_id.split('_')[1] 
            if barcode not in barcode_dict.keys():
                continue

            gene = read.reference_name

            col_idx = int(barcode_dict[barcode].replace('bc','')) - 1
            row_idx = reference_index[gene]

            read_count_matrix[row_idx, col_idx] += 1

    # mask: True = keep
    mask = ~np.all(read_count_matrix == 0, axis=1)

    matrix = read_count_matrix[mask]
    kept_indices = np.where(mask)[0]

    inv_ref = {v: k for k, v in reference_index.items()}
    reference_list = [inv_ref[i] for i in kept_indices]

    inv_barcode = {v: k for k, v in barcode_dict.items()}

    # columns with at least one non-zero
    mask = np.any(matrix != 0, axis=0)
    matrix_filtered = matrix[:, mask]
    kept_indices = np.where(mask)[0]

    barcode_list = [inv_barcode[f'bc{idx+1}'] for idx in kept_indices]

    convert_to_df(matrix_filtered, reference_list, barcode_list, filename)


def pipeline(reference_index, barcode_dict, read_count_matrix, filename, in_bam):
    
    open_bam(in_bam, barcode_dict, read_count_matrix, filename, reference_index)


def open_cluster_file_liver_cells(sample_id):
    cluster_location = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/results_2/cluster_info"
    barcode_dict = {}

    file_name = f'{cluster_location}/cell_clusters_info.csv'
    max_bc_id = 0
    with open(file_name, 'r') as f:
        next(f)
        for line in f:
            line = line.strip('\n').split(',')
            if line[0] != sample_id: continue
            barcode = line[2]
            barcode_id = line[1]
            index = int(barcode_id.replace('bc',''))
            if index > max_bc_id:
                max_bc_id = index   
            
            barcode_dict[barcode] = barcode_id
            

    return barcode_dict, max_bc_id


def main():
    workdir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline"
    _, reference_index = common.create_reference()

    '''
    sample_id = "DMSO"
    depth = 10

    barcode_dict, _ = utils.open_cluster_file(sample_id, depth)        
    bam_location = f"{workdir}/data_1/PIP_2cell_{sample_id}_1_{depth}/mapping"
    in_bam = f"{bam_location}/PIP_2cell_{sample_id}_1_{depth}_steptwo_filter40_dedup.bam"
    
    location = f"{workdir}/results_1/matrices/bam_gene_counts/PIP_2cell_{sample_id}_1_{depth}"
    Path(location).mkdir(parents=True, exist_ok=True)
    filename = f'{location}/gene_count.mx'

    read_count_matrix = np.zeros((len(reference_index), len(barcode_dict)), dtype=int)
    pipeline(reference_index, barcode_dict, read_count_matrix, filename, in_bam)
    '''

    sample_list = ['RHK516' ,'RHK518', 'RHK517','RHK519'] + ['RHK520','RHK522', 'RHK521','RHK523']

    dedup = True

    for sample_id in sample_list:
        print (f"Processing sample: {sample_id}")

        barcode_dict, max_bc_id = open_cluster_file_liver_cells(sample_id)
        print (len(barcode_dict)) 

        bam_location = f"{workdir}/data_2/{sample_id}/mapping"  
        dedup_txt='dedup' if dedup else 'non_dedup'  
        location = f"{workdir}/results_2/matrices/bam_gene/{dedup_txt}/{sample_id}"
        Path(location).mkdir(parents=True, exist_ok=True)
        
        for rna in [True, False]:
            rna_text = '.rRNA' if rna else ''
            filename = f'{location}/gene_count{rna_text}.mx'
            dedup_txt='_dedup'if dedup else '.sorted'
            if rna: 
                in_bam = f"{bam_location}/{sample_id}_rRNA_mtRNA_filter40{dedup_txt}.bam"
            else:
                in_bam = f"{bam_location}/{sample_id}_steptwo_filter40{dedup_txt}.bam"
            
            read_count_matrix = np.zeros((len(reference_index), max_bc_id), dtype=int)
            pipeline(reference_index, barcode_dict, read_count_matrix, filename, in_bam)


if __name__ == "__main__":  
    main()
