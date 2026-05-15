#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from common import create_reference, open_barcode_txt, convert_to_df
import pysam
import argparse


def open_bam(in_bam, barcode_dict, read_count_matrix, filename, reference_index):
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


def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Create Count Matrix From Bam',
        description = "Create a Count Matrix Based on Bam File"
    )

    parser.add_argument('-s', '--sample_id', required = True, type=str, 
                        help='Sample id for the current Conversion')
    
    parser.add_argument('-bam', '--bam_path', required = True, type=str, 
                        help='Path to Bam Files') 
    
    parser.add_argument('-bc', '--barcode', required = True, type=str, 
                        help='Path to File with the list of Barcode') 

    parser.add_argument('-o', '--output_path', required = True, type=str,
                        help='Path to store output') 
    
    parser.add_argument('-r', '--reference', required = True, type=str,
                        help='Path to reference fasta file')
    
    return parser.parse_args()


def main():
    args = get_args()

    barcode_file = args.barcode
    sample_id = args.sample_id
    in_bam = args.bam_path
    location = args.output_path

    Path(location).mkdir(parents=True, exist_ok=True)
    barcode_dict = open_barcode_txt(barcode_file)

    fasta_file = args.reference
        
    reference_index = create_reference(fasta_file, keep_seq=False)
    reference_index = {gene:c for c, gene in enumerate(reference_index)}
    read_count_matrix = np.zeros((len(reference_index), len(barcode_dict)), dtype=int)

    Path(location).mkdir(parents=True, exist_ok=True)
    filename = f'{location}/{sample_id}.gene_count.mx'
    open_bam(in_bam, barcode_dict, read_count_matrix, filename, reference_index)


def pipeline():
    from common import fasta_file, workdir
    reference_index = create_reference(fasta_file, keep_seq=True)
    
    sample_list = [
        "DMS_PIP_DMSO_1in4", "DMS_PIP_NAIN3_1in4",
        "DM_PIP_DMSO_1in4", "DM_PIP_NAIN3_1in4",
        "DMS_PIP_DMSO_3in4", "DMS_PIP_NAIN3_3in4",
        "DM_PIP_DMSO_3in4", "DM_PIP_NAIN3_3in4",
    ]
    
    for sample_id in sample_list:
        print (f"Processing sample: {sample_id}")

        bam_location = f"{workdir}/data_3/{sample_id}/mapping"
        barcode_file = f"{bam_location}/{sample_id}_filter40_barcode.txt"

        barcode_dict = open_barcode_txt(barcode_file)
        
        location = f"{workdir}/results_3/matrices/bam_gene_count/{sample_id}"
        Path(location).mkdir(parents=True, exist_ok=True)

        filename = f'{location}/gene_count.mx'
        in_bam = f"{bam_location}/{sample_id}_filter40_dedup.bam"

        read_count_matrix = np.zeros((len(reference_index), len(barcode_dict)), dtype=int)
        open_bam(in_bam, barcode_dict, read_count_matrix, filename, reference_index)


if __name__ == "__main__":  
    main()
