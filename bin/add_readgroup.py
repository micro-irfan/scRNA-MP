#!/usr/bin/env python3

import pysam
import argparse

sample_dict = {
    'PIP2_PIP_DMSO' : 'DMSO',
    'PIP2_PIP_NAIN3' : 'NAIN3',
}

def write_read_count(batch_id, barcode_dict, read_count):
    barcode_dict = {bc_id:bc for bc, bc_id in barcode_dict.items()}
    read_count = dict(sorted(read_count.items(), key=lambda item: item[1], reverse=True))
    treatment = batch_id
    filename = f'{batch_id}.cell_clusters.read_count.csv'

    header = 'treatment,barcode_id,barcode_seq,cluster,reads'
    with open(filename, 'w') as write_file:
        write_file.write(f'{header}\n')

        for bc_id, count in read_count.items():
            barcode = barcode_dict[bc_id]
            to_write = f'{treatment},{bc_id},{barcode},{count}'
            write_file.write(f'{to_write}\n')

def write_chunk(barcode_dict, sample_id, estimated_chunk_size = 100):
    number_of_bc = len(barcode_dict.keys())
    num_splits = max(1, round(number_of_bc / estimated_chunk_size))
    chunk_size = number_of_bc / num_splits  # not rounded

    ranges = []
    start = 0

    for i in range(num_splits):
        end = start + chunk_size
        ranges.append((round(start), round(end)))
        start = end

    assert ranges[-1][1] == number_of_bc

    for file_number, r in enumerate(ranges):
        lower, upper = r[0], r[1]
        open_new = False
        with open(f"{sample_id}.barcode.{file_number+1}.txt", 'w') as write_file:
            write_file.write('Barcode_ID,Barcode_Seq\n')
            for c, (seq, bc) in enumerate(barcode_dict.items()): 
                if c < lower: continue
                if c >= upper: 
                    open_new = True
                    break

                to_write = f'{bc},{seq}\n'
                write_file.write(to_write)


def add_readgroup(sample_id, in_bam, out_bam, barcode_dict, chunk_size = 100, check_18s=False, threshold=10000):
    """
    Add multiple read groups to a BAM file 
    Add CB tag values

    Parameters:
        in_bam (str): Path to input BAM file.
        out_bam (str): Path to output BAM file.
        barcode_list (list): List of barcode strings (read group IDs).
    """
    multimapped_reads = {}
    read_count = {}
    # Open input BAM for reading
     
    rrna_reference = "human-4V6X-18S" 

    with pysam.AlignmentFile(in_bam, "rb") as infile:
        # Copy header and add @RG lines
        header = infile.header.to_dict()

        # Avoid duplicate RG header entries
        if "RG" not in header:
            header["RG"] = []

        # Add new @RG lines for each barcode
        for barcode, barcode_id in barcode_dict.items():
            count = int(barcode_id.replace('bc','')) - 1
            barcode_meta = {
                "ID": count,
                "BC": barcode,
                "SM": barcode_id,
                "PL": "ILLUMINA"
            }

            read_count[barcode_id] = 0
            multimapped_reads[barcode_id] = 0
            header["RG"].append(barcode_meta)

        # Open output BAM for writing
        with pysam.AlignmentFile(out_bam, "wb", header=pysam.AlignmentHeader.from_dict(header)) as outfile:
            for c, read in enumerate(infile):
                
                if (c+1) % 1000000 == 0:
                    print(f"Processed {c+1} reads...")

                # @LH00504:187:22YLC5LT3:8:1101:9627:1240_AAGCTCCTCATCTAGTAATCGCCAACAT_AGGACG
                if read.is_unmapped:
                    continue
                    
                read_id = read.query_name
                barcode = read_id.split('_')[1] 
                if barcode not in barcode_dict.keys():
                    continue

                if read.reference_name != rrna_reference and check_18s:
                    continue

                read_group = barcode_dict[barcode]

                ## Keeps Only Primary Read 
                if read.is_secondary:
                    multimapped_reads[read_group] += 1
                    continue
                
                read.set_tag("RG", read_group, value_type="Z")
                outfile.write(read)
                read_count[read_group] += 1

    print(f"Done. Wrote BAM with multiple RG tags to {out_bam}")

    low_count = [k for k,v in read_count.items() if v < threshold]
    if len(low_count) > 0:
        barcode_dict = {k:v for k,v in barcode_dict.items() if v not in low_count}

    if not barcode_dict:
        print ("No barcodes passed the threshold")
        return read_count

    write_chunk(barcode_dict, sample_id, chunk_size)
    return read_count


def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Adding Barcode to RG',
        description = ""
    )

    parser.add_argument('--bam', dest = "bam", required = True,
                        help="Bam File To Add ReadGroup")
    
    parser.add_argument('--barcode', dest = "barcode", required = True,
                        help="barcode File From Umi Tools")
    
    parser.add_argument('-s', '--sample_id', required = True,
                        help='Sample ID of Run')    
    
    parser.add_argument('-o', '--output', required = True,
                        help='Output Bam With added RG based on Barcode')    

    parser.add_argument('--chunk_size', type=int, default = 100,
                        help='rough chunk size for next process') 

    args = parser.parse_args()
    return args


def open_cluster_file(barcode_file):
    barcode_dict = {}
    with open(barcode_file, 'r') as f:
        for c, line in enumerate(f):
            barcode = line.strip('\n')
            barcode_dict[barcode] = f'bc{c+1}'

    return barcode_dict


def main():
    args = get_args()
    sample_id = args.sample_id
    bam_file = args.bam
    out_bam = args.output

    # workdir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline"
    # bam_location = f"{workdir}/data_3/{sample_id}/mapping"
    # barcode_file = f"{bam_location}/{sample_id}_filter40_barcode.txt"
    barcode_file = args.barcode
    barcode_dict = open_cluster_file(barcode_file) 

    read_count = add_readgroup(sample_id, bam_file, out_bam, barcode_dict, chunk_size=args.chunk_size)
    write_read_count(sample_id, barcode_dict, read_count)


if __name__ == "__main__":  
    main()