#!/usr/bin/env python3

import pysam
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cluster_location = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/results_1/cluster_info"
def open_cluster_file(batch_id, depth=40):
    barcode_dict = {}
    cluster_dict = {}

    file_name = f'{cluster_location}/{batch_id}_cell_clusters_depth.{depth}.csv'
    with open(file_name, 'r') as f:
        next(f)
        for line in f:
            line = line.strip('\n').split(',')
            if line[0] != batch_id: continue
            barcode = line[2]
            barcode_id = line[1]
            barcode_dict[barcode] = barcode_id
            cluster_dict[barcode_id] = line[3]

    return barcode_dict, cluster_dict


def open_bam(in_bam, barcode_dict, genes_of_interest = []):
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
            if gene not in genes_of_interest:
                continue

            ref_len = read.reference_length

            if gene not in results.keys():
                results[gene] = {v:[] for v in rev_barcode_dict.keys()}

            results[gene][barcode_dict[barcode]] += [ref_len]

    return results 

def assign_group(gene, gene_groups):
    for group, genes in gene_groups.items():
        if gene in genes:
            return group
    return "other"


def create_dataFrame(data, gene_groups):

    rows = []

    for gene, barcode_dict in data.items():
        all_lengths = []
        per_barcode_counts = []

        for cb, lengths in barcode_dict.items():
            lengths = np.asarray(lengths)
            all_lengths.extend(lengths)
            per_barcode_counts.append(len(lengths))

        all_lengths = np.asarray(all_lengths)
        per_barcode_counts = np.asarray(per_barcode_counts)

        rows.append((
            gene,
            all_lengths.mean(),                # avg read length
            per_barcode_counts.mean(),         # avg reads per barcode
            all_lengths.size,                  # total reads (optional, useful)
            len(barcode_dict)                  # number of barcodes
        ))

    df = pd.DataFrame(
        rows,
        columns=[
            "gene",
            "avg_read_length",
            "avg_reads_per_barcode",
            "total_reads",
            "n_barcodes"
        ]
    )

    df["group"] = df["gene"].apply(assign_group, gene_groups=gene_groups)

    return df


def plot_scatter(data, gene_groups, filename):

    df = create_dataFrame(data, gene_groups)

    group_colors = {
        "mean" : "tab:blue",
        "std"  : "tab:orange",
    }

    plt.figure(figsize=(6, 5))

    for group, gdf in df.groupby("group"):
        plt.scatter(
            gdf["avg_read_length"],
            gdf["avg_reads_per_barcode"],
            color=group_colors.get(group, "black"),
            label=group,
            alpha=0.7,
            s=40
        )

    plt.xlabel("Average read length")
    plt.ylabel("Average reads per barcode")
    # plt.yscale("log")   # recommended for counts
    plt.legend(title="Gene group")
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


flatten = lambda nested: [item for sublist in nested for item in sublist]

genes_to_keep_mean = [
    "FAM184B",
    "LMOD3",
    "SEMA3E",
    "ZNF419",
    "PRRG3",
    "PIWIL1",
    "CEBPZOS",
    "NEUROD2",
    "MAVS",
    "CXCL14",
    "DNMT3A",
    "KCNJ3",
    "ADAM10",
    "SLC25A37",
    "TPH1",
    "FOXP1",
    "GPR85",
    "MSX1",
    "RAB12",
    "ARSA",
    "BRAF",
    "LDB1",
    "AP1M1",
    "PDZD8",
    "MYO10",
]

genes_to_keep_std = [
    "RPS27",
    "RPL35",
    "RPL14",
    "RPS2",
    "RPL32",
    "RPL13A",
    "RPL34",
    "RPS29",
    "RPS12",
    "RPS21",
    "RPL35A",
    "RPS5",
    "RPL28",
    "ATP5F1E",
    "RPL9",
    "RPS24",
    "RPLP0",
    "EEF1A1",
    "RPS27A",
    "RPS16",
    "RPL27A",
    "RPL41",
    "HSP90AB1",
    "RPL10",
    "RPL23",
]

genes_to_keep = {
    'mean' : genes_to_keep_mean, 
    'std' : genes_to_keep_std,
}

def main():
    sample_id = "NAIN3"
    depth = 10

    barcode_dict, _ = open_cluster_file(sample_id, depth)    

    genes_of_interest = list(set(flatten([v for _, v in genes_to_keep.items()])))

    workdir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline"
    bam_location = f"{workdir}/data_1/PIP_2cell_NAIN3_1_{depth}/mapping"
    in_bam = f"{bam_location}/PIP_2cell_NAIN3_1_{depth}_steptwo_filter40_dedup.bam"

    data = open_bam(in_bam, barcode_dict, genes_of_interest)

    location = f"{workdir}/results_1/plots/gene_counts"
    Path(location).mkdir(parents=True, exist_ok=True)

    filename = f"{location}/gene_count_v_length.png"
    plot_scatter(data, genes_to_keep, filename)

if __name__ == "__main__":
    main()