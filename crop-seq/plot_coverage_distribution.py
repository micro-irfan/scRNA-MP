import pysam
import matplotlib.pyplot as plt
from extract_sgrna_start_stop import seq1 as ref_seq

from pathlib import Path

method = 'umi'
sample_list = ["SNUCROP_D_notsoR_1", "SNUCROP_D_tsoR_1", "SNUCROP_N_notsoR_1", "SNUCROP_N_tsoR_1"]
for sample_id in sample_list:
    if method == 'umi':
        bam_path = f"/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/raw_sgRNA/results/result_1/bwa/umi/{sample_id}/{sample_id}_addCB_R2.sorted.bam"
    elif method == 'fastp':
        bam_path = f"/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/raw_sgRNA/results/result_1/bwa/fastp/{sample_id}/{sample_id}_R2_fastp.sorted.bam"

    ref_len = len(ref_seq)   # plot from position 1 to ref_len

    bam = pysam.AlignmentFile(bam_path, "rb")

    # one shared count dictionary
    count_dict = {pos: 0 for pos in range(1, ref_len + 1)}

    for read in bam.fetch(until_eof=True):
        # skip unmapped / secondary / supplementary
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        # skip multimapped reads
        if read.mapping_quality == 0:
            continue
        if read.has_tag("NH") and read.get_tag("NH") != 1:
            continue

        # reference positions are 0-based
        for pos0 in read.get_reference_positions():
            pos1 = pos0 + 1

            # only count positions from 1 to ref_len
            if 1 <= pos1 <= ref_len:
                count_dict[pos1] += 1

    bam.close()

    # plot
    x_vals = list(range(1, ref_len + 1))
    y_vals = [count_dict[pos] for pos in x_vals]

    plt.figure(figsize=(12, 5))
    plt.plot(x_vals, y_vals)
    plt.xlabel("Position")
    plt.ylabel("Coverage")
    plt.title(f"Coverage distribution from position 1 to {ref_len} for {sample_id}")
    plt.axvline(x=1609, color="red", linestyle="--")
    plt.axvline(x=1629, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"plots/{sample_id}.{method}.coverage_plot.png", dpi=300)
    plt.close()
