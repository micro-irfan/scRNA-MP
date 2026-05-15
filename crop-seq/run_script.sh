#!/bin/bash

set -euo pipefail

python3 make_matrix.py --input-dir ../bowtie2_2mm \
                       --output-dir ../matrix/bwt2_2mm \
                       --plot-dir ../plots/bwt2_2mm \
                       --threshold 3

## Toy Example
# python3 qc_grna_expression_scanpy.py \
#         --sample-id SNUCROP_D_notsoR_1 \
#         --singlet-csv ../plots/bwt2/combined_singlet_barcodes_by_gRNA_threshold_t3.csv \
#         --expression-root ../expression \
#         --output-dir ../qc/bwt2 --top-genes-to-plot 6


DMSO_tsoR_1, NAIN3_tsoR_1, DMSO_notsoR_1 and NAIN3_notsoR_1
for sample_id in SNUCROP_D_notsoR_1 SNUCROP_N_notsoR_1 SNUCROP_N_tsoR_1 SNUCROP_D_tsoR_1
do 
    python3 qc_grna_expression_scanpy.py \
        --sample-id ${sample_id} \
        --singlet-csv ../plots/bwt2/combined_singlet_barcodes_by_gRNA_threshold_t3.csv \
        --expression-root ../expression \
        --output-dir ../qc/bwt2/${sample_id} --top-genes-to-plot 6

done