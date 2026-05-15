#!/bin/bash

module load nextflow/23.04.2
workdir="/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline"

nextflow run ${workdir}/src/13_mofa3/main.nf \
             -c ${workdir}/src/13_mofa3/nextflow.config \
             --cacheDir /home/users/astar/gis/muhdih/scratch/image \
             --outdir ${workdir}/results_3/salmon_results \
             --batchName salmon_results \
             --samplesheet ${workdir}/src/13_mofa3/sample.csv \
             -resume

SUPPA=/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/mofa/src/4_suppa/SUPPA
samples=('DMS_PIP_NAIN3_3in4' 'DMS_PIP_NAIN3_1in4' 'DM_PIP_NAIN3_3in4' 'DM_PIP_NAIN3_1in4')
referenceDir=/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/ref/Homo_sapiens.GRCh38.115.gtf

module load python/3.8.6
source /home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/analysis/scanpy/bin/activate
python3 salmon_2_mx.py

for sample_id in "${samples[@]}"; do
    echo "Processing $sample"
    path_to_file=/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/results_3/salmon_results/tpm/${sample_id}/isoform_expression_tpm.tsv
    path_to_result=/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/results_3/salmon_results/psiPerIsoform/${sample_id}_isoform_expression_tpm.tsv
    python3 ${SUPPA}/suppa.py psiPerIsoform \
            -g ${referenceDir} \
            -e ${path_to_file}\
            -o ${path_to_result}
    
done
