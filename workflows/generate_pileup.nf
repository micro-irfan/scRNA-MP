#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

process MPILEUP {
    label 'samtools_1'
    publishDir "${params.outdir}/pileup/${sample_id}/", mode: 'move'

    input:
    tuple val(sample_id), path(bam), path(barcode_file)
    path ref
    path refIdx

    output:
    tuple val(sample_id), path("*.pileup")

    script:
    """
    {
    read
        while IFS=',' read -r bc_id barcode
        do
            bc_id=\$(echo "\$bc_id" | tr -d '\\r')
            barcode=\$(echo "\$barcode" | tr -d '\\r')
            samtools view -h -r \${bc_id} ${bam} | \
            samtools mpileup -f ${ref} - > ${sample_id}.\${bc_id}.pileup 
        done
    } < ${barcode_file}
    """
}


process addReadGroup {
    label 'pysam' 

    input:
    tuple val(sample_id), path(bam), path(barcode)

    output:
    tuple val(sample_id), path("${sample_id}.addRG.bam"), path("${sample_id}.barcode.*"), emit: barcode_out
    tuple val(sample_id), path("${sample_id}.cell_clusters.read_count.csv"), emit: read_count

    script:
    """
    python ${baseDir}/../bin/add_readgroup.py -s ${sample_id} \
                                 --bam ${bam} \
                                 --barcode ${barcode}\
                                 -o ${sample_id}.addRG.bam \
                                 --chunk_size 100
    """
}

params.workDir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline"
params.reference = "${params.workDir}/ref/hg38_ncbi_selected_transcriptome.rmdup"
params.directory = "${params.workDir}/results_3/fastp_filter/preprocessing"

// Create channel for input reads 
Channel
    .fromPath(params.samplesheet)
    .splitCsv(header: true)
    .map { row ->
        def sample_id = row.sample_id
        def dir = params.directory
        tuple(
            sample_id,
            file("${dir}/${sample_id}/mapping/${sample_id}_fastp_filter40_addAlltag.bam"),
            file("${dir}/${sample_id}/mapping/${sample_id}_filter40_barcode.txt"),
        )
    }
    .set { bam_files }


workflow {
    addReadGroup(bam_files)

    flattened_ch = addReadGroup.out.barcode_out.flatMap { sample_id, bam_file, barcode_files ->
        def list = barcode_files instanceof List ? barcode_files : [barcode_files]
        list.collect { bc_file ->
            tuple(sample_id, bam_file, bc_file)
        }
    }

    flattened_ch.view()

    // flattened_ch.view()
    ref = "${params.reference}.fa"
    refIdx = "${params.reference}.fa.fai"
    pileup_output = MPILEUP(flattened_ch, ref, refIdx)

    grouped_output = pileup_output
        .groupTuple(by: 0)
        .map { sample_id, file_lists -> 
            tuple(sample_id, file_lists.flatten()) 
        }

}


