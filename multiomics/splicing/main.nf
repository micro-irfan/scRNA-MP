#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

process generateFastq {
    label 'samtools_1'

    input:
    tuple val(sample_id), path(bam), path(barcode_file)

    output:
    tuple val(sample_id), path("*.fq.gz")

    script:
    """
    {
    read
        while IFS=',' read -r bc_id barcode
        do
            bc_id=\$(echo "\$bc_id" | tr -d '\\r')
            barcode=\$(echo "\$barcode" | tr -d '\\r')
            samtools view -h -r \${bc_id} ${bam} | \
            samtools fastq -F 0x900 - | \
            gzip > \${bc_id}.fq.gz
        done
    } < ${barcode_file}
    """
}


process salmonQuantification {
    label 'salmon'
    publishDir "${params.outdir}/salmon/${run_id}/", mode: 'move'

    input:
    tuple val(run_id), val(bc_id), path(fastq)
    path salmon_index

    output:
    tuple val(bc_id), path("${bc_id}.quant.sf")

    script:
    """
    salmon quant \
        -i ${salmon_index} \
        -l A \
        -r ${fastq} \
        --validateMappings \
        -p ${task.cpus} \
        -o ${bc_id}

    cp ${bc_id}/quant.sf ${bc_id}.quant.sf
    """

}

process addReadGroup {
    label 'pysam' 

    input:
    tuple val(sample_id), path(bam)

    output:
    tuple val(sample_id), path("${sample_id}.addRG.bam"), path("${sample_id}.barcode.*"), emit: barcode_out
    tuple val(sample_id), path("${sample_id}.cell_clusters.read_count.csv"), emit: read_count

    script:
    """
    python ${baseDir}/bin/add_readgroup.py -s ${sample_id} \
                                 -b ${bam} \
                                 -o ${sample_id}.addRG.bam \
                                 --chunk_size 100
    """
}

params.workDir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline"
params.directory = "${params.workDir}/data_3"

// Create channel for input reads 
Channel
    .fromPath(params.samplesheet)
    .splitCsv(header: true)
    .map { row ->
        def sample_id = row.sample_id
        def dir = params.directory
        tuple(
            sample_id,
            file("${dir}/${sample_id}/mapping/${sample_id}_filter40_dedup.bam"),
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

    fastq_output = generateFastq(flattened_ch)

    // grouped_output = pileup_output
    //     .groupTuple(by: 0)
    //     .map { sample_id, file_lists -> 
    //         tuple(sample_id, file_lists.flatten()) 
    //     }

    // Channel.fromPath('data/*.fq.gz')
    //     .map { f -> tuple(f.baseName, f) }
    //     .set { reads_ch }

    fastq_output
        .flatMap { sid, reads ->
            reads.collect { r ->
                def fastq_id = r.name.replaceFirst(/\.(fastq|fq)\.gz$/, '')
                tuple(sid, fastq_id, r)
            }
        }
        .set { reads_one_per_fastq_ch }

    reads_one_per_fastq_ch.view()

    salmon_index = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/ref/salmon_hs_index"
    salmonQuantification(reads_one_per_fastq_ch, salmon_index)

}


