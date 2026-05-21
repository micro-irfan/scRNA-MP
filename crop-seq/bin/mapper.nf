#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

process BWA {
    label 'bwa'

    input:
        tuple val(sample_id), path(reads)
        path reference
        val seed_length
        maxRetries 1 

    output:
        tuple val(sample_id), path("${reads.simpleName}.sam")

    script:
        def base = reads.simpleName

        """
        bwa index ${reference}
        bwa mem -t ${task.cpus} -k ${seed_length} ${reference} ${reads} > ${base}.sam
        """
}


process BOWTIE2_BUILD {
    label "bowtie2"
    tag "$sample_id"

    input:
    path(reference)

    output:
    path("*.bt2")

    script:
    """
    bowtie2-build ${reference} rbp_sgrna
    """
}

process COUNT_FILTERED_BARCODES {
    publishDir "${params.outdir}/${method}/${sample_id}", mode: 'copy'
    label 'samtools_8'

    input:
    tuple val(sample_id), path(bam), path(bai)
    val(method)

    output:
    tuple path("${sample_id}.counts.csv")

    script:
    """
    echo "gene,count" > ${sample_id}.counts.csv

    samtools view -h -F 2304 ${bam} | \
    awk 'BEGIN{OFS="\t"}
        /^@/ {next}
        {
            nm = ""
            for (i = 12; i <= NF; i++) {
                if (\$i ~ /^NM:i:/) {
                    split(\$i, a, ":")
                    nm = a[3]
                    break
                }
            }
            if (nm <= 3) {
                print \$3
            }
        }
    ' | \
    LC_ALL=C sort | \
    uniq -c | \
    awk '{OFS=","; print \$2, \$1}' | \
    sort -k1,1nr \
    >> ${sample_id}.counts.csv
    """
}

process CUBATRIE_BAM {
    publishDir "${params.outdir}/cubatrie_bam_${method}_${n_cpus}/${sample_id}", mode: 'copy'
    cpus {n_cpus}
    tag "$sample_id"
    label "cubaTrie_anchors"

    input:
        tuple val(sample_id), path(reads)
        path(reference)
        val(n_cpus)
        val(method)

    output:
        tuple path("${sample_id}.counts.csv"), path("${sample_id}.bam"), path("${sample_id}.bam.bai")

    script:
    int sorting_threads = Math.min((task.cpus / 4) as int, 2)
    int mapping_threads = task.cpus - sorting_threads
    """
    cubaTrie -r ${reference} -i ${reads} -o ${sample_id}.counts.csv -d warn -t ${mapping_threads} --soft-clip --exclude-multihit --m 3 --seed-mm 1 --sam - | \
    samtools sort -@ ${sorting_threads} -o ${sample_id}.bam
    samtools index ${sample_id}.bam 
    """
}

process CUBATRIE_BAM_ANCHORS {
    publishDir "${params.outdir}/cubatrie_bam_${method}_${n_cpus}/${sample_id}", mode: 'copy'
    cpus {n_cpus}
    tag "$sample_id"
    label "cubaTrie_anchors"

    input:
        tuple val(sample_id), path(reads)
        path(reference)
        val(n_cpus)
        val(method)
        val(flank5)
        val(flank3)

    output:
        tuple path("${sample_id}.counts.csv"), path("${sample_id}.bam"), path("${sample_id}.bam.bai")

    script:
    int sorting_threads = Math.min((task.cpus / 4) as int, 2)
    int mapping_threads = task.cpus - sorting_threads
    """
    cubaTrie -r ${reference} \
             -i ${reads} \
             -o ${sample_id}.counts.csv \
             -d warn \
             -t ${mapping_threads} \
             --exclude-multihit \
             -a ${flank5}...${flank3} \
             --anchor-error 3 \
             --soft-clip \
             --m 2 --seed-mm 1 -k 4 \
             --sam - | \
    samtools sort -@ ${sorting_threads} -o ${sample_id}.bam
    samtools index ${sample_id}.bam 
    """
}

process CUTADAPT_FLEX {
    publishDir "${params.outdir}/cutadapt/${sample_id}", mode: 'copy'
    label "cutadapt"
    tag "$sample_id"

    input:
    tuple val(sample_id), file(read)
    val(flank5)
    val(flank3)

    output:
    tuple val(sample_id), path("${sample_id}.trimmed.fastq.gz"), emit: trimmed_reads

    script:
    """
    cutadapt \
        -a ${flank5}...${flank3} \
        -O 6 -e 0.3 \
        --discard-untrimmed \
        --revcomp \
        --max-n 0 \
        -m 20 -M 20 \
        -j ${task.cpus} \
        -o "${sample_id}.trimmed.fastq.gz" \
        ${read}
    """
}


process BOWTIE2_ALIGN_SINGLE_READ {
    label "bowtie2"
    tag "$sample_id"
    publishDir "${params.outdir}/${tool}/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), path(reads)
    path(reference)
    val(tool)

    output:
    tuple val(sample_id), path("${sample_id}.bam"), path("${sample_id}.bam.bai")

    script:
    int sorting_threads = Math.min((task.cpus / 4) as int, 3)
    int mapping_threads = task.cpus - sorting_threads
    """
    bowtie2 -x rbp_sgrna \
            -U ${reads} \
            --end-to-end \
            -N 1 \
            -L 10 \
            -i S,1,0.50 \
            --mp 6,6 \
            --score-min L,-12,0  \
            -p ${mapping_threads} | \
    samtools sort -@ ${sorting_threads} -o ${sample_id}.bam
    samtools index ${sample_id}.bam 
    """
}