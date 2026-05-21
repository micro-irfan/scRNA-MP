#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Default parameters
params.samplesheet = "$projectDir/samplesheet.csv"  // Add this line
params.outdir = "$projectDir/results"
params.threads = Runtime.runtime.availableProcessors()
params.help = false

params.workDir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/raw_sgRNA"
params.directory = "${params.workDir}/data"
params.reference = "${params.workDir}/ref"
params.cubatrie = "/home/users/astar/gis/muhdih/scratch/sgRNA_expression/CubaTrie/cubaTrie"

// Print help message
if (params.help) {
    log.info"""
    ===================================================
    BWA vs cubaTrie Pipeline
    ===================================================
    
    Usage:
    nextflow run main.nf [options]
    
    Options:
      --samplesheet     CSV file containing sample information (default: $params.samplesheet)
      --outdir          Output directory (default: $params.outdir)
      --batchName       Batch name for processing (default: $params.batchName)
      --help            Show this message
    """
    exit 0
}

// Create channel for input reads 
Channel
    .fromPath(params.samplesheet)
    .splitCsv(header: true)
    .map { row ->
        def sample_id = row.sample_id
        def dir = params.directory
        tuple(
            sample_id,
            file("${dir}/${sample_id}/mapping/bowtie_sensitive-local_mp_6-2_rdg_5-3_rfg_5-3_-k_4/${sample_id}_R2_fastp.unmapped.fq.gz"),
        )
    }
    .set { fastp_reads }


include { CUTADAPT_FLEX as CUTADAPT_FASTP } from './bin/mapper'
include { CUBATRIE_BAM as CUBATRIE_BAM_NORMAL } from './bin/mapper'
include { CUBATRIE_BAM as CUBATRIE_BAM_CA } from './bin/mapper'
include { CUBATRIE_BAM_ANCHORS } from './bin/mapper'
include { BOWTIE2_ALIGN_SINGLE_READ } from './bin/mapper'
include { COUNT_FILTERED_BARCODES } from './bin/mapper'
include { BOWTIE2_BUILD } from './bin/mapper'

workflow {
    ref_short = "${params.reference}/target_guides_rbp.fa"

    flank5 = "GGAAAGGACGAAACACCG"
    flank3 = "GTTTAAGAGCTATGCTGG"

    // CUBATRIE_BAM_NORMAL(fastp_reads, ref_short, 8, "cubaTrie")
    // CUBATRIE_BAM_ANCHORS(fastp_reads, ref_short, 8, "cubaTrie_anchors", flank5, flank3)

    ref_long = "${params.reference}/target_guides_rbpWithScaffold.fa"
    long_index_bwt = BOWTIE2_BUILD(ref_long)
    CUTADAPT_FASTP(fastp_reads, flank5, flank3)
    mapped_bam_bwt2 = BOWTIE2_ALIGN_SINGLE_READ(CUTADAPT_FASTP.out.trimmed_reads, long_index_bwt, 'bowtie2')
    COUNT_FILTERED_BARCODES(mapped_bam_bwt2, 'bowtie2')

    // CUBATRIE_BAM_CA(CUTADAPT_FASTP.out.trimmed_reads, ref_short, 8, "cubaTrie_CA")
}
