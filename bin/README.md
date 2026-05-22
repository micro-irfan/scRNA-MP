# bin

Low-level preprocessing helpers used by the pipelines in `workflows/` and by scripts in `src/`.

## What is here
- `run_bowtie2.py`: run Bowtie2 + samtools sort/index/stats for one FASTQ.
- `bowtie_pipeline.py`: plotting/statistics helpers used by `run_bowtie2.py`.
- `analyze_bam_reads.py`: memory-efficient read-assignment analysis from name-sorted BAM; can extract unique reads.
- `add_genetag.sh`: append `XT:Z:<gene>` tag from BAM reference column.
- `extract_add_barcode.py`: add `CB`/`UM` tags from read name, extract barcode lists, validate tags.
- `read_name_to_sequence.py`: prepend barcode+UMI sequence to FASTQ read sequence.
- `count_reads_per_cell_gene.py`: create per-cell/per-gene read counts from tagged BAM.
- `pipseq_check_barcode.py`: validate 28bp barcodes against tier whitelists.
- `add_readgroup.py`: add read-group metadata per barcode and split barcode text chunks.

## Common assumptions
- Read names follow the project format `..._<BARCODE>_<UMI>`.
- `samtools` and `pysam` are available for BAM-aware scripts.
- Some scripts expect BAM sorted by read name (not coordinate-sorted).

## Typical commands
```bash
# 1) Run alignment pipeline
python3 bin/run_bowtie2.py \
  --reference_index /path/to/index_prefix \
  --input_fastq sample.fastq.gz \
  --base_output_dir /path/to/output \
  --bowtie_params "--sensitive-local --mp 6,2 --rdg 5,3 --rfg 5,3" \
  --threads 8

# 2) Add CB/UM tags from read names
python3 bin/extract_add_barcode.py add_tags input.bam tagged.bam

# 3) Count reads per cell and gene
python3 bin/count_reads_per_cell_gene.py -i tagged.bam -o counts.tsv
```

## Notes
- Most scripts can be run directly, but in this repository they are usually called from workflow files.
- If outputs look empty, first verify read-name format and barcode parsing assumptions.
