#!/usr/bin/env python3
"""
Script to run bowtie pipeline with different parameter combinations.
Results are saved to specific folders for each parameter combination.
"""

import subprocess
import itertools
from pathlib import Path
import sys
import os
import argparse

def modify_bowtie_pipeline(reference_index, input_fastq, output_prefix, threads=4, 
                          bowtie_params="--very-sensitive-local --mp 3,1 --rdg 5,1 --rfg 5,1"):
    """
    Simple step-by-step bowtie2 alignment pipeline.
    """
    
    # Output files
    bam_file = f"{output_prefix}.bam"
    temp_bam = f"{output_prefix}.temp.bam"
    unmapped_file = f"{output_prefix}.unmapped.fq.gz"
    log_file = f"{output_prefix}.log"
    stats_file = f"{output_prefix}.stats"
    flagstat_file = f"{output_prefix}.flagstat"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Step 1: Bowtie2 alignment -> BAM
    print("Step 1: Running alignment...")
    cmd1 = f"bowtie2 {bowtie_params} -p {threads} -x {reference_index} -U {input_fastq} --un-gz {unmapped_file} 2> {log_file} | samtools view -bS > {temp_bam}"
    subprocess.run(cmd1, shell=True, check=True)
    
    # Step 2: Sort BAM
    print("Step 2: Sorting BAM...")
    subprocess.run(['samtools', 'sort', '-@', str(threads), '-o', bam_file, temp_bam], check=True)
    
    # Step 3: Generate stats
    print("Step 3: Generating statistics...")
    subprocess.run(f'samtools flagstat {bam_file} > {flagstat_file}', shell=True, check=True)
    subprocess.run(f'samtools stats {bam_file} > {stats_file}', shell=True, check=True)
    
    # Step 4: Index BAM
    print("Step 4: Indexing BAM...")
    subprocess.run(['samtools', 'index', bam_file], check=True)
    
    # Clean up
    os.remove(temp_bam)
    print("Done!")
    
    return {
        'bam': bam_file,
        'stats': stats_file,
        'flagstat': flagstat_file,
        'log': log_file,
        'unmapped': unmapped_file
    }

def run_full_pipeline_with_params(reference_index, input_fastq, base_output_dir, 
                                 bowtie_params, threads=4):
    """Run the complete pipeline with specified parameters.
    
    Args:
        bowtie_params: String containing all bowtie2 parameters (e.g., "--very-sensitive-local --mp 3,1 --rdg 5,1 --rfg 5,1")
    """
    
    # Create parameter-specific output directory name from the parameter string
    param_name = bowtie_params.replace('--', '').replace(' ', '_').replace(',', '-')
    output_dir = Path(f'{base_output_dir}/bowtie_{param_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set output prefix
    input_basename = Path(input_fastq).stem.replace('.fq', '').replace('.fastq', '')
    output_prefix = output_dir / input_basename
    
    print(f"\n{'='*60}")
    print("Running pipeline with parameters:")
    print(f"  Parameters: {bowtie_params}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")
    
    try:
        # Run bowtie2 alignment
        output_files = modify_bowtie_pipeline(
            reference_index, input_fastq, str(output_prefix), threads,
            bowtie_params
        )
        
        # Import and run analysis functions from original pipeline
        sys.path.append(str(Path(__file__).parent))
        from bowtie_pipeline import parse_raw_lengths, calculate_mapped_lengths, plot_distributions, save_statistics_to_csv
        
        # Calculate lengths
        print("Calculating mapped read lengths...")
        raw_lengths, raw_counts = parse_raw_lengths(output_files['stats'])
        mapped_lengths = calculate_mapped_lengths(output_files['bam'])
        
        # Generate plot
        print("Generating plot...")
        plot_file = f"{output_prefix}_length_distributions.png"
        plot_distributions(raw_lengths, raw_counts, mapped_lengths, plot_file)
        
        # Save statistics
        print("Saving statistics to CSV...")
        save_statistics_to_csv(raw_lengths, raw_counts, mapped_lengths, str(output_prefix))
        
        print(f"Pipeline completed successfully for {param_name}")
        return True
        
    except Exception as e:
        print(f"ERROR: Pipeline failed for {param_name}: {str(e)}")
        return False

def main():
    """Main function to run bowtie pipeline with specified parameters."""
    parser = argparse.ArgumentParser(description='Run bowtie pipeline with specified parameters.')
    parser.add_argument('--reference_index', type=str, required=False, help='Path to the bowtie reference index')
    parser.add_argument('--input_fastq', type=str, required=False, help='Path to the input FASTQ file')
    parser.add_argument('--base_output_dir', type=str, required=False, help='Base directory for output files')
    parser.add_argument('--bowtie_params', type=str, required=False, help='Additional bowtie parameters as a single string')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads to use')
    args = parser.parse_args()

    run_full_pipeline_with_params(
        args.reference_index,
        args.input_fastq,
        args.base_output_dir,
        args.bowtie_params,
        args.threads
    )

if __name__ == "__main__":
    main()