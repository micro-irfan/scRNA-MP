#!/usr/bin/env python3
"""
Barcode Whitelist Checker (Modified for whitelist file input)

This script checks barcodes against predefined whitelists organized in 4 tiers.
Each 28bp barcode is split into 4 parts and validated against corresponding tier whitelists.

Modified to read barcodes from a whitelist file (first column) instead of individual files.

Usage:
    python pipseq_check_barcode.py <whitelist_file>
    
Example:
    python pipseq_check_barcode.py PIP_2cell_DMSO_1_40_whitelist.txt
    python pipseq_check_barcode.py PIP_2cell_DMSO_1_40_whitelist.txt -o valid_barcodes.txt

Input format:
    Tab-delimited file with barcodes in the first column (28bp each)
    Format: BARCODE\tVARIANTS\tTOTAL_COUNT\tVARIANT_COUNTS

Output:
    - Validation results for each barcode
    - Summary statistics
    - List of valid barcodes (only those that pass the check)
"""

import argparse
import sys

def check_barcodes(input_file, output_file=None):
    """
    Check barcodes against tier whitelists.
    
    Args:
        input_file: Path to whitelist file with barcodes in first column
        output_file: Optional path to save valid barcodes
    """
    # Define tier whitelists
    tier1 = set(['AGAAACCA', 'CCTTTACA', 'AACTGCCT', 'AAGAGTAT', 'AAACTACA', 'ATTACCTT', 'AAGCCTTC', 'CTCCTCCA', 'TCTAAACT', 'AAGTCCAA', 'TCCGACAC', 'GAGAAACC', 'AAGCTCCT', 'AGACCTCA', 'GATTACTT', 'CCACCTCT', 'TAACTTCT', 'ACTCATAC', 'CTGTTTCC', 'TCCTATAT', 'CACTAACC', 'AGCTCCAC', 'AGTAGTTA', 'TCTATTCC', 'GTGTCACC', 'AGGACACA', 'CTTTGGAC', 'CCTATTTA', 'TAGTCTCT', 'CTTTCACT', 'AGTGCTTC', 'ATACTCTC', 'CAATTCTC', 'ATTTCCAT', 'CAAGGGTT', 'GTCTTCCT', 'CTGGGTAT', 'CAAACATT', 'CAGGTTGC', 'GTCCTTGC', 'GATTGGGA', 'TTGGGTCC', 'ATAAGCTA', 'GAGGGTCA', 'AGAGGTGC', 'CTGTGACC', 'GTCCACTA', 'CTTAGTGT', 'GAGTGTAC', 'GTTGTCCG', 'TCTTTGAC', 'CCTTTGTC', 'GTGAACTC', 'AAGGGACC', 'AATACATC', 'AAACAAAC', 'CTGTTAAA', 'GATGTGGC', 'GCTTCTCC', 'GTTCTGCT', 'AAAGAGGC', 'AAGTTGTC', 'TAGCCACA', 'CTTTATCC', 'GTAAACAA', 'CTTCTACG', 'CCATCCAC', 'CCTCATGA', 'CTAGACTA', 'ACCAGTTT', 'AGTTGAAC', 'AGTTTGTA', 'TGCTTCAT', 'GAGGAGTG', 'TTAGCTGC', 'AAATTCCG', 'GAAATACG', 'AGTCACAA', 'TACTGAAT', 'AGACGAGG', 'CCTACGCT', 'AAACCGCC', 'AATATGAC', 'GACACCTG', 'CTGTTGTG', 'CTAACGCC', 'TTCACTGG', 'GTCTAATC', 'TATGTGAA', 'GTGAGGCA', 'TATCTGTC', 'GTGGTGCT', 'TTGCTCTA', 'TTTGTACA', 'TATCCACG', 'GCCTGGTA'])
    print("Tier 1 barcodes loaded. There are", len(tier1), "barcodes.")
    
    tier2 = set(['AGGAAA', 'AGGTAA', 'AGTGGA', 'ATGTTG', 'GGTTTC', 'GTAGAG', 'GTTAGT', 'GTTTGG', 'TAGCGA', 'TATTGG', 'TGGGTT', 'TTGGTA', 'AAGAGA', 'AAAGTG', 'TAAGGC', 'AAAGGC', 'AATAGC', 'TAAGCC', 'TATGCC', 'GTTGCT', 'CAGTTG', 'GAAAGG', 'CACAAG', 'TACAGA', 'GAAGAA', 'ACAAAG', 'AGAAGG', 'GCGTTT', 'TGAAAG', 'TGAGAA', 'GATGAA', 'CACGAA', 'ACGGTT', 'GATTTC', 'TAGTCT', 'TTTCTC', 'CGCAAA', 'CATCTA', 'GGTCTA', 'AAGGTG', 'AAAGAC', 'AGAAAC', 'TTAACG', 'TGAACC', 'AGTTAC', 'AAACCG', 'TAACCC', 'GCACTA', 'AGACGT', 'ACATGT', 'ATCAAC', 'TTCGAA', 'GACAAT', 'TGCTTT', 'TGCTAG', 'GTGATC', 'GATATG', 'GTAATC', 'GAAATC', 'GCTGTA', 'TGTATC', 'CTGAAG', 'ATGCAC', 'GTACAA', 'AAACAC', 'AAGCAC', 'GAACAG', 'GTTCAC', 'ACCTTT', 'AACTGA', 'CCGTAT', 'ATCTGA', 'TCAAAG', 'TCGATT', 'GCTAAG', 'GAGATA', 'CTGGTA', 'CTTGTT', 'CTTTAG', 'CTTCGA', 'CTAAAG', 'CTATGG', 'TCTGTG', 'CACATT', 'TCAGTC', 'CCAAAT', 'AATACC', 'ACTTCC', 'ACAACC', 'CCTAAT', 'ACAGCA', 'CTTGAC', 'CAATAC', 'GCTCTT', 'TTGGCA', 'TCTACC'])
    print("Tier 2 barcodes loaded. There are", len(tier2), "barcodes.")
    
    tier3 = set(['AAGGTG', 'AGGAAA', 'AGGTAA', 'AGTGGA', 'ATGCAC', 'ATGTTG', 'GGTTTC', 'GTAATC', 'GTACAA', 'GTAGAG', 'GTGATC', 'GTTAGT', 'GTTTGG', 'TAGAAC', 'TAGCGA', 'TATTGG', 'TGGGTT', 'TGTAAC', 'TGTATC', 'TTAACG', 'TTGGCA', 'TTGGTA', 'AAGAGA', 'AAAGTG', 'TAAGGC', 'AAACCG', 'AAACAC', 'AAAGAC', 'AAAGGC', 'AATAGC', 'AAGCAC', 'AGTTAC', 'TAAGCC', 'TATGCC', 'CTGAAG', 'CTGGTA', 'GTTGCT', 'GCTGTA', 'CAGTTG', 'CTTGAC', 'CTTGTT', 'CTTTAG', 'CTAAAG', 'CTATGG', 'GCTAAG', 'TAACCC', 'AATACC', 'ACAGCA', 'ACATGT', 'ACTTCC', 'TCAGTC', 'CAAACA', 'CAATAC', 'TGCTTT', 'TGCTAG', 'TCTGTG', 'CTTCCA', 'CCTAAT', 'AACTGA', 'GAAAGG', 'GAACAG', 'CACAAG', 'CACATT', 'TACAGA', 'ACCATA', 'TCCATA', 'GTTCAC', 'TCTACC', 'TTCCAG', 'GAAGAA', 'ACAAAG', 'ACAACC', 'CCAAAT', 'GACAAT', 'GCACTA', 'ATCAAC', 'GAAACC', 'GAAATC', 'GATATG', 'AGAAAC', 'AGAAGG', 'AGACGT', 'GATACC', 'TCAAAG', 'ACCTTT', 'AAGCTC', 'GCGTTT', 'CTTCGA', 'ATCTGA', 'TGAAAG', 'TGAACC', 'TGAGAA', 'ATCTTC', 'AAACTC', 'TACTCA', 'CACGAA'])
    print("Tier 3 barcodes loaded. There are", len(tier3), "barcodes.")
    
    tier4 = set(['CCTATTTA', 'CAGTTTAA', 'TCCTATAT', 'TTAGCAAT', 'GCCAACAT', 'ACCAGTTT', 'AGTAGTTA', 'CCTTTACA', 'ACAAGTAG', 'ACACCAAG', 'GGCTATAA', 'ATGCATAT', 'TTGCATTC', 'AATAAGGA', 'TTCTAGGA', 'AGTAATGG', 'ACTAATTG', 'ATCAGGGA', 'GACACAAA', 'CCATATGA', 'ATATGCAA', 'GCTAAGTT', 'TGCACCAG', 'CAAACATT', 'AAACTACA', 'ATCCTAGT', 'TACCTAAG', 'TGGCTAGT', 'CACAACCT', 'ATAACAGG', 'GGCAAGGT', 'GGTTAGGG', 'GTCAGGTT', 'CTCAAACA', 'AATATGAC', 'AACAGAAC', 'ACCACAGA', 'ACTAGAGC', 'GATGCAGA', 'GTCAAGAG', 'TCCAGAAG', 'GGTTACAC', 'AAACTGTG', 'ACAGGCCA', 'CAAGGAAT', 'CAAGGTAC', 'GCTATGGG', 'AACAAATG', 'CCTTTGTC', 'TTTGCCAG', 'GCCTGGTA', 'GTCTGGAA', 'TCTGATTT', 'ACAAAGAT', 'AACTGCCT', 'CTCACATC', 'GTCTAATC', 'AAATAGCA', 'AATGTATG', 'CGTGTACA', 'GATGTGGC', 'TATGTGAA', 'CGTGGGAT', 'GATGGTTA', 'AAAGAGGC', 'ACAGATAA', 'ATAGATGT', 'CCAGACAG', 'GAAGATAT', 'TGGGAATT', 'AGTTTGTA', 'GGAGGTTT', 'GGAGTAAG', 'ACCTGAAG', 'TACTGAAT', 'CAAGGGTT', 'CTGTTAAA', 'GAGTGTAC', 'TAATGTGG', 'CTGGGTAT', 'ACATGGAC', 'ATATGGGT', 'GAAAGACA', 'CACAAGTA', 'CGTGTGTT', 'TGAATAGG', 'GGGTCATT', 'TCATACCA', 'TCAAATGG', 'GGGAGATG', 'GGGATTAC', 'TGGAAAGC', 'TGGTGTCT', 'GTGGTGCT', 'GTTCTGCT', 'AAGGATGA'])
    print("Tier 4 barcodes loaded. There are", len(tier4), "barcodes.")
    print("="*60)

    # Statistics counters
    total_barcodes = 0
    valid_lines_list = []  # Store entire lines, not just barcodes
    invalid_length = 0
    tier1_fail = 0
    tier2_fail = 0
    tier3_fail = 0
    tier4_fail = 0

    try:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                original_line = line.rstrip('\n\r')  # Preserve original line without trailing newlines
                line_stripped = line.strip()
                
                if not line_stripped:  # Skip empty lines
                    continue
                
                # Extract barcode from first column (tab-delimited)
                fields = line_stripped.split('\t')
                if not fields:
                    continue
                
                barcode = fields[0].strip()
                if not barcode:
                    continue
                    
                total_barcodes += 1
                
                if len(barcode) != 28:
                    print(f"Line {line_num}: Invalid length {len(barcode)} (expected 28) - {barcode}")
                    invalid_length += 1
                    continue
                
                # Split barcode into 4 parts: 8bp + 6bp + 6bp + 8bp
                part1 = barcode[0:8]    # First 8bp
                part2 = barcode[8:14]   # Next 6bp
                part3 = barcode[14:20]  # Next 6bp
                part4 = barcode[20:28]  # Last 8bp
                
                # Check each part against corresponding tier
                in_tier1 = part1 in tier1
                in_tier2 = part2 in tier2
                in_tier3 = part3 in tier3
                in_tier4 = part4 in tier4
                
                # Count failures
                if not in_tier1:
                    tier1_fail += 1
                if not in_tier2:
                    tier2_fail += 1
                if not in_tier3:
                    tier3_fail += 1
                if not in_tier4:
                    tier4_fail += 1
                
                # Check if all parts match (barcode is valid)
                if in_tier1 and in_tier2 and in_tier3 and in_tier4:
                    print(f"Line {line_num}: VALID - {barcode}")
                    valid_lines_list.append(original_line)  # Store entire original line
                # Note: Invalid barcodes are NOT printed (filtered out)
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total barcodes:        {total_barcodes}")
    print(f"Valid barcodes:        {len(valid_lines_list)} ({100*len(valid_lines_list)/total_barcodes:.2f}%)" if total_barcodes > 0 else "Valid barcodes:        0")
    print(f"Invalid barcodes:      {total_barcodes - len(valid_lines_list) - invalid_length} ({100*(total_barcodes - len(valid_lines_list) - invalid_length)/total_barcodes:.2f}%)" if total_barcodes > 0 else "Invalid barcodes:      0")
    print(f"Invalid length:        {invalid_length}")
    
    if total_barcodes > 0:
        print(f"\nTier failures:")
        print(f"  Tier 1 (pos 1-8):    {tier1_fail} ({100*tier1_fail/total_barcodes:.2f}%)")
        print(f"  Tier 2 (pos 9-14):   {tier2_fail} ({100*tier2_fail/total_barcodes:.2f}%)")
        print(f"  Tier 3 (pos 15-20):  {tier3_fail} ({100*tier3_fail/total_barcodes:.2f}%)")
        print(f"  Tier 4 (pos 21-28):  {tier4_fail} ({100*tier4_fail/total_barcodes:.2f}%)")
    print("="*60)
    
    # Output valid lines (only those that passed)
    if valid_lines_list:
        print(f"\n{'='*60}")
        print(f"VALID LINES ({len(valid_lines_list)} total)")
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    for line in valid_lines_list:
                        f.write(line + '\n')
                print(f"\nValid lines (with all columns) saved to: {output_file}")
            except Exception as e:
                print(f"Error writing to output file: {e}")
    else:
        print("\nNo valid barcodes found.")

def main():
    """Main function to parse arguments and call the barcode checking function."""
    parser = argparse.ArgumentParser(
        description="Check barcodes against tier whitelists. Each 28bp barcode is validated against 4 tier whitelists. "
                    "Reads barcodes from the first column of a tab-delimited whitelist file.",
        epilog="Example: python pipseq_check_barcode.py PIP_2cell_DMSO_1_40_whitelist.txt -o valid_barcodes.txt"
    )
    parser.add_argument("input_file", help="Input whitelist file with barcodes in first column (tab-delimited, 28bp each)")
    parser.add_argument("-o", "--output", help="Output file to save valid barcodes", default=None)
    
    args = parser.parse_args()
    check_barcodes(args.input_file, args.output)


if __name__ == "__main__":
    main()