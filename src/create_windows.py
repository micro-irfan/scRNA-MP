#!/usr/bin/env python3

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
from collections import OrderedDict
import os
import argparse
import create_utils as utils
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
from common import convert_to_df, create_reference, open_barcode_txt


# Custom file name and directory
log = utils.get_logger(
    name=__name__,
    log_dir="my_logs",        # folder (will be created if missing)
    log_file=f"make_matrix_{ts}.log",  # file name
    level="DEBUG"             # file min level (DEBUG/INFO/...)
)

def read_mpileup(file, reference):
    pileup = utils.create_pileup_dict_by_transrcipt(reference)
    gene_set = set()
    with open(file, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue

            fields = line.strip().split('\t')
            if len(fields) < 5:
                continue  # skip incomplete lines

            data = utils.Pileup(fields)
            data.read_pileup()
            
            gene = data.tx_id
            pos = data.pos
            pileup[gene][pos] = data
            gene_set.add(gene)
    
    tmp = {}
    for gene, gene_data in pileup.items():
        if gene not in gene_set: continue
        tmp[gene] = gene_data
    
    return tmp, gene_set


def write_output(gene_data, sample_id, barcode_id, filename):
    with open(filename, 'w') as write_file:
        write_file.write('sample_id,barcode_id,tx_id,pos,cov,mut,mutrate,bases\n')

        for gene, pos_data in gene_data.items():
            if not pos_data: continue

            for pos, d in pos_data.items():
                if d.mutrate < 0: continue
                if d.cov == 0: continue
                to_write = f'{sample_id},{barcode_id},{gene},{pos},{d.cov},{d.mut},{d.mutrate},{d.base_count}\n'
                write_file.write(to_write)


def _worker(method: str, barcode_id: str, sample_id: str, filepath: Path, reference: Dict, save_location: str) -> Tuple[str, Path]:
    """
    Run read_mpileup on one file and write a per-barcode pickle to tmp_outdir.
    Returns (barcode_id, out_path) on success; raises on failure.
    """
    gene_data, gene_set = read_mpileup(filepath, reference)

    reference = {gene:seq_len for gene,seq_len in reference.items() if gene in list(gene_set)}

    func = {
        'rolling' : utils.create_rolling_windows,
        'fixed' : utils.create_fixed_windows,
        'single_base' : utils.create_single_base,
    }

    gene_data = func[method](gene_data, reference)

    filename = f'{save_location}/{sample_id}-{barcode_id}.window.10.csv'
    write_output(gene_data, sample_id, barcode_id, filename)


def run_parallel(
    method: str,
    sample_id: str,
    files: List[str],
    barcode_to_keep: Set[str],
    reference: Dict, 
    tmp_location: str,
    save_location: str,
    max_workers: int = None,
) -> Dict[str, Any]:
    """
    Parallelizes _worker, which writes per-barcode pickle files.
    Returns: results dict mapping barcode_id -> loaded object.
    """
    tmp_base = Path(tmp_location)
    tmp_base.mkdir(parents=True, exist_ok=True)

    # Build work list (stable order)
    work: List[Tuple[str, Path]] = []
    for f in files:
        f_str = str(f)
        barcode_id = f_str.split("/")[-1].split(".")[1]
        if barcode_id not in barcode_to_keep:
            continue

        in_path = tmp_base / f_str
        work.append((barcode_id, in_path))

    log.info(f"{sample_id}: Dispatching {len(work)} files to {max_workers or 'auto'} workers")

    # Submit jobs; no return values, so keep a map to know which future corresponds to which barcode
    future_to_bid: Dict[Any, str] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for bid, in_path in work:
            fut = ex.submit(_worker, method, bid, sample_id, in_path, reference, save_location)
            future_to_bid[fut] = bid

        done_count = 0
        total = len(future_to_bid)
        for fut in as_completed(future_to_bid):
            bid = future_to_bid[fut]
            try:
                # result() returns None; we call it to surface exceptions if any
                fut.result()
                done_count += 1
                if done_count % 50 == 0 or done_count == total:
                    log.info(f"{sample_id}: Completed {done_count}/{total} (last: {bid})")
            except Exception as e:
                log.exception(f"{sample_id}: Worker failed for {bid}: {e}")


def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Create Matrix Table For Mutrate / Coverage / Mutation',
        description = ""
    )
    
    parser.add_argument('-s', '--sample_id', required = True,
                        help='Comma Separated List of Sample IDs')
    
    parser.add_argument('-t', '--threads', required = False, type=int, default=8,
                        help='Number of Workers')  

    parser.add_argument('-m', '--method', required = False, type=str, default='single_base',
                        help='Rolling Window, Fixed Window Or Single Base')      
    
    parser.add_argument('-w', '--workdir', required = True, type=str, default='',
                        help='Path to Pileup files generated from Samtools')
    
    parser.add_argument('-o', '--output_path', required = True, type=str, default='',
                        help='Path to store output') 
    
    parser.add_argument('-b', '--barcode', dest = "barcode", required = True,
                        help="barcode File From Umi Tools")
    
    parser.add_argument('-r', '--reference', required = True, type=str, default='',
                        help='Path to reference fasta file')
    
    parser.add_argument('--keep-tmp', action="store_false", dest='keep_tmp', 
						help='Keep Temporary files (Default: Does not Keep)')
    
    args = parser.parse_args()
    return args


def generate_pseudobulk(sample_id, mut_mat, cov_mat, window_label, path_to_pseudobulk):
    import create_pseudobulk as cp

    coverages = [10,20,50,100,200]
    pseudobulk = {}
    for coverage in coverages:
        print (f"Calculating Pseudobulk for {coverage} cov!")
        raw_matrix_filtered = cp.create_coverage_mask(mut_mat, cov_mat, min_cov=coverage)
        pseudobulk[coverage] = cp.generate_psuedobulk_by_cluster(raw_matrix_filtered)

    filename = f'{path_to_pseudobulk}/{sample_id}.pseudobulk.filtered.byWindows.allCells.csv'
    cp.write_pseudobulk(pseudobulk, filename, window_label, coverages=coverages)
    log.info (f'Psuedobulk Saved to {filename}')


def generate_matrices(sample_id, path_to_windows, path_to_matrices, method, reference):
    import create_matrix as mm
    
    coverage = 10
    log.info (f'Generating Matrices For {sample_id} at {coverage} coverage')

    files = [f for f in os.listdir(path_to_windows) if f.endswith('.csv') and sample_id in f] 
    log.info(f"Number of Files To Read: {len(files)}")
    files.sort()

    min_bases = 1 if method == 'single_base' else 6

    results = OrderedDict()
    for _, f in enumerate(files):
        bc_id = f.split('.')[0].split('-')[1]
        f = f'{path_to_windows}/{f}'
        results[bc_id] = mm.open_pileup(f, reference, min_count=min_bases)

    barcode_idx = mm.generate_barcode_idx(results)
    mutrate_matrix, cov_matrix, mut_matrix, row_labels = mm.generate_matrix(results, reference)        
    mutrate_matrix_filtered, cov_matrix_filtered, mut_matrix_filtered, row_labels_filtered = mm.filter_nan_rows(mutrate_matrix, 
                                                                                                                cov_matrix, 
                                                                                                                mut_matrix,
                                                                                                                row_labels)
    
    log.info (f'Mutation Rate Matrix Shape: {mutrate_matrix_filtered.shape}')
    log.info (f'Coverage Matrix Shape: {cov_matrix_filtered.shape}')
    log.info (f'Mutation Matrix Shape: {mut_matrix_filtered.shape}')

    location_to_save = f'{path_to_matrices}/{sample_id}'
    Path(location_to_save).mkdir(parents=True, exist_ok=True)

    cell_txt = "AllCells"
    filename = f"{location_to_save}/{sample_id}.mutrate.matrix{coverage}.{cell_txt}.csv"
    convert_to_df(mutrate_matrix_filtered, row_labels_filtered, barcode_idx, filename)
    log.info (f'MutRate Matrix Saved to {filename}')

    filename = f"{location_to_save}/{sample_id}.coverage.matrix{coverage}.{cell_txt}.csv"
    convert_to_df(cov_matrix_filtered, row_labels_filtered, barcode_idx, filename)
    log.info (f'Coverage Matrix Saved to {filename}')

    filename = f"{location_to_save}/{sample_id}.mutant.matrix{coverage}.{cell_txt}.csv"
    convert_to_df(mut_matrix_filtered, row_labels_filtered, barcode_idx, filename)
    log.info (f'Mutation Matrix Saved to {filename}')

    del results
 
    return mutrate_matrix_filtered, cov_matrix_filtered, row_labels_filtered, barcode_idx


def delete_files(temp_files_to_remove):
    for fname in temp_files_to_remove:
        os.remove(fname)      
    log.info("Files Generated From PileUp removed!")


def main():
    args = get_args()
    sample_id = args.sample_id
    method = args.method
    threads = args.threads
    barcode_file = args.barcode
    pileup_location = args.workdir

    # Define Path To Output
    path_to_results = args.output_path
    path_to_windows = f"{path_to_results}/make_windows/{method}"
    path_to_matrices = f"{path_to_results}/matrices/{method}"
    path_to_pseudobulk = f"{path_to_results}/pseudobulk/{method}"
    
    fasta_file = args.reference
    if not fasta_file:
        from common import fasta_file
        
    reference_list = create_reference(fasta_file)
    reference_seqlen = create_reference(fasta_file, keep_seq=True)

    tmp_location = f"{pileup_location}/{sample_id}"
    files = [f for f in os.listdir(tmp_location) if f.endswith('.pileup')]
    files.sort()
    log.info(f"Number of Files To Process: {len(files)}")

    location_to_save = f"{path_to_windows}/{sample_id}"
    Path(location_to_save).mkdir(parents=True, exist_ok=True)

    barcode_dict = open_barcode_txt(barcode_file)
    barcode_to_keep = set(barcode_dict.values())
    log.info(f"Number of Files To Process: {len(barcode_to_keep)}/{len(files)}")

    ## Create Windows 
    run_parallel(
        method=method,
        sample_id=sample_id,
        files=files,
        barcode_to_keep=barcode_to_keep,
        tmp_location=tmp_location,
        reference=reference_list,
        max_workers=threads,
        save_location=location_to_save,
    )

    ## Generate Matrices From Window Information
    mut_mat, cov_mat, window_label, _ = generate_matrices(sample_id, 
                                                            location_to_save, 
                                                            path_to_matrices, 
                                                            method, 
                                                            reference_seqlen)
    

    if args.keep_tmp:
        temp_files_to_remove = [
            str(f)
            for f in Path(location_to_save).glob(f"*{sample_id}*.csv")
        ]        
        delete_files(temp_files_to_remove)
    
    ## Generate Pseudobulk For DMSO Samples
    if 'DMSO' not in sample_id: 
        return
    
    Path(path_to_pseudobulk).mkdir(parents=True, exist_ok=True)
    generate_pseudobulk(sample_id, mut_mat, cov_mat, window_label, path_to_pseudobulk)


if __name__ == "__main__":  
    main()
