#!/usr/bin/env python3

import common as utils
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse

def fixed_window_mean_by_gene_first_position(
    reactivity: np.ndarray,
    coverage: np.ndarray,
    reference_list: list,
    *,
    window_size: int = 10,
    min_valid: int = 6,
    sep: str = "-",
    sort_within_gene: bool = True,
):
    """
    Fixed (non-overlapping) window mean per gene for (n_rows x n_cells) matrices.

    Windows never cross genes. Missing positions are allowed; windows are formed
    over the ordered *available rows* within each gene.

    Threshold is per-window-per-cell: require >= min_valid finite reactivity values.

    Returns
    -------
    r_win : (n_windows_total, n_cells)
    c_win : (n_windows_total, n_cells)
    win_ref : list[str]  e.g. "GENE-100..109" (based on positions in that window)
    """

    assert reactivity.shape == coverage.shape
    n_rows, n_cells = reactivity.shape
    assert len(reference_list) == n_rows

    # 1) Group row indices by gene, store (pos, row_idx)
    by_gene = defaultdict(list)
    for i, ref in enumerate(reference_list):
        gene, pos_str = ref.rsplit(sep, 1)   # rsplit handles gene names that might contain '-'
        pos = int(pos_str)
        by_gene[gene].append((pos, i))

    r_out = []
    c_out = []
    win_ref = []

    # 2) Process each gene separately
    for gene, pos_idx in by_gene.items():
        if sort_within_gene:
            pos_idx.sort(key=lambda x: x[0])  # sort by position

        positions = np.array([p for p, _ in pos_idx], dtype=int)
        idxs      = np.array([j for _, j in pos_idx], dtype=int)

        rg = reactivity[idxs, :]  # (n_gene_rows, n_cells)
        cg = coverage[idxs, :]

        n_gene_rows = rg.shape[0]
        n_win = n_gene_rows // window_size
        if n_win == 0:
            continue  # not enough rows for one window in this gene

        # Trim to exact multiple
        trim = n_win * window_size
        rg = rg[:trim, :]
        cg = cg[:trim, :]
        pos_trim = positions[:trim]

        # Reshape to (n_win, window_size, n_cells)
        rw = rg.reshape(n_win, window_size, n_cells)
        cw = cg.reshape(n_win, window_size, n_cells)

        valid_counts = np.sum(np.isfinite(rw), axis=1)  # (n_win, n_cells)

        rm = np.nanmean(rw, axis=1)
        cm = np.nanmean(cw, axis=1)

        ok = valid_counts >= min_valid
        rm[~ok] = np.nan
        cm[~ok] = np.nan

        # window reference labels: gene-start..end (based on positions in that window)
        pos_w = pos_trim.reshape(n_win, window_size)
        for w in range(n_win):  
            win_ref.append(f"{gene}{sep}{pos_w[w,0]}")

        r_out.append(rm)
        c_out.append(cm)

    if not r_out:
        # No windows created
        return (
            np.empty((0, n_cells), dtype=float),
            np.empty((0, n_cells), dtype=float),
            []
        )

    return np.vstack(r_out), np.vstack(c_out), win_ref


def fixed_window_mean_by_gene_absolute(
    reactivity: np.ndarray,
    coverage: np.ndarray,
    reference_list: list,
    *,
    window_size: int = 10,
    min_valid: int = 6,
    sep: str = "-",
    sort_within_gene: bool = True,
    label_as_range: bool = True,     # "GENE-10..19" vs "GENE-10"
    include_empty_bins: bool = False, # if True, output bins even when no rows exist
    max_bin_strategy: str = "max_pos",# "max_pos" or "max_bin_seen"
):
    """
    Absolute-position fixed-grid window mean per gene.

    Binning:
      bin_start = floor(pos / window_size) * window_size
    so windows are anchored to absolute coordinates: 0,10,20,...

    Missing positions are fine.

    If include_empty_bins=True, outputs every bin from 0..last_bin for each gene,
    filling bins with no rows as all-NaN vectors (and counts=0 -> fail min_valid).
    """

    assert reactivity.shape == coverage.shape
    n_rows, n_cells = reactivity.shape
    assert len(reference_list) == n_rows

    by_gene = defaultdict(list)
    for i, ref in enumerate(reference_list):
        gene, pos_str = ref.rsplit(sep, 1)
        pos = int(pos_str)
        by_gene[gene].append((pos, i))

    r_out, c_out, win_ref = [], [], []

    for gene, pos_idx in by_gene.items():
        if sort_within_gene:
            pos_idx.sort(key=lambda x: x[0])

        positions = np.array([p for p, _ in pos_idx], dtype=int)
        idxs      = np.array([j for _, j in pos_idx], dtype=int)

        # ABSOLUTE bins: 0, 10, 20, ...
        bin_start = (positions // window_size) * window_size

        bins = defaultdict(list)
        for b, row_idx in zip(bin_start, idxs):
            bins[int(b)].append(int(row_idx))

        if include_empty_bins:
            if max_bin_strategy == "max_pos":
                last_bin = int((positions.max() // window_size) * window_size)
            elif max_bin_strategy == "max_bin_seen":
                last_bin = max(bins.keys())
            else:
                raise ValueError("max_bin_strategy must be 'max_pos' or 'max_bin_seen'")

            all_bins = range(0, last_bin + window_size, window_size)
        else:
            all_bins = sorted(bins.keys())

        for b in all_bins:
            rows = bins.get(int(b), [])
            if len(rows) == 0:
                # Empty bin: emit NaNs (keeps alignment across datasets)
                rm = np.full((n_cells,), np.nan, dtype=float)
                cm = np.full((n_cells,), np.nan, dtype=float)
            else:
                rows = np.array(rows, dtype=int)
                rw = reactivity[rows, :]
                cw = coverage[rows, :]

                valid_counts = np.sum(np.isfinite(rw), axis=0)  # per-cell
                rm = np.nanmean(rw, axis=0)
                cm = np.nanmean(cw, axis=0)

                ok = valid_counts >= min_valid
                rm[~ok] = np.nan
                cm[~ok] = np.nan

            label_window_size = min(positions.max(), b + window_size - 1)
            if label_as_range:
                win_ref.append(f"{gene}{sep}{b}:{label_window_size}")
            else:
                win_ref.append(f"{gene}{sep}{b}")

            r_out.append(rm)
            c_out.append(cm)

    if not r_out:
        return (
            np.empty((0, n_cells), dtype=float),
            np.empty((0, n_cells), dtype=float),
            []
        )

    return np.vstack(r_out), np.vstack(c_out), win_ref


def get_indices(reference_list, win_ref):
    ref_to_idx = {ref: i for i, ref in enumerate(reference_list)}
    indices = [ref_to_idx[ref] for ref in win_ref]
    return indices

def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Convert Single Base to Fixed Windows (10 Bases)',
        description = ""
    )

    parser.add_argument('-s', '--sample_id', required = True,
                        help='Sample ID for current File')
    
    parser.add_argument('--gene_first_position', action="store_true",
						help="Uses gene's first position to create windows (Default: Uses absolute 0-based as starting point)")
    
    parser.add_argument('--window_size', required = False, type=int, default=10,
                        help='Size of Windows (Default: 10)')
    
    parser.add_argument('--min_valid', required = False, type=int, default=6,
                        help='Number of bases required in a window (Default: 6)') 
    
    parser.add_argument('-w', '--work_path', required = False, type=str, default='',
                        help='Path to store output') 
    
    parser.add_argument('-c', '--coverage', required = False, default = '50,100', type=str,
                        help='Coverage to Analyze, e.g., 10,20 - comma separated (Default: 50,100)')  
    
    return parser.parse_args()


def main():
    args = get_args()
    sample_id = args.sample_id
    cell_txt = "AllCells"
    
    path_to_results = args.work_path
    if not args.gene_first_position:
        func = fixed_window_mean_by_gene_absolute
        txt = 'absolute'
    else:
        func = fixed_window_mean_by_gene_first_position
        txt = 'firstPos'
    
    output_location = f'{path_to_results}/single_base_to_fixed_{txt}'
    Path(output_location).mkdir(parents=True, exist_ok=True)

    coverage_to_analyze = [int(c) for c in args.coverage.split(',')]

    print(f'Analyzing {sample_id}')
    matrix_location = f'{path_to_results}/matrices/single_base/{sample_id}'
    filename = f"{matrix_location}/{sample_id}.coverage.matrix10.{cell_txt}.csv"
    cov_barcode_list, cov_reference_list, matrix_cov = utils.open_file(filename)

    for coverage in coverage_to_analyze:
        normalized_location = f'{path_to_results}/normalized_mtx/single_base/{sample_id}'
        filename = f"{normalized_location}/{sample_id}.normalized_reactivity.matrix{coverage}.{cell_txt}.gene_level.csv" 
        rx_barcode_list, rx_reference_list, matrix_rx = utils.open_file(filename)

        ## Assert that the barcodes are the same
        assert (all([x == y for x,y in zip(cov_barcode_list, rx_barcode_list)])), [(x,y) for x,y in zip(cov_barcode_list, rx_barcode_list)]

        indices_to_keep = get_indices(cov_reference_list, rx_reference_list)
        matrix_cov_filtered = matrix_cov[indices_to_keep, :]

        r_win, c_win, win_ref = func(
            reactivity=matrix_rx,
            coverage=matrix_cov_filtered,
            reference_list=rx_reference_list,
            window_size=args.window_size,
            min_valid=args.min_valid,
            sep='-',
            sort_within_gene=True,
        )   

        filename = f"{output_location}/{sample_id}.fixed_single_base.matrix{coverage}.{cell_txt}.csv"
        utils.convert_to_df(r_win, win_ref, rx_barcode_list, filename)

        filename = f"{output_location}/{sample_id}.coverage.matrix{coverage}.{cell_txt}.csv"
        utils.convert_to_df(c_win, win_ref, rx_barcode_list, filename)


if __name__ == "__main__":
    main()