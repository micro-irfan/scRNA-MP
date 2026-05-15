#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional


def _winsorize_cols(matrix: np.ndarray, low: float, high: float) -> np.ndarray:
    if low == 0 and high == 0:
        return matrix

    lo_pct = low * 100.0
    hi_pct = 100.0 - high * 100.0

    lo = np.nanpercentile(matrix, lo_pct, axis=0)   # shape (n_cols,)
    hi = np.nanpercentile(matrix, hi_pct, axis=0)   # shape (n_cols,)

    return np.clip(matrix, lo[None, :], hi[None, :])


def _group_rows_by_gene(reference_list: List[str], sep: str = "-") -> Dict[str, List[int]]:
    groups = defaultdict(list)
    for i, ref in enumerate(reference_list):
        if sep in ref:
            gene, _ = ref.rsplit(sep, 1)
        else:
            gene = ref
        groups[gene].append(i)
    return groups


def _nan_iqr(x: np.ndarray) -> float:
    q75 = np.nanpercentile(x, 75)
    q25 = np.nanpercentile(x, 25)
    return q75 - q25


def _compute_scale_from_vector(vec: np.ndarray, use_99: bool) -> Optional[float]:
    """
    Given a 1D vector (may contain NaNs), compute the robust scale:
    - threshold = max(P95/P99, 1.5*IQR)
    - trim values >= threshold
    - scale = mean(values > P90 of trimmed)
    Returns None if scale cannot be computed.
    """
    if np.isnan(vec).all():
        return None

    iqr = _nan_iqr(vec)
    pct_val = np.nanpercentile(vec, 99.0 if use_99 else 95.0)
    thr = max(pct_val, 1.5 * iqr)

    keep = (vec < thr) 
    trimmed = vec[keep]
    if np.isfinite(trimmed).sum() == 0:
        return None

    p90 = np.nanpercentile(trimmed, 90.0)
    top = trimmed[trimmed > p90]
    scale = np.nanmean(top) if top.size else p90

    if not np.isfinite(scale) or scale == 0:
        return None

    return float(scale)


def winsorized_normalization_by_gene(
    matrix: np.ndarray,
    reference_list: List[str],
    cluster_map: Dict[str, List[int]],
    *,
    bottom_winsorize: float = 0.0,
    top_winsorize: float = 0.0,
    transcriptome_winsorize: bool = False,
    gene_winsorize: bool = False,
    sep: str = "-"
) -> np.ndarray:
    """
    Per-gene robust normalization, but using cluster membership to determine which
    columns (cells) to process together.

    cluster_mode:
      - "per_column": compute scale per column (same as your original), but iterate columns grouped by cluster.
      - "shared": compute one scale from all columns in the cluster (within the gene), apply to all columns in that cluster.

    Returns a (rows x cols) array of normalized values.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("matrix must be a 2D NumPy array")
    
    if len(reference_list) != matrix.shape[0]:
        raise ValueError("reference_list length must equal number of rows (matrix.shape[0])")
    
    # Optional global (transcriptome-wide) winsorization per column
    W = _winsorize_cols(matrix, bottom_winsorize, top_winsorize) if transcriptome_winsorize else matrix.copy()
    norm = np.full_like(W, np.nan)

    # Per-gene processing
    groups = _group_rows_by_gene(reference_list, sep=sep)
    ncols = W.shape[1]

    for _, row_idx in groups.items():
        rows = np.asarray(row_idx, dtype=int)
        G = W[rows, :] # (ng x ncols)

        # Optional gene-level winsorization (per column within this gene)
        if gene_winsorize: G = _winsorize_cols(G, bottom_winsorize, top_winsorize)

        g_out = np.empty_like(G)
        g_out[:] = np.nan

        # One scale per cluster (per gene), then apply to all columns in that cluster
        # Build scales per cluster from a concatenated view across the cluster's columns
        if cluster_map:
            for _, cols in cluster_map.items():
                cols = [c for c in cols if 0 <= c < ncols]
                if not cols: continue
                
                # Flatten the cluster's columns 
                vec = G[:, cols].ravel()
                use_99 = (len(cols) >= 500)  
                scale = _compute_scale_from_vector(vec, use_99)
                
                if scale is None:
                    # leave those columns unchanged
                    g_out[:, cols] = G[:, cols]
                else:
                    g_out[:, cols] = G[:, cols] / scale

        else:
            all_cols = list(range(ncols))

            # Flatten entire matrix
            vec = G[:, all_cols].ravel()
            use_99 = (ncols >= 500)
            scale = _compute_scale_from_vector(vec, use_99)

            if scale is None:
                g_out[:, all_cols] = G[:, all_cols]
            else:
                g_out[:, all_cols] = G[:, all_cols] / scale
            
        norm[rows, :] = g_out

    return norm


def winsorized_normalization(matrix, to_scale=True):
    '''
    This helps to limit the influence of very small (possibly outlier) values without completely removing them. 
    It's common in robust statistics when you want to reduce the effect of extreme values.
    '''
    p5  = np.nanpercentile(matrix, 5, axis=0)
    p95 = np.nanpercentile(matrix, 95, axis=0)

    winsorized_matrix = np.where(matrix < p5, p5, matrix)
    winsorized_matrix = np.where(winsorized_matrix > p95, p95, winsorized_matrix)

    if not to_scale:
        return winsorized_matrix

    # Normalize each column: (x - min) / (max - min)
    col_min = np.nanmin(winsorized_matrix, axis=0)
    col_max = np.nanmax(winsorized_matrix, axis=0)

    range_ = col_max - col_min
    range_[range_ == 0] = 1  # prevent div by zero
    normalized_matrix = (winsorized_matrix - col_min) / range_

    return normalized_matrix


def open_pseudobulk_file(cluster, filename):
    print(f"Opening {Path(filename).name}!")
    df = pd.read_csv(filename, index_col=0)

    if cluster not in df.columns:
        raise KeyError(f"Cluster '{cluster}' not found. Available: {list(df.columns)}")

    # Ensure numeric; coerce non-numeric to NaN if any
    ref_values = pd.to_numeric(df[cluster], errors="coerce").to_numpy(dtype=float)  # shape: (n_rows,)
    index_list = df.index.tolist()

    return ref_values, index_list


def filter_nan_rows(matrix_raw, reference_list):
    keep_mask = ~np.all(np.isnan(matrix_raw), axis=1)
    matrix_raw = matrix_raw[keep_mask]
    
    keep_indices = np.nonzero(keep_mask)[0]
    reference_list = [reference_list[i] for i in keep_indices]
    
    return matrix_raw, reference_list


def create_coverage_masks(
    sample_id,
    mut_matrix,
    cov_matrix,
    reference_list,
    min_cov,
    path_to_results,
    method='fixed',    
    ref_fill_value=np.nan,   # use np.nan if you want missing refs to become NaN
):
    """
    mut_matrix:      (n_rows, n_cols) float/num
    cov_matrix:      same shape as mut_matrix
    cluster_dict:    {'C0': [bc, ...], 'C1': [...], ...}
    barcode_index:   {barcode: column_index}
    reference_list:  row labels aligned to mut_matrix rows
    min_cov:         scalar threshold
    """

    # 1) Coverage-mask the mutation matrix
    mask = (cov_matrix >= min_cov)
    mut_masked = mut_matrix.astype(float, copy=True)  # allow NaNs
    mut_masked[~mask] = np.nan

    # Start result as masked matrix; we'll overwrite the columns we compute
    result = mut_masked.copy()

    sample_id = sample_id.replace('NAIN3', 'DMSO')
    
    path_to_pseudobulk = f"{path_to_results}/pseudobulk/{method}" 
    filename = f'{path_to_pseudobulk}/{sample_id}.pseudobulk.filtered.byWindows.allCells.csv' 
    ref_values, ref_labels = open_pseudobulk_file('All', filename)  

    lookup = dict(zip(ref_labels, ref_values))
    ref_vec = np.array([lookup.get(k, ref_fill_value) for k in reference_list], dtype=float)
    result = result - ref_vec[:, None]

    assert not np.array_equal(result, mut_masked, equal_nan=True)
    return result