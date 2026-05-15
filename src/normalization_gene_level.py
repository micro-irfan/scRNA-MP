#!/usr/bin/env python3

from collections import defaultdict
import numpy as np
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

            vec = G[:, all_cols].ravel()
            use_99 = (ncols >= 500)
            scale = _compute_scale_from_vector(vec, use_99)

            if scale is None:
                g_out[:, all_cols] = G[:, all_cols]
            else:
                g_out[:, all_cols] = G[:, all_cols] / scale
            
        norm[rows, :] = g_out

    return norm