#!/usr/bin/env python3

import numpy as np
import pandas as pd
from collections import defaultdict
import re
from common import convert_to_df
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy.stats import beta, pearsonr
from matplotlib.patches import Patch
from sklearn.mixture import GaussianMixture


ALL_CELLS = 'AllCells'
ACCEPTED_COVERAGE = 10


# Optional: Hartigan's dip test if the 'diptest' package is installed
try:
    from diptest import diptest
    HAS_DIPTEST = True
except ImportError:
    HAS_DIPTEST = False
    diptest = None

def analyze_and_filter_multimodal_rows(
    X,
    reference_list,
    max_components=3,
    alpha=0.05,
    min_samples_for_tests=20,
    random_state=42
):
    """
    Row-wise multimodality analysis + filtering.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        2D matrix with shape (n_rows, n_samples).
        Values are assumed to be proportions in [0, 1], but this is
        only used for interpretation (no transformation is applied).
    max_components : int, default=3
        Maximum number of mixture components to consider in GMM.
    alpha : float, default=0.05
        Significance level for the dip test (if available).
    min_samples_for_tests : int, default=20
        Minimum number of non-NaN samples in a row to run tests.
    random_state : int, default=42
        Random seed for GMM.

    Returns
    -------
    results : pd.DataFrame
        Per-row statistics:
          - row: original row index (or integer index)
          - n_samples: non-NaN samples used
          - n_unique: number of unique values
          - dip_stat, dip_p (if diptest installed)
          - best_k_bic: GMM components with lowest BIC
          - bic_k1, bic_k2, ... bic_kK
          - likely_multimodal: boolean flag
          - note: text comment
    X_filtered : same type as X
        Subset of rows from X that passed the multimodality criteria
        (results['likely_multimodal'] == True).
    keep_mask : np.ndarray[bool]
        Boolean mask over rows of X indicating which rows were kept.
    """

    # Preserve DataFrame index/columns if present
    is_df = isinstance(X, pd.DataFrame)
    if is_df:
        row_index = X.index
        col_index = X.columns
        X_values = X.values
    else:
        row_index = reference_list
        col_index = None
        X_values = np.asarray(X)

    n_rows, n_cols = X_values.shape
    records = []

    for i in range(n_rows):
        row = X_values[i, :]
        # Drop NaNs
        row = row[np.isfinite(row)]
        n = len(row)

        if n > 0:
            row = np.log1p(row)
            row = (row - np.mean(row)) / (np.std(row) + 1e-8)

        n_unique = len(np.unique(row))

        # Prepare default record
        rec = {
            "row": row_index[i],
            "n_samples": n,
            "n_unique": n_unique,
            "dip_stat": np.nan,
            "dip_p": np.nan,
            "best_k_bic": np.nan,
            "likely_multimodal": False,
            "note": ""
        }

        # Pre-allocate BIC columns
        for k in range(1, max_components + 1):
            rec[f"bic_k{k}"] = np.nan

        # If not enough data or constant row, skip tests
        if n < min_samples_for_tests or n_unique <= 1:
            rec["note"] = "Too few samples or constant row"
            records.append(rec)
            continue

        # 1) Hartigan's dip test (if available)
        if HAS_DIPTEST:
            try:
                dip, p_dip = diptest(row)
                rec["dip_stat"] = float(dip)
                rec["dip_p"] = float(p_dip)
            except Exception as e:
                rec["note"] += f"[diptest error: {e}] "
        else:
            rec["note"] += "[diptest not installed] "

        # 2) Gaussian Mixture Models + BIC
        row_reshaped = row.reshape(-1, 1)
        bics = []
        ks_tested = []

        for k in range(1, max_components + 1):
            # Need at least k samples for k components
            if n <= k:
                continue
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=random_state
                )
                gmm.fit(row_reshaped)
                bic = gmm.bic(row_reshaped)
                rec[f"bic_k{k}"] = float(bic)
                bics.append(bic)
                ks_tested.append(k)
            except Exception as e:
                rec["note"] += f"[GMM k={k} error: {e}] "

        if bics:
            best_idx = int(np.argmin(bics))
            best_k = ks_tested[best_idx]
            rec["best_k_bic"] = best_k
        else:
            rec["note"] += "[no valid GMM fits] "

        # 3) Combine evidence into 'likely_multimodal'
        likely_multi = False

        # GMM: if best k > 1, strong hint at multimodality
        if not np.isnan(rec["best_k_bic"]) and rec["best_k_bic"] > 1:
            likely_multi = True

        # Dip test: reject unimodality
        if HAS_DIPTEST and np.isfinite(rec["dip_p"]):
            if rec["dip_p"] < alpha:
                likely_multi = True

        rec["likely_multimodal"] = bool(likely_multi)
        records.append(rec)

    # Build results DataFrame
    results = pd.DataFrame.from_records(records)

    # Build keep_mask based on criteria
    keep_mask = results["likely_multimodal"].values.astype(bool)

    # # Apply mask to original X
    # if is_df:
    #     X_filtered = X.loc[keep_mask]
    # else:
    #     X_filtered = X_values[keep_mask, :]

    return keep_mask, results


def open_matrices(filename, sep=','):    
    print(f"Opening {filename.split('/')[-1]}!")
    df = pd.read_csv(filename, index_col=0, sep=sep)
    return df

def add_source_prefix(df, source, sep="__"):
    df2 = df.copy()
    df2.columns = [f"{source}{sep}{c}" for c in df2.columns]
    return df2

def check_header(cov_df, rxt_df):
    cov_bc_list = list(cov_df.columns)
    rtx_bc_list = list(rxt_df.columns)
    assert (all([x == y for x,y in zip(rtx_bc_list, cov_bc_list)])), [(x,y) for x,y in zip(rtx_bc_list, cov_bc_list)]


def plot_beta_with_scatter(
    matrix,
    second_matrix,
    random_indices,
    n_rows_to_plot,
    filename,
    reference_list,
    *,
    cluster_labels=None,           # NEW: list/array of length = n_cols
    cluster_palette=None,          # NEW: optional dict {label: color}; else auto
    xlim=(0, 1),
    ylim=None,
    point_size=12,
    alpha_pts=0.7,
    to_scale=False
):
    """
    For each selected row index:
      - Left: scatter of matrix[idx] (x) vs second_matrix[idx] (y), colored by cluster of each column
      - Right: histogram of matrix[idx] with fitted Beta(a,b) (floc=0,fscale=1)
    """

    X = np.asarray(matrix, dtype=float)
    Y = np.asarray(second_matrix, dtype=float)
    n_rows, n_cols = X.shape

    if Y.shape != X.shape:
        raise ValueError("matrix and second_matrix must have the same shape")
    if cluster_labels is not None and len(cluster_labels) != n_cols:
        raise ValueError("cluster_labels length must equal number of columns in matrix")
    if len(reference_list) != n_rows:
        raise ValueError("reference_list length must equal number of rows in matrix")

    # Build cluster → color LUT
    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels)
        uniq = list(dict.fromkeys(cluster_labels))  # stable unique
        if cluster_palette is None:
            # use a default qualitative cycle
            prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            # extend if needed
            if len(prop_cycle) < len(uniq):
                # fall back to tab20 if we run out
                from matplotlib import cm
                cmap = cm.get_cmap("tab20", len(uniq))
                prop_cycle = [cmap(i) for i in range(len(uniq))]
            lut = {lab: prop_cycle[i % len(prop_cycle)] for i, lab in enumerate(uniq)}
        else:
            lut = dict(cluster_palette)
            # ensure all labels covered
            missing = [lab for lab in uniq if lab not in lut]
            if missing:
                raise ValueError(f"cluster_palette missing labels: {missing}")
        # precompute column colors
        col_colors = np.array([lut[lab] for lab in cluster_labels], dtype=object)
        legend_handles = [Patch(facecolor=lut[lab], label=str(lab)) for lab in uniq]
    else:
        col_colors = None
        legend_handles = None

    fig, axes = plt.subplots(
        n_rows_to_plot, 2,
        figsize=(12, 3.2 * n_rows_to_plot),
        gridspec_kw={"width_ratios": [1.1, 1.6]}
    )
    if n_rows_to_plot == 1:
        axes = np.array([axes])

    x_plot = np.linspace(0, 1, 200)
    eps = 1e-6

    for i, idx in enumerate(random_indices[:n_rows_to_plot]):
        xvals = X[idx]
        yvals = Y[idx]

        # Keep only finite pairs so scatter aligns properly
        mask = np.isfinite(xvals) & np.isfinite(yvals)
        x_scatter = xvals[mask]
        y_scatter = yvals[mask]

        ax_scatter = axes[i, 0]
        if x_scatter.size == 0:
            ax_scatter.text(0.5, 0.5, "No finite data", ha="center", va="center")
        else:
            if col_colors is not None:
                # color by cluster of each COLUMN, filtered by the same mask
                point_colors = col_colors[mask]
                ax_scatter.scatter(
                    x_scatter, y_scatter,
                    s=point_size, alpha=alpha_pts, c=point_colors, edgecolors="none"
                )
                # add legend once per row (or only on the first row to avoid repeats)
                ax_scatter.legend(handles=legend_handles, title="Cluster", loc="best", frameon=False)
            else:
                ax_scatter.scatter(
                    x_scatter, y_scatter,
                    s=point_size, alpha=alpha_pts, edgecolors="none"
                )

            # Axis limits
            if xlim is not None:
                ax_scatter.set_xlim(*xlim)
            if ylim is not None:
                ax_scatter.set_ylim(*ylim)

            # Correlation (guard for constant arrays)
            if np.std(x_scatter) < 1e-12 or np.std(y_scatter) < 1e-12:
                corr_txt = "constant axis"
            else:
                r, p = pearsonr(x_scatter, y_scatter)
                corr_txt = f"r={r:.2f} (p={p:.1e})"

            ax_scatter.set_title(f"{reference_list[idx]} • Scatter\nn={x_scatter.size} | {corr_txt}")
            ax_scatter.set_xlabel("Normalized Reactivity")
            ax_scatter.set_ylabel("Coverage")

        # --- Right: Histogram + Beta fit ---
        ax_dist = axes[i, 1]
        row_primary = xvals[np.isfinite(xvals)]

        # scale-to-[0,1] if requested (shared across all clusters for this row)
        max_x = float(np.nanmax(row_primary)) + eps if to_scale else 1.0
        row01_all = np.clip(row_primary / max_x, eps, 1 - eps)

        if row01_all.size == 0:
            ax_dist.text(0.5, 0.5, "No finite data", ha="center", va="center")
            ax_dist.set_title(f"{reference_list[idx]} • Distribution\nn=0")
        else:
            if cluster_labels is None:
                # original single-distribution behavior
                try:
                    print('Plotting Normally', cluster_labels)
                    a, b, _, _ = beta.fit(row01_all, floc=0, fscale=1)
                    y_pdf = beta.pdf(x_plot, a, b)
                    ax_dist.hist(row01_all, bins=20, density=True, alpha=0.5, label="Row data")
                    ax_dist.plot(x_plot, y_pdf, label=f"Beta({a:.2f}, {b:.2f})")
                    ax_dist.set_xlim(0, 1)
                    ax_dist.set_title(
                        f"{reference_list[idx]} • Distribution\nn={row01_all.size} | Beta({a:.2f}, {b:.2f})"
                    )
                    ax_dist.legend(frameon=False)
                except Exception:
                    ax_dist.hist(row01_all, bins=20, density=True, alpha=0.5, label="Row data")
                    ax_dist.set_xlim(0, 1)
                    ax_dist.set_title(f"{reference_list[idx]} • Distribution\nBeta fit error")
                    ax_dist.legend(frameon=False)
            else:
                print('Plotting By Clusters')
                # per-cluster overlays
                clabs = np.asarray(cluster_labels)
                # columns valid for this row (finite xvals)
                valid_cols = np.isfinite(xvals)
                
                # --- combined distribution over all clusters ---
                try:
                    if np.allclose(np.std(row01_all, ddof=1), 0):
                        raise RuntimeError("constant")
                    a_all, b_all, _, _ = beta.fit(row01_all, floc=0, fscale=1)
                    # combined histogram (light grey)
                    ax_dist.hist(
                        row01_all, bins=20, density=True,
                        alpha=0.25, color="lightgrey",
                        label=f"All (n={row01_all.size}) | Beta({a_all:.2f}, {b_all:.2f})"
                    )
                    # combined Beta curve (black)
                    ax_dist.plot(
                        x_plot, beta.pdf(x_plot, a_all, b_all),
                        color="black"
                    )
                except Exception:
                    # fall back to just a combined histogram
                    ax_dist.hist(
                        row01_all, bins=20, density=True,
                        alpha=0.25, color="lightgrey",
                        label=f"All (n={row01_all.size})"
                    )

                # iterate in the same order used for colors/legend
                for lab in [*dict.fromkeys(clabs)]:
                    cols = valid_cols & (clabs == lab)
                    vals = xvals[cols]
                    if vals.size == 0:
                        continue
                    vals01 = np.clip(vals / max_x, eps, 1 - eps)

                    # beta fit per cluster (guard tiny/constant)
                    try:
                        if np.allclose(np.std(vals01, ddof=1), 0):
                            raise RuntimeError("constant")
                        a, b, _, _ = beta.fit(vals01, floc=0, fscale=1)
                        ax_dist.plot(x_plot, beta.pdf(x_plot, a, b), color=lut[lab])

                        # histogram per cluster
                        ax_dist.hist(
                            vals01, bins=20, density=True, alpha=0.35,
                            label=f"{lab} (n={vals01.size}) | Beta({a:.2f}, {b:.2f})", color=lut[lab]
                        )
                    except Exception:
                        # skip fit if unstable
                        pass

                ax_dist.set_xlim(0, 1)
                title_extra = f" (scaled by max={max_x - eps:.3g})" if to_scale else ""
                ax_dist.set_title(f"{reference_list[idx]} • Distribution{title_extra}")
                ax_dist.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()




def threshold_and_drop_all_nan_rows(path, X):
    """
    Convert values < X to NaN and drop rows that are all NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (numeric values).
    X : float
        Threshold.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = open_matrices(path)

    # Convert values < X to NaN
    df[df < X] = np.nan

    # Drop rows where all columns are NaN
    df = df.dropna(axis=0, how="all")

    return df


def percent_feature_set(counts, gene_names, pattern=r"^MT-"):
    """
    Compute percentage of counts per cell coming from genes
    matching a prefix pattern (e.g. 'MT-').

    Parameters
    ----------
    counts : np.ndarray
        Gene x cell count matrix (n_genes, n_cells)
    gene_names : list or array
        Gene names corresponding to rows
    pattern : str
        Prefix to match (default 'MT-')

    Returns
    -------
    percent : np.ndarray
        Percentage per cell (length n_cells)
    """
    import re
    gene_names = np.array(gene_names)

    # Identify matching genes
    # mask = np.char.startswith(gene_names.astype(str), pattern)
    pattern = re.compile(pattern)
    mask = np.array([bool(pattern.match(str(g))) for g in gene_names])
    
    # Sum mitochondrial counts per cell
    mt_counts = counts[mask, :].sum(axis=0)

    # Total counts per cell
    total_counts = np.nansum(counts, axis=0)

    # Avoid division by zero
    percent = np.divide(
        mt_counts,
        total_counts,
        out=np.zeros_like(mt_counts, dtype=float),
        where=total_counts != 0
    ) * 100
    
    return percent


def drop_near_constant_windows(
    window_by_cell: np.ndarray,
    *,
    iqr_eps: float = 1e-3,
    min_valid_cells: int = 10,
    min_valid_frac: float = 0.10,
    return_iqr: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Flag and drop near-constant windows using a robust IQR across cells.

    Parameters
    ----------
    window_by_cell
        2D array (n_windows x n_cells). Can contain NaNs.
        Each row is a window summary per cell (e.g., window-mean reactivity per cell).
    iqr_eps
        Windows with IQR < iqr_eps are considered near-constant (after NaN handling).
    min_valid_cells
        Require at least this many finite values in the window (row) to score.
    min_valid_frac
        Also require at least this fraction of cells to be finite to score.
    return_iqr
        If True, also return the per-window IQR array (NaN for windows not scored).

    Returns
    -------
    keep_mask
        Boolean array (n_windows,) True for windows to keep.
    iqr
        Optional float array (n_windows,) with computed IQRs (NaN if not scored).
    """
    X = np.asarray(window_by_cell, dtype=float)
    if X.ndim != 2:
        raise ValueError("window_by_cell must be a 2D array (n_windows x n_cells)")

    n_windows, n_cells = X.shape
    finite = np.isfinite(X)
    n_valid = finite.sum(axis=1)

    # windows must have enough valid values to be meaningfully assessed
    valid_enough = (n_valid >= min_valid_cells) & (n_valid >= int(np.ceil(min_valid_frac * n_cells)))

    # compute robust IQR across cells per window (ignore NaNs)
    q25 = np.full(n_windows, np.nan)
    q75 = np.full(n_windows, np.nan)

    if np.any(valid_enough):
        q25[valid_enough] = np.nanpercentile(X[valid_enough], 25, axis=1)
        q75[valid_enough] = np.nanpercentile(X[valid_enough], 75, axis=1)

    iqr = q75 - q25

    # keep if either not scorable (you decide) or scorable and not near-constant
    # Here: if not scorable -> drop (conservative). Change to True to keep them.
    keep_mask = valid_enough & (iqr >= iqr_eps)

    if return_iqr:
        return keep_mask, iqr
    return keep_mask, None


def reorder_columns(matrix, cluster_label, barcode_label):
    '''
    reorder columns based on cluster labels order
    '''
    # unique labels in the order they first appear
    unique_labels = []
    for lab in cluster_label:
        if lab not in unique_labels:
            unique_labels.append(lab)

    # build new column order
    new_order = [i for lab in unique_labels for i, l in enumerate(cluster_label) if l == lab]

    # reorder matrix and labels
    matrix_reordered = matrix[:, new_order]
    cluster_label_reordered = [cluster_label[i] for i in new_order]
    barcode_label_reordered = [barcode_label[i] for i in new_order]

    return matrix_reordered, cluster_label_reordered, barcode_label_reordered


def calculate_nan_by_clusters(matrix, labels):
    total_elements = matrix.size
    nan_count = np.isnan(matrix).sum()
    nan_percent = nan_count / total_elements * 100

    print("Total elements:", total_elements)
    print("NaN count:", nan_count)
    print("NaN percent:", nan_percent, "%")

    nan_stats = {}
    for lab in set(labels):
        cols = [i for i, l in enumerate(labels) if l == lab]   # indices for this cluster
        submatrix = matrix[:, cols]
        total = submatrix.size
        count = np.isnan(submatrix).sum()
        nan_stats[lab] = {
            "nan_count": count,
            "nan_percent": count / total * 100
        }   

    print(nan_stats)


def write_genes_windows(
    reference_list,
    print_gene_name=False,
    output_file=None
):
    """
    Group windows by gene and optionally write summary to file.

    Parameters
    ----------
    reference_list : list
        List like ["GENE-123", "GENE-124", ...]
    print_gene_name : bool
        Whether to print gene positions
    output_file : str or None
        If provided, write results to this file
    """

    gene_dict = {}
    gene_positions = defaultdict(list)

    # Parse reference list
    for i, entry in enumerate(reference_list):
        pos = int(entry.split('-')[-1])
        gene = entry.rsplit('-', 1)[0]

        gene_dict.setdefault(gene, []).append(i)
        gene_positions[gene].append(pos)

    # Sort genes by number of windows (descending)
    gene_dict = dict(sorted(
        gene_dict.items(),
        key=lambda item: len(item[1]),
        reverse=True
    ))

    keep_indices = []
    gene_list = []

    for gene, indices in gene_dict.items():
        keep_indices.extend(indices)
        gene_list.append(gene)

    total_genes = len(gene_list)
    total_windows = len(keep_indices)

    # Prepare output text
    lines = []
    lines.append("Gene Summary")
    lines.append("=" * 40)

    if print_gene_name:
        for c, gene in enumerate(gene_positions):
            to_write_gene_pos = ','.join([str(i) for i in gene_positions[gene]])
            lines.append(
                f"{c}\t{gene}\t{len(gene_positions[gene])}\t{to_write_gene_pos}"
            )

    lines.append("\nGene List (sorted by window count):")
    lines.append(str(gene_list))
    lines.append(f"\nTotal Number of Genes : {total_genes}")
    lines.append(f"Total Number of Windows : {total_windows}")

    output_text = "\n".join(lines)

    # Write to file if requested
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(output_text)

    # Also print to console
    print(output_text)

    return gene_list, keep_indices


def parse_template(template, sep="_"):
    parts = template.split(sep)
    
    mapping = {}
    for i, part in enumerate(parts):
        match = re.match(r"\{(.+?)\}", part)
        if match:
            key = match.group(1)
            if key != "":   # ignore {}
                mapping[key] = i
    
    return mapping


def parse_sample(sample, mapping, sep="_"):
    parts = sample.split(sep)
    
    return {
        key: parts[idx] if idx < len(parts) else None
        for key, idx in mapping.items()
    }


#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

def bh_fdr(pvals):
    """
    Benjamini–Hochberg FDR correction for a 1D array-like of p-values.
    NaNs are ignored and remain NaN in the output.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    # mask of valid p-values
    mask = np.isfinite(p) & (p >= 0)
    m = int(mask.sum())
    if m == 0:
        return q

    # order indices by p, push invalids to the end
    order = np.argsort(np.where(mask, p, np.inf))
    # take the first m positions = valid p-values in ascending order
    ranked = p[order[:m]]

    # BH adjustment on the valid slice only
    ranks = np.arange(1, m + 1, dtype=float)
    adj = ranked * m / ranks

    # enforce monotonicity (non-increasing when going from high to low ranks)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    # scatter back to original positions
    q[order[:m]] = adj
    return q

def cliffs_delta(x, y):
    # assumes finite arrays
    x = np.asarray(x); y = np.asarray(y)
    # fast approximate δ via U statistic relation:
    # AUC = U / (n_x*n_y); δ = 2*AUC - 1
    U, _ = mannwhitneyu(x, y, alternative='two-sided')
    auc = U / (len(x)*len(y))
    return 2*auc - 1


def filter_rows_by_cluster_diffs(
    matrix,                 # pd.DataFrame (rows=features, cols=cells) OR 2D np.array
    cluster_labels,         # list/array of cluster label for each column (e.g. ['C0','C1','C2',...])
    clusters=None,          # optional explicit order like ['C0','C1','C2']
    alpha=0.05,
    min_cells_per_cluster=5,
    effect_thresh=0.2,      # |Cliff's δ| threshold
    return_pairwise=False
):
    if isinstance(matrix, np.ndarray):
        df = pd.DataFrame(matrix)
    else:
        df = matrix.copy()

    col_labels = np.asarray(cluster_labels)
    if clusters is None:
        clusters = pd.unique(col_labels)

    # indices of columns per cluster
    cols_by_cluster = {c: np.where(col_labels == c)[0] for c in clusters}

    # Precompute which rows are “testable” (enough finite values per cluster)
    def get_cluster_values(rowvals, idx):
        vals = rowvals[idx]
        return vals[np.isfinite(vals)]

    n_rows = df.shape[0]
    k = len(clusters)

    if k == 2:
        cA, cB = clusters[0], clusters[1]
        pvals = np.full(n_rows, np.nan)
        deltas = np.full(n_rows, np.nan)
        testable = np.zeros(n_rows, dtype=bool)

        for i, (_, row) in enumerate(df.iterrows()):
            xa = get_cluster_values(row.values, cols_by_cluster[cA])
            xb = get_cluster_values(row.values, cols_by_cluster[cB])
            if (xa.size >= min_cells_per_cluster) and (xb.size >= min_cells_per_cluster):
                testable[i] = True
                try:
                    U, p = mannwhitneyu(xa, xb, alternative='two-sided')
                except TypeError:
                    U, p = mannwhitneyu(xa, xb)
                pvals[i] = p
                try:
                    deltas[i] = cliffs_delta(xa, xb)
                except Exception:
                    deltas[i] = np.nan

        qvals = bh_fdr(pvals)
        keep = (qvals < alpha) & (np.abs(deltas) >= effect_thresh)

        res = pd.DataFrame({
            "mw_p": pvals,
            "mw_q": qvals,
            "testable": testable,
            "delta": deltas,
            "keep": keep
        }, index=df.index)

        # ensure boolean 1D mask
        keep_mask = np.asarray(keep, dtype=bool)

        if return_pairwise:
            # keep naming consistent with k>2 case
            res[f"q_{cA}_vs_{cB}"] = qvals
            res[f"delta_{cA}_vs_{cB}"] = deltas

        return keep_mask, res

    kw_p = np.full(n_rows, np.nan)
    testable = np.zeros(n_rows, dtype=bool)

    # 1) Kruskal–Wallis gate
    for i, (_, row) in enumerate(df.iterrows()):
        groups = []
        ok = True
        for c in clusters:
            v = get_cluster_values(row.values, cols_by_cluster[c])
            if v.size < min_cells_per_cluster:
                ok = False
                break
            groups.append(v)
        if not ok:
            continue
        testable[i] = True
        try:
            _, p = kruskal(*groups, nan_policy='omit')
        except Exception:
            p = np.nan
        kw_p[i] = p

    kw_q = bh_fdr(kw_p)
    passes_kw = (kw_q < alpha)

    # 2) Pairwise post-hoc on rows that pass KW
    pairs = [(clusters[0], clusters[1]), (clusters[0], clusters[2]), (clusters[1], clusters[2])]
    pair_p = {pair: np.full(n_rows, np.nan) for pair in pairs}
    pair_delta = {pair: np.full(n_rows, np.nan) for pair in pairs}

    rows_to_check = np.where(passes_kw & testable)[0]
    for i in rows_to_check:
        row = df.iloc[i].values
        for pair in pairs:
            a, b = pair
            xa = get_cluster_values(row, cols_by_cluster[a])
            xb = get_cluster_values(row, cols_by_cluster[b])
            # check min cells again (after dropping NaNs)
            if (xa.size < min_cells_per_cluster) or (xb.size < min_cells_per_cluster):
                continue
            try:
                U, p = mannwhitneyu(xa, xb, alternative='two-sided')
            except TypeError:
                # for older SciPy without 'method' kw
                U, p = mannwhitneyu(xa, xb)
            pair_p[pair][i] = p
            try:
                pair_delta[pair][i] = cliffs_delta(xa, xb)
            except Exception:
                pair_delta[pair][i] = np.nan

    # FDR across *all* tested pairs jointly
    all_pair_p = np.column_stack([pair_p[p] for p in pairs])
    all_pair_q = np.full_like(all_pair_p, np.nan, dtype=float)

    # vectorize BH over columns, but compute across rows independently
    # Flatten, BH, then reshape so that multiplicity is over all pairwise tests for all rows
    flat_p = all_pair_p.ravel()
    flat_q = bh_fdr(flat_p)
    all_pair_q = flat_q.reshape(all_pair_p.shape)

    # Decide keep per row: any pair (q<alpha & |δ| >= effect_thresh)
    good = np.zeros(n_rows, dtype=bool)
    best_pair = np.array([None]*n_rows, dtype=object)
    best_q = np.full(n_rows, np.nan)
    best_delta = np.full(n_rows, np.nan)

    for r in range(n_rows):
        qs = all_pair_q[r, :]
        ds = np.column_stack([pair_delta[p][r] for p in pairs]).ravel()
        cond = (qs < alpha) & (np.abs(ds) >= effect_thresh)
        if np.any(cond):
            k = np.nanargmin(np.where(cond, qs, np.nan))
            good[r] = True
            best_pair[r] = f"{pairs[k][0]} vs {pairs[k][1]}"
            best_q[r] = qs[k]
            best_delta[r] = ds[k]

    # Build results DataFrame
    res = pd.DataFrame({
        "kw_p": kw_p,
        "kw_q": kw_q,
        "passes_kw": passes_kw,
        "keep": good,
        "best_pair": best_pair,
        "best_pair_q": best_q,
        "best_pair_delta": best_delta
    }, index=df.index)

    # Add per-pair q and δ (optional)
    if return_pairwise:
        for j, pair in enumerate(pairs):
            res[f"q_{pair[0]}_vs_{pair[1]}"] = all_pair_q[:, j]
            res[f"delta_{pair[0]}_vs_{pair[1]}"] = np.column_stack([pair_delta[pair]])[:,0]

    kept_df = df.loc[res["keep"].values]

    keep_mask = res["keep"].to_numpy() 
    # kept_rows_array = kept_df.to_numpy()

    # kept_rows -> 2D DataFrame (subset of original rows)
    # keep_mask  -> 1D boolean array aligned to your original row order
    # keep_idx   -> the row labels/indices to keep (use list(keep_idx) if you want a list)
    # kept_rows_array -> 2D NumPy array of the kept rows (None if you passed a DataFrame and keep it that way)

    return keep_mask, res


def quantile_normalize_columns(sub):
    """
    Quantile normalize columns of a 2D numpy array.
    Rows = positions, columns = cells

    NaNs are preserved and ignored during ranking/averaging.
    """
    sub = np.asarray(sub, dtype=float)
    out = sub.copy()

    n_rows, n_cols = sub.shape

    # Sort each column, ignoring NaNs by pushing them to the end
    sorted_vals = np.full((n_rows, n_cols), np.nan)

    valid_masks = []
    sort_orders = []

    for j in range(n_cols):
        col = sub[:, j]
        valid = np.isfinite(col)
        valid_masks.append(valid)

        valid_idx = np.where(valid)[0]
        sort_idx_local = valid_idx[np.argsort(col[valid_idx], kind="mergesort")]
        sort_orders.append(sort_idx_local)

        sorted_col = col[sort_idx_local]
        sorted_vals[:len(sorted_col), j] = sorted_col

    # Mean per rank across columns
    rank_means = np.nanmean(sorted_vals, axis=1)

    # Put averaged ranks back into original row order per column
    out[:] = np.nan
    for j in range(n_cols):
        sort_idx_local = sort_orders[j]
        if len(sort_idx_local) == 0:
            continue
        out[sort_idx_local, j] = rank_means[:len(sort_idx_local)]

    return out


def group_rows_by_gene(gene_pos_labels, sep='-'):
    groups = defaultdict(list)
    for i, label in enumerate(gene_pos_labels):
        if sep in label:
            gene, _ = label.rsplit(sep, 1)
        else:
            gene = label
        groups[gene].append(i)
    return groups



def winsorize_scale_by_gene_per_cell(
    X,
    gene_pos_labels,
    lower=5,
    upper=95,
    sep='-',
    min_finite=3,
    scale=True
):
    """
    Winsorize per gene per cell (column-wise), then optionally scale to [0, 1].

    Parameters
    ----------
    X : np.ndarray
        Shape (n_positions, n_cells)
    gene_pos_labels : list-like
        Row labels like GENE-123
    lower, upper : float
        Percentiles for winsorization
    sep : str
        Separator between gene and position
    min_finite : int
        Minimum finite values required in a gene-column block
    scale : bool
        Whether to min-max scale each gene-column block to [0, 1]

    Returns
    -------
    np.ndarray
        Same shape as X
    """
    X = np.asarray(X, dtype=float).copy()
    groups = group_rows_by_gene(gene_pos_labels, sep=sep)

    for _, idxs in groups.items():
        sub = X[idxs, :].copy()   # shape: (positions_of_gene, n_cells)

        for j in range(sub.shape[1]):
            col = sub[:, j]
            finite = np.isfinite(col)

            if finite.sum() < min_finite:
                sub[:, j] = np.nan
                continue

            vals = col[finite]

            lo = np.nanpercentile(vals, lower)
            hi = np.nanpercentile(vals, upper)

            # winsorize this gene-column
            clipped = col.copy()
            clipped[finite] = np.clip(vals, lo, hi)

            if scale:
                cmin = np.nanmin(clipped[finite])
                cmax = np.nanmax(clipped[finite])

                if cmax > cmin:
                    clipped[finite] = (clipped[finite] - cmin) / (cmax - cmin)
                else:
                    # all same value after clipping
                    clipped[finite] = 0.0

            sub[:, j] = clipped

        X[idxs, :] = sub

    return X


def filter_genes_by_min_bases(X, gene_pos_labels, min_bases=3, sep='-'):
    """
    Keep only genes with at least `min_bases` rows.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_positions, n_cells)
    gene_pos_labels : list-like
        Row labels like GENE-123
    min_bases : int
        Minimum number of rows/bases required per gene
    sep : str
        Separator between gene and position

    Returns
    -------
    keep_indices : np.ndarray
    """
    X = np.asarray(X)
    groups = group_rows_by_gene(gene_pos_labels, sep=sep)

    keep_indices = []
    for _, idxs in groups.items():
        if len(idxs) >= min_bases:
            keep_indices.extend(idxs)

    keep_indices = np.array(sorted(keep_indices))
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[keep_indices] = True

    return mask


def normalize_by_gene(
    X,
    gene_pos_labels,
    method="winsor",
    sep='-',
    lower=5,
    upper=95,
    min_finite=2
):
    """
    Normalize each gene block separately.

    Parameters
    ----------
    X : np.ndarray
        shape = (n_positions, n_cells)
    gene_pos_labels : list[str]
        row labels like GENE-123
    method : str
        'quantile', 'winsor', or 'none'
    """
    X = np.asarray(X, dtype=float).copy()

    if method == "winsor":
        return winsorize_scale_by_gene_per_cell(
            X, gene_pos_labels,
            lower=lower, upper=upper,
            sep=sep, min_finite=min_finite
        )

    elif method == "quantile":
        return quantile_normalize_columns(X)

    elif method == "none":
        return X

    else:
        raise ValueError("method must be one of: 'quantile', 'winsor', 'none'")
    

class MatrixFiltering:

    pattern_mt = r"^MT-" # r"^(RPL|RPS)" # r"(RPS|RPL|MT-)"
    mt_threshold = 10
    read_threshold = 10000
    poor_windows_filter = 0.7

    def __init__(self, 
                 args, 
                 batch_id,
                 matrices, 
                 path_to_output,
                 **kwargs):
        
        self.batch_id = batch_id
        self.imputed_r_mat = None
        self.template = args.template
        self.args = args

        reactivity_df = matrices.r_df
        coverage_df = matrices.c_df
        

        self.r_mat = reactivity_df.to_numpy(dtype=float)
        self.c_mat = coverage_df.to_numpy(dtype=float)
        self.r_index = reactivity_df.index.tolist()
        self.bc_index = np.array(reactivity_df.columns)

        self.labels = {} # Store Treatment and Volume For eg.
        self.path_to_output = path_to_output

        self._check_equal_index(coverage_df)

        readcount_df = matrices.rc_df
        self.rc_mat = readcount_df.to_numpy(dtype=float)
        print (self.rc_mat)
        self.gene_names = readcount_df.index.tolist() 

        """
        Certain Functions (Very Old Functions) are untested in this pipeline
        Untested Functions are labeled "Untested"
        Functions are in order - rearrange if required
        """
        filtering_task = {
            "fitler_readcount": (self._filter_by_readcount, False),
            "fitler_mt_genes": (self._fitler_mt_genes, False),
            "remove_near_constant": (self._remove_near_constant, False),
            "remove_poor_windows_1" : (self._remove_poor_windows, True),
            "remove_poor_cells" : (self._remove_poor_cells, True),
            "filter_by_genes" : (self._filter_row_by_genes, False),
            "remove_low_bases": (self._remove_low_bases, False),
            "second_normalization": (self._second_normalization, False),
            # "remove_poor_windows_2" : (self._remove_poor_windows, True),
            "highly_variable_window" : (self._highly_variable_window, False),
            "calculate_nan_by_clusters" : (self._calculate_nan_by_clusters, True),
            "write_genes_window" : (self._write_genes_windows, True),
            "generate_labels": (self._generate_labels, True),
            "impute" : (self._impute, True),
        }

        self.hvw_method = kwargs.get("highly_variable_window", None)
        self.normalisation = kwargs.get("second_normalization", 'winsor')
        self.min_bases = kwargs.get("remove_low_bases", 5)
        self.percent_mt = percent_feature_set(self.rc_mat, self.gene_names, pattern=self.pattern_mt)
        for key, (func, to_run) in filtering_task.items():            
            if to_run or kwargs.get(key, False):
                print (f'FITLERING MATRIX: Running {key} step!')
                func()

    def _remove_low_bases(self):
        '''
        Remove Genes with less than min_bases, shrink number of features
        '''
        mask = filter_genes_by_min_bases(self.r_mat, self.r_index, min_bases=self.min_bases)
        self.__apply_window_mask(mask, "Remove Low Number of Bases in Genes")

    def _second_normalization(self):
        print (f"Running Second Normalzation Method: {self.normalisation}")
        self.r_mat = normalize_by_gene(self.r_mat, self.r_index, method=self.normalisation)


    def _generate_labels(self):
        mapping = parse_template(self.template)
        self.label_count = len(mapping.keys())
        for key, ind in mapping.items():
            self.labels[key] = [i.split('_')[ind] for i in self.bc_index]
    

    def _impute(self, impute_method='soft_impute'):
        print ("Imputing!!")
        if impute_method == 'iterative_impute':
            from sklearn.impute import IterativeImputer
            from sklearn.experimental import enable_iterative_imputer
            imputer = IterativeImputer(random_state=0)

        elif impute_method == 'soft_impute':
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median',
                                    keep_empty_features=True)  

        # Fit and transform
        self.imputed_r_mat = imputer.fit_transform(self.r_mat)
        self.__save_imputed_files()


    def __save_imputed_files(self):
        path_to_matrices = f"{self.path_to_output}/"
        filename = f"{path_to_matrices}/{self.batch_id}.filtered.matrix{self.args.coverage}.{ALL_CELLS}.csv"
        convert_to_df(self.r_mat, self.r_index, self.bc_index, filename)

        filename = f"{path_to_matrices}/{self.batch_id}.imputed_filtered.matrix{self.args.coverage}.{ALL_CELLS}.csv"
        convert_to_df(self.imputed_r_mat, self.r_index, self.bc_index, filename)


    def _filter_row_by_genes(self):
        ## TODO ##
        pass

    def _highly_variable_window(self):
        ## TODO - Add Cluster By Treatment First

        func_dict = {
            'compare_cluster' : filter_rows_by_cluster_diffs,
            'multimodal' : analyze_and_filter_multimodal_rows,
        }

        cluster_label = [i.split('_')[parse_template(self.template)['treatment']] for i in self.bc_index]
        if self.hvw_method == 'compare_cluster':
            arg_2 = cluster_label
        elif self.hvw_method == 'multimodal':
            arg_2 = self.r_index
        
        mask, stats = func_dict[self.hvw_method](
            self.r_mat,
            arg_2
        )

        self.__apply_window_mask(mask, task=self.hvw_method)
        cluster_label = [i.split('_')[parse_template(self.template)['treatment']] for i in self.bc_index]

        print (f'Post {self.hvw_method} Matrix Shape ::', self.r_mat.shape, self.c_mat.shape, self.rc_mat.shape)
        print (stats)

        n_rows_to_plot = 5
        random_indices = np.random.choice(range(self.r_mat.shape[0]), size=n_rows_to_plot, replace=False)
        filename = f"{self.path_to_output}/{self.batch_id}.random5.distribution.png"

        plot_beta_with_scatter(
            matrix=self.r_mat,
            second_matrix=self.c_mat,         # same shape as matrix
            random_indices=random_indices,
            n_rows_to_plot=n_rows_to_plot,
            filename=filename,
            reference_list=self.r_index,
            cluster_labels=cluster_label,
            xlim=None,                         # scatter x-limits (primary axis)
            ylim=None,                         # let y auto-scale; set like (0,1) if you prefer
            point_size=12,
            to_scale=True
        )


    def _remove_poor_windows(self):
        """
        keep_only_filled_rows > cells
        """

        threshold = self.r_mat.shape[1]*self.poor_windows_filter
        print (f'Keeping Gene-Position with > {threshold} windows / {self.r_mat.shape[1]} Windows/Bases')
        
        non_zero_counts = np.sum(~np.isnan(self.r_mat), axis=1)

        print ( np.sum(non_zero_counts > threshold) )
        print ( np.sum(non_zero_counts == self.r_mat.shape[1]) )

        mask = non_zero_counts >= threshold
        self.__apply_window_mask(mask)
        

    def _remove_near_constant(self):
        mask, _ = drop_near_constant_windows(self.r_mat, iqr_eps=1e-3, min_valid_cells=20, return_iqr=True)
        self.__apply_window_mask(mask, "Remove Near Constant")


    def __apply_window_mask(self, mask, task="Remove Poor Windows"):
        self.r_mat = self.r_mat[mask]
        self.c_mat = self.c_mat[mask]
        keep_indices = np.nonzero(mask)[0]
        self.r_index = [self.r_index[i] for i in keep_indices]
        print (f'Post {task} Matrix Shape ::', self.r_mat.shape, self.c_mat.shape, self.rc_mat.shape)


    def _remove_poor_cells(self, nan_threshold=0.2):
        """
        Remove columns (cells) that have more than `nan_threshold` fraction of NaNs.
        """
        self.r_mat = np.asarray(self.r_mat)

        n_rows = self.r_mat.shape[0]
        nan_counts = np.sum(np.isnan(self.r_mat), axis=0)
        nan_fraction = nan_counts / n_rows

        print(f"Keeping columns with ≤ {nan_threshold*100:.1f}% NaN")

        mask = nan_fraction <= nan_threshold
        self.__apply_cell_mask(mask, task="Remove Poor Cells")


    def _check_equal_index(self, coverage_df):
        c_index = coverage_df.index.tolist()
        assert (all([x == y for x,y in zip(self.r_index, c_index)]))

    def _calculate_nan_by_clusters(self):
        for label in self.labels:
            calculate_nan_by_clusters(self.r_mat, label)

    def _write_genes_windows(self):
        filename = f'{self.path_to_output}/genes_summary.csv'
        write_genes_windows(self.r_index, print_gene_name=True, output_file=filename)

    def _filter_by_readcount(self):
        mask = np.nansum(self.rc_mat, axis=0) > self.read_threshold 
        self.__apply_cell_mask(mask, task='Filter ReadCount')

    def _fitler_mt_genes(self):
        mask = self.percent_mt > self.mt_threshold
        self.__apply_cell_mask(mask, task='High MT Content')

    def __apply_cell_mask(self, mask, task = 'Filter ReadCount'):
        self.c_mat = self.c_mat[:, mask]
        self.r_mat = self.r_mat[:, mask]
        self.rc_mat = self.rc_mat[:, mask]
        self.bc_index = self.bc_index[mask]
        self.percent_mt = self.percent_mt[mask]
        print (f'Post {task} Matrix Shape ::', self.r_mat.shape, self.c_mat.shape, self.rc_mat.shape)


        