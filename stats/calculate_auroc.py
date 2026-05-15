#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn import metrics
from typing import List, Optional
import plot as plot
from pathlib import Path
from itertools import combinations
import warnings

warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

def open_matrices(filename, sep=','):    
    print(f"Opening {filename.split('/')[-1]}!")
    df = pd.read_csv(filename, index_col=0, sep=sep)
    df = df[df.index.str.contains("MT-RNR|18S")]
    return df

def add_source_prefix(df, source, sep="__"):
    df2 = df.copy()
    df2.columns = [f"{source}{sep}{c}" for c in df2.columns]
    return df2

def read_dot_bracket(file_path, gene_name):
    """
    Read a file containing dot-bracket notation and extract the structure for a specific gene.
    
    Parameters:
    -----------
    file_path : str
        Path to the file containing dot-bracket notation (e.g., 'references/Dot bracket for ribo for mouse.txt')
    gene_name : str
        Name of the gene to extract (e.g., '18S', '12S', '16S')
    
    Returns:
    --------
    str
        Dot-bracket notation string for the specified gene, or None if gene not found
    
    Example:
    --------
    >>> structure = read_dot_bracket('references/Dot bracket for ribo for mouse.txt', '18S')
    >>> print(structure[:50])  # Print first 50 characters
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Look for the gene name (it will be on a line starting with '>')
        target_header = f'>{gene_name}'
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if this line contains the gene name
            if line.startswith(target_header):
                # The dot-bracket notation should be on the next line(s)
                dot_bracket = ''
                
                # Read subsequent lines until we hit another header or end of file
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    
                    # Stop if we encounter another gene header
                    if next_line.startswith('>'):
                        break
                    
                    # Accumulate the dot-bracket notation
                    dot_bracket += next_line
                
                return dot_bracket if dot_bracket else None
        
        # Gene not found
        print(f"Gene '{gene_name}' not found in file.")
        return None
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def calculate_cell_correlation(
    df,
    gene: str,     
    method: str = 'pearson',
    min_shared_positions: int = 100,
) -> pd.DataFrame:
    """
    Calculate pairwise correlation between cells within the same cluster for a given gene.

    For each pair of cells in the same cluster, computes correlation of their
    reactivity values across all positions of the specified gene.
    """
    df = df[df.index.str.contains(gene)]
    reactivity_matrix = df.to_numpy(dtype=float)
    cell_names = df.columns.to_numpy(dtype=str)
    clusters = ['DM', 'DMS']

    # Store correlation results
    corr_results = []
    for cluster in clusters:
        print(f"\nProcessing cluster: {cluster}")
        mask = np.char.startswith(cell_names, f"{cluster}_")
        cluster_reactivity  = reactivity_matrix[:, mask]

        cluster_cell_indices = np.where(mask)[0]
        cluster_cell_names = [cell_names[i] for i in cluster_cell_indices]
        n_cells = len(cluster_cell_indices)
        print(f"  Number of cells: {n_cells}")

        # Calculate pairwise correlations
        n_pairs = 0
        for i, j in combinations(range(n_cells), 2):
            cell1_vals = cluster_reactivity[:, i]
            cell2_vals = cluster_reactivity[:, j]

            # Find shared non-NaN positions
            valid_mask = ~np.isnan(cell1_vals) & ~np.isnan(cell2_vals)
            n_shared = np.sum(valid_mask)

            if n_shared >= min_shared_positions:
                x = cell1_vals[valid_mask]
                y = cell2_vals[valid_mask]

                if method == 'pearson':
                    # Pearson correlation
                    corr = np.corrcoef(x, y)[0, 1]
                elif method == 'spearman':
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(x, y)
                elif method == 'kendall':
                    from scipy.stats import kendalltau
                    corr, _ = kendalltau(x, y)
                else:
                    raise ValueError(f"Unknown method: {method}. Use 'pearson', 'spearman', or 'kendall'.")
            else:
                corr = np.nan

            corr_results.append({
                'cell1': cluster_cell_names[i],
                'cell2': cluster_cell_names[j],
                'cluster': cluster,
                'correlation': corr,
                'n_shared': n_shared
            })
            n_pairs += 1

        valid_corrs = [r['correlation'] for r in corr_results[-n_pairs:] if not np.isnan(r['correlation'])]
        if valid_corrs:
            print(f"  Pairs calculated: {n_pairs}")
            print(f"  Valid correlations: {len(valid_corrs)}")
            print(f"  Mean correlation: {np.mean(valid_corrs):.4f}")

    # Create correlation dataframe
    corr_df = pd.DataFrame(corr_results)

    print(f"\n{'='*60}")
    print(f"Total pairs: {len(corr_df)}")
    valid_corrs = corr_df['correlation'].dropna()
    print(f"Valid correlations: {len(valid_corrs)}")
    if len(valid_corrs) > 0:
        print(f"Overall mean correlation: {valid_corrs.mean():.4f}")
        print(f"Correlation range: [{valid_corrs.min():.4f}, {valid_corrs.max():.4f}]")

    save_mean_as = f'mean_corr_{gene}'

    # For each cell, calculate mean of all its pairwise correlations
    cell_mean_corr = {}
    for cell in cell_names:
        # Get correlations where this cell is either cell1 or cell2
        cell_corrs = corr_df[
            (corr_df['cell1'] == cell) | (corr_df['cell2'] == cell)
        ]['correlation'].dropna()

        if len(cell_corrs) > 0:
            cell_mean_corr[cell] = cell_corrs.mean()
        else:
            cell_mean_corr[cell] = np.nan

    cell_mean_corr_series = pd.Series(
        cell_mean_corr,
        index=cell_names,
        name=f"MEAN_CORR_{gene}"
    )        

    cell_mean_corr_series = cell_mean_corr_series.dropna()

    print(f"Cells with valid mean correlation for {save_mean_as}: {len(cell_mean_corr_series)} / {len(cell_names)}")

    return corr_df, cell_mean_corr_series


def calculate_auroc(
    df,
    gene: str,
    dot_bracket: str,
    len_filter: Optional[float] = 0.8, 
    min_positions: Optional[int] = None,
    skip_positions: Optional[List[int]] = None,
):
    """
    Calculate AUROC for each cell based on reactivity vs RNA secondary structure.

    For each cell, AUROC is computed using reactivity values as predictions and
    dot-bracket structure labels as ground truth (1.0 for unpaired '.', 0.0 for
    paired '(' or ')').
    """
    df = df[df.index.str.contains(gene)]
    df[["gene", "pos"]] = df.index.to_series().str.extract(r'(.+)-(\d+)$')
    df["pos"] = df["pos"].astype("Int64")
    print (df["pos"])

    reactivity_matrix = df.drop(columns=["gene", "pos"]).to_numpy(dtype=float)
    if not min_positions: 
        min_positions = int(len_filter * len(dot_bracket))

    len_filter = round(min_positions / len(dot_bracket) * 100, 3)
    if len_filter > 80:
        return None, None
    
    print (f'Length of {min_positions} ({len_filter}%) of {gene} required!')

    # Create dot-bracket dataframe with position labels
    dotbracket_df = pd.DataFrame({
        "pos": range(1, len(dot_bracket) + 1),
        "dbrt": list(dot_bracket),
    })

    # Assign labels: 1.0 for '.', 0.0 for '(' or ')'
    dotbracket_df["dbrt_label"] = dotbracket_df["dbrt"].apply(
        lambda x: 1.0 if x == '.' else 0.0
    )

    merged_df = df.merge(
        dotbracket_df,
        left_on='pos',
        right_on='pos',
        how='inner'
    )

    # Apply skip_positions filter
    if skip_positions is not None:
        skip_set = set(skip_positions)
        merged_df = merged_df[~merged_df['pos'].isin(skip_set)]
        print(f"Positions after skipping: {len(merged_df)}")

    if len(merged_df) < min_positions:
        raise ValueError(
            f"Only {len(merged_df)} positions available after filtering. "
            f"Need at least {min_positions}."
        )

    # Get the row indices in the reactivity matrix
    dbrt_labels = merged_df['dbrt_label'].values
    cell_names = list(df.drop(columns=["gene", "pos"]).columns)
    n_cells = len(cell_names)
    print(f"Calculating AUROC for {n_cells} cells...")

    # Calculate AUROC for each cell
    auroc_values = []
    valid_count = 0

    invalid_reasons = {
        'short' : 0,
        'no unique labels': 0,
        'fail' : 0,
    }

    for cell_idx in range(n_cells):
        cell_reactivity = reactivity_matrix[:, cell_idx]

        # Get non-NaN positions
        valid_mask = ~np.isnan(cell_reactivity)
        n_valid = np.sum(valid_mask)

        if n_valid >= min_positions:
            pred = cell_reactivity[valid_mask]
            label = dbrt_labels[valid_mask]

            # Check if we have both classes
            unique_labels = np.unique(label)
            if len(unique_labels) < 2:
                auroc_values.append(np.nan)
                invalid_reasons['no unique labels'] += 1
            else:
                try:
                    fpr, tpr, _ = metrics.roc_curve(label, pred)
                    auc = metrics.auc(fpr, tpr)
                    auroc_values.append(round(auc, 3))
                    valid_count += 1
                except Exception:
                    auroc_values.append(np.nan)
                    invalid_reasons['fail'] += 1
        else:
            auroc_values.append(np.nan)
            invalid_reasons['short'] += 1

    # Create series with cell names as index
    auroc_series = pd.Series(auroc_values, index=cell_names, name=f'AUROC_{gene}')

    print(f"\n{'='*60}")
    print(f"AUROC calculation complete for {gene}:")
    print(f"  Cells with valid AUROC: {valid_count} / {n_cells}")
    all_aurocs = auroc_series.dropna()
    threshold = 0.5
    valid_aurocs = all_aurocs[all_aurocs >= threshold]
    if len(valid_aurocs) > 0:
        print(f"  Number of Non-Nan Cells: {len(all_aurocs)}")
        print(f"  Number of Passed Cells at {len_filter}: {len(valid_aurocs)}")
        print(f"  Mean AUROC: {valid_aurocs.mean():.3f}")
        print(f"  Median AUROC: {valid_aurocs.median():.3f}")
        print(f"  AUROC range: [{all_aurocs.min():.3f}, {all_aurocs.max():.3f}]")

    assert not invalid_reasons['no unique labels']
    assert not invalid_reasons['fail']

    row = {
        "gene": gene,
        "len_filter%": len_filter,
        "min_positions": min_positions,
        "gene_len": len(dot_bracket),

        "n_cells_total": int(len(auroc_series)),
        "n_cells_non_nan": int(len(all_aurocs)),
        "n_cells_pass": int(len(valid_aurocs)),

        "mean_auroc_pass": float(valid_aurocs.mean()) if len(valid_aurocs) else np.nan,
        "median_auroc_pass": float(valid_aurocs.median()) if len(valid_aurocs) else np.nan,

        "min_auroc_non_nan": float(all_aurocs.min()) if len(all_aurocs) else np.nan,
        "max_auroc_non_nan": float(all_aurocs.max()) if len(all_aurocs) else np.nan,

        # store as a string so it fits in one CSV cell
        "short_mappings" : int(invalid_reasons['short'])
    }

    return auroc_series, row


def check_header(cov_df, rxt_df):
    cov_bc_list = list(cov_df.columns)
    rtx_bc_list = list(rxt_df.columns)
    assert (all([x == y for x,y in zip(rtx_bc_list, cov_bc_list)])), [(x,y) for x,y in zip(rtx_bc_list, cov_bc_list)]


def threshold_mask(matrix, threshold):
    """
    Return boolean mask where values >= threshold
    """
    return matrix >= threshold


workdir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline"
cellline = 'results_1'

def main():
    reactivity = True
    if cellline == 'results_3':
        list_of_samples = [
            ['DMS_PIP_NAIN3_1in4', 'DM_PIP_NAIN3_1in4', ],
            ['DMS_PIP_NAIN3_3in4', 'DM_PIP_NAIN3_3in4', ],  
            ['DMS_PIP_DMSO_1in4', 'DM_PIP_DMSO_1in4', ],
            ['DMS_PIP_DMSO_3in4', 'DM_PIP_DMSO_3in4', ],
        ]
    else: 
        list_of_samples = [
            ['NAIN3']
        ]

    coverage_threshold = 50
    txt = '_reactivity' if reactivity else ''
    location_for_summary = f'{workdir}/{cellline}/plots/auroc{txt}/cov{coverage_threshold}'

    auroc_results = []
    for sample_list in list_of_samples:

        batch_id = sample_list[0].replace(f"{sample_list[0].split('_')[0]}_", '')
        assert all([batch_id in sample_id for sample_id in sample_list])
        print (f"Batch ID: {batch_id}")
        
        location_to_save = f'{location_for_summary}/{batch_id}'
        Path(f'{location_to_save}/').mkdir(parents=True, exist_ok=True)

        rows = pipeline(batch_id, sample_list, location_to_save, coverage_threshold, reactivity=reactivity)
        auroc_results += rows

    # after loops:
    summary_df = pd.DataFrame(auroc_results)

    # optional: stable ordering
    summary_df = summary_df.sort_values(["batch_id", "gene", "len_filter%"]).reset_index(drop=True)

    summary_df.to_csv(f"{location_for_summary}/auroc_summary.csv", index=False)
    print("Wrote:", "auroc_summary.csv")


def pipeline(batch_id, sample_list, location_to_save, coverage_threshold, reactivity=False):
    
    len_filter_to_test = [200, 400, 600, 800, 1000, 1200]
    depth = 10
    
    if reactivity:
        to_skip = all(['DMSO' in i for i in sample_list])
        if to_skip: 
            return []

    reactivity_df = {}
    coverage_df = {}
    for sample_id in sample_list:
        print (f"Analyzing Sample: {sample_id}")
        if cellline == 'results_3':
            if reactivity:
                matrix_location = f'{workdir}/{cellline}/normalized_mtx/single_base'
                filename = f'{matrix_location}/{sample_id}.normalized_reactivity.matrix{coverage_threshold}.AllCells.gene_level.csv'
            else:   
                matrix_location = f'{workdir}/{cellline}/matrices/single_base'
                filename = f"{matrix_location}/{sample_id}/{sample_id}.mutrate.matrix10.AllCells.csv"
        else:
            matrix_location = f'{workdir}/{cellline}/countercheck/single_base'
            filename = f'{matrix_location}/{depth}/{sample_id}.normalized_reactivity.matrix{coverage_threshold}.AllCells.gene_level.csv'

        react_df = open_matrices(filename)
        react_df.index = react_df.index.str.replace(r":.*$", "", regex=True)
        reactivity_df[sample_id] = add_source_prefix(react_df, sample_id)

        if cellline == 'results_3':
            matrix_location = f'{workdir}/{cellline}/matrices/single_base'
            filename = f"{matrix_location}/{sample_id}/{sample_id}.coverage.matrix10.AllCells.csv"
        else:
            matrix_location = f'{workdir}/{cellline}/matrices/single_base'
            filename = f"{matrix_location}/{depth}/{sample_id}/{sample_id}.coverage.matrix10.AllCells.csv"
        cov_df = open_matrices(filename)
        cov_df.index = cov_df.index.str.replace(r":.*$", "", regex=True)
        coverage_df[sample_id] = add_source_prefix(cov_df, sample_id)   
        
    reactivity_merged = pd.concat(reactivity_df, axis=1, join="outer") 
    reactivity_merged.columns = np.array([i[1] for i in reactivity_merged.columns])

    reactivity_matrix = reactivity_merged.to_numpy(dtype=float)
    reactivity_masked = reactivity_matrix.astype(float).copy()

    if not reactivity:
        coverage_merged = pd.concat(coverage_df, axis=1, join="outer") 
        coverage_merged.columns = np.array([i[1] for i in coverage_merged.columns])
        check_header(coverage_merged, reactivity_merged)

        coverage_matrix = coverage_merged.to_numpy(dtype=float)
        mask = threshold_mask(coverage_matrix, threshold=coverage_threshold)
        reactivity_masked[~mask] = np.nan

    row_labels = reactivity_merged.index.tolist()
    barcode_list = list(reactivity_merged.columns)
    df = pd.DataFrame(reactivity_masked, index=row_labels, columns=barcode_list)
    reactivity_merged = df.dropna(how="all")

    filename = "./notebooks/Dot bracket for ribo for human.txt"
    dot_bracket_12S = read_dot_bracket(filename, '12S')
    dot_bracket_16S = read_dot_bracket(filename, '16S')

    dot_bracket = {
        'MT-RNR1' : dot_bracket_12S,
        'MT-RNR2' : dot_bracket_16S,
    }

    cell_names = list(reactivity_merged.columns)
    n_cells = {'DM':0, 'DMS':0}
    for name in cell_names:
        treatment = name.split('_')[0]    
        n_cells[treatment] += 1

    
    rows = [] 
    for len_filter in len_filter_to_test:
        auc_by_gene = {}
        for gene in ['MT-RNR1', 'MT-RNR2']:
            row = {}
            s, row = calculate_auroc(
                reactivity_merged,
                gene,
                dot_bracket[gene],
                min_positions=len_filter,
            )
            if not row: continue

            row['batch_id'] = batch_id
            rows.append(row)
            s = s.dropna()
            if cellline == 'results_3':
                dm_array  = s[s.index.str.startswith("DM_")].to_numpy()
                dms_array = s[s.index.str.startswith("DMS_")].to_numpy()

                auc_by_gene[gene] = {
                    "DM" : dm_array,
                    'DMS' : dms_array
                }
            else:
                all_cells_array = s.to_numpy()
                auc_by_gene[gene] = {
                    "All" : all_cells_array,
                }

        filename = f'{location_to_save}/{batch_id}_AUROC.{len_filter}.vp'
        plot.plot_auroc_violins_from_dict(auc_by_gene, filename, batch_id, n_cells)
        del auc_by_gene

    corr_by_gene = {}
    for gene in ['MT-RNR1', 'MT-RNR2']:
        corr_df, s = calculate_cell_correlation(reactivity_merged, gene)
        if cellline == 'results_3':
            dm_array  = s[s.index.str.startswith("DM_")].to_numpy()
            dms_array = s[s.index.str.startswith("DMS_")].to_numpy()

            corr_by_gene[gene] = {
                "DM" : dm_array,
                'DMS' : dms_array
            }
        else:
            all_cells_array = s.to_numpy()
            auc_by_gene[gene] = {
                "All" : all_cells_array,
            }

        filename = f'{location_to_save}/{batch_id}_CORR.{gene}.corr' 
        plot.plot_corr(corr_df, filename)
    
    filename = f'{location_to_save}/{batch_id}_CORR.vp'
    plot.plot_auroc_violins_from_dict(corr_by_gene, filename, batch_id, n_cells, title_prefix="CORR_")
    
    print (len(dot_bracket_12S))
    print (len(dot_bracket_16S))

    return rows


if __name__ == "__main__":
    main()