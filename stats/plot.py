#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr(corr_df, filename):
                                                                                                                               
    # Get unique clusters                                                                                                                                                  
    clusters = corr_df['cluster'].unique()                                                                                                                                 
                                                                                                                                                                            
    # Create subplots                                                                                                                                                      
    fig, axes = plt.subplots(1, len(clusters), figsize=(6 * len(clusters), 5))                                                                                             
    if len(clusters) == 1:                                                                                                                                                 
        axes = [axes]                                                                                                                                                      
                                                                                                                                                                            
    cluster_colors = {'DM': '#E41A1C', 'DMS': '#377EB8'}                                                                                                             
                                                                                                                                                                            
    for idx, cluster in enumerate(clusters):                                                                                                                               
        # Filter data for this cluster                                                                                                                                     
        cluster_df = corr_df[corr_df['cluster'] == cluster]                                                                                                                
                                                                                                                                                                            
        # Get unique cells in this cluster                                                                                                                                 
        cluster_cells = pd.unique(cluster_df[['cell1', 'cell2']].values.ravel())                                                                                           
                                                                                                                                                                            
        # Create correlation matrix for this cluster                                                                                                                       
        corr_matrix = pd.DataFrame(index=cluster_cells, columns=cluster_cells, dtype=float)                                                                                
                                                                                                                                                                            
        # Fill the matrix                                                                                                                                                  
        for _, row in cluster_df.iterrows():                                                                                                                               
            corr_matrix.loc[row['cell1'], row['cell2']] = row['correlation']                                                                                               
            corr_matrix.loc[row['cell2'], row['cell1']] = row['correlation']                                                                                               
                                                                                                                                                                            
        # Set diagonal to 1                                                                                                                                                
        np.fill_diagonal(corr_matrix.values, 1.0)                                                                                                                          
                                                                                                                                                                            
        # Plot heatmap                                                                                                                                                     
        sns.heatmap(corr_matrix.astype(float),                                                                                                                             
                    ax=axes[idx],                                                                                                                                          
                    cmap='RdBu_r', center=0,                                                                                                                               
                    vmin=-1, vmax=1,                                                                                                                                       
                    xticklabels=False, yticklabels=False,                                                                                                                  
                    cbar_kws={'label': 'Correlation', 'shrink': 0.8},                                                                                                      
                    square=True)                                                                                                                                           
                                                                                                                                                                            
        axes[idx].set_title(f'{cluster}\n(n={len(cluster_cells)} cells)',                                                                                                  
                            color=cluster_colors.get(cluster, 'black'), fontweight='bold')                                                                                 
        axes[idx].set_xlabel('Cells')                                                                                                                                      
        axes[idx].set_ylabel('Cells')                                                                                                                                      
                                                                                                                                                                            
    plt.suptitle('Cell-Cell Correlation Heatmaps by Cluster', fontsize=14, y=1.02)                                                                                         
    plt.tight_layout()                                                                                                                                                     
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')


def plot_auroc_violins_from_dict(
    auc_by_gene: dict,
    filename: str, 
    batch_id: str,
    n_cells: dict,
    *,
    genes: list | None = None,
    sample_order: list | None = None,
    ncols: int = 2,
    figsize: tuple = (12, 4),
    jitter: float = 0.08,
    point_size: float = 10,
    mean_marker_size: float = 70,
    ylim: tuple | None = None,
    title_prefix: str = "AUROC_",
    seed: int = 0,
):
    """
    Parameters
    ----------
    auc_by_gene : dict
        Nested dict: {gene: {sample_id: array_like_of_auroc_values}}
        Example:
          {"MT-RNR1": {"K562": [..], "HEK293T":[..]}, "MT-RNR2": {...}}

    genes : list, optional
        Which genes to plot (order). Default = keys in auc_by_gene.

    sample_order : list, optional
        Which sample_ids to show (order). Default = union of sample_ids across genes.

    Returns
    -------
    fig, axes
    """
    rng = np.random.default_rng(seed)

    if genes is None:
        genes = list(auc_by_gene.keys())
    else:
        genes = list(genes)

    # union of sample_ids across genes (preserves first-seen order)
    if sample_order is None:
        seen = set()
        sample_order = []
        for g in genes:
            for sid in auc_by_gene.get(g, {}).keys():
                if sid not in seen:
                    sample_order.append(sid)
                    seen.add(sid)
    else:
        sample_order = list(sample_order)

    n = len(genes)
    ncols = min(ncols, max(1, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for i, gene in enumerate(genes):
        ax = axes[i]
        per_sample = auc_by_gene.get(gene, {})

        # build data in the requested sample order
        data = []
        means = []
        for sid in sample_order:
            vals = np.asarray(per_sample.get(sid, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            data.append(vals)
            means.append(np.nanmean(vals) if vals.size else np.nan)

        positions = np.arange(1, len(sample_order) + 1)

        # violins
        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.9,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_alpha(0.5)
            body.set_edgecolor("black")
            body.set_linewidth(1.2)

        # jitter points + mean diamonds
        for sid, x0, vals, m in zip(sample_order, positions, data, means):
            if vals.size:
                xj = x0 + rng.uniform(-jitter, jitter, size=vals.size)
                ax.scatter(xj, vals, s=point_size, alpha=0.8, edgecolors="none")
            if np.isfinite(m):
                ax.scatter([x0], [m], marker="D", s=mean_marker_size,
                           color="red", edgecolor="white", linewidth=0.8, zorder=5)

            n = len(vals)
            if n > 0:
                y_min, y_max = ax.get_ylim()
                y_pos = y_min + (y_max - y_min) * 0.05  # near bottom
                ax.text(
                    x0, y_pos,
                    f"n={n}/{n_cells[sid]}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black"
                )

        ax.set_title(f"{batch_id}_{title_prefix}{gene}")
        ax.set_xticks(positions)
        ax.set_xticklabels(sample_order)
        ax.set_xlabel("Treatment")
        ax.set_ylabel(f"AUROC_{gene}")

        # y-limits
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            all_vals = np.concatenate([v for v in data if v.size], axis=0) if any(v.size for v in data) else None
            if all_vals is not None and all_vals.size:
                lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
                pad = max(0.01, (hi - lo) * 0.15)
                ax.set_ylim(lo - pad, hi + pad)

        ax.grid(False)

    # hide extra axes
    for j in range(len(genes), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
