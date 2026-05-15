#!/usr/bin/env python3

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm, colors as mcolors
import colorsys
from matplotlib.patches import Patch
from scipy.stats import beta, pearsonr, gaussian_kde
import re
from pathlib import Path
from cluster_cells_utils import MatrixFiltering

HEX7 = re.compile(r'^#[0-9a-fA-F]{6}$')

jointplot_default = {
    'y' : "Normalized Reactivity",
    'x' : "Coverage"
}

def sample_columns_per_cluster(coverage, reactivity, cluster_label, k, random_state=None):
    rng = np.random.default_rng(random_state)
    
    coverage = np.asarray(coverage)
    reactivity = np.asarray(reactivity)
    cluster_label = np.asarray(cluster_label)

    # Unique clusters
    clusters = np.unique(cluster_label)

    selected_cols = []

    for c in clusters:
        col_indices = np.where(cluster_label == c)[0]

        # If cluster has fewer than k columns, take all
        if len(col_indices) <= k:
            chosen = col_indices
        else:
            chosen = rng.choice(col_indices, size=k, replace=False)

        selected_cols.extend(chosen)

    selected_cols = np.array(sorted(selected_cols))
    print(selected_cols)
    
    # Subset the matrix
    coverage_subset = coverage[:, selected_cols]
    reactivity_subset = reactivity[:, selected_cols]

    label_subset = cluster_label[selected_cols]

    return coverage_subset, reactivity_subset, label_subset, selected_cols


def prepare_1d(coverage, reactivity, cluster_label, sample=True):
    if sample: 
        k = 10
        coverage, reactivity, cluster_label, _ = sample_columns_per_cluster(coverage, reactivity, cluster_label, k, random_state=42)

    n_rows, n_cols = coverage.shape

    coverage_flat = coverage.ravel()
    reactivity_flat = reactivity.ravel()
    cluster_expanded = np.tile(cluster_label, n_rows)

    return reactivity_flat, coverage_flat, cluster_expanded   


def scatter_with_marginals(
    coverage, reactivity, cluster_label, *,
    bins=20, s=2, alpha=0.2,
    label=jointplot_default, filename=None
):

    x, y, cluster_label = prepare_1d(coverage, reactivity, cluster_label)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cluster_label = np.asarray(cluster_label)

    # Keep only paired finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    clusters = cluster_label[mask]

    n = x.size

    # -------------------------------
    # Combined correlation
    # -------------------------------
    if n >= 2 and np.std(x) > 0 and np.std(y) > 0:
        r_all, p_all = pearsonr(x, y)
        combined_corr_txt = f"ALL: r = {r_all:.2f}; p = {p_all:.1e}"
    else:
        combined_corr_txt = "ALL: r = n/a"

    # -------------------------------
    # Per-cluster correlations
    # -------------------------------
    unique_clusters = np.unique(clusters)
    per_cluster_corr_lines = []

    for c in unique_clusters:
        xc = x[clusters == c]
        yc = y[clusters == c]

        if len(xc) >= 3 and np.std(xc) > 0 and np.std(yc) > 0:
            rc, pc = pearsonr(xc, yc)
            per_cluster_corr_lines.append(f"{c}: r={rc:.2f}")
        else:
            per_cluster_corr_lines.append(f"{c}: r=n/a")

    per_cluster_text = "\n".join(per_cluster_corr_lines)

    # -------------------------------
    # Plot Layout
    # -------------------------------
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4],
                          hspace=0.05, wspace=0.05)

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx   = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy   = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # -------------------------------
    # Color map for clusters
    # -------------------------------
    cmap = cm.get_cmap("tab10", len(unique_clusters))
    color_dict = {c: cmap(i) for i, c in enumerate(unique_clusters)}

    # -------------------------------
    # Scatter (NOW COLORED)
    # -------------------------------
    for c in unique_clusters:
        mask_c = (clusters == c)
        ax_scatter.scatter(x[mask_c], y[mask_c],
                           s=s, alpha=alpha, color=color_dict[c],
                           label=str(c))

    ax_scatter.legend(title="Cluster")

    ax_scatter.set_xlabel(label['x'])
    ax_scatter.set_ylabel(label['y'])

    # main text block
    combined_plus_cluster = f"n = {n}\n{combined_corr_txt}\n\n{per_cluster_text}"
    ax_scatter.text(0.02, 0.98, combined_plus_cluster,
                    transform=ax_scatter.transAxes,
                    ha="left", va="top")

    # Set x-axis limits
    ax_scatter.set_xlim(np.min(x), np.max(x))
    ax_histx.set_xlim(np.min(x), np.max(x))

    # -------------------------------
    # KDE marginals
    # -------------------------------
    if n > 1:
        for c in unique_clusters:
            mask_c = clusters == c
            x_c = x[mask_c]
            y_c = y[mask_c]

            # Skip clusters with < 2 points (kde cannot compute)
            if len(x_c) < 2 or np.std(x_c) == 0 or np.std(y_c) == 0:
                continue

            # KDE for X
            kde_x = gaussian_kde(x_c)
            xx = np.linspace(np.min(x), np.max(x), 200)
            ax_histx.plot(xx, kde_x(xx), color=color_dict[c], lw=2, alpha=0.9)

            # KDE for Y
            kde_y = gaussian_kde(y_c)
            yy = np.linspace(np.min(y), np.max(y), 200)
            ax_histy.plot(kde_y(yy), yy, color=color_dict[c], lw=2, alpha=0.9)

    ax_histx.hist(x, bins=bins, density=True, alpha=0.3, color="gray")
    ax_histy.hist(y, bins=bins, density=True, alpha=0.3, color="gray", orientation="horizontal")

    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    ax_histx.tick_params(axis="x", which="both", length=0)
    ax_histy.tick_params(axis="y", which="both", length=0)

    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



class PlotResults:

    genes_to_print = ['ALB', 'HNF4A', 'HNF1B', 'CYP3A4', 'KRT19', 'KRT7', 'MT-RNR1', 'MT-RNR2', 'GPR85']

    def __init__(self, args, batch_id, filtered_matrix: MatrixFiltering, path_to_output: str):
        super().__init__() 

        self.args = args
        self.batch_id = batch_id
        self.path_to_output = path_to_output
        self.cluster_list = filtered_matrix.labels['treatment']
        self.stacked_matrix = filtered_matrix.imputed_r_mat
        self.reference_list = filtered_matrix.r_index
        self.filtered_matrix = filtered_matrix

    ###################
    ##### HEATMAP #####
    ###################

    def generate_matrix_heatmap(self):
        jointplot_label = {
            'y' : "Normalized Reactivity",
            'x' : "Coverage"
        }

        self.setup_row_colours()
        labels = self.filtered_matrix.labels.keys()
        for label in labels:
            cluster_label = self.filtered_matrix.labels[label]
            self.setup_col_colours(cluster_label) 

            analysis = 'Normalized_Reactivity'
            filename = f"{self.path_to_output}/{self.batch_id}.heatmap.{analysis}.{label}.png"
            matrix_to_print = self.filtered_matrix.r_mat.copy()
            
            # compute 95th percentile
            p95 = np.nanpercentile(matrix_to_print, 99)
            p5 = np.nanpercentile(matrix_to_print, 1)

            # clip values above 95th percentile
            r_matrix_to_print = np.clip(matrix_to_print, p5, p95)
            self.plot_heatmap_preimpute(r_matrix_to_print, filename, analysis)

            analysis = 'Coverage'
            filename = f"{self.path_to_output}/{self.batch_id}.heatmap.{analysis}.{label}.png"
            matrix_to_print = self.filtered_matrix.c_mat.copy()
            
            # compute 95th percentile
            p95 = np.nanpercentile(matrix_to_print, 99)

            # clip values above 95th percentile
            c_matrix_to_print = np.clip(matrix_to_print, None, p95)
            self.plot_heatmap_preimpute(c_matrix_to_print, filename, analysis)

            scatter_with_marginals(r_matrix_to_print, 
                                   c_matrix_to_print, 
                                   cluster_label, 
                                   label = jointplot_label, 
                                   filename = f"{self.path_to_output}/raw_reactivity_scatter.{label}.png")


    def plot_heatmap_preimpute(self, matrix, filename, title):
        """
        matrix          : 2D numpy array (rows=transcripts, cols=cells)
        filename        : output path
        label_every     : show every k-th non-empty gene label (1 = show all)
        """
        from matplotlib.transforms import blended_transform_factory

        # ---- colormap (paint NaNs black)
        cmap = plt.cm.coolwarm.copy()
        cmap.set_bad("black")

        # ---- build sparse y labels: only first row per gene (strip suffix after '-')
        genes = [gene.replace(f"-{gene.split('-')[-1]}", '') for gene in self.reference_list]

        seen = set()
        ylabels = []
        for g in genes:
            if g in seen:
                ylabels.append("")
            else:
                seen.add(g)
                ylabels.append(g)

        # ---- draw clustermap (no seaborn colorbar)
        g = sns.clustermap(
            matrix,
            col_colors=self.col_colors.to_numpy(),
            row_colors=self.row_colors.to_numpy(),
            cmap=cmap,
            row_cluster=False,
            col_cluster=False,
            figsize=(8, 6),
            cbar_pos=None
        )

        g.ax_heatmap.set_title(title, fontsize=14, pad=40)

        # remove ticks on heatmap axis
        g.ax_heatmap.set_xticks([])
        g.ax_heatmap.set_yticks([])

        # row-colors axis (normalize to a single Axes)
        row_ax = g.ax_row_colors
        if isinstance(row_ax, (list, tuple, np.ndarray)):
            row_ax = row_ax[0]

        # sync y-limits so text aligns to row centers (i + 0.5)
        y0, y1 = g.ax_heatmap.get_ylim()
        row_ax.set_ylim(y0, y1)


        # clean the strip panel
        row_ax.set_xticks([])

        n_rows, n_cols = matrix.shape
        row_ax.set_yticks(np.arange(n_rows) + 0.5)
        row_ax.tick_params(axis='y', width=0.3)

        for sp in row_ax.spines.values():
            sp.set_visible(False)

        # give modest room on the left for labels; leave room on the right for colorbar
        g.fig.subplots_adjust(left=0.22, right=0.92, top=0.97, bottom=0.05)

        # ---- external slim colorbar aligned to heatmap panel
        hb = g.ax_heatmap.get_position()          # bbox in figure coords
        pad = 0.008                               # small gap between heatmap and cbar
        cbar_w = 0.015                            # slim width
        cax = g.fig.add_axes([hb.x1 + pad, hb.y0, cbar_w, hb.height])

        mappable = g.ax_heatmap.collections[0]
        cb = g.fig.colorbar(mappable, cax=cax)    # vertical
        cb.ax.tick_params(labelsize=8)

        g.fig.savefig(filename, dpi=300)
        plt.close(g.fig)
        
    ######################
    ##### CLUSTERMAP #####
    ######################

    def plot_clustermap(self):
        corr_matrix, cluster_list = PlotResults._clean_matrix_pre_clusterplot(self.stacked_matrix, self.cluster_list)
        self.setup_col_colours(cluster_list)

        plt.figure(figsize=(8,5))
        g = sns.clustermap(corr_matrix, 
                           col_colors=self.col_colors.to_numpy(), 
                           cmap='coolwarm', 
                           figsize=(8, 8), 
                           row_cluster=True, 
                           col_cluster=True)

        # Remove axis ticks
        g.ax_heatmap.set_xticks([])
        g.ax_heatmap.set_yticks([])

        # Save figure before showing it
        plt.savefig(f"{self.path_to_output}/{self.batch_id}.clustermap.png", dpi=300, bbox_inches='tight')
        plt.close()  # good practice if running in a loop or notebook     

    
    @staticmethod
    def _clean_matrix_pre_clusterplot(stacked_matrix, cluster_list):
        # Compute correlation matrix between columns
        constant_rows = np.all(stacked_matrix == stacked_matrix[0, :], axis=0)
        print("Indices of constant rows:", np.nonzero(constant_rows)[0])

        corr_matrix = np.corrcoef(stacked_matrix, rowvar=False) 
        
        all_nan_cols = np.all(np.isnan(corr_matrix), axis=0)
        nan_col_indices = np.nonzero(all_nan_cols)[0]

        # Remove columns
        matrix_cleaned = np.delete(corr_matrix, all_nan_cols, axis=1)
        
        # Remove rows
        matrix_cleaned = np.delete(matrix_cleaned, all_nan_cols, axis=0)
        cluster_list = [item for i, item in enumerate(cluster_list) if i not in nan_col_indices]
        PlotResults._column_corrcoef(stacked_matrix)

        return matrix_cleaned, cluster_list


    @staticmethod
    def _column_corrcoef(data, check_nan=True):
        """
        Computes the correlation matrix between columns of a 2D NumPy array.
        
        Parameters:
        - data (np.ndarray): 2D array where columns are variables, rows are observations
        - check_nan (bool): whether to warn if NaNs are present (default True)

        Returns:
        - corr (np.ndarray): correlation matrix (columns x columns)
        """
        data = np.asarray(data)

        # Find columns with zero standard deviation
        constant_columns = (np.std(data, axis=0) == 0)

        # Number of constant columns
        num_constant_columns = np.sum(constant_columns)

        print(f"Number of constant columns: {num_constant_columns}")

        # Ensure data is 2D
        if data.ndim != 2:
            raise ValueError(f"Input data must be 2D, got shape {data.shape}")

        # Optional NaN check
        if check_nan and np.isnan(data).any():
            print("Warning: NaNs detected in data. Correlation matrix may contain NaNs.")


    #######################################
    ##### Helper Function For Colours #####
    #######################################

    def setup_col_colours(self, cluster_list):
        colours_list = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']

        unique_labels = []
        for lab in cluster_list:
            if lab not in unique_labels:
                unique_labels.append(lab)

        cmap_clusters = {}
        for c, lab in enumerate(unique_labels):
            cmap_clusters[lab] = colours_list[c]

            
        # cmap_clusters = {'C1':'#1f77b4', 'C2':'#ff7f0e', 'C3':'#2ca02c', 'C0':'#d62728'}
        cluster_colors = pd.Series(cluster_list, name='Cluster').map(cmap_clusters).fillna('#aaaaaa')
        
        library = pd.Series([self.batch_id]*len(cluster_list), name='Library')
        library_colors = library.map({self.batch_id:'#7f7f7f'}).fillna('#7f7f7f')

        # N x K DataFrame (rows = columns/samples of your matrix)
        col_colors = pd.concat([cluster_colors, library_colors], axis=1).T
        col_colors = col_colors.applymap(PlotResults.normalize_color)   # sanitize
        self.col_colors = col_colors  # keep as DataFrame

    @staticmethod
    def normalize_color(x, default='#aaaaaa'):
        # convert to a valid hex like '#rrggbb' or fall back to default
        if pd.isna(x):
            return default
        if isinstance(x, (tuple, list)):  # RGBA tuple
            try:
                return mcolors.to_hex(x)
            except Exception:
                return default
        s = str(x).strip()
        if HEX7.match(s) or mcolors.is_color_like(s):
            return mcolors.to_hex(s)
        return default

    def setup_row_colours(self, skip_row_color=False, txt=None):
        if skip_row_color:
            default_color = ClusterPlot.normalize_color("#808080")

            self.row_colors = pd.Series(
                [default_color] * len(self.reference_list),
                index=self.reference_list,
                name="Group"
            )
            self.handles = [Patch(facecolor=default_color, label="All")]
            return

        # Strip the suffix after "-" to get group name
        labels = pd.Series([gene.replace(f"-{gene.split('-')[-1]}", '') for gene in self.reference_list],
                            index=self.reference_list, name='Group')
        
        def big_tab_palette(n):
            cols = []
            for name in ["tab20", "tab20b", "tab20c"]:
                cmap = cm.get_cmap(name)
                cols.extend([mcolors.to_hex(cmap(i)) for i in range(cmap.N)])
            return cols[:n] if n <= len(cols) else cols

        def big_distinct_palette(n, sat=0.70, val=0.95):
            # 1) Start with curated qualitative sets
            base = []
            for name in ["tab20", "tab20b", "tab20c"]:
                cmap = cm.get_cmap(name)
                base.extend([mcolors.to_hex(cmap(i)) for i in range(cmap.N)])

            # 2) If more needed, add golden-angle HSV hues
            if n > len(base):
                need = n - len(base)
                phi = 0.6180339887498949  # golden ratio frac
                h = 0.0
                extras = []
                for _ in range(need):
                    h = (h + phi) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(h, sat, val)
                    extras.append(mcolors.to_hex((r, g, b)))
                base.extend(extras)

            # 3) Sort by hue for nicer spacing (optional)
            def hue_key(hexcol):
                r, g, b = mcolors.to_rgb(hexcol)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                return h
            base_sorted = sorted(base[:n], key=hue_key)
            return base_sorted

        unique_groups = sorted(labels.unique())
        palette = big_distinct_palette(len(unique_groups))
        lut = dict(zip(unique_groups, palette))

        row_colors = labels.map(lut)
        row_colors = row_colors.map(PlotResults.normalize_color)        # sanitize
        self.row_colors = row_colors
        self.handles = [Patch(facecolor=lut[g], label=g) for g in lut]
        if len(self.handles) > 1 :
            self.plot_group_legend(txt=txt)

    
    def plot_group_legend(self, txt=None, ncol=3, title="Genes & Colors"):
        fig, ax = plt.subplots(figsize=(5, min(5, len(self.handles)//ncol)))
        ax.axis("off")
        ax.legend(
            handles=self.handles,
            loc="center",
            ncol=ncol,
            frameon=False,
            columnspacing=1.6,
            handlelength=1.8,
            handletextpad=0.6,
            borderaxespad=0.2,
            title=title,
        )
        # plt.tight_layout()
        txt = '' if not txt else f'_{txt}'
        plt.savefig(f"{self.path_to_output}/{self.batch_id}.colour_label{txt}.png", dpi=300, bbox_inches='tight')
        plt.close()  