#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_completeness_v_measure, silhouette_score,
                             confusion_matrix)

from scipy.optimize import linear_sum_assignment
from cluster_cells_plot import PlotResults
from pathlib import Path


def plot_unsupervised_clustering(method, pca_components, y_pred, umap_df, location, var):
    # 1) KMeans on PCs
    K = umap_df['Cluster'].nunique()

    # 2) Align to ground-truth label space for readable comparison
    y_true_raw = umap_df['Cluster'].to_numpy()
    le = LabelEncoder().fit(y_true_raw)
    y_true = le.transform(y_true_raw)
    y_pred_relab, _ = relabel_to_match(y_true, y_pred)
    y_pred_names = le.inverse_transform(y_pred_relab)

    # 3) Metrics + confusion matrix
    ari = adjusted_rand_score(y_true, y_pred_relab)
    nmi = normalized_mutual_info_score(y_true, y_pred_relab)
    cm  = confusion_matrix(y_true, y_pred_relab, labels=np.arange(K))

    print(f"ARI: {ari:.3f}   NMI: {nmi:.3f}")
    print("Confusion matrix (truth rows x predicted cols in truth order):\n", cm)

    # 4) Prepare a plotting frame
    plot_df = umap_df.copy()
    plot_df['Truth'] = y_true_raw
    plot_df['Pred']  = y_pred_names
    plot_df['Match'] = plot_df['Truth'] == plot_df['Pred']

    # 5) Color (truth) + Shape (pred)
    truth_order = list(np.unique(plot_df['Truth']))
    palette = sns.color_palette("Set2", n_colors=len(truth_order))
    color_map = dict(zip(truth_order, palette))

    pred_order = list(np.unique(plot_df['Pred']))
    markers = ['o','s','^','D','P','X','v','<','>','*']  # will cycle if >10
    marker_map = {lab: markers[i % len(markers)] for i, lab in enumerate(pred_order)}

    _, ax = plt.subplots(figsize=(7,5))

    for pred_lab in pred_order:
        sub = plot_df[plot_df['Pred'] == pred_lab]
        ax.scatter(sub['UMAP1'], sub['UMAP2'],
                c=sub['Truth'].map(color_map),
                marker=marker_map[pred_lab],
                s=30, alpha=0.9, edgecolor='none', label=f"pred {pred_lab}")

    # Halo the mismatches
    mm = plot_df[~plot_df['Match']]
    if not mm.empty:
        ax.scatter(mm['UMAP1'], mm['UMAP2'],
                facecolors='none', edgecolors='k', s=70, linewidth=1.0, label='Mismatch')

    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"Truth (color) vs {method} (shape) on UMAP For {pca_components} PCs ({var*100:.2f}% Variance) \nARI: {ari:.3f}, NMI: {nmi:.3f}")

    # Two legends: colors for Truth, shapes for Pred
    color_handles = [Line2D([0],[0], marker='o', color='w',
                            markerfacecolor=color_map[lab], markersize=8, label=str(lab))
                    for lab in truth_order]
    shape_handles = [Line2D([0],[0], marker=marker_map[lab], linestyle='None', color='k',
                            markersize=8, label=str(lab))
                    for lab in pred_order]

    leg1 = ax.legend(handles=color_handles, title="Truth (color)", loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, title=f"{method} (shape)", loc='lower left', bbox_to_anchor=(1.01, 0.0))
    plt.tight_layout()
    plt.savefig(f"{location}.unsupervised.{method}.png", dpi=300, bbox_inches='tight')
    plt.close()  # good practice if running in a loop or notebook


# ---- helper: relabel predicted clusters to best match ground truth for display ----
def relabel_to_match(y_true_int, y_pred):
    """Return y_pred_relab where cluster IDs are permuted to maximize matches with y_true_int."""
    y_pred = np.asarray(y_pred)
    up, ut = np.unique(y_pred), np.unique(y_true_int)
    # Build cost matrix as (max_count - overlap) so Hungarian solves a min problem
    C = np.zeros((len(up), len(ut)), dtype=int)
    for i, p in enumerate(up):
        for j, t in enumerate(ut):
            C[i, j] = np.sum((y_pred == p) & (y_true_int == t))
    # convert to cost
    maxc = C.max() if C.size else 0
    cost = maxc - C
    ri, cj = linear_sum_assignment(cost)
    mapping = {up[i]: ut[j] for i, j in zip(ri, cj)}
    y_pred_relab = np.array([mapping.get(v, v) for v in y_pred])
    return y_pred_relab, mapping


# ---- helper: safe silhouette (requires >=2 clusters and no noise label -1) ----
def safe_silhouette(X, labels):
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or np.any(labels == -1):
        return np.nan
    # each cluster must have at least 2 samples
    if any((labels == u).sum() < 2 for u in uniq):
        return np.nan
    return float(silhouette_score(X, labels))


def save_cluster_assignments_with_truth(
    barcode_list,
    y_pred,
    y_true,
    method_name,
    location
):

    df = pd.DataFrame({
        "barcode": barcode_list,
        "predicted": y_pred,
        "true": y_true,
    })

    # match column: True if correct prediction
    df["match"] = df["predicted"] == df["true"]

    outfile=f"{location}.{method_name}_cluster_assignments.csv"

    df.to_csv(outfile, index=False)


def evaluate_clusterings(cluster_list, 
                         barcode_list,
                         pca_components,
                         var,
                         location='',
                         umap_df=None, *, n_neighbors_spectral=15):
    """
    pca_selected : array-like (n_samples, n_pcs)  <-- use this for clustering
    cluster_list : array-like of ground-truth labels (strings or ints)
    umap_df      : optional DataFrame with columns ['UMAP1','UMAP2'] for visualization
    """
    X = umap_df[['UMAP1','UMAP2']] 
    y_true_raw = np.asarray(cluster_list)
    le = LabelEncoder()
    y_true = le.fit_transform(y_true_raw)         # ints 0..K-1
    K = len(np.unique(y_true))

    results = []
    label_sets = {}  # store predicted labels for later visualization

    algos = {
        "kmeans":        lambda: KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X),
        "gmm":           lambda: GaussianMixture(n_components=K, random_state=42).fit(X).predict(X),
        "agglomerative": lambda: AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X),
        "spectral":      lambda: SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                                                    n_neighbors=n_neighbors_spectral,
                                                    assign_labels="kmeans", random_state=42).fit_predict(X),
        # density-based (no K required). You may tune eps/min_samples.
        "dbscan":        lambda: DBSCAN(eps=0.8, min_samples=10).fit_predict(X),
    }

    for name, fn in algos.items():
        try:
            y_pred = fn()
        except Exception as e:
            results.append({"method": name, "error": str(e)})
            continue

        if location and name in ["kmeans", "gmm", "spectral"]:
            print(f"Plotting clustering results for {name}...")
            plot_unsupervised_clustering(name, pca_components, y_pred, umap_df, location, var)
            save_cluster_assignments_with_truth(
                barcode_list,
                y_pred,
                y_true,
                method_name=name,
                location=location
            )

        # External metrics (ignore label identities)
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)

        # Internal metric (on the same X you clustered)
        sil = safe_silhouette(X, y_pred)

        # For readable confusion matrix, relabel to match ground truth as best as possible
        y_pred_relab, mapping = relabel_to_match(y_true, y_pred)

        # Save
        label_sets[name] = {
            "raw": y_pred,
            "relabeled_to_truth_space": y_pred_relab,
            "mapping_to_truth_ids": mapping
        }

        # Confusion matrix in truth ID space (0..K-1), can convert to original labels later
        cm = confusion_matrix(y_true, y_pred_relab, labels=np.arange(K))

        results.append({
            "method": name,
            "n_clusters_found": int(len(np.unique(y_pred[y_pred != -1]))) if np.any(y_pred != -1) else int(len(np.unique(y_pred))),
            "ARI": float(ari),
            "NMI": float(nmi),
            "Homogeneity": float(h),
            "Completeness": float(c),
            "V_measure": float(v),
            "Silhouette_on_PCs": sil,
            "has_noise_label": bool(np.any(y_pred == -1)),
            "confusion_matrix_(truth_rows_pred_cols)": cm
        })

    df = pd.DataFrame(results)

    # For readability, convert confusion matrices to string; you can keep numpy arrays if you prefer
    def pretty_cm(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    if "confusion_matrix_(truth_rows_pred_cols)" in df.columns:
        df["confusion_matrix_(truth_rows_pred_cols)"] = df["confusion_matrix_(truth_rows_pred_cols)"].map(pretty_cm)

    return df, label_sets, le


class PlotResultsClustering(PlotResults): 

    def initialize_pca(self):
        rc_mat = self.filtered_matrix.rc_mat
        gene_list = self.filtered_matrix.gene_names

        gene_count = np.sum(
            (np.isfinite(rc_mat)) & (rc_mat != 0),
            axis=0
        )

        read_sums = np.nansum(rc_mat, axis=0)
        p95 = np.nanpercentile(read_sums, 99)
        read_sums = np.clip(read_sums, None, p95)
        to_plot = {
            'GeneCount' : gene_count,
            'ReadCount' : read_sums,
            'Percent.MT' : self.filtered_matrix.percent_mt
        }

        genes_index = {
            gene: gene_list.index(gene)
            for gene in self.genes_to_print
            if gene in gene_list
        }

        genes_to_plot = { 
            gene : rc_mat[i] for gene, i in genes_index.items() 
        }

        genes_to_plot = {
            gene: np.clip(values, None, np.nanpercentile(values, 95))
            for gene, values in genes_to_plot.items()
        }

        to_plot.update(genes_to_plot)
        to_plot.update(self.filtered_matrix.labels)

        to_plot, matrix_cleaned = self.sanitise_matrix(to_plot)
        self.pca_matrix = matrix_cleaned
        self.to_plot_umap = to_plot
        self.to_plot_pca = list(self.filtered_matrix.labels.keys()) + ['ReadCount', 'Percent.MT']


    def run_pca(self, unsupervised_clustering = False):
        from sklearn.decomposition import PCA

        self.pca = PCA(random_state=42)
        self.pca_components = self.pca.fit_transform(self.pca_matrix)
        
        self.print_principal_components()

        for key in self.to_plot_pca:
            values = self.to_plot_umap[key]
            self.plot_pca(values, hue=key)

        for i in range(1,5):
            self.plot_pca_loadings(pc=i, top_n=25)

        if unsupervised_clustering:
            self.unsupervised_clustering()

        for n_components in [5, 10, 20, 40]:
            pca_selected = self.pca_components[:, :n_components]
            var = np.sum(self.pca.explained_variance_ratio_[:n_components])
            self.plot_umap(pca_selected, n_components, round(var*100, 2))

    
    def plot_umap(self, pca_selected, n_components_80, percent):
        import umap

        # UMAP those components
        reducer = umap.UMAP(
            n_neighbors=30,     # change this value as needed
            min_dist=0.2,       # change this value as needed
            n_components=2,     # change the output dimension of the embedding (e.g. 2D or 3D)
            random_state=42
        )

        umap_embedding = reducer.fit_transform(pca_selected)

        for analysis, list_to_plot in self.to_plot_umap.items():
            s = pd.Series(list_to_plot)
            is_numeric = pd.to_numeric(s, errors="coerce").notna().all()

            n_umap = umap_embedding.shape[0]
            n_vals = len(list_to_plot)
            print("analysis:", analysis, "n_umap:", n_umap, "n_vals:", n_vals)

            # hard fail if mismatch
            assert n_vals == n_umap, f"{analysis}: length mismatch! embedding={n_umap}, vals={n_vals}"

            umap_df = pd.DataFrame({
                'UMAP1': umap_embedding[:,0],
                'UMAP2': umap_embedding[:,1],
                analysis : list_to_plot
            })

            # Plot UMAP
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(
                data=umap_df,
                x='UMAP1',
                y='UMAP2',
                hue=analysis,            # numeric column
                palette='viridis',       # continuous colormap
                legend= not is_numeric,            # remove categorical legend
                ax=ax,
                s=15
            )

            if is_numeric: 
                # Add colorbar manually
                norm = plt.Normalize(
                    umap_df[analysis].min(),
                    umap_df[analysis].max()
                )

                sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm) 
                sm.set_array([]) 
                fig.colorbar(sm, ax=ax, label=analysis)

            else:
                ax.legend(
                    title=analysis,
                    bbox_to_anchor=(1.02, 1),   # x slightly >1 moves outside
                    loc="upper left",
                    borderaxespad=0
                )
                            
            plt.title(f"{analysis} UMAP from top {n_components_80} PCA components (≥{percent}% variance) for {self.batch_id}")            
            plt.savefig(f"{self.path_to_output}/{self.batch_id}.{analysis}.{str(int(percent))}var.umap.png", dpi=300, bbox_inches='tight')
            plt.close() 


    def unsupervised_clustering(self):
        cluster_list = self.to_plot_umap['treatment']

        for n_components in [5, 10, 20, 40]:
            self.print_features_per_component(n_components)
            pca_selected = self.pca_components[:, :n_components]
            var = np.sum(self.pca.explained_variance_ratio_[:n_components])
            self.check_clustering(pca_selected, cluster_list, n_components, var)

    
    def check_clustering(self, pca_selected, cluster_list, pca_components, var):
        import umap

        # UMAP those components
        reducer = umap.UMAP(
            n_neighbors=30,     # change this value as needed
            min_dist=0.2,       # change this value as needed
            n_components=2,     # change the output dimension of the embedding (e.g. 2D or 3D)
            random_state=42
        )

        umap_embedding = reducer.fit_transform(pca_selected)

        umap_df = pd.DataFrame({
            'UMAP1': umap_embedding[:,0],
            'UMAP2': umap_embedding[:,1],
            'Cluster': cluster_list
        })
        
        path_to_output = f'{self.path_to_output}/{pca_components}'
        Path(path_to_output).mkdir(parents=True, exist_ok=True)
        file_prefix = f'{path_to_output}/{self.batch_id}.pca'
        scores, labels_by_method, label_encoder = evaluate_clusterings(
            cluster_list, self.filtered_matrix.bc_index, pca_components, var, umap_df=umap_df, location=file_prefix,
        )

        # print(scores.sort_values("ARI", ascending=False))
        # To get the best method’s relabeled predictions:
        best_method = scores.sort_values("ARI", ascending=False).iloc[0]["method"]
        y_pred_best = labels_by_method[best_method]["relabeled_to_truth_space"]            # integers in truth space
        y_pred_best_names = self.safe_inverse_transform(label_encoder, y_pred_best) 

        print (best_method)
        print (y_pred_best)
        print (y_pred_best_names)

        scores.to_csv(f'{self.path_to_output}/{pca_components}/preds.csv', index=False)
    
    def safe_inverse_transform(self, le, arr, fill_value="Unknown"):
        arr = np.asarray(arr)
        known = np.isin(arr, le.classes_)
        output = np.full(arr.shape, fill_value, dtype=object)
        output[known] = le.inverse_transform(arr[known])
        return output


    def print_features_per_component(self, n_components):
        with open(f'{self.path_to_output}/pca_top_features_per_component.{n_components}.txt', 'w') as write_file:
            write_file.write("PC\tPosition\tLoading\tContribution\n")
            for pc in range(n_components):  # first 10 PCs
                top_features_idx = self.cumulative_contribution(pc, cutoff=0.80)
                
                for pos, load, contrib in top_features_idx:
                    write_file.write(f"PC{pc+1}\t{pos}\t{load:.4f}\t{contrib:.4f}\n")

        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            index=self.reference_list,            # feature names
            columns=[f"PC{i+1}" for i in range(self.pca.n_components_)]
        )

        ## Sort By PC1 and PC2 loadings and save
        loadings_df = loadings_df.reindex(loadings_df["PC1"].abs().sort_values(ascending=False).index)
        loadings_df.to_csv(f'{self.path_to_output}/pca_loadings.pc1.csv')

        loadings_df = loadings_df.reindex(loadings_df["PC2"].abs().sort_values(ascending=False).index)
        loadings_df.to_csv(f'{self.path_to_output}/pca_loadings.pc2.csv')

    
    def cumulative_contribution(self, pc, cutoff=0.80):
        loadings = self.pca.components_[pc]
        contrib = loadings**2

        # Sort descending by contribution
        sorted_idx = np.argsort(contrib)[::-1]
        cum = np.cumsum(contrib[sorted_idx])

        # Select indices where cumulative <= cutoff
        cutoff_mask = cum <= cutoff
        selected_sorted_idx = sorted_idx[cutoff_mask]

        selected_features = [
            (self.reference_list[i], loadings[i], contrib[i])
            for i in selected_sorted_idx
        ]

        return selected_features


    def plot_pca_loadings(self, pc=1, top_n=20, figsize=(8,6)):
        """
        Plot PCA loadings for a selected principal component.

        Parameters
        ----------
        pca : fitted sklearn PCA object
            Must have pca.components_ and pca.explained_variance_ratio_.
        feature_names : list or array of strings, optional
            Names of original features (e.g., positions or genes).
            If None, they will be numbered 0..n_features-1.
        pc : int (1-indexed)
            Which principal component to visualize (PC1 = 1, PC2 = 2, etc.)
        top_n : int
            Number of top absolute loadings to show.
        figsize : tuple
            Size of the matplotlib figure.
        """

        pc_idx = pc - 1  # convert to 0-index
        loadings = self.pca.components_[pc_idx]

        df = pd.DataFrame({
            "feature": self.reference_list,
            "loading": loadings,
            "abs_loading": np.abs(loadings)
        })

        df = df.sort_values("abs_loading", ascending=False).head(top_n)

        _, ax = plt.subplots(figsize=figsize)
        ax.barh(df["feature"], df["loading"], color='steelblue')
        ax.set_title(f"PCA Loadings for PC{pc}  (explains {self.pca.explained_variance_ratio_[pc_idx]*100:.2f}%)")
        ax.set_xlabel("Loading weight")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()  # most important features on top
        plt.tight_layout()
        plt.savefig(f"{self.path_to_output}.pca_loadings.PC{pc}.top{top_n}.png", dpi=300, bbox_inches='tight')
        plt.close()  # good practice if running in a loop or notebook


    def plot_pca(self, cluster_list, hue='Cluster'):
        # Plot PCA scatterplot
        is_numeric = np.isfinite(pd.to_numeric(cluster_list, errors="coerce")).all()
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x=self.pca_components[:, 0],
                        y=self.pca_components[:, 1],
                        hue=cluster_list,
                        legend= not is_numeric,
                        palette='Set2',  # or your custom palette
                        s=40, alpha=0.6, ax=ax)
        
        
        if is_numeric: 
            # Add colorbar manually
            norm = plt.Normalize(
                np.array(cluster_list).min(),
                np.array(cluster_list).max()
            )

            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm) 
            sm.set_array([]) 
            fig.colorbar(sm, ax=ax, label=hue)

        else:
            ax.legend(
                title=hue,
                bbox_to_anchor=(1.02, 1),   # x slightly >1 moves outside
                loc="upper left",
                borderaxespad=0
            )

        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'PCA plot for {self.batch_id}')

        plt.legend()

        # Save figure before showing it
        plt.savefig(f"{self.path_to_output}/{self.batch_id}.{hue}.pca.png", dpi=300, bbox_inches='tight')
        plt.close()  # good practice if running in a loop or notebook


    def print_principal_components(self):
        # Variance explained (ratio per component)
        explained_var = self.pca.explained_variance_ratio_

        # Convert to percentage
        explained_var_percent = explained_var * 100

        # Optional: put into a dataframe for easy viewing / saving
        df_pca_var = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(explained_var))],
            'Variance (%)': explained_var_percent
        })

        X = 20  # number of components you want
        df_pca_var.head(X).to_csv(f'{self.path_to_output}/pca_variance_first_{X}_components.csv', index=False)
        
        n_components = self.pca_components.shape[1]

        df_scores = pd.DataFrame(
            self.pca_components,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )

        barcode_list = self.filtered_matrix.bc_index
        df_scores.insert(0, "barcode", barcode_list)   # add barcode column first
        df_scores.to_csv(f'{self.path_to_output}/pca_scores_per_cell.csv', index=False)
        print("Saved pca_scores_per_cell.csv")
    

    def sanitise_matrix(self, to_plot, to_clip=False, winsorize=False):
        all_nan_cols = np.all(self.filtered_matrix.imputed_r_mat == 0, axis=0)
        nan_col_indices = np.nonzero(all_nan_cols)[0]
        
        # Remove columns
        matrix_cleaned = np.delete(self.filtered_matrix.imputed_r_mat, all_nan_cols, axis=1)

        if winsorize:
            from normalization_utils import winsorized_normalization 
            matrix_cleaned = winsorized_normalization(matrix_cleaned.T, to_scale = False).T

        if to_clip:
            matrix_cleaned = np.clip(matrix_cleaned,
                             np.nanpercentile(matrix_cleaned),
                             np.nanpercentile(matrix_cleaned, 99))

        matrix_cleaned = matrix_cleaned.T

        for key, value in to_plot.items():
            to_plot[key] = [item for i, item in enumerate(value) if i not in nan_col_indices]

        return to_plot, matrix_cleaned