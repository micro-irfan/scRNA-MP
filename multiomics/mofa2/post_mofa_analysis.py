#!/usr/bin/env python3

import mofax as mfx
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import pandas as pd
import seaborn as sns


def save_plot(plot_func, filename, *args, dpi=300, close=True, verbose=True, add_title=None, **kwargs):
    """
    Run a plotting function and save the figure.

    Parameters
    ----------
    plot_func : callable
        Plotting function (e.g. mfx.plot_r2)
    filename : str
        Output file name
    *args :
        Arguments passed to the plotting function
    dpi : int
        Resolution of the saved figure
    close : bool
        Close figure after saving
    **kwargs :
        Keyword arguments passed to the plotting function
    """
    try:
        plt.figure(figsize=(7, 6))
        plot_func(*args, **kwargs)
        if add_title:
            plt.title(add_title)
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"[OK] Saved: {filename}")

    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to save plot: {filename}")
            print(f"Function: {plot_func.__name__}")
            print(f"Error: {e}")
            traceback.print_exc()
    
    finally:
        if close:
            plt.close()


def plotting(m, number_of_factors, location):

    # mfx.plot_r2(m, factors=list(range(10)), cmap="Blues")
    filename = f"{location}/plot_r2.png"
    save_plot(
        mfx.plot_r2,
        filename,
        m,
        factors=list(range(10)), cmap="Blues"
    )

    # mfx.plot_r2(m, factors=list(range(5)), cmap="Blues", group_label="treatment")
    filename = f"{location}/plot_r2.treatment.png"
    save_plot(
        mfx.plot_r2,
        filename,
        m,
        factors=list(range(number_of_factors)), cmap="Blues", group_label="treatment"
    )

    filename = f"{location}/plot_r2.volume.png"
    save_plot(
        mfx.plot_r2,
        filename,
        m,
        factors=list(range(number_of_factors)), cmap="Blues", group_label="volume"
    )


    filename = f"{location}/plot_r2_barplot.treatment.png"
    save_plot(
        mfx.plot_r2_barplot,
        filename,
        m,
        factors=list(range(number_of_factors)), 
        x="Group", groupby="Factor", group_label="treatment", palette="winter"
    )

    filename = f"{location}/plot_r2_pvalues.treatment.png"
    save_plot(
        mfx.plot_r2_pvalues,
        filename,
        m,
        factors=list(range(number_of_factors)), 
        group_label="treatment",
        n_iter=10,
    )

    filename = f"{location}/plot_factors_scatter.treatment.png"
    save_plot(
        mfx.plot_factors_scatter,
        filename,
        m,
        x=0, y=range(1, number_of_factors), 
        size=8, alpha=.75, 
        color="treatment", 
        legend='brief', ncols=2,
    )

    filename = f"{location}/plot_factors_scatter.individual.png"
    save_plot(
        mfx.plot_factors_scatter,
        filename,
        m,
        x="treatment", y=range(number_of_factors), group_label="treatment", color='treatment',
        alpha=.1,
        rotate_x_labels=90, ncols=2,
        
    )

    filename = f"{location}/plot_weights.png"
    save_plot(
        mfx.plot_weights,
        filename,
        m,
        n_features=5,
    )

    for view in ['Reactivity', 'RNA', 'Splicing']:
        filename = f"{location}/plot_weights_heatmap.{view}.png"
        save_plot(
            mfx.plot_weights_heatmap,
            filename,
            m,
            n_features=5,
            view=view,
            factors=range(number_of_factors), 
            xticklabels_size=6, w_abs=True,
            cmap="Greys", cluster_factors=True
        )

    filename = f"{location}/plot_weights_correlation.png"
    save_plot(
        mfx.plot_weights_correlation,
        filename,
        m
    )

    
    filename = f"{location}/plot_factors_matrix.treatment.png"
    save_plot(
        mfx.plot_factors_matrix,
        filename,
        m,
        agg="mean", factors=list(range(number_of_factors)), 
        linewidths=0.01, linecolor="#FFFFFF33",
        vmax=10,
        group_label="treatment"
    )

    filename = f"{location}/plot_factors_correlation.png"
    save_plot(
        mfx.plot_factors_correlation,
        filename,
        m,
        add_title="Pearson r",
    )

    covariates = pd.get_dummies(m.metadata.treatment)
    filename = f"{location}/plot_factors_covariates_correlation.png"
    save_plot(
        mfx.plot_factors_covariates_correlation,
        filename,
        m,
        add_title="Pearson r",
        covariates=covariates
    )

    filename = f"{location}/plot_factors_covariates_correlation.pval.png"
    save_plot(
        mfx.plot_factors_covariates_correlation,
        filename,
        m,
        covariates=covariates,
        pvalues=True, cmap=sns.light_palette("#FF0000")
    )


def get_factors(m):
    r2 = m.get_r2()
    print(r2.groupby("View")["R2"].sum())
    print(r2.pivot(index="Factor", columns="View", values="R2"))

    pivot = r2.pivot(index="Factor", columns="View", values="R2").fillna(0)

    # keep factors where non-RNA contribution is meaningful
    keep = pivot.index[
        (pivot[["Reactivity", "Splicing"]].sum(axis=1) > 0.8) &
        (pivot["RNA"] < 8)
    ].tolist()

    if len(keep) < 2:
        keep = pivot.index[
            (pivot[["Reactivity", "Splicing"]].sum(axis=1) > 0.8)
        ].tolist()

    keep = [int(i.replace('Factor', '')) - 1 for i in keep]
    print("Keeping factors:", keep)
    return keep


def run_umap(m, number_of_factors, location, factors=None):
    if not factors:
        factors = list(range(number_of_factors))
        txt = 'all_factors'
    else:
        txt = f'{"-".join([str(i) for i in factors])}_factors'
        if len(factors) == 1:
            return

    m.run_umap(factors=factors)
    plot_df = m.samples_metadata

    for color_col in ['treatment', 'volume']:
        # Plot
        plt.figure(figsize=(7, 6))
        if color_col is not None and color_col in plot_df.columns:
            for val, subdf in plot_df.groupby(color_col):
                plt.scatter(subdf["UMAP1"], subdf["UMAP2"], s=10, label=val)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(plot_df["UMAP1"], plot_df["UMAP2"], s=10)

        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title("UMAP from MOFA latent factors")
        plt.tight_layout()

        filename = f"{location}/umap.{txt}.{color_col}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    columns_to_drop = ['UMAP1', 'UMAP2']
    m.samples_metadata.drop(columns=columns_to_drop, inplace=True)


def open_model(filename):
    m = mfx.mofa_model(filename)

    ## Add Treatment And Volume to Meta Data
    idx = m.metadata.index.to_series()
    m.metadata["treatment"] = idx.str.extract(r"^(DM|DMS)_", expand=False)
    m.metadata["volume"] = idx.str.extract(r"_(\din\d)__", expand=False)

    print(f"""\
    Cells: {m.shape[0]}
    Features: {m.shape[1]}
    Groups of cells: {', '.join(m.groups)}
    Views: {', '.join(m.views)}
    """)

    print (m.get_r2(factors=list(range(10)))) # .sort_values("R2", ascending=False))
    
    return m


workdir = '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline'
cellline = 'results_3'

def pipeline():
    # combined_15000_removePoorCells _ impute _fsb_psi7_2000g_scale combined_removePoorCells_impute
    batch_id = 'combined_15000_removePoorCells'
    psi_reactivity = False 
    txt = '.psi_reactivity' if psi_reactivity else ''

    # 3in4_removePoorCells_impute_sb_psi7
    to_append = "_sb_psi7_1000g_Zscale" # "_fsb_psi7_2000g_scale"
    location = f'{workdir}/{cellline}/mofa_results/{batch_id}_impute{to_append}'
    filename = f"{location}/mofa_model{txt}.hdf5"

    m = open_model(filename)

    df_factor = m.get_weights(df=True) 
    number_of_factors = len(df_factor.columns.to_list())
    if number_of_factors < 2: 
        print ("Number of Factors Too Small!")
        return 

    path_to_output = f'{location}/analysis{txt}'
    Path(path_to_output).mkdir(parents=True, exist_ok=True)
    plotting(m, number_of_factors, path_to_output)

    run_umap(m, number_of_factors, path_to_output)
    
    keep = get_factors(m)
    run_umap(m, number_of_factors, path_to_output, factors=keep)
    # run_umap(m, number_of_factors, path_to_output, factors=[1,3])

    from mofa_support import run_mofa_treatment_pipeline, run_mofa_contribution_pipeline
    factors_df, merged_df, stats_df, weights_dict = run_mofa_treatment_pipeline(
        m,
        metadata=None,               # or pass your own DataFrame
        treatment_col="treatment",
        outdir=path_to_output,
        alpha=0.05,
        plot_all_factors=False,     # only significant factors
        max_factors_to_plot=None,
        top_n_weights=20,
        top_n_tables=50
    )

    print(stats_df.head(10))

    res = run_mofa_contribution_pipeline(
        m,
        metadata=None,                  # or your metadata DataFrame
        treatment_col="treatment",
        outdir=path_to_output,
        normalization="zscore",         # "zscore", "minmax", or "none"
        alpha=0.05,
        top_n_bar=20,
        top_n_grouped=12,
        n_neighbors=15,
        n_pcs=20
    )

    print(res)


if __name__ == "__main__":
    pipeline()