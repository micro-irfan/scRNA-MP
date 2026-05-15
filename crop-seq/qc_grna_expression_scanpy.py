import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_target_gene(grna_name):
    if "_sg" in grna_name:
        target = grna_name.split("_sg", 1)[0]
    else:
        target = grna_name

    # POSCON guides are prefixed (e.g., POSCON_DBR1_sg1); expression matrix uses the gene symbol.
    if target.upper().startswith("POSCON_"):
        target = target.split("_", 1)[1]

    return target


def find_expression_tsv(expression_root, sample_id):
    sample_dir = Path(expression_root) / sample_id / "expression"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Expression directory not found: {sample_dir}")

    matches = sorted(sample_dir.glob("*_exp_fastp.tsv"))
    if not matches:
        raise FileNotFoundError(f"No expression TSV found in: {sample_dir}")
    return matches[0]


def load_singlet_assignments(singlet_csv, sample_id):
    df = pd.read_csv(singlet_csv)
    if "gRNA" not in df.columns:
        raise ValueError(f"'gRNA' column not found in: {singlet_csv}")
    if sample_id not in df.columns:
        raise ValueError(f"Sample '{sample_id}' not found in: {singlet_csv}")

    barcode_to_grna = {}
    duplicate_conflicts = []

    for _, row in df.iterrows():
        grna = str(row["gRNA"]).strip()
        if not grna:
            continue

        barcodes_raw = row[sample_id]
        if pd.isna(barcodes_raw):
            continue

        for barcode in str(barcodes_raw).split(";"):
            bc = barcode.strip()
            if not bc:
                continue

            prev = barcode_to_grna.get(bc)
            if prev is not None and prev != grna:
                duplicate_conflicts.append((bc, prev, grna))
                continue
            barcode_to_grna[bc] = grna

    if duplicate_conflicts:
        print(
            f"Warning: {len(duplicate_conflicts)} conflicting barcode assignments "
            "found; keeping first assignment for each barcode."
        )

    assign_df = pd.DataFrame(
        {
            "barcode": list(barcode_to_grna.keys()),
            "gRNA": list(barcode_to_grna.values()),
        }
    )
    assign_df["target_gene"] = assign_df["gRNA"].map(parse_target_gene)
    return assign_df


def build_anndata_from_expression(expression_tsv, assignments_df):
    import scanpy as sc

    expr_df = pd.read_csv(expression_tsv, sep="\t", index_col=0)
    expr_df.index = expr_df.index.astype(str)
    expr_df.columns = expr_df.columns.astype(str)

    assignment_indexed = assignments_df.set_index("barcode")
    overlap = [bc for bc in expr_df.index if bc in assignment_indexed.index]
    if not overlap:
        raise RuntimeError("No overlapping barcodes between assignments and expression matrix.")

    expr_sub = expr_df.loc[overlap].copy()
    obs = assignment_indexed.loc[overlap].copy()

    adata = sc.AnnData(expr_sub)
    adata.obs = obs

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def extract_gene_vector(adata, gene):
    if gene not in adata.var_names:
        return None
    x = adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).ravel()


def gene_group_label(target_gene, gene, negcon_gene):
    if target_gene == gene:
        return f"targeting_{gene}"
    if target_gene == negcon_gene:
        return "non_targeting_NEGCON"
    return "other_targeting"


def plot_gene_panel_and_summarize(
    adata,
    genes,
    negcon_gene,
    min_cells_per_group,
    output_plot,
    overlay_points=True,
    overlay_sgrna_means=True,
    title_suffix_by_gene=None,
):
    import matplotlib.pyplot as plt

    summary_rows = []
    valid_genes = []

    n_genes = len(genes)
    n_cols = 2
    n_rows = int(np.ceil(n_genes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, gene in enumerate(genes):
        ax = axes[i]
        expr_vec = extract_gene_vector(adata, gene)
        if expr_vec is None:
            summary_rows.append(
                {
                    "gene": gene,
                    "status": "skipped_gene_not_found_in_expression_matrix",
                    "n_targeting": 0,
                    "n_negcon": 0,
                    "n_other_targeting": 0,
                    "mean_targeting": np.nan,
                    "mean_negcon": np.nan,
                    "mean_other_targeting": np.nan,
                    "median_targeting": np.nan,
                    "median_negcon": np.nan,
                    "median_other_targeting": np.nan,
                    "delta_mean_target_minus_negcon": np.nan,
                    "mannwhitney_pvalue_target_vs_negcon": np.nan,
                }
            )
            ax.axis("off")
            ax.set_title(f"{gene} (not found)")
            continue

        tmp = adata.obs.copy()
        tmp["expr"] = expr_vec
        tmp["group"] = tmp["target_gene"].map(
            lambda g: gene_group_label(g, gene, negcon_gene)
        )

        order = [f"targeting_{gene}", "non_targeting_NEGCON", "other_targeting"]
        groups = {grp: tmp.loc[tmp["group"] == grp, "expr"].values for grp in order}

        counts = {grp: len(vals) for grp, vals in groups.items()}
        if counts[f"targeting_{gene}"] < min_cells_per_group or counts["non_targeting_NEGCON"] < min_cells_per_group:
            summary_rows.append(
                {
                    "gene": gene,
                    "status": "skipped_low_group_size",
                    "n_targeting": counts[f"targeting_{gene}"],
                    "n_negcon": counts["non_targeting_NEGCON"],
                    "n_other_targeting": counts["other_targeting"],
                    "mean_targeting": np.nan,
                    "mean_negcon": np.nan,
                    "mean_other_targeting": np.nan,
                    "median_targeting": np.nan,
                    "median_negcon": np.nan,
                    "median_other_targeting": np.nan,
                    "delta_mean_target_minus_negcon": np.nan,
                    "mannwhitney_pvalue_target_vs_negcon": np.nan,
                }
            )
            ax.axis("off")
            ax.set_title(
                f"{gene} skipped (n_target={counts[f'targeting_{gene}']}, "
                f"n_negcon={counts['non_targeting_NEGCON']})"
            )
            continue

        try:
            from scipy.stats import mannwhitneyu

            stat = mannwhitneyu(
                groups[f"targeting_{gene}"],
                groups["non_targeting_NEGCON"],
                alternative="two-sided",
            )
            p_value = float(stat.pvalue)
        except Exception:
            p_value = np.nan

        means = {grp: float(np.mean(vals)) if len(vals) > 0 else np.nan for grp, vals in groups.items()}
        medians = {grp: float(np.median(vals)) if len(vals) > 0 else np.nan for grp, vals in groups.items()}

        summary_rows.append(
            {
                "gene": gene,
                "status": "ok",
                "n_targeting": counts[f"targeting_{gene}"],
                "n_negcon": counts["non_targeting_NEGCON"],
                "n_other_targeting": counts["other_targeting"],
                "mean_targeting": means[f"targeting_{gene}"],
                "mean_negcon": means["non_targeting_NEGCON"],
                "mean_other_targeting": means["other_targeting"],
                "median_targeting": medians[f"targeting_{gene}"],
                "median_negcon": medians["non_targeting_NEGCON"],
                "median_other_targeting": medians["other_targeting"],
                "delta_mean_target_minus_negcon": means[f"targeting_{gene}"] - means["non_targeting_NEGCON"],
                "mannwhitney_pvalue_target_vs_negcon": p_value,
            }
        )
        valid_genes.append(gene)

        data_to_plot = [groups[grp] for grp in order]
        ax.boxplot(data_to_plot, labels=order, showfliers=False)

        target_tmp = tmp.loc[tmp["group"] == order[0]].copy()
        sgrnas = sorted(target_tmp["gRNA"].unique())
        sgrna_palette = [
            "#d62728",
            "#2ca02c",
            "#9467bd",
            "#ff7f0e",
            "#8c564b",
            "#e377c2",
        ]
        sgrna_color_map = {
            sgrna: sgrna_palette[idx % len(sgrna_palette)]
            for idx, sgrna in enumerate(sgrnas)
        }

        if overlay_points:
            # Targeting group: color by sgRNA identity.
            for sgrna, sub in target_tmp.groupby("gRNA"):
                vals = sub["expr"].values
                if len(vals) == 0:
                    continue
                x = np.random.normal(loc=1, scale=0.05, size=len(vals))
                ax.scatter(
                    x,
                    vals,
                    s=14,
                    alpha=0.45,
                    c=sgrna_color_map.get(sgrna, "#d62728"),
                    linewidths=0,
                    zorder=2,
                )

            # Other groups: single-color jitter.
            for pos, grp, color in [
                (2, order[1], "#1f77b4"),
                (3, order[2], "#7f7f7f"),
            ]:
                vals = groups[grp]
                if len(vals) == 0:
                    continue
                x = np.random.normal(loc=pos, scale=0.06, size=len(vals))
                ax.scatter(
                    x,
                    vals,
                    s=10,
                    alpha=0.30,
                    c=color,
                    linewidths=0,
                    zorder=2,
                )

        if overlay_sgrna_means:
            sgrna_means = target_tmp.groupby("gRNA")["expr"].mean().sort_index()
            if not sgrna_means.empty:
                offsets = np.linspace(-0.12, 0.12, len(sgrna_means))
                for off, sgrna, mean_val in zip(offsets, sgrna_means.index, sgrna_means.values):
                    ax.scatter(
                        1 + off,
                        mean_val,
                        s=60,
                        c=sgrna_color_map.get(sgrna, "#ff7f0e"),
                        marker="D",
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=4,
                    )

        title = f"{gene} (target vs NEGCON)"
        if title_suffix_by_gene and gene in title_suffix_by_gene:
            title = f"{title} | {title_suffix_by_gene[gene]}"
        ax.set_title(title)
        ax.set_ylabel("log1p(normalized expression)")
        ax.tick_params(axis="x", rotation=25)

        if sgrnas:
            from matplotlib.lines import Line2D

            sgrna_n = target_tmp["gRNA"].value_counts().to_dict()
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=sgrna_color_map[sgrna],
                    markeredgecolor="none",
                    markersize=5.5,
                    label=f"{sgrna} (n={int(sgrna_n.get(sgrna, 0))})",
                )
                for sgrna in sgrnas
            ]
            ax.legend(
                handles=handles,
                title="targeting sgRNA",
                loc="upper left",
                fontsize=6.5,
                title_fontsize=7.5,
                frameon=False,
            )

    for j in range(len(genes), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close(fig)

    return pd.DataFrame(summary_rows), valid_genes


def choose_all_target_genes(assignments_df, negcon_gene):
    genes = (
        assignments_df.loc[assignments_df["target_gene"] != negcon_gene, "target_gene"]
        .dropna()
        .astype(str)
        .unique()
    )
    return sorted(genes)


def compute_gene_summary_row(adata, gene, negcon_gene, min_cells_per_group):
    expr_vec = extract_gene_vector(adata, gene)
    if expr_vec is None:
        return {
            "gene": gene,
            "status": "skipped_gene_not_found_in_expression_matrix",
            "n_targeting": 0,
            "n_negcon": 0,
            "n_other_targeting": 0,
            "mean_targeting": np.nan,
            "mean_negcon": np.nan,
            "mean_other_targeting": np.nan,
            "median_targeting": np.nan,
            "median_negcon": np.nan,
            "median_other_targeting": np.nan,
            "delta_mean_target_minus_negcon": np.nan,
            "mannwhitney_pvalue_target_vs_negcon": np.nan,
        }

    tmp = adata.obs.copy()
    tmp["expr"] = expr_vec
    tmp["group"] = tmp["target_gene"].map(lambda g: gene_group_label(g, gene, negcon_gene))

    order = [f"targeting_{gene}", "non_targeting_NEGCON", "other_targeting"]
    groups = {grp: tmp.loc[tmp["group"] == grp, "expr"].values for grp in order}
    counts = {grp: len(vals) for grp, vals in groups.items()}

    if counts[f"targeting_{gene}"] < min_cells_per_group or counts["non_targeting_NEGCON"] < min_cells_per_group:
        return {
            "gene": gene,
            "status": "skipped_low_group_size",
            "n_targeting": counts[f"targeting_{gene}"],
            "n_negcon": counts["non_targeting_NEGCON"],
            "n_other_targeting": counts["other_targeting"],
            "mean_targeting": np.nan,
            "mean_negcon": np.nan,
            "mean_other_targeting": np.nan,
            "median_targeting": np.nan,
            "median_negcon": np.nan,
            "median_other_targeting": np.nan,
            "delta_mean_target_minus_negcon": np.nan,
            "mannwhitney_pvalue_target_vs_negcon": np.nan,
        }

    try:
        from scipy.stats import mannwhitneyu

        stat = mannwhitneyu(
            groups[f"targeting_{gene}"],
            groups["non_targeting_NEGCON"],
            alternative="two-sided",
        )
        p_value = float(stat.pvalue)
    except Exception:
        p_value = np.nan

    means = {grp: float(np.mean(vals)) if len(vals) > 0 else np.nan for grp, vals in groups.items()}
    medians = {grp: float(np.median(vals)) if len(vals) > 0 else np.nan for grp, vals in groups.items()}

    return {
        "gene": gene,
        "status": "ok",
        "n_targeting": counts[f"targeting_{gene}"],
        "n_negcon": counts["non_targeting_NEGCON"],
        "n_other_targeting": counts["other_targeting"],
        "mean_targeting": means[f"targeting_{gene}"],
        "mean_negcon": means["non_targeting_NEGCON"],
        "mean_other_targeting": means["other_targeting"],
        "median_targeting": medians[f"targeting_{gene}"],
        "median_negcon": medians["non_targeting_NEGCON"],
        "median_other_targeting": medians["other_targeting"],
        "delta_mean_target_minus_negcon": means[f"targeting_{gene}"] - means["non_targeting_NEGCON"],
        "mannwhitney_pvalue_target_vs_negcon": p_value,
    }


def summarize_all_target_genes(adata, target_genes, negcon_gene, min_cells_per_group):
    rows = []
    for gene in target_genes:
        rows.append(
            compute_gene_summary_row(
                adata=adata,
                gene=gene,
                negcon_gene=negcon_gene,
                min_cells_per_group=min_cells_per_group,
            )
        )
    return pd.DataFrame(rows)


def choose_top_genes_to_plot(summary_df, top_n):
    ok_df = summary_df.loc[summary_df["status"] == "ok"].copy()
    if ok_df.empty:
        return []

    ok_df = ok_df.sort_values(
        by=["delta_mean_target_minus_negcon", "mannwhitney_pvalue_target_vs_negcon"],
        ascending=[True, True],
        na_position="last",
    )
    return list(ok_df["gene"].head(top_n))


def choose_top_genes_by_n_targeting(summary_df, top_n):
    ok_df = summary_df.loc[summary_df["status"] == "ok"].copy()
    if ok_df.empty:
        return []

    ok_df = ok_df.sort_values(
        by=["n_targeting", "delta_mean_target_minus_negcon", "mannwhitney_pvalue_target_vs_negcon"],
        ascending=[False, True, True],
        na_position="last",
    )
    return list(ok_df["gene"].head(top_n))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "One-sample CROP-seq QC: summarize target-gene expression vs NEGCON for all "
            "targeted genes, then plot top X genes."
        )
    )
    parser.add_argument("--sample-id", required=True, help="Sample ID, e.g. SNUCROP_D_notsoR_1")
    parser.add_argument(
        "--singlet-csv",
        required=True,
        help="Path to combined_singlet_barcodes_by_gRNA_threshold_t*.csv",
    )
    parser.add_argument(
        "--expression-root",
        default="../expression",
        help="Root expression directory containing <sample>/expression/*_exp_fastp.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default="../qc",
        help="Output directory for assignment table, plot, and summary CSV",
    )
    parser.add_argument(
        "--negcon-gene",
        default="NEGCON",
        help="Target gene label used for non-targeting guides (default: NEGCON).",
    )
    parser.add_argument(
        "--min-cells-per-group",
        type=int,
        default=1,
        help="Minimum cells required in targeting and NEGCON groups for a gene summary to be testable.",
    )
    parser.add_argument(
        "--top-genes-to-plot",
        type=int,
        default=12,
        help=(
            "Number of top genes to plot after summarizing all targeted genes. "
            "Ranked by most negative mean(targeting - NEGCON)."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_df = load_singlet_assignments(args.singlet_csv, args.sample_id)
    if assignments_df.empty:
        raise RuntimeError(f"No singlet assignments found for sample: {args.sample_id}")

    assignment_out = output_dir / f"{args.sample_id}.barcode_to_gRNA_target_gene.csv"
    assignments_df.to_csv(assignment_out, index=False)
    print(f"Wrote {assignment_out}")

    expr_tsv = find_expression_tsv(args.expression_root, args.sample_id)
    print(f"Using expression file: {expr_tsv}")
    adata = build_anndata_from_expression(expr_tsv, assignments_df)

    target_genes = choose_all_target_genes(assignments_df, negcon_gene=args.negcon_gene)
    if not target_genes:
        raise RuntimeError("No non-NEGCON target genes found in assigned barcodes.")
    print(f"Summarizing all targeted genes: n={len(target_genes)}")

    summary_df = summarize_all_target_genes(
        adata=adata,
        target_genes=target_genes,
        negcon_gene=args.negcon_gene,
        min_cells_per_group=args.min_cells_per_group,
    )

    summary_out = output_dir / f"{args.sample_id}.target_vs_NEGCON.expression_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Wrote {summary_out}")

    if summary_df.empty:
        print("Warning: no summary rows were produced.")
    else:
        status_counts = summary_df["status"].value_counts(dropna=False).to_dict()
        print(f"Gene status counts: {status_counts}")

    top_genes = choose_top_genes_to_plot(summary_df, top_n=args.top_genes_to_plot)
    if not top_genes:
        print("Skipping plot: no genes passed status=ok.")
        return

    print(f"Top genes to plot: {', '.join(top_genes)}")
    plot_out = output_dir / f"{args.sample_id}.target_vs_NEGCON.top{len(top_genes)}.expression_panel.png"
    _, valid_genes = plot_gene_panel_and_summarize(
        adata=adata,
        genes=top_genes,
        negcon_gene=args.negcon_gene,
        min_cells_per_group=args.min_cells_per_group,
        output_plot=plot_out,
    )
    print(f"Wrote {plot_out}")
    print(f"Completed plotted genes: {', '.join(valid_genes) if valid_genes else 'none'}")

    top_n_targeting_genes = choose_top_genes_by_n_targeting(
        summary_df, top_n=args.top_genes_to_plot
    )
    if not top_n_targeting_genes:
        print("Skipping n_targeting-ranked plot: no genes passed status=ok.")
        return

    print(f"Top genes by n_targeting: {', '.join(top_n_targeting_genes)}")
    plot_n_targeting_out = (
        output_dir
        / f"{args.sample_id}.target_vs_NEGCON.top{len(top_n_targeting_genes)}.by_n_targeting.expression_panel.png"
    )
    _, valid_genes_n_targeting = plot_gene_panel_and_summarize(
        adata=adata,
        genes=top_n_targeting_genes,
        negcon_gene=args.negcon_gene,
        min_cells_per_group=args.min_cells_per_group,
        output_plot=plot_n_targeting_out,
        title_suffix_by_gene={
            gene: f"n_targeting={int(summary_df.loc[summary_df['gene'] == gene, 'n_targeting'].iloc[0])}"
            for gene in top_n_targeting_genes
            if not summary_df.loc[summary_df["gene"] == gene, "n_targeting"].empty
        },
    )
    print(f"Wrote {plot_n_targeting_out}")
    print(
        "Completed n_targeting plotted genes: "
        f"{', '.join(valid_genes_n_targeting) if valid_genes_n_targeting else 'none'}"
    )


if __name__ == "__main__":
    main()
