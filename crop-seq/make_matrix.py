import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

def plot_stacked_dict(
    dict_by_threshold,
    max_bin=5,
    ylabel="Count",
    title=None,
    output_file=None,
    show_plot=False,
):
    """
    dict_by_threshold:
        {
            threshold1: {0:..., 1:..., 2:..., ..., '>5':...},
            threshold2: {0:..., 1:..., 2:..., ..., '>5':...},
            ...
        }
    """

    bin_order = list(range(max_bin + 1)) + [f">{max_bin}"]
    thresholds = sorted(dict_by_threshold.keys())

    x = np.arange(len(thresholds))
    bottom = np.zeros(len(thresholds))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for b in bin_order:
        vals = [dict_by_threshold[t].get(b, 0) for t in thresholds]
        plt.bar(x, vals, bottom=bottom, label=str(b))
        bottom += np.array(vals)

    plt.xticks(x, thresholds)
    plt.xlabel("Threshold")
    plt.ylabel(ylabel)
    plt.title(title if title else f"Stacked bar plot of {ylabel.lower()} by threshold")
    plt.legend(title="Features per cell", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
    if show_plot:
        plt.show()
    plt.close()

def count_and_proportion_features_per_cell_np(binary_mat, max_bin=5):
    """
    binary_mat: feature x cell NumPy array
                values should be 0/1

    Returns:
        count_dict: number of cells with 0, 1, 2, ..., max_bin, >max_bin features
        prop_dict: proportion of cells with 0, 1, 2, ..., max_bin, >max_bin features
    """

    binary_mat = np.asarray(binary_mat)

    # Sum down rows/features to get number of features per cell
    features_per_cell = np.nansum(binary_mat, axis=0)

    total_cells = binary_mat.shape[1]

    count_dict = {}
    prop_dict = {}

    for i in range(max_bin + 1):
        count = int(np.sum(features_per_cell == i))
        count_dict[i] = count
        prop_dict[i] = count / total_cells

    count = int(np.sum(features_per_cell > max_bin))
    count_dict[f">{max_bin}"] = count
    prop_dict[f">{max_bin}"] = count / total_cells

    return count_dict, prop_dict


def binarize_matrix(df, threshold):
    return (df >= threshold).astype(int)


def save_binary_csv_hide_zeros(binary_df, output_csv):
    out = binary_df.replace(0, "")
    out.to_csv(output_csv)


def summarize_singlet_cells(binary_df, sample_id, threshold):
    features_per_cell = binary_df.sum(axis=0)
    singlet_cols = features_per_cell == 1

    singlet_mat = binary_df.loc[:, singlet_cols]
    unique_gRNAs_in_singlets = int((singlet_mat.sum(axis=1) > 0).sum())
    singlet_cell_count = int(singlet_cols.sum())
    total_cells = int(binary_df.shape[1])

    return {
        "sample_id": sample_id,
        "threshold": threshold,
        "total_cells": total_cells,
        "cells_features_per_cell_eq_1": singlet_cell_count,
        "unique_gRNAs_in_features_per_cell_eq_1_cells": unique_gRNAs_in_singlets,
    }


def collect_singlet_barcodes_by_grna(binary_df):
    features_per_cell = binary_df.sum(axis=0)
    singlet_cols = features_per_cell == 1
    singlet_mat = binary_df.loc[:, singlet_cols]

    barcodes_by_grna = defaultdict(list)
    for barcode in singlet_mat.columns:
        positive_grnas = singlet_mat.index[singlet_mat[barcode] == 1]
        if len(positive_grnas) != 1:
            continue
        barcodes_by_grna[positive_grnas[0]].append(barcode)

    for grna in barcodes_by_grna:
        barcodes_by_grna[grna] = sorted(barcodes_by_grna[grna])

    return dict(barcodes_by_grna)


def build_combined_singlet_barcode_matrix(singlet_barcodes_by_sample):
    all_grnas = sorted(
        {
            grna
            for sample_dict in singlet_barcodes_by_sample.values()
            for grna in sample_dict.keys()
        }
    )
    sample_ids = sorted(singlet_barcodes_by_sample.keys())

    combined = pd.DataFrame("", index=all_grnas, columns=sample_ids)
    for sample_id in sample_ids:
        for grna, barcodes in singlet_barcodes_by_sample[sample_id].items():
            combined.at[grna, sample_id] = ";".join(barcodes)

    combined.index.name = "gRNA"
    return combined


def build_combined_singlet_barcode_count_matrix(singlet_barcodes_by_sample):
    all_grnas = sorted(
        {
            grna
            for sample_dict in singlet_barcodes_by_sample.values()
            for grna in sample_dict.keys()
        }
    )
    sample_ids = sorted(singlet_barcodes_by_sample.keys())

    combined = pd.DataFrame(0, index=all_grnas, columns=sample_ids, dtype=int)
    for sample_id in sample_ids:
        for grna, barcodes in singlet_barcodes_by_sample[sample_id].items():
            combined.at[grna, sample_id] = len(barcodes)

    combined.index.name = "gRNA"
    return combined


def parse_gene_from_grna(grna):
    if "_sg" in grna:
        return grna.split("_sg", 1)[0]
    return grna


def collapse_grna_count_matrix_to_gene(count_df):
    tmp = count_df.copy()
    tmp["gene"] = [parse_gene_from_grna(grna) for grna in tmp.index]
    collapsed = tmp.groupby("gene", sort=True).sum(numeric_only=True)
    collapsed.index.name = "gene"
    return collapsed


def aggregate_count_dicts(count_dicts, max_bin):
    bin_order = list(range(max_bin + 1)) + [f">{max_bin}"]
    combined_count = {b: 0 for b in bin_order}

    for count_dict in count_dicts:
        for b in bin_order:
            combined_count[b] += int(count_dict.get(b, 0))

    total_cells = int(sum(combined_count.values()))
    if total_cells == 0:
        combined_prop = {b: 0.0 for b in bin_order}
    else:
        combined_prop = {b: combined_count[b] / total_cells for b in bin_order}

    return combined_count, combined_prop


def save_combined_threshold_count_prop_plot(
    sample_count_dicts,
    threshold,
    max_bin,
    plot_dir,
    show_plot=False,
):
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        plot_dir
        / f"combined_features_per_cell_threshold_t{threshold}.count_proportion.png"
    )

    sample_ids = list(sample_count_dicts.keys())
    if not sample_ids:
        raise RuntimeError("No sample count dictionaries provided for combined plot.")

    bin_order = list(range(max_bin + 1)) + [f">{max_bin}"]
    x = np.arange(len(sample_ids))

    count_matrix = {
        b: np.array([int(sample_count_dicts[sample_id].get(b, 0)) for sample_id in sample_ids])
        for b in bin_order
    }

    prop_matrix = {}
    for b in bin_order:
        vals = []
        for sample_id in sample_ids:
            count_dict = sample_count_dicts[sample_id]
            total = int(sum(int(count_dict.get(k, 0)) for k in bin_order))
            val = int(count_dict.get(b, 0)) / total if total > 0 else 0.0
            vals.append(val)
        prop_matrix[b] = np.array(vals, dtype=float)

    fig_width = max(14, 0.65 * len(sample_ids) + 8)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))
    for ax, metric_matrix, ylabel, title in [
        (axes[0], count_matrix, "Count", "Per-sample features per cell (count)"),
        (
            axes[1],
            prop_matrix,
            "Proportion",
            "Per-sample features per cell (proportion)",
        ),
    ]:
        bottom = np.zeros(len(sample_ids), dtype=float)
        for b in bin_order:
            vals = metric_matrix[b]
            ax.bar(x, vals, bottom=bottom, label=str(b))
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(sample_ids, rotation=60, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (t={threshold})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Features per cell", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig)

    return out_path


def summarize_features_per_cell_by_threshold(count_df, thresholds, max_bin=5):
    count_by_threshold = {}
    prop_by_threshold = {}

    for threshold in thresholds:
        binary_df = binarize_matrix(count_df, threshold=threshold)
        count_dict, prop_dict = count_and_proportion_features_per_cell_np(
            binary_df.values, max_bin=max_bin
        )
        count_by_threshold[threshold] = count_dict
        prop_by_threshold[threshold] = prop_dict

    return count_by_threshold, prop_by_threshold


def save_features_per_cell_plots(
    sample_id,
    count_df,
    plot_dir,
    thresholds,
    max_bin=5,
    show_plot=False,
):
    count_by_threshold, prop_by_threshold = summarize_features_per_cell_by_threshold(
        count_df=count_df,
        thresholds=thresholds,
        max_bin=max_bin,
    )

    plot_dir.mkdir(parents=True, exist_ok=True)

    count_plot_path = plot_dir / f"{sample_id}.features_per_cell.count.png"
    prop_plot_path = plot_dir / f"{sample_id}.features_per_cell.proportion.png"

    plot_stacked_dict(
        count_by_threshold,
        max_bin=max_bin,
        ylabel="Count",
        title=f"{sample_id}: Features per cell (count)",
        output_file=count_plot_path,
        show_plot=show_plot,
    )
    plot_stacked_dict(
        prop_by_threshold,
        max_bin=max_bin,
        ylabel="Proportion",
        title=f"{sample_id}: Features per cell (proportion)",
        output_file=prop_plot_path,
        show_plot=show_plot,
    )

    return count_plot_path, prop_plot_path

def open_matrix_bam(bam_file, require_nm0=True):
    try:
        import pysam
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pysam is required to read BAM files. Install it with: pip install pysam"
        ) from exc

    results = defaultdict(lambda: defaultdict(int))

    with pysam.AlignmentFile(str(bam_file), "rb") as bam:
        for read in bam:
            if read.is_unmapped:
                continue

            if require_nm0:
                if not read.has_tag("NM") or read.get_tag("NM") != 0:
                    continue

            barcode = extract_barcode(read.query_name)
            if barcode is None:
                continue

            ref_name = bam.get_reference_name(read.reference_id)
            if ref_name is None:
                continue

            sgrna = ref_name.split(":")[0]

            results[barcode][sgrna] += 1

    return {bc: dict(sgrnas) for bc, sgrnas in results.items()}


def extract_barcode(query_name):
    parts = query_name.split("_")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return None


def open_matrix(sam_file):
    results = {}
    with open(sam_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("@"):
                continue

            fields = line.strip("\n").split("\t")
            if len(fields) < 3:
                continue

            barcode = extract_barcode(fields[0])
            if barcode is None:
                continue

            sgrna = fields[2].split(":")[0]
            if barcode not in results:
                results[barcode] = {sgrna: 1}
            else:
                if sgrna not in results[barcode]:
                    results[barcode][sgrna] = 1
                else:
                    results[barcode][sgrna] += 1

    return results


def dict_to_matrix(results, sgrna_index=None, barcode_header=None, fill_value=0):
    if sgrna_index is None:
        sgrna_index = sorted({sgrna for d in results.values() for sgrna in d})
    if barcode_header is None:
        barcode_header = sorted(results.keys())

    mat = np.full(
        (len(sgrna_index), len(barcode_header)),
        fill_value,
        dtype=int
    )

    sgrna_to_i = {sgrna: i for i, sgrna in enumerate(sgrna_index)}
    barcode_to_j = {barcode: j for j, barcode in enumerate(barcode_header)}

    for barcode, sgrna_counts in results.items():
        j = barcode_to_j.get(barcode)
        if j is None:
            continue

        for sgrna, count in sgrna_counts.items():
            i = sgrna_to_i.get(sgrna)
            if i is None:
                continue

            mat[i, j] = count

    df = pd.DataFrame(mat, index=sgrna_index, columns=barcode_header)

    return mat, df


def discover_sample_bams(input_dir):
    sample_bams = {}

    for sample_dir in sorted(input_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        preferred = sample_dir / f"{sample_dir.name}.bam"
        if preferred.exists():
            sample_bams[sample_dir.name] = preferred
            continue

        bams = sorted(sample_dir.glob("*.bam"))
        if bams:
            sample_bams[sample_dir.name] = bams[0]

    return sample_bams


def load_barcode_list(barcode_file):
    barcodes = []
    seen = set()

    with open(barcode_file, "r", encoding="utf-8") as fh:
        for line in fh:
            bc = line.strip()
            if not bc or bc in seen:
                continue
            seen.add(bc)
            barcodes.append(bc)

    return barcodes


def discover_sample_barcode_files(barcode_dir):
    sample_barcode_files = {}

    for sample_dir in sorted(barcode_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        mapping_dir = sample_dir / "mapping"
        preferred = mapping_dir / f"{sample_dir.name}_filter40_barcode.txt"
        if preferred.exists():
            sample_barcode_files[sample_dir.name] = preferred
            continue

        candidates = []
        if mapping_dir.exists():
            candidates.extend(sorted(mapping_dir.glob("*barcode*.txt")))
        candidates.extend(sorted(sample_dir.glob("*barcode*.txt")))
        if candidates:
            sample_barcode_files[sample_dir.name] = candidates[0]

    return sample_barcode_files


def resolve_barcode_dir(input_dir, barcode_dir_arg=None):
    if barcode_dir_arg:
        candidate = Path(barcode_dir_arg)
        return candidate if candidate.exists() else None

    candidates = [
        input_dir.parent / "barcode",
        Path("barcode"),
        Path("../barcode"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_and_save_matrices(
    sample_id,
    bam_path,
    expected_barcodes,
    output_dir,
    plot_dir,
    threshold,
    require_nm0,
    max_bin=5,
    plot_thresholds=None,
    show_plot=False,
):
    results = open_matrix_bam(bam_path, require_nm0=require_nm0)
    _, count_df = dict_to_matrix(results, barcode_header=expected_barcodes)
    binary_df = binarize_matrix(count_df, threshold=threshold)

    expected_barcodes = expected_barcodes or []
    expected_set = set(expected_barcodes)
    bam_set = set(results.keys())
    missing_from_bam = len(expected_set - bam_set)
    unexpected_in_bam = len(bam_set - expected_set) if expected_set else 0

    output_dir.mkdir(parents=True, exist_ok=True)

    count_out = output_dir / f"{sample_id}.gRNA_count_matrix.csv"
    binary_out = output_dir / f"{sample_id}.gRNA_binary_matrix_t{threshold}.csv"

    count_df.to_csv(count_out)
    save_binary_csv_hide_zeros(binary_df, binary_out)
    singlet_summary = summarize_singlet_cells(
        binary_df=binary_df,
        sample_id=sample_id,
        threshold=threshold,
    )
    threshold_count_dict, _ = count_and_proportion_features_per_cell_np(
        binary_df.values, max_bin=max_bin
    )
    singlet_barcodes_by_grna = collect_singlet_barcodes_by_grna(binary_df)

    if plot_thresholds is None:
        plot_thresholds = [threshold]

    count_plot_out, prop_plot_out = save_features_per_cell_plots(
        sample_id=sample_id,
        count_df=count_df,
        plot_dir=plot_dir,
        thresholds=plot_thresholds,
        max_bin=max_bin,
        show_plot=show_plot,
    )

    return (
        count_out,
        binary_out,
        count_plot_out,
        prop_plot_out,
        singlet_summary,
        singlet_barcodes_by_grna,
        threshold_count_dict,
        {
            "expected_total": len(expected_barcodes),
            "missing_from_bam": missing_from_bam,
            "unexpected_in_bam": unexpected_in_bam,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create per-sample gRNA matrices from BAM files: raw count matrix and "
            "binary matrix using threshold >=3 (default)."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="bowtie2",
        help="Directory containing per-sample folders with BAM files (default: bowtie2).",
    )
    parser.add_argument(
        "--output-dir",
        default="matrix",
        help="Directory to write matrix CSV outputs (default: matrix).",
    )
    parser.add_argument(
        "--plot-dir",
        default="plots",
        help="Directory to write features-per-cell plot outputs (default: plots).",
    )
    parser.add_argument(
        "--barcode-dir",
        default=None,
        help=(
            "Directory containing per-sample barcode lists; when available, these "
            "barcodes define matrix columns so missing BAM barcodes are kept as 0."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Binarization threshold; values >= threshold become 1 (default: 3).",
    )
    parser.add_argument(
        "--plot-thresholds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help=(
            "Threshold(s) used for features-per-cell plots. "
            "Default: use --threshold only."
        ),
    )
    parser.add_argument(
        "--max-bin",
        type=int,
        default=8,
        help="Maximum exact bin shown in features-per-cell plots (default: 8).",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plot windows while running (default: save only).",
    )
    parser.add_argument(
        "--no-require-nm0",
        action="store_true",
        help="Do not require NM==0 alignments when counting.",
    )
    parser.add_argument(
        "sample_ids",
        nargs="*",
        help="Optional sample IDs. If omitted, process all discovered samples.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    require_nm0 = False #not args.no_require_nm0

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_sample_bams = discover_sample_bams(input_dir)
    if not all_sample_bams:
        raise RuntimeError(f"No sample BAM files found under: {input_dir}")

    barcode_dir = resolve_barcode_dir(input_dir=input_dir, barcode_dir_arg=args.barcode_dir)
    all_sample_barcode_files = {}
    if barcode_dir is not None:
        all_sample_barcode_files = discover_sample_barcode_files(barcode_dir)
        print(f"Using barcode directory: {barcode_dir}")
    else:
        print("Barcode directory not found; using BAM-observed barcodes only.")

    if args.sample_ids:
        selected = {}
        missing = []
        for sample_id in args.sample_ids:
            bam = all_sample_bams.get(sample_id)
            if bam is None:
                missing.append(sample_id)
            else:
                selected[sample_id] = bam
        if missing:
            print(f"Skipping missing sample IDs: {', '.join(missing)}")
    else:
        selected = all_sample_bams

    if not selected:
        raise RuntimeError("No valid samples selected to process.")

    singlet_summary_rows = []
    singlet_barcodes_by_sample = {}
    sample_threshold_count_dicts = {}

    for sample_id, bam_path in selected.items():
        print(f"[{sample_id}] Processing {bam_path}")
        (
            count_out,
            binary_out,
            count_plot_out,
            prop_plot_out,
            singlet_summary,
            singlet_barcodes_by_grna,
            threshold_count_dict,
            barcode_stats,
        ) = build_and_save_matrices(
            sample_id=sample_id,
            bam_path=bam_path,
            expected_barcodes=load_barcode_list(all_sample_barcode_files[sample_id])
            if sample_id in all_sample_barcode_files
            else None,
            output_dir=output_dir,
            plot_dir=plot_dir,
            threshold=args.threshold,
            require_nm0=require_nm0,
            max_bin=args.max_bin,
            plot_thresholds=args.plot_thresholds,
            show_plot=args.show_plot,
        )
        print(f"[{sample_id}] Wrote {count_out}")
        print(f"[{sample_id}] Wrote {binary_out}")
        print(f"[{sample_id}] Wrote {count_plot_out}")
        print(f"[{sample_id}] Wrote {prop_plot_out}")
        if barcode_stats["expected_total"] > 0:
            print(
                f"[{sample_id}] Barcode list total={barcode_stats['expected_total']}, "
                f"missing in BAM={barcode_stats['missing_from_bam']} (counted as zero gRNA/cell), "
                f"unexpected in BAM={barcode_stats['unexpected_in_bam']}"
            )
        singlet_summary_rows.append(singlet_summary)
        singlet_barcodes_by_sample[sample_id] = singlet_barcodes_by_grna
        sample_threshold_count_dicts[sample_id] = threshold_count_dict

    plot_dir.mkdir(parents=True, exist_ok=True)
    singlet_summary_out = (
        plot_dir
        / f"combined_singlet_gRNA_summary_threshold_t{args.threshold}.csv"
    )
    pd.DataFrame(singlet_summary_rows).to_csv(singlet_summary_out, index=False)
    print(f"[combined] Wrote {singlet_summary_out}")

    singlet_barcode_matrix_out = (
        plot_dir
        / f"combined_singlet_barcodes_by_gRNA_threshold_t{args.threshold}.csv"
    )
    combined_singlet_barcode_df = build_combined_singlet_barcode_matrix(
        singlet_barcodes_by_sample
    )
    combined_singlet_barcode_df.to_csv(singlet_barcode_matrix_out)
    print(f"[combined] Wrote {singlet_barcode_matrix_out}")

    singlet_barcode_count_matrix_out = (
        plot_dir
        / f"combined_singlet_barcode_counts_by_gRNA_threshold_t{args.threshold}.csv"
    )
    combined_singlet_barcode_count_df = build_combined_singlet_barcode_count_matrix(
        singlet_barcodes_by_sample
    )
    
    # combined_singlet_barcode_count_df.to_csv(singlet_barcode_count_matrix_out)

    save_binary_csv_hide_zeros(combined_singlet_barcode_count_df, singlet_barcode_count_matrix_out)
    print(f"[combined] Wrote {singlet_barcode_count_matrix_out}")

    singlet_barcode_gene_count_matrix_out = (
        plot_dir
        / f"combined_singlet_barcode_counts_by_gene_threshold_t{args.threshold}.csv"
    )
    combined_singlet_barcode_gene_count_df = collapse_grna_count_matrix_to_gene(
        combined_singlet_barcode_count_df
    )
    combined_singlet_barcode_gene_count_df.to_csv(singlet_barcode_gene_count_matrix_out)
    print(f"[combined] Wrote {singlet_barcode_gene_count_matrix_out}")

    combined_count_prop_plot_out = save_combined_threshold_count_prop_plot(
        sample_count_dicts=sample_threshold_count_dicts,
        threshold=args.threshold,
        max_bin=args.max_bin,
        plot_dir=plot_dir,
        show_plot=args.show_plot,
    )
    print(f"[combined] Wrote {combined_count_prop_plot_out}")


if __name__ == "__main__":
    main()
