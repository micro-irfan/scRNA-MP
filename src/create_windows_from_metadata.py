#!/usr/bin/env python3

"""
Create window/matrix outputs for selected barcodes grouped by metadata.

This script mirrors the matrix outputs from create_windows.py, but instead of
using all bcX entries from one barcode list, it:
1) Reads a metadata CSV containing columns:
   - cellbarcodes (semicolon-separated barcodes)
   - sample
   - sample_RBP
2) Maps each barcode sequence to bcX for each sample via sample barcode files
3) Aggregates raw pileup files (sample.bcX.pileup) within each (sample, sample_RBP) group
4) Writes grouped window CSVs and final mutrate/coverage/mutant matrices
   per sample
"""

from __future__ import annotations

import argparse
import ast
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

import create_matrix as mm
import create_utils as cu
from common import convert_to_df, create_reference, open_barcode_txt, struct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="create_windows_from_metadata.py",
        description=(
            "Aggregate per-cell pileups by metadata groups (sample + sample_RBP) "
            "and generate per-sample matrices matching create_windows.py output format."
        ),
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help=(
            "Deprecated legacy output sample ID. Ignored when metadata contains "
            "sample values (default behavior)."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="CSV with columns: cellbarcodes, sample, sample_RBP.",
    )
    parser.add_argument(
        "--pileup-root",
        required=True,
        help="Directory containing per-sample subdirs of *.pileup files.",
    )
    parser.add_argument(
        "--barcode-root",
        required=True,
        help=(
            "Root directory to resolve each sample's barcode file "
            "(e.g., <root>/<sample>/mapping/<sample>_filter40_barcode.txt)."
        ),
    )
    parser.add_argument("--reference", required=True, help="Reference FASTA path.")
    parser.add_argument("--output-path", required=True, help="Output root directory.")
    parser.add_argument(
        "--method",
        default="single_base",
        choices=["single_base", "fixed", "rolling"],
        help="Windowing method (default: single_base).",
    )
    parser.add_argument(
        "--cellbarcodes-col",
        default="cellbarcodes",
        help="Metadata column name for barcode list (default: cellbarcodes).",
    )
    parser.add_argument(
        "--sample-col",
        default="sample",
        help="Metadata column name for sample (default: sample).",
    )
    parser.add_argument(
        "--sample-rbp-col",
        default="sample_RBP",
        help="Metadata column name for sample/RBP group (default: sample_RBP).",
    )
    parser.add_argument(
        "--barcode-sep",
        default=";",
        help="Separator for cellbarcodes string (default: ';').",
    )
    parser.add_argument(
        "--coverage",
        type=int,
        default=10,
        help="Coverage value used in output matrix filenames (default: 10).",
    )
    parser.add_argument(
        "--min-bases-window",
        type=int,
        default=6,
        help=(
            "For fixed/rolling methods, minimum contributing bases per window "
            "to match create_windows.py matrix filtering behavior (default: 6)."
        ),
    )
    parser.add_argument(
        "--no-write-windows",
        action="store_false",
        dest="write_windows",
        help="Do not write grouped window CSV files under make_windows/<method>/<sample>/.",
    )
    parser.set_defaults(write_windows=True)
    return parser.parse_args()


def _parse_cellbarcodes(value: object, sep: str) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

    return [x.strip() for x in text.split(sep) if x.strip()]


def _sanitize_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _make_group_id(sample: str, sample_rbp: str) -> str:
    sample = str(sample)
    sample_rbp = str(sample_rbp)
    if sample_rbp.startswith(f"{sample}_"):
        return _sanitize_id(sample_rbp)
    return _sanitize_id(f"{sample}_{sample_rbp}")


def _load_metadata_groups(
    metadata_csv: Path,
    cellbarcodes_col: str,
    sample_col: str,
    sample_rbp_col: str,
    barcode_sep: str,
) -> List[Tuple[str, str, List[str], str]]:
    df = pd.read_csv(metadata_csv)
    required = {cellbarcodes_col, sample_col, sample_rbp_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata column(s): {sorted(missing)}")

    groups: List[Tuple[str, str, List[str], str]] = []
    grouped = df.groupby([sample_col, sample_rbp_col], sort=False, dropna=False)
    for (sample, sample_rbp), sub in grouped:
        sample = str(sample).strip()
        sample_rbp = str(sample_rbp).strip()
        if not sample or sample.lower() == "nan":
            continue
        if not sample_rbp or sample_rbp.lower() == "nan":
            continue

        seen = OrderedDict()
        for v in sub[cellbarcodes_col].tolist():
            for bc in _parse_cellbarcodes(v, barcode_sep):
                seen[bc] = None
        barcodes = list(seen.keys())
        if not barcodes:
            continue

        group_id = _make_group_id(sample, sample_rbp)
        groups.append((sample, sample_rbp, barcodes, group_id))

    if not groups:
        raise RuntimeError("No valid groups found in metadata CSV.")
    return groups


def _resolve_barcode_file(barcode_root: Path, sample: str) -> Path:
    candidates = [
        barcode_root / sample / "mapping" / f"{sample}_filter40_barcode.txt",
        barcode_root / sample / f"{sample}_filter40_barcode.txt",
        barcode_root / f"{sample}_filter40_barcode.txt",
    ]
    for c in candidates:
        if c.exists():
            return c

    rec_hits = sorted(barcode_root.glob(f"**/{sample}_filter40_barcode.txt"))
    if rec_hits:
        return rec_hits[0]

    raise FileNotFoundError(
        f"Could not resolve barcode file for sample '{sample}' under {barcode_root}"
    )


def _resolve_sample_pileup_files(pileup_root: Path, sample: str) -> Dict[str, Path]:
    sample_dir = pileup_root / sample
    if not sample_dir.exists():
        raise FileNotFoundError(f"Pileup directory not found for sample '{sample}': {sample_dir}")

    bc_to_file: Dict[str, Path] = {}
    for path in sorted(sample_dir.glob("*.pileup")):
        tokens = path.stem.split(".")
        bc_token = None
        for tok in tokens:
            if re.fullmatch(r"bc\d+", tok, flags=re.IGNORECASE):
                bc_token = tok
                break
        if bc_token is None and len(tokens) >= 2 and re.fullmatch(r"bc\d+", tokens[1], flags=re.IGNORECASE):
            bc_token = tokens[1]
        if bc_token is None:
            continue
        bc_to_file[bc_token.lower()] = path
    return bc_to_file


def _parse_pileup_file(path: Path) -> Dict[str, Dict[int, Tuple[int, int]]]:
    # gene -> pos -> (cov, mut)
    out: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 5:
                continue

            data = cu.Pileup(fields)
            data.read_pileup()
            if data.cov <= 0:
                continue

            acc = out[data.tx_id][data.pos]
            acc[0] += int(data.cov)
            acc[1] += int(data.mut)

    final: Dict[str, Dict[int, Tuple[int, int]]] = {}
    for gene, pos_map in out.items():
        final[gene] = {pos: (vals[0], vals[1]) for pos, vals in pos_map.items()}
    return final


def _merge_counts(
    per_bc_counts: Iterable[Dict[str, Dict[int, Tuple[int, int]]]]
) -> Dict[str, Dict[int, struct]]:
    # gene -> pos -> [cov, mut]
    merged: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for bc_counts in per_bc_counts:
        for gene, pos_map in bc_counts.items():
            for pos, (cov, mut) in pos_map.items():
                dst = merged[gene][pos]
                dst[0] += int(cov)
                dst[1] += int(mut)

    # convert to objects with .cov and .mut attributes for create_utils functions
    out: Dict[str, Dict[int, struct]] = {}
    for gene, pos_map in merged.items():
        out[gene] = {}
        for pos, (cov, mut) in pos_map.items():
            out[gene][pos] = struct({"cov": int(cov), "mut": int(mut)})
    return out


def _apply_window_method(
    merged_counts: Dict[str, Dict[int, struct]],
    reference_seqlen: Dict[str, int],
    method: str,
    min_bases_window: int,
) -> Dict[str, Dict[int, object]]:
    if method == "single_base":
        return cu.create_single_base(merged_counts, reference_seqlen)

    if method == "fixed":
        windowed = cu.create_fixed_windows(merged_counts, reference_seqlen)
    elif method == "rolling":
        windowed = cu.create_rolling_windows(merged_counts, reference_seqlen)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if min_bases_window <= 1:
        return windowed

    filtered = {}
    for gene, pos_map in windowed.items():
        kept = {p: d for p, d in pos_map.items() if getattr(d, "base_count", 0) >= min_bases_window}
        if kept:
            filtered[gene] = kept
    return filtered


def _write_window_csv(
    path: Path,
    sample_label: str,
    group_id: str,
    gene_data: Dict[str, Dict[int, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        wf.write("sample_id,barcode_id,tx_id,pos,cov,mut,mutrate,bases\n")
        for gene, pos_map in gene_data.items():
            for pos, d in pos_map.items():
                cov = float(getattr(d, "cov", 0))
                mut = float(getattr(d, "mut", 0))
                mutrate = float(getattr(d, "mutrate", -1))
                bases = int(getattr(d, "base_count", 1))
                if mutrate < 0 or cov == 0:
                    continue
                wf.write(f"{sample_label},{group_id},{gene},{pos},{cov},{mut},{mutrate},{bases}\n")


def main() -> None:
    args = parse_args()

    metadata_csv = Path(args.metadata_csv)
    pileup_root = Path(args.pileup_root)
    barcode_root = Path(args.barcode_root)
    output_root = Path(args.output_path)

    reference_seqlen = create_reference(args.reference, keep_seq=True)

    groups = _load_metadata_groups(
        metadata_csv=metadata_csv,
        cellbarcodes_col=args.cellbarcodes_col,
        sample_col=args.sample_col,
        sample_rbp_col=args.sample_rbp_col,
        barcode_sep=args.barcode_sep,
    )
    print(f"[metadata] Groups loaded: {len(groups)}")

    # Organize groups by sample to process/write one sample at a time.
    groups_by_sample: Dict[str, List[Tuple[str, str, List[str], str]]] = OrderedDict()
    for sample, sample_rbp, barcode_list, group_id in groups:
        if sample not in groups_by_sample:
            groups_by_sample[sample] = []
        groups_by_sample[sample].append((sample, sample_rbp, barcode_list, group_id))

    windows_base_dir = output_root / "make_windows" / args.method
    matrices_base_dir = output_root / "matrices" / args.method
    windows_base_dir.mkdir(parents=True, exist_ok=True)
    matrices_base_dir.mkdir(parents=True, exist_ok=True)

    any_sample_written = False

    cell_txt = "AllCells"
    cov_token = args.coverage
    for sample, sample_groups in groups_by_sample.items():
        bc_file = _resolve_barcode_file(barcode_root, sample)
        bc_map = open_barcode_txt(str(bc_file))  # barcode_seq -> bcX
        pileup_map = _resolve_sample_pileup_files(pileup_root, sample)  # bcx(lower) -> path
        print(
            f"[sample={sample}] barcodes in mapping={len(bc_map)}, "
            f"pileup files={len(pileup_map)}"
        )

        # Cache parsed bc pileups per sample only, then release memory after writing.
        parsed_cache: Dict[str, Dict[str, Dict[int, Tuple[int, int]]]] = {}
        used_group_ids: set = set()
        grouped_results: "OrderedDict[str, Dict[str, Dict[int, object]]]" = OrderedDict()
        summary_rows: List[dict] = []

        for _, sample_rbp, barcode_list, group_id in sample_groups:
            # Ensure unique group ID after sanitization within each sample.
            base_gid = group_id
            suffix = 2
            while group_id in used_group_ids:
                group_id = f"{base_gid}_{suffix}"
                suffix += 1
            used_group_ids.add(group_id)

            per_bc_counts = []
            missing_in_barcode = []
            missing_pileup = []

            for bc_seq in barcode_list:
                bc_id = bc_map.get(bc_seq)
                if bc_id is None:
                    missing_in_barcode.append(bc_seq)
                    continue

                if bc_id in parsed_cache:
                    bc_counts = parsed_cache[bc_id]
                else:
                    pileup_file = pileup_map.get(bc_id.lower())
                    if pileup_file is None:
                        missing_pileup.append(bc_id)
                        continue
                    bc_counts = _parse_pileup_file(pileup_file)
                    parsed_cache[bc_id] = bc_counts

                per_bc_counts.append(bc_counts)

            if not per_bc_counts:
                print(
                    f"[skip] group={group_id} sample={sample} sample_RBP={sample_rbp} "
                    f"(no usable pileups; missing barcode={len(missing_in_barcode)}, "
                    f"missing pileup={len(missing_pileup)})"
                )
                continue

            merged_counts = _merge_counts(per_bc_counts)
            windowed = _apply_window_method(
                merged_counts=merged_counts,
                reference_seqlen=reference_seqlen,
                method=args.method,
                min_bases_window=args.min_bases_window,
            )

            if args.write_windows:
                windows_dir = windows_base_dir / sample
                windows_dir.mkdir(parents=True, exist_ok=True)
                window_file = windows_dir / f"{sample}-{group_id}.window.10.csv"
                _write_window_csv(window_file, sample, group_id, windowed)

            grouped_results[group_id] = windowed
            summary_rows.append(
                {
                    "group_id": group_id,
                    "sample": sample,
                    "sample_RBP": sample_rbp,
                    "barcodes_requested": len(barcode_list),
                    "barcodes_used": len(per_bc_counts),
                    "missing_in_barcode_txt": len(missing_in_barcode),
                    "missing_pileup_files": len(missing_pileup),
                }
            )
            print(
                f"[group={group_id}] used={len(per_bc_counts)}/{len(barcode_list)} "
                f"(missing barcode={len(missing_in_barcode)}, missing pileup={len(missing_pileup)})"
            )

        if not grouped_results:
            print(f"[sample={sample}] No usable groups. Skipping matrix write.")
            continue

        any_sample_written = True
        matrices_dir = matrices_base_dir / sample
        matrices_dir.mkdir(parents=True, exist_ok=True)

        barcode_idx = mm.generate_barcode_idx(grouped_results)
        mutrate_matrix, cov_matrix, mut_matrix, row_labels = mm.generate_matrix(grouped_results, reference_seqlen)
        mutrate_matrix, cov_matrix, mut_matrix, row_labels = mm.filter_nan_rows(
            mutrate_matrix, cov_matrix, mut_matrix, row_labels
        )

        mutrate_out = matrices_dir / f"{sample}.mutrate.matrix{cov_token}.{cell_txt}.csv"
        coverage_out = matrices_dir / f"{sample}.coverage.matrix{cov_token}.{cell_txt}.csv"
        mutant_out = matrices_dir / f"{sample}.mutant.matrix{cov_token}.{cell_txt}.csv"

        convert_to_df(mutrate_matrix, row_labels, barcode_idx, str(mutrate_out))
        convert_to_df(cov_matrix, row_labels, barcode_idx, str(coverage_out))
        convert_to_df(mut_matrix, row_labels, barcode_idx, str(mutant_out))

        summary_out = matrices_dir / f"{sample}.group_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_out, index=False)

        print(f"[sample={sample}] mutrate matrix:  {mutrate_out}")
        print(f"[sample={sample}] coverage matrix: {coverage_out}")
        print(f"[sample={sample}] mutant matrix:   {mutant_out}")
        print(f"[sample={sample}] group summary:   {summary_out}")
        print(
            f"[sample={sample}] shape mutrate={mutrate_matrix.shape}, "
            f"coverage={cov_matrix.shape}, mutant={mut_matrix.shape}"
        )

    if not any_sample_written:
        raise RuntimeError("No groups produced usable aggregated data. Nothing to write.")


if __name__ == "__main__":
    main()
