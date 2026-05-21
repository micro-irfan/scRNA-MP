#!/usr/bin/env python3

"""
Build RBP cell metadata CSV from selected_rbp_transcriptome_window_raw.pkl.

This script converts the notebook workflow in notebooks/mutation_analysis.ipynb
into a reproducible CLI:
1) Load ShapeData pickle
2) Split into DMSO and NAIN3 sample groups
3) Normalize/clean cellbarcodes and sample_RBP fields
4) Write combined metadata CSV with columns:
   - cellbarcodes
   - sample_RBP
   - sample
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _add_shapemap_to_sys_path() -> None:
    this_file = Path(__file__).resolve()
    candidates = [
        Path("shapemap_util"),
        Path("../shapemap_util"),
        this_file.parents[2] / "shapemap_util",  # repo/scRNA-MP/src -> repo
    ]
    for candidate in candidates:
        p = candidate.resolve()
        if p.exists() and str(p) not in sys.path:
            sys.path.append(str(p))


_add_shapemap_to_sys_path()

try:
    from shape_data import ShapeData
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import 'shape_data'. Ensure 'shapemap_util/' is present and "
        "run from repo root (or set PYTHONPATH to include shapemap_util)."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="create_rbp_cell_metadata.py",
        description=(
            "Generate metadata CSV (cellbarcodes, sample_RBP, sample) from a "
            "ShapeData pickle used by notebooks/mutation_analysis.ipynb."
        ),
    )
    parser.add_argument(
        "--pkl-path",
        default="selected_rbp_transcriptome_window_raw.pkl",
        help=(
            "Path to ShapeData pickle. If not found, script also checks "
            "notebooks/<filename>."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="rbp_cells_sample_metadata.csv",
        help="Output CSV path (default: rbp_cells_sample_metadata.csv).",
    )
    parser.add_argument(
        "--dmso-samples",
        nargs="+",
        default=["dmso1", "dmso2"],
        help="Sample names to include in DMSO group.",
    )
    parser.add_argument(
        "--nain3-samples",
        nargs="+",
        default=["nain31", "nain32"],
        help="Sample names to include in NAIN3 group.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help=(
            "Skip ShapeData.get_cell_stats()/get_position_stats() calls. "
            "These are not required for CSV creation but were present in notebook."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    return parser.parse_args()


def resolve_pickle_path(pkl_path: str) -> Path:
    p = Path(pkl_path)
    if p.exists():
        return p

    fallback = Path("notebooks") / p.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find pickle file: {p} or {fallback}"
    )


def load_shapedata(pkl_path: str, verbose: bool = True) -> ShapeData:
    p = resolve_pickle_path(pkl_path)
    data = ShapeData.from_pickle(str(p))

    if verbose:
        print("=== ShapeData Quick Metadata ===")
        print(f"file: {p.resolve()}")
        print(f"shape (positions x cells): {data.shape}")
        print(f"coverage nnz: {data.coverage.nnz:,}")
        if data.mutrate is not None:
            print(f"mutrate nnz: {data.mutrate.nnz:,}")
        if isinstance(data.cells, pd.DataFrame):
            print(f"cells columns ({len(data.cells.columns)}): {list(data.cells.columns)}")

    return data


def _is_na_scalar(x: object) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))


def _strip_prefix_scalar(value: object, sample: object) -> object:
    if _is_na_scalar(value) or _is_na_scalar(sample):
        return value
    value_str = str(value)
    sample_str = str(sample)
    prefix = f"{sample_str}_"
    return value_str[len(prefix):] if value_str.startswith(prefix) else value_str


def _collapse_cellbarcodes(value: object, sample: object) -> object:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        cleaned = [_strip_prefix_scalar(v, sample) for v in value]
        return ";".join(str(v) for v in cleaned)

    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                cleaned = [_strip_prefix_scalar(v, sample) for v in parsed]
                return ";".join(str(v) for v in cleaned)
        except (ValueError, SyntaxError):
            pass

    return _strip_prefix_scalar(value, sample)


def _prep_cells(cells_df: pd.DataFrame) -> pd.DataFrame:
    df = cells_df.copy().reset_index()

    if "cellbarcodes" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "cellbarcodes"})

    if "sample" not in df.columns and "sample_RBP" in df.columns:
        df["sample"] = df["sample_RBP"].astype(str).str.split("_", n=1).str[0]

    if "sample_RBP" not in df.columns and {"sample", "RBP"}.issubset(df.columns):
        df["sample_RBP"] = df["sample"].astype(str) + "_" + df["RBP"].astype(str)

    missing = [c for c in ["cellbarcodes", "sample_RBP", "sample"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after prep: {missing}")

    df["cellbarcodes"] = [
        _collapse_cellbarcodes(v, s) for v, s in zip(df["cellbarcodes"], df["sample"])
    ]
    df["sample_RBP"] = [
        _strip_prefix_scalar(v, s) for v, s in zip(df["sample_RBP"], df["sample"])
    ]

    return df[["cellbarcodes", "sample_RBP", "sample"]]


def build_rbp_cell_metadata_csv(
    rbp_shapedata_nain3: ShapeData,
    rbp_shapedata_dmso: ShapeData,
    output_csv: Path,
) -> pd.DataFrame:
    nain3_df = _prep_cells(rbp_shapedata_nain3.cells)
    dmso_df = _prep_cells(rbp_shapedata_dmso.cells)

    combined_df = pd.concat([nain3_df, dmso_df], axis=0, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print(f"Rows: {len(combined_df):,}")
    print(combined_df.head())
    return combined_df


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    rbp_shapedata = load_shapedata(args.pkl_path, verbose=verbose)

    rbp_shapedata_dmso = rbp_shapedata.filter_cells(
        cell_filters={"sample": list(args.dmso_samples)}
    )
    rbp_shapedata_nain3 = rbp_shapedata.filter_cells(
        cell_filters={"sample": list(args.nain3_samples)}
    )

    if not args.skip_stats:
        rbp_shapedata_dmso.get_cell_stats()
        rbp_shapedata_dmso.get_position_stats()
        rbp_shapedata_nain3.get_cell_stats()
        rbp_shapedata_nain3.get_position_stats()

    build_rbp_cell_metadata_csv(
        rbp_shapedata_nain3=rbp_shapedata_nain3,
        rbp_shapedata_dmso=rbp_shapedata_dmso,
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
