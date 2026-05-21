#!/usr/bin/env python3
"""
Compute pairwise Hamming distances from a FASTA file and export pairs
whose distance is below a user threshold.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "List FASTA sequence pairs with Hamming distance below a threshold."
        )
    )
    p.add_argument(
        "--fasta",
        default="target_guides_rbp.fa",
        help="Input FASTA file.",
    )
    p.add_argument(
        "--max-distance",
        type=int,
        required=True,
        help="Keep only pairs with hamming_distance < this value.",
    )
    p.add_argument(
        "--output-csv",
        default="hamming_pairs_below_threshold.csv",
        help="Output CSV path.",
    )
    return p.parse_args()


def read_fasta(path: str) -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records: List[Tuple[str, str]] = []
    curr_id = None
    curr_seq_parts: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if curr_id is not None:
                    records.append((curr_id, "".join(curr_seq_parts).upper()))
                curr_id = s[1:].strip().split()[0]
                curr_seq_parts = []
            else:
                if curr_id is None:
                    raise ValueError("Invalid FASTA: sequence line appears before header.")
                curr_seq_parts.append(s)

    if curr_id is not None:
        records.append((curr_id, "".join(curr_seq_parts).upper()))

    if not records:
        raise ValueError(f"No FASTA records found in: {path}")
    return records


def hamming_distance(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def main() -> None:
    args = parse_args()
    if args.max_distance < 0:
        raise ValueError("--max-distance must be >= 0")

    records = read_fasta(args.fasta)
    n = len(records)

    rows: List[dict] = []
    skipped_unequal_len = 0
    total_pairs = 0

    for i in range(n):
        id1, seq1 = records[i]
        for j in range(i + 1, n):
            id2, seq2 = records[j]
            total_pairs += 1

            if len(seq1) != len(seq2):
                skipped_unequal_len += 1
                continue

            dist = hamming_distance(seq1, seq2)
            if dist < args.max_distance:
                rows.append(
                    {
                        "seq_id_1": id1,
                        "seq_id_2": id2,
                        "length": len(seq1),
                        "hamming_distance": dist,
                    }
                )

    rows.sort(
        key=lambda r: (
            int(r["hamming_distance"]),
            str(r["seq_id_1"]),
            str(r["seq_id_2"]),
        )
    )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=["seq_id_1", "seq_id_2", "length", "hamming_distance"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] fasta: {os.path.abspath(args.fasta)}")
    print(f"[DONE] sequences: {n}")
    print(f"[DONE] total_pairs: {total_pairs}")
    print(f"[DONE] skipped_unequal_length_pairs: {skipped_unequal_len}")
    print(f"[DONE] kept_pairs_below_threshold: {len(rows)}")
    print(f"[DONE] threshold_rule: hamming_distance < {args.max_distance}")
    print(f"[DONE] output_csv: {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
