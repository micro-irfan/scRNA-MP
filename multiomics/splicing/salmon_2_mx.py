#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import os

# ----------------------------
# Helpers
# ----------------------------
_TX_VER_RE = re.compile(r"\.\d+$")  # remove .1 .2 ... at end


def strip_tx_version(tx_id: str) -> str:
    return _TX_VER_RE.sub("", tx_id)


def parse_gtf_attributes(attr: str) -> dict:
    """
    Parse the 9th GTF column (attributes) into a dict.
    Example:
      gene_id "ENSMUSG..."; transcript_id "ENSMUST..."; gene_name "Actb";
    """
    d = {}
    # key "value";
    for m in re.finditer(r'(\S+)\s+"([^"]+)"\s*;', attr):
        d[m.group(1)] = m.group(2)
    return d


def load_transcript_annotations(gtf_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      transcript_annot: transcript_id, transcript_name
      tx2gene: transcript_id, gene_symbol (gene_name if available else gene_id)
    """
    gtf_path = Path(gtf_path)

    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
        dtype={"seqname": str, "source": str, "feature": str, "strand": str, "attribute": str},
    )

    # Focus on transcript-level rows (same as the R code)
    gtf = gtf.loc[gtf["feature"] == "transcript", ["attribute"]].copy()

    attrs = gtf["attribute"].map(parse_gtf_attributes)
    tx_id = attrs.map(lambda d: d.get("transcript_id"))
    tx_name = attrs.map(lambda d: d.get("transcript_name"))
    gene_id = attrs.map(lambda d: d.get("gene_id"))
    gene_name = attrs.map(lambda d: d.get("gene_name"))

    transcript_annot = pd.DataFrame(
        {
            "transcript_id": tx_id,
            "transcript_name": tx_name,
        }
    ).dropna(subset=["transcript_id"]).drop_duplicates()

    # tx2gene-style map: transcript_id -> gene symbol/name
    # If gene_name is missing, fall back to gene_id
    tx2gene = pd.DataFrame(
        {
            "transcript_id": tx_id,
            "gene_symbol": gene_name.where(gene_name.notna(), gene_id),
        }
    ).dropna(subset=["transcript_id", "gene_symbol"]).drop_duplicates()

    # Normalize transcript IDs by removing version suffix (optional but matches your R approach)
    transcript_annot["transcript_id"] = transcript_annot["transcript_id"].map(strip_tx_version)
    tx2gene["transcript_id"] = tx2gene["transcript_id"].map(strip_tx_version)

    # If duplicates occur after stripping version, keep first (or you can group/resolve differently)
    transcript_annot = transcript_annot.drop_duplicates(subset=["transcript_id"], keep="first")
    tx2gene = tx2gene.drop_duplicates(subset=["transcript_id"], keep="first")

    return transcript_annot, tx2gene


def read_salmon_quant(quant_sf: str | Path) -> pd.DataFrame:
    q = pd.read_csv(quant_sf, sep="\t")
    # Standardize transcript ID (strip version)
    q["transcript_id"] = q["Name"].map(strip_tx_version)
    return q


def find_salmon_quants(quant_root: str | Path) -> dict[str, Path]:
    """
    Find quant.sf recursively and name each file by its parent folder, like tximport does:
      names(files) <- basename(dirname(files))
    """
    quant_root = Path(quant_root)


    files = [f for f in os.listdir(quant_root) if f.endswith('.quant.sf')]
    files.sort()

    out = {}
    for c, f in enumerate(files):
        # if c >= 10: break
        sample = f.split('/')[-1].split('.')[0]
        out[sample] = f'{quant_root}/{f}'
        
    return out


# ----------------------------
# Main "equivalents"
# ----------------------------
def annotate_single_quant(
    quant_sf: str | Path,
    transcript_annot: pd.DataFrame,
    out_path: str | Path,
) -> pd.DataFrame:
    q = read_salmon_quant(quant_sf)


    annot_map = transcript_annot.set_index("transcript_id")["transcript_name"]
    q["transcript_name"] = q["transcript_id"].map(annot_map)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eps = 1e-8
    q = q[q["TPM"].abs() > eps]

    q.to_csv(out_path, sep="\t", index=False)
    return q


def build_transcript_matrices(
    quant_files: dict[str, Path],
    transcript_annot: pd.DataFrame,
    *,
    use_counts: str = "lengthScaledTPM",  # or "lengthScaledTPM"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
      counts matrix (transcripts x samples)
      tpm matrix (transcripts x samples)

    use_counts:
      - "NumReads": Salmon's estimated counts (recommended)
      - "lengthScaledTPM": approximate tximport-like scaling (see note below)
    """
    annot_map = transcript_annot.set_index("transcript_id")["transcript_name"]

    counts_parts = []
    tpm_parts = []

    # Optional: store effective lengths if you want lengthScaledTPM
    efflen_parts = []

    for sample, f in sorted(quant_files.items()):
        q = read_salmon_quant(f)

        # Keep only what we need
        q = q[["transcript_id", "TPM", "NumReads", "EffectiveLength"]].copy()
        q = q.drop_duplicates(subset=["transcript_id"], keep="first").set_index("transcript_id")

        eps = 1e-8
        q = q[q["TPM"].abs() > eps]

        tpm_parts.append(q["TPM"].rename(sample))
        counts_parts.append(q["NumReads"].rename(sample))
        efflen_parts.append(q["EffectiveLength"].rename(sample))

    tpm = pd.concat(tpm_parts, axis=1).fillna(0.0)
    numreads = pd.concat(counts_parts, axis=1).fillna(0)
    efflen = pd.concat(efflen_parts, axis=1).fillna(0.0)

    # Choose counts output
    if use_counts == "NumReads":
        counts = numreads

    elif use_counts == "lengthScaledTPM":
        # ensure numeric
        tpm = tpm.apply(pd.to_numeric, errors="coerce")

        # align libsize to TPM columns
        libsize = numreads.sum(axis=0).apply(pd.to_numeric, errors="coerce").reindex(tpm.columns)

        # representative length per transcript, aligned to TPM rows
        rep_len = efflen.median(axis=1, skipna=True).apply(pd.to_numeric, errors="coerce").reindex(tpm.index)

        # avoid divide-by-zero global median (ignore 0 / nonpositive)
        global_med = rep_len.where(rep_len > 0).median(skipna=True)

        base_counts = tpm.mul(libsize / 1e6, axis=1)

        rep_len_norm = (rep_len / global_med).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        counts = (
            base_counts
            .mul(rep_len_norm, axis=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # # Approximation:
        # # 1) compute a "representative length" per transcript across samples (median effective length)
        # # 2) convert TPM back to counts using library size ~= sum(NumReads) per sample
        # #
        # # NOTE: tximport's method has specific details; this is a practical approximation.
        # rep_len = efflen.median(axis=1, skipna=True)
        # libsize = numreads.sum(axis=0)  # per-sample

        # # counts ~ TPM * rep_len * libsize / 1e6 / (rep_len?)  -> here we use:
        # # TPM = (scaled_count / sum_scaled_count)*1e6; with rep_len we approximate scaled_count ~ count/rep_len
        # # A common practical back-transform used is: count ~ TPM * libsize / 1e6
        # # and then length-scale by rep_len. We'll do: count ~ TPM * libsize / 1e6
        # # then adjust by rep_len/median(rep_len) to preserve length scaling.
        # # If you want strictly count ~ TPM * efflen * libsize / 1e6, use that variant instead.
        # base_counts = tpm.mul(libsize / 1e6, axis=1)
        # # length adjust: multiply by rep_len normalized (avoids huge inflation for long tx)
        # rep_len_norm = rep_len / rep_len.median(skipna=True)
        # counts = base_counts.mul(rep_len_norm, axis=0).fillna(0.0)

        print (counts)

    else:
        raise ValueError("use_counts must be 'NumReads' or 'lengthScaledTPM'")

    # Add annotations as columns like the R output (transcript_id + transcript_name first)
    counts_out = counts.copy()
    # counts_out.insert(0, "transcript_name", counts_out.index.map(annot_map))
    counts_out.insert(0, "transcript_id", counts_out.index)

    tpm_out = tpm.copy()
    # tpm_out.insert(0, "transcript_name", tpm_out.index.map(annot_map))
    tpm_out.insert(0, "transcript_id", tpm_out.index)

    return counts_out.reset_index(drop=True), tpm_out.reset_index(drop=True)


def build_gene_matrices_from_transcripts(
    counts_tx: pd.DataFrame,
    tpm_tx: pd.DataFrame,
    tx2gene: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Python equivalent of tximport(..., tx2gene=..., ignoreTxVersion=TRUE) gene-level summarization:
      - counts: sum transcript counts per gene (this matches common tx2gene aggregation)
      - TPM: sum transcript TPM per gene (common practice; preserves "per million" scale)
    """
    tx_to_gene = tx2gene.set_index("transcript_id")["gene_symbol"]

    # Identify sample columns (everything except the first two metadata cols)
    count_cols = [c for c in counts_tx.columns if c not in ("transcript_id", "transcript_name")]
    tpm_cols = [c for c in tpm_tx.columns if c not in ("transcript_id", "transcript_name")]

    counts_tx = counts_tx.copy()
    tpm_tx = tpm_tx.copy()

    counts_tx["gene_symbol"] = counts_tx["transcript_id"].map(tx_to_gene)
    tpm_tx["gene_symbol"] = tpm_tx["transcript_id"].map(tx_to_gene)

    # Drop transcripts that don't map
    counts_gene = (
        counts_tx.dropna(subset=["gene_symbol"])
        .groupby("gene_symbol", as_index=True)[count_cols]
        .sum()
    )

    tpm_gene = (
        tpm_tx.dropna(subset=["gene_symbol"])
        .groupby("gene_symbol", as_index=True)[tpm_cols]
        .sum()
    )

    return counts_gene, tpm_gene

def write_tsv_no_index_header(df, path, sep="\t", float_fmt=None):
    """
    Write a DataFrame to TSV with:
      - index written as first column
      - NO header cell for the index
      - clean column headers (no leading tab)
    """
    with open(path, "w") as f:
        # Write column headers ONLY
        f.write(sep.join(map(str, df.columns)) + "\n")

        for idx, row in df.iterrows():
            if float_fmt:
                values = [
                    float_fmt.format(v) if isinstance(v, float) else str(v)
                    for v in row
                ]
            else:
                values = map(str, row)

            f.write(str(idx) + sep + sep.join(values) + "\n")

# ----------------------------
# Example usage (edit paths)
# ----------------------------
if __name__ == "__main__":
    workdir = "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/"
    gtf_path = Path(f"{workdir}/ref/Homo_sapiens.GRCh38.115.gtf").expanduser()
    transcript_annot, tx2gene = load_transcript_annotations(gtf_path)

    sample_list = ['DMS_PIP_NAIN3_3in4', 'DMS_PIP_NAIN3_1in4', 'DM_PIP_NAIN3_3in4', 'DM_PIP_NAIN3_1in4']

    for sample_id in sample_list:
        quant_root = Path(f"{workdir}/cellline/results_3/salmon_results/salmon/{sample_id}").expanduser()

        out_path = Path(f"{workdir}/cellline/results_3/salmon_results/tpm/{sample_id}")
        Path(out_path).mkdir(parents=True, exist_ok=True)

        # 2) Multi-sample transcript matrices (counts + TPM)
        quant_files = find_salmon_quants(quant_root)

        counts_tbl, tpm_tbl = build_transcript_matrices(
            quant_files,
            transcript_annot,
            use_counts="lengthScaledTPM",  # recommended; see note above if you want lengthScaledTPM approx
        )

        counts_tbl.to_csv(f"{out_path}/isoform_expression_counts.tsv", sep="\t", index=False)

        tpm_out = tpm_tbl.set_index("transcript_id")
        tpm_out.index.name = None

        write_tsv_no_index_header(tpm_out, f"{out_path}/isoform_expression_tpm.tsv")

        # 3) Gene-level matrices (tx2gene aggregation)
        gene_counts, gene_tpm = build_gene_matrices_from_transcripts(counts_tbl, tpm_tbl, tx2gene)
        gene_counts.to_csv(f"{out_path}/gene_expression_counts.tsv", sep="\t")
        gene_tpm.to_csv(f"{out_path}/gene_expression_tpm.tsv", sep="\t")
