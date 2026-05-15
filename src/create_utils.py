#!/usr/bin/env python3


import logging
import logging.handlers
import os
from pathlib import Path
from typing import Union, Optional  # <-- use Union/Optional on 3.8
from collections import namedtuple
import numpy as np


class struct:

    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)


def get_logger(
    name: str = __name__,
    log_dir: Union[str, Path] = "logs",
    log_file: str = "app.log",
    level: Optional[Union[str, int]] = None,
) -> logging.Logger:
    """
    Create (or reuse) a configured logger.

    Env overrides:
      LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL  (file handler min level)
      LOG_CONSOLE_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL (console min level)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    env_level = os.getenv("LOG_LEVEL")
    if isinstance(level, str):
        level = level.upper()
    if env_level:
        level = env_level

    numeric_level = getattr(logging, (level or "DEBUG"), logging.DEBUG)
    logger.setLevel(logging.DEBUG)  # keep logger open; handlers filter

    file_fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter("%(levelname)s | %(message)s")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.handlers.RotatingFileHandler(
        filename=str(log_dir / log_file),
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(numeric_level)     # DEBUG+ (or your chosen level) to file
    fh.setFormatter(file_fmt)

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, os.getenv("LOG_CONSOLE_LEVEL", "INFO"), logging.INFO))
    ch.setFormatter(console_fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # quiet noisy deps (optional)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

    return logger


def create_pileup_dict_by_transrcipt(reference, cut_off=0, default_value = None):        
    pileup = {gene:{p+1:default_value for p in range(seq_len-cut_off)} for gene, seq_len in reference.items()}
    return pileup    


class Pileup:
    
    bases = 'ACGT'

    def __init__(self, field):
        self.tx_id = field[0]
        self.pos = int(field[1])
        self.ref = field[2]
        self.cov = int(field[3])
        self.pileup = field[4]
        self.mutrate = 0

    def read_pileup(self):
        self.mut = 0
        self.mutation = {k: 0 for k in self.bases}  # e.g., {'A':0,'C':0,'G':0,'T':0,'N':0}
        self.mutrate = 0.0
        self.ref_base = 0

        if self.cov == 0 or not self.pileup:
            return

        s = self.pileup
        i = 0
        while i < len(s):
            ch = s[i]

            # start-of-read marker: skip '^' and the following one char (mapQ)
            if ch == '^':
                i += 2
                continue

            # end-of-read marker: skip
            if ch == '$':
                i += 1
                continue

            # match to reference at this position ('.' or ','): not a mutation
            if ch == '.' or ch == ',':
                i += 1
                continue

            # deletion padding from previous position: not a new event here
            if ch == '*':
                i += 1
                continue

            # insertion or deletion starting at this position
            if ch == '+' or ch == '-':
                i += 1  # move past +/- sign

                # parse the length
                j = i
                while j < len(s) and s[j].isdigit():
                    j += 1
                if j == i:
                    # no digits found; malformed pileup, bail out safely
                    i += 1
                    continue

                indel_len = int(s[i:j])
                i = j  # now i points to the first base of the indel sequence

                # grab the indel sequence (may be shorter if truncated)
                indel_seq = s[i:i+indel_len]
                # count ONE event per indel (don’t re-count its bases)
                key = f"{ch}{indel_len}{indel_seq}"
                self.mutation[key] = self.mutation.get(key, 0) + 1
                self.mut += 1

                # skip over the indel sequence completely
                i += indel_len
                continue

            # mismatches (letters): these represent bases differing from ref
            up = ch.upper()
            if up in self.bases:        # e.g., A/C/G/T/N
                self.mutation[up] += 1
                self.mut += 1
                i += 1
                continue

            # anything else (unexpected symbols): skip
            i += 1

        # compute rates
        self.mutrate = self.mut / self.cov
        # ref_base = reads covering position minus reads that showed a mutation event at this locus
        self.ref_base = self.cov - self.mut

WINDOW_LEN = 10
Mutation = namedtuple('Mutation', 'mut cov mutrate base_count')

def create_fixed_windows(data, reference, window_length=WINDOW_LEN, min_bases=1):
    """
    Create non-overlapping fixed windows of length `window_length`.
    Each window aggregates coverage and mutation across positions in that window.
    
    Parameters
    ----------
    data : dict
        {gene: {pos: Mutation(cov, mut, mutrate)}}
    reference : dict
        {gene: seq_len}
    window_length : int
        Length of each fixed window.

    Returns
    -------
    results : dict
        {gene: {window_start: Mutation(mean_mut, mean_cov, mutrate)}}
    """
    results = {}
    for gene, gene_data in data.items():
        results[gene] = {}
        seq_len = reference[gene]

        # iterate in fixed, non-overlapping windows
        # e.g. 0, window_length, 2*window_length, ...
        # only take full windows (no partial at the end)
        last_start = seq_len - window_length
        if last_start < 0:
            # sequence shorter than a single window → skip
            continue

        for window_start in range(1, seq_len+1, window_length):
            coverage_vals = []
            mutation_vals = []

            actual_len = min(window_length, seq_len - window_start + 1)
            for offset in range(actual_len):
                p = window_start + offset
                tmp = gene_data.get(p)

                # if we require all positions to exist
                if not tmp:
                    continue

                if tmp.cov == 0:
                    continue

                cov = int(tmp.cov)
                mut = int(tmp.mut)

                # filter out crazy positions
                if cov > 0 and (mut / cov) > 0.2:
                    continue

                coverage_vals.append(cov)
                mutation_vals.append(mut)

            # no valid positions or incomplete window → skip
            base_count = len(coverage_vals)
            if base_count < min_bases:
                continue

            mean_cov = np.nanmean(coverage_vals)
            mean_mut = np.nanmean(mutation_vals)

            if mean_cov == 0:
                mutrate = -1
            else:
                mutrate = mean_mut / mean_cov

            results[gene][window_start] = Mutation(mean_mut, mean_cov, mutrate, base_count)

    return results


def create_single_base(data, reference):
    """
    Compute single-base mutation statistics (no windows).

    Parameters
    ----------
    data : dict
        {gene: {pos: Mutation(cov, mut, mutrate)}}
    reference : dict
        {gene: seq_len}

    Returns
    -------
    results : dict
        {gene: {pos: Mutation(mut, cov, mutrate)}}
        Default positions remain Mutation(0, 0, -1) from the initializer.
    """
    results = {}

    for gene, gene_data in data.items():
        results[gene] = {}
        seq_len = reference[gene]

        for pos, tmp in gene_data.items():
            # basic bounds check (in case there are weird positions)
            if pos < 0 or pos >= seq_len:
                continue

            if not tmp:
                continue

            # skip if zero coverage
            if tmp.cov == 0:
                continue

            cov = int(tmp.cov)
            mut = int(tmp.mut)

            if cov == 0:
                continue

            # filter out crazy positions
            mutrate = mut / cov
            if mutrate > 0.2:
                continue

            # store as-is (no averaging, it's single base)
            results[gene][pos] = Mutation(mut, cov, mutrate, 1)

    return results


def create_rolling_windows(data, reference, window_length=WINDOW_LEN):
    results = create_pileup_dict_by_transrcipt(reference, window_length, default_value=Mutation(0, 0, -1, 0))
    
    for gene, gene_data in data.items():
        seq_len = reference[gene]
        for pos, tmp in gene_data.items():
            if pos > seq_len - window_length: break
            if not tmp: continue

            skip = False
            coverage = []
            mutation = []
            for i in range(window_length):
                p = pos+i
                tmp = gene_data[p]
                if not tmp:
                    skip = True
                    break

                if tmp.cov == 0: continue

                cov = int(tmp.cov)
                mut = int(tmp.mut)
                if mut/cov > 0.2: continue

                coverage.append(cov)
                mutation.append(mut)
            
            if skip or not coverage: continue

            base_count = len(coverage)
            coverage = np.nanmean(coverage)
            mutation = np.nanmean(mutation)

            mutrate = mutation/coverage 
            results[gene][pos] = Mutation(mutation, coverage, mutrate, base_count)

    return  results 
