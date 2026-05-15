#!/usr/bin/env python3

import pandas as pd
import gzip



flatten = lambda nested: [item for sublist in nested for item in sublist]

class struct:

    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)


def open_file(filename, barcode_to_keep=[]):    
    print (f'Opening {filename}')
    df = pd.read_csv(filename, index_col=0)
    index_list = df.index.tolist()

    header = list(df.columns)
    matrix = df.to_numpy(dtype=float)

    if barcode_to_keep:
        indices_to_keep = [header.index(x) for x in barcode_to_keep]
        matrix = matrix[:, indices_to_keep]
        header = [header[i] for i in indices_to_keep]
        
    return header, index_list, matrix


def convert_to_df(matrix, row_labels, barcode_index, filename):
    if isinstance(barcode_index, dict):
        barcode_list = [bc_id for bc_id, _ in barcode_index.items()]
    else:
        barcode_list = barcode_index

    df = pd.DataFrame(matrix, index=row_labels, columns=barcode_list)
    df_cleaned = df[~(df.isna()).all(axis=1)]
    df_cleaned.to_csv(filename)


def create_opener(file):
    opener = gzip.open if file.endswith('.gz') else open
    mode = 'rt' if file.endswith('.gz') else 'r'
    return opener, mode


def readfq(fp): # this is a generator function
    last = None # this is a buffer keeping the last unprocessed line
    while True: # mimic closure; is it a bad idea?
        if not last: # the first record or a record following a fastq
            for l in fp: # search for the start of the next record
                if l[0] in '>@': # fasta/q header line
                    last = l[:-1] # save this line
                    break
        if not last: break
        name, seqs, last = last[1:].partition(" ")[0], [], None
        for l in fp: # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+': # this is a fasta record
            yield name, ''.join(seqs), None # yield a fasta record
            if not last: break
        else: # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp: # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq): # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs); # yield a fastq record
                    break
            if last: # reach EOF before reading enough quality
                yield name, seq, None # yield a fasta record instead
                break


def create_reference(fasta_file, keep_seq = True):
    opener, mode = create_opener(fasta_file)
    
    seq_id_list = []
    seq_list = []
    count = 0
    with opener(fasta_file, mode) as f:
        for name, seq, _ in readfq(f):
            count += 1
            if (count) % 10000 == 0:
                print(f"Processed {count} transcript...")
            
            seq_id_list.append(name)
            seq_list.append(len(seq))

    if keep_seq:
        seq_id_list = {k:v for k,v in zip(seq_id_list, seq_list)}

    print ("References Processed")
    return seq_id_list


def open_matrices(filename, sep=','):    
    print(f"Opening {filename.split('/')[-1]}!")
    df = pd.read_csv(filename, index_col=0, sep=sep)
    return df


def add_source_prefix(df, source, sep="__"):
    df2 = df.copy()
    df2.columns = [f"{source}{sep}{c}" for c in df2.columns]
    return df2


def check_header(cov_df, rxt_df):
    cov_bc_list = list(cov_df.columns)
    rtx_bc_list = list(rxt_df.columns)
    assert (all([x == y for x,y in zip(rtx_bc_list, cov_bc_list)])), [(x,y) for x,y in zip(rtx_bc_list, cov_bc_list)]


def open_barcode_txt(barcode_file):
    barcode_dict = {}
    with open(barcode_file, 'r') as f:
        for c, line in enumerate(f):
            barcode = line.strip('\n')
            barcode_dict[barcode] = f'bc{c+1}'

    return barcode_dict