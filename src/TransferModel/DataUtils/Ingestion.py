# Parses fasta files into a pandas dataframe

from collections import defaultdict
from typing import List

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# Returns dataframe
def fastaFileToPandas(path: str = None, rowNames: List[str] = None, types: List[str] = None):
    """
    Reads in fasta file and returns a pandas dataframe
    Args:
        path: path to fasta file
        rowNames: indicates row names for fasta file
        types: indicates dtype of each row (optional)

    Returns: A pandas dataframe with sequences mapping to the "seq" column, and column data indexed by rowNames
    """
    assert str

    data = defaultdict(list)  # {rowNames.prop : [seq1.prop, seq2.prop, ...]}

    for seq_record in SeqIO.parse(path, 'fasta'):
        for ele, k in zip(seq_record.description.split('|'), rowNames):
            data[k].append(ele.strip())
        data['seq'].append(str(seq_record.seq))

    return pd.DataFrame.from_dict(data, dtype=types) if types else pd.DataFrame.from_dict(data)

def pandasToFastaFile(df, path: str = None):
    outFile = open(path, "w")

    for i, row in df.iterrows():
        feature_sequence = SeqRecord(Seq(row['seq']), id=row['Accession'],
            description="|{}|{}|{}".format(row['Protein'], row['Host'], row['Genotype']))
        SeqIO.write(feature_sequence, outFile, 'fasta')

    outFile.close()


def loadFastaFiles(fastaMetaData):
    """
    Load in a single pandas dataframe over multiple fasta files
    Args:
        fastaMetaData: A dictionary of fasta files such that keys path and rowNames exist and are valid inputs to fastaFileToPandas

    Returns:
        A single dataframe joined by columns over all fasta files parsed

    """
    df = pd.concat([fastaFileToPandas(d['path'], d['rowNames']) for d in fastaMetaData], axis=0)
    return df
