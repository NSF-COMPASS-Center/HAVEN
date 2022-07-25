# Parses fasta files into a pandas dataframe

from Bio import SeqIO
from typing import List
from collections import defaultdict
import pandas as pd


# path: path to fasta file
# rowNames: indicates row names for fasta file
# types: indicates dtype of each row (optional)
# Returns dataframe
def fastaFileToPandas(path: str = None, rowNames: List[str] = None, types: List[str] = None):
    assert str

    data = defaultdict(list)  # {rowNames.prop : [seq1.prop, seq2.prop, ...]}

    for seq_record in SeqIO.parse(path, 'fasta'):
        for ele, k in zip(seq_record.description.split('|'), rowNames):
            data[k].append(ele.strip())
        data['seq'].append(str(seq_record.seq))

    return pd.DataFrame.from_dict(data, dtype=types) if types else pd.DataFrame.from_dict(data)


# See hepTest.yaml for formatting
def loadFastaFiles(fastaMetaData):
    df = pd.concat([fastaFileToPandas(d['path'], d['rowNames']) for d in fastaMetaData], axis=0)
    return df


