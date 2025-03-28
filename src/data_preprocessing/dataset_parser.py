from Bio import SeqIO
import re
import pandas as pd

# Column names at various stages of dataset curation
TAX_ID = "tax_id"
SEQUENCE = "seq"


# Parse uniprot fasta file
# input fasta file
# output: csv file with columns [id_col, "tax_id", "seq"]
def parse_uniprot_fasta_file(input_file_path, id_col):
    sequences = []
    i = 0
    no_id_count = 0
    print("START: Parsing fasta file {input_file_path}")
    # parse fasta file to extract uniprot_id, tax_id of virus/organism, and protein sequence
    with open(input_file_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            i += 1
            try:
                match = re.search(r"..\|(\S+)\|.+OX=(\d+).+", record.description)
                uniprot_id = match.group(1).strip()
                tax_id = match.group(2).strip()
                sequences.append({id_col: uniprot_id, TAX_ID: tax_id, SEQUENCE: str(record.seq)})
            except AttributeError:
                no_id_count += 1
                print(record.description)
    print("END: Parsing fasta file")
    print(len(sequences))
    print(f"Number of records parsed = {i}")
    print(f"Number of records with no tax_id = {no_id_count}")
    return pd.DataFrame(sequences)

# Parse uniref90 fasta file
# input fasta file
# output: csv file with columns [id_col, "tax_id", "seq"]
def parse_uniref_fasta_file(input_file_path, id_col):
    sequences = []
    i = 0
    no_id_count = 0
    print("START: Parsing fasta file {input_file_path}")
    # parse fasta file to extract uniref90_id, tax_id of virus/organism, and protein sequence
    with open(input_file_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            i += 1
            print(i)
            try:
                match = re.search(r".+TaxID=(\d+).+", record.description)
                tax_id = match.group(1).strip()
                sequences.append({id_col: record.id, TAX_ID: tax_id, SEQUENCE: str(record.seq)})
            except AttributeError:
                no_id_count += 1
    print("END: Parsing fasta file")
    print(len(sequences))
    print(f"Number of records parsed = {i}")
    print(f"Number of records with no tax_id = {no_id_count}")
    return pd.DataFrame(sequences)

