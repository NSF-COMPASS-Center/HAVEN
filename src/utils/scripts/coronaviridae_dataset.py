import pandas as pd
import os
from Bio import SeqIO
import argparse
import re


PARSED_SEQ_CSV_FILENAME = "coronaviridae_seq_parsed.csv"

# Parse fasta file
# input fasta file
# output: csv file with columns ["uniref90_id", "tax_id", "seq"]
def parse_fasta_file(fasta_file_path, output_directory):
    sequences = []
    i = 0
    no_id_count = 0
    print("START: Parsing fasta file")
    # parse fasta file to extract uniprot_id, protein sequence
    with open(fasta_file_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            i += 1
            print(i)
            try:
                match = re.search(r"\|(\w+)\|", record.id)
                uniprot_id = match.group(1).strip()
                sequences.append({"uniprot_id": uniprot_id, "seq": str(record.seq)})
            except AttributeError:
                no_id_count += 1
    print("END: Parsing fasta file")
    print(len(sequences))
    print(f"Number of records parsed = {i}")
    print(f"Number of records with no tax_id = {no_id_count}")
    df = pd.DataFrame(sequences)


    # write the parsed dataframe to a csv file
    print(f"Writing to file {PARSED_SEQ_CSV_FILENAME}")
    df.to_csv(os.path.join(output_directory, PARSED_SEQ_CSV_FILENAME), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Parse and create Coronaviridae Dataset')
    parser.add_argument('-if', '--input_file', required=True,
                        help="Absolute path to the input file in fasta format\n")
    parser.add_argument('-od', '--output_dir', required=True,
                        help="Absolute path to output directory to create intermediate file.\n")
    args = parser.parse_args()
    return args


def main():
    config = parse_args()
    input_file_path = config.input_file
    output_dir = config.output_dir
    parse_fasta_file(input_file_path, output_dir)


if __name__ == '__main__':
    main()
    exit(0)
