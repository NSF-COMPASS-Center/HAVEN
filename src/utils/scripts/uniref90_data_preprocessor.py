import argparse
import os
from data_processing import uniref_dataset_processor

# names of all intermediary files to be created
UNIREF90_DATA_CSV_FILENAME = "uniref90_viridae.csv"
UNIREF90_DATA_HOST_UNIPROT_MAPPING_FILENAME = "uniref90_viridae_uniprot_hosts.csv"
UNIREF90_DATA_HOST_VIRUSHOSTDB_MAPPING_FILENAME = "uniref90_viridae_virushostdb_hosts.csv"

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the UniRef90 protein sequences fasta file')
    parser.add_argument("-if", "--input_file", required=True,
                        help="Absolute path to input protein sequences fasta file.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory where the generated file will be saved.\n")
    args = parser.parse_args()
    return args


def pre_process_uniref90(input_file_path, output_dir):

    viridae_csv_file_path = os.path.join(output_dir, UNIREF90_DATA_CSV_FILENAME)
    # 1. Parse the Fasta file
    uniref_dataset_processor.parse_fasta_file(input_file_path=input_file_path,
                                              output_file_path=viridae_csv_file_path)

    # 2. Get hosts of the virus from which the protein sequences were sampled
    uniref_dataset_processor.get_virus_hosts_from_uniprot(input_file_path=viridae_csv_file_path,
                                                          output_file_path=os.path.join(output_dir, UNIREF90_DATA_HOST_UNIPROT_MAPPING_FILENAME))


def main():
    config = parse_args()
    input_file_path = config.input_file
    output_dir = config.output_dir
    pre_process_uniref90(input_file_path, output_dir)
    return


if __name__ == '__main__':
    main()
    exit(0)
