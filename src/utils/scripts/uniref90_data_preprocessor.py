import argparse
import os
from pathlib import Path
from data_processing import uniref_dataset_processor

# names of all intermediary files to be created
UNIREF90_DATA_CSV_FILENAME = "uniref90_viridae.csv"
UNIREF90_DATA_HOST_UNIPROT_MAPPING_FILENAME = "uniref90_viridae_uniprot_hosts.csv"
UNIREF90_DATA_HOST_VIRUSHOSTDB_MAPPING_FILENAME = "uniref90_viridae_virushostdb_hosts.csv"

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the UniRef90 protein sequences dataset.\nOnly one of the below options can be selected at runtime.')
    parser.add_argument("-if", "--input_file", required=True,
                        help="Absolute path to input file depending on the option(s) selected.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory where the generated file will be saved.\n")
    parser.add_argument("--fasta_to_csv", action="store_true",
                        help="Convert the input fasta file to csv format.\n")
    parser.add_argument("--host_map_uniprot", action="store_true",
                        help="Get hosts of virus from UniProt.\n")
    parser.add_argument("--host_map_virushostdb",
                        help="Get hosts of virus from VirusHostDB mapping using the absolute path to the mapping file.\n")
    parser.add_argument("--prune_dataset", action="store_true",
                        help="Remove sequences without hosts of virus from the input csv dataset file.\n")
    parser.add_argument("--taxon_dir",
                        help="Absolute path to the NCBI taxon directory.")
    parser.add_argument("--taxon_metadata", action="store_true",
                        help="Get taxonomy metadata using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--filter_species", action="store_true",
                        help="Filter for virus and virus hosts with rank of species.")
    parser.add_argument("--filter_mammals_aves", action="store_true",
                        help="Filter for virus hosts belonging to mammalia OR aves family using the absolute path to the NCBI taxon directory provided in --taxon_dir.")

    args = parser.parse_args()
    return args


def pre_process_uniref90(config):
    output_dir = config.output_dir

    # 1. Parse the Fasta file
    ## config.input_file_path points
    if config.fasta_to_csv:
        uniref_dataset_processor.parse_fasta_file(input_file_path=config.input_file,
                                              output_file_path=os.path.join(output_dir, UNIREF90_DATA_CSV_FILENAME))

    # 2. Get hosts of the virus from which the protein sequences were sampled using either uniprot or virushostdb
    #    depending on the input.
    # 2A. Host mapping from UniProt
    if config.host_map_uniprot:
        uniref_dataset_processor.get_virus_hosts_from_uniprot(input_file_path=config.input_file,
                                                              output_file_path=os.path.join(output_dir,
                                                                                            UNIREF90_DATA_HOST_UNIPROT_MAPPING_FILENAME))

    # 2B. Host mapping from VirusHostDB
    if config.host_map_virushostdb is not None:
        uniref_dataset_processor.get_virus_hosts_from_virushostdb(input_file_path=config.input_file,
                                                                  output_file_path=os.path.join(output_dir, UNIREF90_DATA_HOST_VIRUSHOSTDB_MAPPING_FILENAME),
                                                                  virushostdb_mapping_file=config.host_map_virushostdb)
    # 3. Remove sequences with no hosts
    if config.prune_dataset:
        input_file_path = config.input_file
        pruned_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_pruned.csv")
        uniref_dataset_processor.remove_sequences_w_no_hosts(input_file_path=input_file_path,
                                                             output_file_path=pruned_dataset_file_path)

    # 4. Get taxonomy metadata (rank of virus and virus hosts) from NCBI
    if config.taxon_metadata:
        input_file_path = config.input_file
        metadata_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_metadata.csv")
        uniref_dataset_processor.get_virus_metadata(input_file_path=input_file_path,
                                                             taxon_metadata_dir_path=config.taxon_dir,
                                                             output_file_path=metadata_dataset_file_path)

    # 5. Filter for virus and virus_hosts at species level
    if config.filter_species:
        input_file_path = config.input_file
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_species.csv")
        get_sequences_at_species_level(input_file_path=input_file_path,
                                       output_file_path=filtered_dataset_file_path)

    # 5. Filter for virus_hosts belonging to mammals OR aves
    if config.filter_mammals_aves:
        input_file_path = config.input_file
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_mammals_or_aves.csv")
        get_sequences_from_mammals_aves_hosts(input_file_path=input_file_path,
                                              taxon_metadata_dir_path=config.taxon_dir,
                                       output_file_path=filtered_dataset_file_path)

def main():
    config = parse_args()
    pre_process_uniref90(config)
    return


if __name__ == "__main__":
    main()
    exit(0)
