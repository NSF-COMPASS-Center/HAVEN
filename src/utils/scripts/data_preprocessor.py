import argparse
import os
from pathlib import Path
from data_processing import base_dataset_processor, uniref_dataset_processor, uniprot_dataset_processor
from utils import external_sources_utils

# names of all intermediary files to be created
# UNIREF90_DATA_CSV_FILENAME = "uniref90_viridae.csv"
# UNIREF90_DATA_HOST_UNIPROT_MAPPING_FILENAME = "uniref90_viridae_uniprot_hosts.csv"
# EMBL_HOST_MAPPING_FILENAME = "embl_hosts.csv"
# UNIREF90_DATA_HOST_EMBL_MAPPING_FILENAME = "uniref90_viridae_embl_hosts.csv"
# UNIREF90_DATA_HOST_VIRUSHOSTDB_MAPPING_FILENAME = "uniref90_viridae_virushostdb_hosts.csv"

# names of all intermediary files to be created
UNIREF = "uniref"
UNIREF90_ID = "uniref90_id"
UNIREF90_DATA_CSV_FILENAME = "coronaviridae_s_uniref90.csv"

# Uniprot filenames
UNIPROT = "uniprot"
UNIPROT_ID = "uniprot_id"
UNIPROT_DATA_CSV_FILENAME = "coronaviridae_s_uniprot.csv"



def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess the UniRef90 protein sequences dataset.\nOnly one of the below options can be selected at runtime.')
    parser.add_argument("-it", "--input_type", required=True,
                        help="Type of input sequences. Supported values: uniref, uniprot\n")
    parser.add_argument("-if", "--input_file", required=True,
                        help="Absolute path to input file depending on the option(s) selected.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory where the generated file will be saved.\n")
    parser.add_argument("--fasta_to_csv", action="store_true",
                        help="Convert the input fasta file to csv format.\n")
    parser.add_argument("--uniprot_metadata", action="store_true",
                        help="Get metadata (hosts and embl reference id) of virus from UniProt.\n")
    parser.add_argument("--host_map_embl", action="store_true",
                        help="Get hosts of virus from EMBL.\n")
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
    # parser.add_argument("--filter_mammals_aves", action="store_true",
    #                     help="Filter for virus hosts belonging to mammalia OR aves family using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--filter_vertebrates", action="store_true",
                        help="Filter for virus hosts belonging to Vertebrata clade using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--merge_sequence_data",
                        help="Join the metadata from the input_file with the sequence data from the provided absolute file path.")
    # parser.add_argument("--remove_multi_host_sequences", action="store_true",
    #                     help="Remove sequences with more than one host.")
    # parser.add_argument("--remove_single_host_viruses", action="store_true",
    #                     help="Remove viruses with only one host.")

    args = parser.parse_args()
    return args


def pre_process(config, id):
    input_file_path = config.input_file
    output_dir = config.output_dir

    # 2B. Host mapping from EMBL
    if config.host_map_embl:
        embl_host_mapping_filepath = os.path.join(output_dir, Path(input_file_path).stem + "_embl_host_mapping.csv")
        dataset_embl_hosts_mapping_filepath = os.path.join(output_dir, Path(input_file_path).stem + "_embl_hosts.csv")
        base_dataset_processor.get_virus_hosts_from_embl(input_file_path=input_file_path,
                                                           embl_mapping_filepath=os.path.join(output_dir, embl_host_mapping_filepath),
                                                           output_file_path=os.path.join(output_dir, dataset_embl_hosts_mapping_filepath),
                                                         id=id)
    # 3. Remove sequences with no hosts
    if config.prune_dataset:
        pruned_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_pruned.csv")
        base_dataset_processor.remove_sequences_w_no_hosts(input_file_path=input_file_path,
                                                             output_file_path=pruned_dataset_file_path)

    # 4. Get taxonomy metadata (rank of virus and virus hosts) from NCBI
    if config.taxon_metadata:
        metadata_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_metadata.csv")
        base_dataset_processor.get_virus_metadata(input_file_path=input_file_path,
                                                    taxon_metadata_dir_path=config.taxon_dir,
                                                    output_file_path=metadata_dataset_file_path,
                                                  id=id)

    # 5. Filter for virus and virus_hosts at species level
    if config.filter_species:
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_species.csv")
        base_dataset_processor.get_sequences_at_species_level(input_file_path=input_file_path,
                                                                output_file_path=filtered_dataset_file_path)

    # 6. Filter for virus_hosts belonging to Vertebrata clade
    if config.filter_vertebrates:
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_vertebrates.csv")
        base_dataset_processor.get_sequences_from_vertebrata_hosts(input_file_path=input_file_path,
                                                                     taxon_metadata_dir_path=config.taxon_dir,
                                                                     output_file_path=filtered_dataset_file_path)

    # 7. Merge the metadata with the sequence data
    if config.merge_sequence_data:
        sequence_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_w_seq.csv")
        base_dataset_processor.join_metadata_with_sequences_data(input_file_path=input_file_path,
                                                                   sequence_data_file_path=config.merge_sequence_data,
                                                                   output_file_path=sequence_dataset_file_path,
                                                                 id=id)


def pre_process_uniref90(config):
    input_file_path = config.input_file
    output_dir = config.output_dir

    # 1. Parse the Fasta file
    if config.fasta_to_csv is not None:
        uniref_dataset_processor.parse_uniref_fasta_file(input_file_path=input_file_path,
                                                         output_file_path=os.path.join(output_dir,
                                                                                       UNIREF90_DATA_CSV_FILENAME))

    # 2A. Metadata (host, embl ref id) from UniProt
    if config.uniprot_metadata:
        uniprot_metadata_file_path = os.path.join(output_dir, Path(input_file_path).stem + "uniprot_metadata.csv")
        base_dataset_processor.get_metadata_from_uniprot(input_file_path=input_file_path,
                                                           output_file_path=os.path.join(output_dir,
                                                                                         uniprot_metadata_file_path),
                                                         id=UNIREF90_ID,
                                                         query_uniprot=external_sources_utils.query_uniref)
    pre_process(config, UNIREF90_ID)


def pre_process_uniprot(config):
    input_file_path = config.input_file
    output_dir = config.output_dir

    # 1. Parse the Fasta file
    if config.fasta_to_csv is not None:
        uniprot_dataset_processor.parse_uniprot_fasta_file(input_file_path=input_file_path,
                                                          output_file_path=os.path.join(output_dir,
                                                                                        UNIPROT_DATA_CSV_FILENAME))

        # 2A. Metadata (host, embl ref id) from UniProt
    if config.uniprot_metadata:
        uniprot_metadata_file_path = os.path.join(output_dir, Path(input_file_path).stem + "uniprot_metadata.csv")
        base_dataset_processor.get_metadata_from_uniprot(input_file_path=input_file_path,
                                                         output_file_path=os.path.join(output_dir,
                                                                                       uniprot_metadata_file_path),
                                                         id=UNIPROT_ID,
                                                         query_uniprot=)
    pre_process(config, UNIPROT_ID)

def main():
    config = parse_args()
    if config.input_type == UNIREF:
        pre_process_uniref90(config)
    elif config.input_type == UNIPROT:
        pre_process_uniprot(config)
    return


if __name__ == "__main__":
    main()
    exit(0)
