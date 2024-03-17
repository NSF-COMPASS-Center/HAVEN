import argparse
import os
from pathlib import Path
from data_processing import uniref_dataset_processor

# names of all intermediary files to be created
# UNIREF90_DATA_CSV_FILENAME = "uniref90_viridae.csv"
# UNIREF90_DATA_HOST_UNIPROT_MAPPING_FILENAME = "uniref90_viridae_uniprot_hosts.csv"
# EMBL_HOST_MAPPING_FILENAME = "embl_hosts.csv"
# UNIREF90_DATA_HOST_EMBL_MAPPING_FILENAME = "uniref90_viridae_embl_hosts.csv"
# UNIREF90_DATA_HOST_VIRUSHOSTDB_MAPPING_FILENAME = "uniref90_viridae_virushostdb_hosts.csv"

# names of all intermediary files to be created
UNIREF90_DATA_CSV_FILENAME = "coronaviridae_s_uniref90.csv"
UNIPROT_DATA_CSV_FILENAME = "coronaviridae_s_uniprot.csv"
UNIREF90_DATA_HOST_UNIPROT_MAPPING_FILENAME = "coronaviridae_s_uniref90_uniprot_hosts.csv"
EMBL_HOST_MAPPING_FILENAME = "coronaviridae_s_uniref90_embl_host_mapping.csv"
UNIREF90_DATA_HOST_EMBL_MAPPING_FILENAME = "coronaviridae_s_uniref90_embl_hosts.csv"
UNIREF90_DATA_HOST_VIRUSHOSTDB_MAPPING_FILENAME = "uniref90_viridae_virushostdb_hosts.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess the UniRef90 protein sequences dataset.\nOnly one of the below options can be selected at runtime.')
    parser.add_argument("-if", "--input_file", required=True,
                        help="Absolute path to input file depending on the option(s) selected.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory where the generated file will be saved.\n")
    parser.add_argument("--fasta_to_csv",
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
    parser.add_argument("--filter_mammals_aves", action="store_true",
                        help="Filter for virus hosts belonging to mammalia OR aves family using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--filter_vertebrates", action="store_true",
                        help="Filter for virus hosts belonging to Vertebrata clade using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--merge_sequence_data",
                        help="Join the metadata from the input_file with the sequence data from the provided absolute file path.")
    parser.add_argument("--remove_multi_host_sequences", action="store_true",
                        help="Remove sequences with more than one host.")
    parser.add_argument("--remove_single_host_viruses", action="store_true",
                        help="Remove viruses with only one host.")

    args = parser.parse_args()
    return args


def pre_process_uniref90(config):
    input_file_path = config.input_file
    output_dir = config.output_dir

    # 1. Parse the Fasta file
    if config.fasta_to_csv is not None:
        if config.fasta_to_csv == "uniref":
            uniref_dataset_processor.parse_uniref_fasta_file(input_file_path=input_file_path,
                                                  output_file_path=os.path.join(output_dir, UNIREF90_DATA_CSV_FILENAME))
        elif config.fasta_to_csv == "uniprot":
            uniref_dataset_processor.parse_uniprot_fasta_file(input_file_path=input_file_path,
                                                             output_file_path=os.path.join(output_dir,
                                                                                           UNIPROT_DATA_CSV_FILENAME))

    # 2. Get hosts of the virus from which the protein sequences were sampled using either uniprot or virushostdb
    #    depending on the input.
    # 2A. Metadata (host, embl ref id) from UniProt
    if config.uniprot_metadata:
        uniref_dataset_processor.get_metadata_from_uniprot(input_file_path=input_file_path,
                                                              output_file_path=os.path.join(output_dir,
                                                                                            UNIREF90_DATA_HOST_UNIPROT_MAPPING_FILENAME))
    # 2B. Host mapping from EMBL
    if config.host_map_embl:
        uniref_dataset_processor.get_virus_hosts_from_embl(input_file_path=input_file_path,
                                                           embl_mapping_filepath=os.path.join(output_dir, EMBL_HOST_MAPPING_FILENAME),
                                                           output_file_path=os.path.join(output_dir, UNIREF90_DATA_HOST_EMBL_MAPPING_FILENAME))
    # 2C. Host mapping from VirusHostDB
    if config.host_map_virushostdb is not None:
        uniref_dataset_processor.get_virus_hosts_from_virushostdb(input_file_path=input_file_path,
                                                                  output_file_path=os.path.join(output_dir,
                                                                                                UNIREF90_DATA_HOST_VIRUSHOSTDB_MAPPING_FILENAME),
                                                                  virushostdb_mapping_file=config.host_map_virushostdb)
    # 3. Remove sequences with no hosts
    if config.prune_dataset:
        pruned_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_pruned.csv")
        uniref_dataset_processor.remove_sequences_w_no_hosts(input_file_path=input_file_path,
                                                             output_file_path=pruned_dataset_file_path)

    # 4. Get taxonomy metadata (rank of virus and virus hosts) from NCBI
    if config.taxon_metadata:
        metadata_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_metadata.csv")
        uniref_dataset_processor.get_virus_metadata(input_file_path=input_file_path,
                                                    taxon_metadata_dir_path=config.taxon_dir,
                                                    output_file_path=metadata_dataset_file_path)

    # 5. Filter for virus and virus_hosts at species level
    if config.filter_species:
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_species.csv")
        uniref_dataset_processor.get_sequences_at_species_level(input_file_path=input_file_path,
                                                                output_file_path=filtered_dataset_file_path)

    # 6A. Filter for virus_hosts belonging to family of mammals OR aves
    if config.filter_mammals_aves:
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_mammals_or_aves.csv")
        uniref_dataset_processor.get_sequences_from_mammals_aves_hosts(input_file_path=input_file_path,
                                                                       taxon_metadata_dir_path=config.taxon_dir,
                                                                       output_file_path=filtered_dataset_file_path)
    # 6B. Filter for virus_hosts belonging to Vertebrata clade
    if config.filter_vertebrates:
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_vertebrates.csv")
        uniref_dataset_processor.get_sequences_from_vertebrata_hosts(input_file_path=input_file_path,
                                                                     taxon_metadata_dir_path=config.taxon_dir,
                                                                     output_file_path=filtered_dataset_file_path)

    # 7. Merge the metadata with the sequence data
    if config.merge_sequence_data:
        sequence_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_w_seq.csv")
        uniref_dataset_processor.join_metadata_with_sequences_data(input_file_path=input_file_path,
                                                                   sequence_data_file_path=config.merge_sequence_data,
                                                                   output_file_path=sequence_dataset_file_path)

    # 8. Remove sequences with more than one host
    if config.remove_multi_host_sequences:
        uniref_dataset_processor.remove_duplicate_sequences(input_file_path=input_file_path,
                                                            output_file_path=os.path.join(output_dir, Path(
                                                                input_file_path).stem + "_wo_multi_host_seq.csv"),
                                                            filtered_file_path=os.path.join(output_dir, Path(
                                                                input_file_path).stem + "_multi_host_seq.csv"))

    # 9. Remove viruses with only one host
    if config.remove_single_host_viruses:
        uniref_dataset_processor.remove_sequences_of_virus_with_one_host(input_file_path=input_file_path,
                                                                         output_file_path=os.path.join(output_dir, Path(
                                                                             input_file_path).stem + "_wo_single_host_virus.csv"),
                                                                         filtered_file_path=os.path.join(output_dir,
                                                                                                         Path(
                                                                                                             input_file_path).stem + "_single_host_virus.csv"))


def main():
    config = parse_args()
    pre_process_uniref90(config)
    return


if __name__ == "__main__":
    main()
    exit(0)
