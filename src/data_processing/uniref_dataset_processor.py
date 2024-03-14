import itertools

import pandas as pd
import os
import numpy as np
import re
from ast import literal_eval
from Bio import SeqIO
from multiprocessing import Pool
from itertools import repeat
from pathlib import Path
import ast
from utils import external_sources_utils

N_CPU = 2
EMBL_QUERY_PAYLOAD_SIZE = 200

# Virus Host DB keywords
VIRUS_HOST_DB_VIRUS_TAX_ID = "virus tax id"
VIRUS_HOST_DB_VIRUS_NAME = "virus name"
VIRUS_HOST_DB_HOST_TAX_ID = "host tax id"
VIRUS_HOST_DB_HOST_NAME = "host name"

## NCBI Taxonomy keywords
NAME = "Name"
RANK = "Rank"
NCBI_TAX_ID = "TaxID"
TAXONKIT_DB = "TAXONKIT_DB"
SPECIES = "species"

# Column names at various stages of dataset curation
UNIREF90_ID = "uniref90_id"
TAX_ID = "tax_id"
SEQUENCE = "seq"
HOST_TAX_IDS = "host_tax_ids"
UNIPROT_HOST_TAX_IDS ="uniprot_host_tax_ids"
EMBL_REF_ID = "embl_ref_id"
EMBL_HOST_NAME ="embl_host_name"
HOST_COUNT = "host_count"
VIRUS_NAME = "virus_name"
VIRUS_TAXON_RANK = "virus_taxon_rank"
VIRUS_HOST_TAX_ID = "virus_host_tax_id"
VIRUS_HOST_NAME = "virus_host_name"
VIRUS_HOST_TAXON_RANK = "virus_host_taxon_rank"


# Parse fasta file
# input fasta file
# output: csv file with columns ["uniref90_id", "tax_id", "seq"]
def parse_fasta_file(input_file_path, output_file_path):
    sequences = []
    i = 0
    no_id_count = 0
    print("START: Parsing fasta file")
    # parse fasta file to extract uniref90_id, tax_id of virus/organism, and protein sequence
    with open(input_file_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            i += 1
            print(i)
            try:
                match = re.search(r".+TaxID=(\d+).+", record.description)
                tax_id = match.group(1).strip()
                sequences.append({UNIREF90_ID: record.id, TAX_ID: tax_id, SEQUENCE: str(record.seq)})
            except AttributeError:
                no_id_count += 1
    print("END: Parsing fasta file")
    print(len(sequences))
    print(f"Number of records parsed = {i}")
    print(f"Number of records with no tax_id = {no_id_count}")
    df = pd.DataFrame(sequences)

    # write the parsed dataframe to a csv file
    print(f"Writing to file {output_file_path}")
    df.to_csv(output_file_path, index=False)


# Get hosts of virus from virus_host db using virus tax id
# input: parsed csv file of all sequences
# output: csv file with hosts of virus. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
# The sequences will be joined back and compiled into one dataset at a later stage
def get_virus_hosts_from_virushostdb(input_file_path, output_file_path, virushostdb_mapping_file):
    print("START: Get virus hosts from Virus Host DB")
    # read the mapping data
    mapping_df = pd.read_csv(virushostdb_mapping_file, sep="\t")
    print(f"Virus Host DB mapping dataset size = {mapping_df.shape}")

    # read the parsed uniref90_data csv file
    df = pd.read_csv(input_file_path)
    print(f"Uniref90 dataset size = {df.shape}")

    # retain only uniref90_ids to save memory
    df = df[[UNIREF90_ID, TAX_ID]]

    # join the two dfs on the virus tax id
    mapped_df = df.merge(mapping_df, how="left", left_on=TAX_ID, right_on=VIRUS_HOST_DB_VIRUS_TAX_ID)
    print(f"Mapped dataset size = {mapped_df.shape}")

    # rename "host tax id" to HOST_TAX_IDS
    mapped_df.rename(columns={VIRUS_HOST_DB_VIRUS_NAME: VIRUS_NAME,
                              VIRUS_HOST_DB_HOST_TAX_ID: HOST_TAX_IDS,
                              VIRUS_HOST_DB_HOST_NAME: VIRUS_HOST_NAME}, inplace=True)

    # retain only [UNIREF90_ID, TAX_ID, HOST_TAX_IDS]
    mapped_df = mapped_df[[UNIREF90_ID, TAX_ID, VIRUS_NAME, HOST_TAX_IDS, VIRUS_HOST_NAME]]

    # remove records with no hosts
    mapped_df = mapped_df[~mapped_df[HOST_TAX_IDS].isna()]
    print(f"Mapped dataset size after removing sequences with no hosts =  {mapped_df.shape}")

    # aggregate the sequences with multiple hosts to one record with a list of host tax ids
    mapped_df_agg = mapped_df.groupby([UNIREF90_ID, TAX_ID]).agg({HOST_TAX_IDS: lambda x: list(x)})
    mapped_df_agg.reset_index(inplace=True)
    print(f"Mapped dataset size after aggregating hosts =  {mapped_df_agg.shape}")

    mapped_df_agg.to_csv(output_file_path, index=False)
    print(f"Written to file {output_file_path}")
    print("END: Get virus hosts from Virus Host DB")


# Get hosts of virus from UniPROT using uniref90_id of protein sequences
# input: parsed csv file of all sequences
# output: csv file with hosts of virus. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
# Use multiprocessing to speed up the process
# Note: this method drops the sequence information to save memory.
# The sequences will be joined back and compiled into one dataset at a later stage
def get_virus_hosts_from_uniprot(input_file_path, output_file_path):
    print("START: Get virus hosts from UniProt")
    # read the parsed uniref90_data csv file
    df = pd.read_csv(input_file_path)
    print(f"Read Uniref90 dataset size = {df.shape}")

    # retain only uniref90_ids to save memory
    df = df[[UNIREF90_ID, TAX_ID]]

    # read the existing output file, if it exists, to pick up from where the previous execution left.
    if Path(output_file_path).is_file():
        df_host = pd.read_csv(output_file_path, on_bad_lines=None, converters={2: literal_eval},
                              names=[UNIREF90_ID, TAX_ID, HOST_TAX_IDS, EMBL_REF_ID])
        df_host = df_host[[TAX_ID, UNIREF90_ID]]
        print(f"Number of records already processed = {df_host.shape[0]}")

        # remove the uniref_ids which have already been processed in the previous executions.
        processed_hosts = list(df_host[UNIREF90_ID].unique())
        df = df[~df[UNIREF90_ID].isin(processed_hosts)]
    print(f"Number of records TO BE processed = {df.shape[0]}")

    # split into sub dfs for parallel processing
    dfs = np.array_split(df, N_CPU)
    print(f"Number of sub dfs = {len(dfs)}")
    for i in range(N_CPU):
        print(f"Size of dfs[{i}] = {dfs[i].shape}")

    # multiprocessing for parallelization
    cpu_pool = Pool(N_CPU)
    cpu_pool.starmap(get_uniprot_virus_hosts, zip(dfs, repeat(output_file_path)))

    cpu_pool.close()
    cpu_pool.join()
    print(f"Written to file {output_file_path}")
    print("END: Get virus hosts from UniProt")


# call another method which will query UniProt to get hosts of the virus
# write the retrieved host ids to the output file
def get_uniprot_virus_hosts(df, output_file_path):
    # get virus hosts
    for row in df.iterrows():
        row = row[1]
        # query uniprot
        uniref90_id = row[UNIREF90_ID]
        tax_id = row[TAX_ID]
        host_tax_ids, embl_entry_id = external_sources_utils.query_uniprot(uniref90_id)
        print(f"{uniref90_id}: {len(host_tax_ids) if host_tax_ids is not None else None}, {embl_entry_id}")

        # write output to file
        f = open(output_file_path, mode="a")
        f.write(",".join([str(uniref90_id), str(tax_id), "\"" + str(host_tax_ids) + "\"", str(embl_entry_id)]) + "\n")
        f.close()


# Get hosts of virus from EMBL using embl_id of protein sequences
# input: csv file with embl ids. Columns = ["uniref90_id", "tax_id", "host_tax_ids", "embl_id"]
# output: csv file with hosts of virus. Columns = ["uniref90_id", "tax_id", "uniprot_host_tax_ids", "embl_id", "embl_host_names"]
# Use multiprocessing to speed up the process
def get_virus_hosts_from_embl(input_file_path, embl_mapping_filepath, output_file_path):
    print("START: Get virus hosts from EMBL")
    # read the input file
    df = pd.read_csv(input_file_path, on_bad_lines=None, converters={2: literal_eval},
                     names=[UNIREF90_ID, TAX_ID, HOST_TAX_IDS, EMBL_REF_ID])
    print(f"Read Uniref90 dataset size = {df.shape}")

    embl_ref_ids = list(df[EMBL_REF_ID].unique())
    print(f"Number of unique EMBL reference ids = {len(embl_ref_ids)}")

    # read the existing output file, if it exists, to pick up from where the previous execution left.
    if Path(embl_mapping_filepath).is_file():
        embl_mapping_df = pd.read_csv(embl_mapping_filepath, names=[EMBL_REF_ID, EMBL_HOST_NAME])
        processed_embl_ref_ids = set(embl_mapping_df[EMBL_REF_ID].unique())
        print(f"Number of EMBL reference ids already processed = {len(processed_embl_ref_ids)}")

        # remove the embl_reference_ids which have already been processed in the previous executions.
        embl_ref_ids = list(set(embl_ref_ids) - processed_embl_ref_ids)
    print(f"Number of unique EMBL reference ids TO BE processed = {len(embl_ref_ids)}")

    # split into sub dfs for parallel processing
    embl_ref_ids_sublists = np.array_split(np.array(embl_ref_ids), len(embl_ref_ids) / EMBL_QUERY_PAYLOAD_SIZE + 1)
    print(f"Number of sub lists = {len(embl_ref_ids_sublists)}")
    # for i in range(N_CPU):
    #      print(f"Size of embl_ref_ids_sublists[{i}] = {embl_ref_ids_sublists[i].shape}")

    # multiprocessing for parallelization
    cpu_pool = Pool(N_CPU)
    cpu_pool.starmap(get_embl_virus_host, zip(embl_ref_ids_sublists, repeat(embl_mapping_filepath)))

    cpu_pool.close()
    cpu_pool.join()
    print(f"EMBL host mapping written to file {embl_mapping_filepath}")

    # Map the embl_reference_ids in the dataset to the respective hosts from embl based on the embl_reference_ids
    embl_mapping_df = pd.read_csv(embl_mapping_filepath, names=[EMBL_REF_ID, EMBL_HOST_NAME])
    # join the two dfs on the embl_reference_id
    mapped_df = df.merge(embl_mapping_df, how="left", on=EMBL_REF_ID)
    print(f"Mapped dataset size = {mapped_df.shape}")
    mapped_df.to_csv(output_file_path, index=False)
    print(f"Written to file {output_file_path}")
    print("END: Get virus hosts from EMBL")


# call another method which will query EMBL to get hosts of the virus
# write the retrieved host names to the output mapping file
def get_embl_virus_host(embl_ref_ids, output_file_path):
    # get virus hosts from EMBL
    embl_mapping = external_sources_utils.query_embl(embl_ref_ids, temp_dir=os.path.dirname(output_file_path))
    # write output to file
    embl_mapping_dict = {
        EMBL_REF_ID: list(embl_mapping.keys()),
        EMBL_HOST_NAME: list(embl_mapping.values())
    }
    pd.DataFrame.from_dict(embl_mapping_dict).to_csv(output_file_path, mode="a", index=False, header=False)


# # remove sequences with no hosts of the virus from which the sequences were sampled
# # input: Dataset in csv file containing sequences with host_tax_ids. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
# # output: Dataframe with sequences containing atleast one host_tax_ids. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
# # Used only for mapping with UniProt.
# # In case of VirusHost DB, the dataset is already pruned to remove sequences without hosts during the mapping stage itself.
# def remove_sequences_w_no_hosts(input_file_path, output_file_path):
#     print("START: Remove sequences with no hosts")
#     df = pd.read_csv(input_file_path, on_bad_lines=None, converters={2: literal_eval},
#                      names=[UNIREF90_ID, TAX_ID, HOST_TAX_IDS])
#
#     # count the number of hosts for each sequence
#     df[HOST_COUNT] = df.apply(lambda x: len(x[HOST_TAX_IDS]), axis=1)
#     print(f"Dataset size = {df.shape}")
#
#     # Filter for sequences with atleast one host
#     df = df[df[HOST_COUNT] > 0]
#     print(f"Dataset after excluding proteins with no virus host = {df.shape[0]}")
#     # drop the host_count column
#     df.drop(columns=[HOST_COUNT], inplace=True)
#     df.to_csv(output_file_path, index=False)
#     print(f"Written to file {output_file_path}")
#     print("END: Remove sequences with no hosts")
#     return


# remove sequences with no EMBL hosts of the virus from which the sequences were sampled
# input: Dataset in csv file containing sequences with embl_host_name. Columns = [uniref90_id, tax_id, host_tax_ids, embl_ref_id, embl_host_name]
# output: Dataframe with sequences containing embl_host_name. Columns = [uniref90_id, tax_id, uniprot_host_tax_ids, embl_ref_id, embl_host_name]
# Used only for mapping with UniProt and EMBL.
# In case of VirusHost DB, the dataset is already pruned to remove sequences without hosts during the mapping stage itself.
def remove_sequences_w_no_hosts(input_file_path, output_file_path):
    print("START: Remove sequences with no hosts")
    df = pd.read_csv(input_file_path)
    print(f"Dataset size = {df.shape}")

    # renaming host_tax_ids to uniprot_host_tax_ids
    df.rename(columns={HOST_TAX_IDS: UNIPROT_HOST_TAX_IDS}, inplace=True)

    df = df[~df[EMBL_HOST_NAME].isna()]
    print(f"Dataset after excluding proteins with no virus host name from EMBL = {df.shape[0]}")

    # Additional pruning:
    # Remove sequences with duplicate EMBL reference ids
    embl_ref_id_counts = df[EMBL_REF_ID].value_counts()
    non_unique_embl_ref_ids = embl_ref_id_counts[embl_ref_id_counts > 1]
    print(f"Number of non-unique EMBL reference ids = {non_unique_embl_ref_ids.shape}")
    print(f"Non-unique EMBL reference ids = {non_unique_embl_ref_ids}")
    df = df[~df[EMBL_REF_ID].isin(non_unique_embl_ref_ids.index)]

    print(f"Dataset after excluding non-unique EMBL reference ids = {df.shape[0]}")
    df.to_csv(output_file_path, index=False)
    print(f"Written to file {output_file_path}")
    print("END: Remove sequences with no hosts")
    return


# # Get taxonomy name and rank of the virus and its host for each sequence record.
# # Input: Dataframe with exploded host_tax_ids. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
# # Output: Dataset with metadata. Columns = ["uniref90_id", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# def get_virus_metadata(input_file_path, taxon_metadata_dir_path, output_file_path):
#     print("START: Retrieving virus and virus host metadata using pytaxonkit")
#     # Set TAXONKIT_DB environment variable
#     os.environ[TAXONKIT_DB] = taxon_metadata_dir_path
#
#     # Read input dataset
#     df = pd.read_csv(input_file_path)
#     print(f"Read dataset size = {df.shape[0]}")
#
#     # convert HOST_TAX_IDS column to list type
#     df[HOST_TAX_IDS] = df[HOST_TAX_IDS].apply(ast.literal_eval)
#     # Explode the hosts column
#     df = df.explode(HOST_TAX_IDS)
#     # convert the HOST_TAX_IDS column to int64 for merge with the taxonomy metadata df
#     df[HOST_TAX_IDS] = df[HOST_TAX_IDS].astype("int64")
#     print(f"Dataset size after exploding {HOST_TAX_IDS} column = {df.shape[0]}")
#     print(f"Number of unique viral protein sequences = {len(df[UNIREF90_ID].unique())}")
#
#     # Retrieve name and rank of all unique viruses in the dataset
#     virus_tax_ids = df[TAX_ID].unique()
#     print(f"Number of unique virus tax ids = {len(virus_tax_ids)}")
#     virus_metadata_df = external_sources_utils.get_taxonomy_name_rank(virus_tax_ids)
#     print(f"Size of virus metadata dataset = {virus_metadata_df.shape[0]}")
#
#     # Retrieve name and rank of all unique virus_hosts in the dataset
#     virus_host_tax_ids = df[HOST_TAX_IDS].unique()
#     print(f"Number of unique virus host tax ids = {len(virus_host_tax_ids)}")
#     virus_host_metadata_df = external_sources_utils.get_taxonomy_name_rank(virus_host_tax_ids)
#     print(f"Size of virus host metadata dataset = {virus_host_metadata_df.shape[0]}")
#
#     # Merge df with virus_metadata_df to map metadata of viruses
#     df_w_metadata = pd.merge(df, virus_metadata_df, left_on=TAX_ID, right_on=NCBI_TAX_ID, how="left")
#     df_w_metadata.drop(columns=[NCBI_TAX_ID], inplace=True)
#     df_w_metadata.rename(columns={NAME: VIRUS_NAME, RANK: VIRUS_TAXON_RANK}, inplace=True)
#     print(f"Dataset size after merge with virus metadata = {df_w_metadata.shape}")
#
#     # Merge df with virus_metadata_df to map metadata of virus hosts
#     df_w_metadata = pd.merge(df_w_metadata, virus_host_metadata_df, left_on=HOST_TAX_IDS, right_on=NCBI_TAX_ID,
#                              how="left")
#     df_w_metadata.drop(columns=[NCBI_TAX_ID], inplace=True)
#     df_w_metadata.rename(columns={NAME: VIRUS_HOST_NAME, RANK: VIRUS_HOST_TAXON_RANK}, inplace=True)
#     print(f"Dataset size after merge with virus host metadata = {df_w_metadata.shape}")
#     df_w_metadata.to_csv(output_file_path, index=False)
#     print(f"Written to file {output_file_path}")
#     print("END: Retrieving virus and virus host metadata using pytaxonkit")


# Get taxonomy name and rank of the virus and its host for each sequence record.
# Input: Dataframe with exploded host_tax_ids. Columns = [uniref90_id, tax_id, uniprot_host_tax_ids, embl_ref_id, embl_host_name]
# Output: Dataset with metadata. Columns = [uniref90_id, tax_id, embl_ref_id, embl_host_name, virus_name, virus_taxon_rank, virus_host_tax_id, virus_host_name, virus_host_taxon_rank]
def get_virus_metadata(input_file_path, taxon_metadata_dir_path, output_file_path):
    print("START: Retrieving virus and virus host metadata using pytaxonkit")
    # Set TAXONKIT_DB environment variable
    os.environ[TAXONKIT_DB] = taxon_metadata_dir_path

    # Read input dataset
    df = pd.read_csv(input_file_path)
    print(f"Read dataset size = {df.shape[0]}")

    # drop UNIPROT_HOST_TAX_IDS
    df.drop(columns=UNIPROT_HOST_TAX_IDS, inplace=True)

    # convert EMBL_HOST_NAME column to list type
    df[EMBL_HOST_NAME] = df[EMBL_HOST_NAME].apply(ast.literal_eval)

    # Create a new virus_host_name column by extracting the host name from the embl_host_name column
    # 1. Take the first element (assuming there is only one element in the list (TODO: double check) e.g. ['Homo sapiens']
    # 2. Split by ';' and take the first element in case of noisy host names e.g. ['Homo sapiens; sex: M; age: 7 months']
    df[VIRUS_HOST_NAME] = df[EMBL_HOST_NAME].apply(lambda x: x[0].split(";")[0])

    print(f"Number of unique viral protein sequences = {len(df[UNIREF90_ID].unique())}")

    # Retrieve name and rank of all unique viruses in the dataset
    virus_tax_ids = df[TAX_ID].unique()
    print(f"Number of unique virus tax ids = {len(virus_tax_ids)}")
    virus_metadata_df = external_sources_utils.get_taxonomy_name_rank_from_id(virus_tax_ids)
    print(f"Size of virus metadata dataset = {virus_metadata_df.shape[0]}")

    # Retrieve name and rank of all unique virus_hosts in the dataset
    embl_virus_host_names = df[VIRUS_HOST_NAME].unique()
    print(f"Number of unique embl_virus_host_names = {len(embl_virus_host_names)}")
    virus_host_metadata_df = external_sources_utils.get_taxonomy_name_rank_from_name(embl_virus_host_names)
    print(f"Size of virus host metadata dataset = {virus_host_metadata_df.shape[0]}")

    # Merge df with virus_metadata_df to map metadata of viruses
    df_w_metadata = pd.merge(df, virus_metadata_df, left_on=TAX_ID, right_on=NCBI_TAX_ID, how="left")
    df_w_metadata.drop(columns=[NCBI_TAX_ID], inplace=True)
    df_w_metadata.rename(columns={NAME: VIRUS_NAME, RANK: VIRUS_TAXON_RANK}, inplace=True)
    print(f"Dataset size after merge with virus metadata = {df_w_metadata.shape}")

    # Merge df with virus_metadata_df to map metadata of virus hosts
    df_w_metadata = pd.merge(df_w_metadata, virus_host_metadata_df, left_on=VIRUS_HOST_NAME, right_on=NAME,
                             how="left")
    df_w_metadata.drop(columns=[NAME], inplace=True)
    df_w_metadata.rename(columns={NCBI_TAX_ID: VIRUS_HOST_TAX_ID, RANK: VIRUS_HOST_TAXON_RANK}, inplace=True)
    print(f"Dataset size after merge with virus host metadata = {df_w_metadata.shape}")
    df_w_metadata.to_csv(output_file_path, index=False)
    print(f"Written to file {output_file_path}")
    print("END: Retrieving virus and virus host metadata using pytaxonkit")


# Filter for records with virus_name and virus_host_name at "Species" level
# Input: Dataset with metadata containing columns = [virus_taxon_rank, virus_host_taxon_rank]
# Output: Filtered dataset with metadata with the same columns as in the input dataset
def get_sequences_at_species_level(input_file_path, output_file_path):
    print("START: Filter records with virus and virus hosts at 'species' level taxonomy.")
    df = pd.read_csv(input_file_path)
    print(f"Dataset size before filter: {df.shape[0]}")

    # Filter for virus rank == Species
    df = df[df[VIRUS_TAXON_RANK] == SPECIES]
    print(f"Dataset size after virus at species level filter: {df.shape[0]}")

    # Filter for virus_host rank == Species
    df = df[df[VIRUS_HOST_TAXON_RANK] == SPECIES]
    print(f"Dataset size after virus_host at species level filter: {df.shape[0]}")

    df.to_csv(output_file_path, index=False)
    print(f"Writing to file {output_file_path}")
    print("END: Filter records with virus and virus hosts at 'species' level taxonomy.")


# # Filter for sequences with virus hosts belonging to the class of Mammals OR Aves (birds)
# # Input: Dataset with metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# # Output: Filtered dataset with metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# def get_sequences_from_mammals_aves_hosts(input_file_path, taxon_metadata_dir_path, output_file_path):
#     print("START: Filter records with virus hosts belonging to 'mammals' OR 'aves' family.")
#     # Set TAXONKIT_DB environment variable
#     os.environ["TAXONKIT_DB"] = taxon_metadata_dir_path
#
#     # Read input file
#     df = pd.read_csv(input_file_path)
#
#     # Get all unique host tax ids
#     host_tax_ids = df[HOST_TAX_IDS].unique()
#     print(f"Number of unique host tax ids = {len(host_tax_ids)}")
#
#     # Get taxids belonging to the class of mammals and aves
#     mammals_aves_tax_ids = external_sources_utils.get_mammals_aves_tax_ids(host_tax_ids)
#     print(f"Number of unique mammalia or aves tax ids = {len(mammals_aves_tax_ids)}")
#     # Filter
#     print(f"Dataset size before filtering for mammals and aves: {df.shape}")
#     df = df[df[HOST_TAX_IDS].isin(mammals_aves_tax_ids)]
#     print(f"Dataset size after filtering for mammals and aves: {df.shape}")
#
#     df.to_csv(output_file_path, index=False)
#     print(f"Writing to file {output_file_path}")
#     print("END: Filter records with virus hosts belonging to 'mammals' OR 'aves' family.")


# Filter for sequences with virus hosts belonging to the clade of vertebrata
# Input: Dataset with metadata with column "virus_host_tax_id"
# Output: Filtered dataset with same columns as in the input dataset
def get_sequences_from_vertebrata_hosts(input_file_path, taxon_metadata_dir_path, output_file_path):
    print("START: Filter records with virus hosts belonging to 'vertebrata' clade.")
    # Set TAXONKIT_DB environment variable
    os.environ["TAXONKIT_DB"] = taxon_metadata_dir_path

    # Read input file
    df = pd.read_csv(input_file_path)

    # Get all unique host tax ids
    host_tax_ids = df[VIRUS_HOST_TAX_ID].unique()
    print(f"Number of unique host tax ids = {len(host_tax_ids)}")

    # Get taxids belonging to the clade of vertebrata
    # split into sublists for parallel processing
    host_tax_ids_sublists = np.array_split(np.array(host_tax_ids), N_CPU)
    for i in range(N_CPU):
         print(f"Size of host_tax_ids_sublists[{i}] = {host_tax_ids_sublists[i].shape}")

    # multiprocessing for parallelization
    cpu_pool = Pool(N_CPU)
    vertebrata_tax_ids_sublists = cpu_pool.map(external_sources_utils.get_vertebrata_tax_ids, host_tax_ids_sublists)
    # flatten the list of sub_lists into one list
    vertebrata_tax_ids = list(itertools.chain.from_iterable(vertebrata_tax_ids_sublists))
    cpu_pool.close()
    cpu_pool.join()
    print(f"Number of unique vertebrata tax ids = {len(vertebrata_tax_ids)}")
    # Filter
    print(f"Dataset size before filtering for vertebrata: {df.shape}")
    df = df[df[VIRUS_HOST_TAX_ID].isin(vertebrata_tax_ids)]
    print(f"Dataset size after filtering for vertebrata: {df.shape}")

    df.to_csv(output_file_path, index=False)
    print(f"Writing to file {output_file_path}")
    print("END: Filter records with virus hosts belonging to vertebrata' clade.")


# Join metadata dataset with sequence data from the parsed fasta file
# Input: Metadata dataset. Columns = ["uniref90_id", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# Output: Dataset written to csv file. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
def join_metadata_with_sequences_data(input_file_path, sequence_data_file_path, output_file_path):
    print("START: Joining metadata with sequences data.")
    metadata_df = pd.read_csv(input_file_path)
    print(f"Metadata dataset size = {metadata_df.shape[0]}")

    sequence_data_df = pd.read_csv(sequence_data_file_path)
    print(f"Sequence dataset size = {sequence_data_df.shape[0]}")

    merged_df = pd.merge(metadata_df, sequence_data_df[[UNIREF90_ID, SEQUENCE]], how="left", on=UNIREF90_ID)
    print(f"Size of dataset after merge of metadata with sequence data = {merged_df.shape[0]}")
    merged_df.to_csv(output_file_path, index=False)
    print(f"Written to file {output_file_path}")
    print("END: Joining metadata with sequences data.")


# Remove sequences of virus with only one host
# Input: Dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# Output: Filtered dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
def remove_sequences_of_virus_with_one_host(input_file_path, output_file_path, filtered_file_path):
    print("START: Remove sequences of viruses with one host.")

    # Read input file
    df = pd.read_csv(input_file_path)

    # group by virus name and count the number of unique hosts for each virus
    agg_df = df.groupby([VIRUS_NAME])[VIRUS_HOST_NAME].nunique()
    # list of viruses with only one unique host
    viruses_with_one_host = agg_df[agg_df == 1].index.tolist()

    print(f"Number of viruses with one host = {len(viruses_with_one_host)}")
    filtered_df = df[df[VIRUS_NAME].isin(viruses_with_one_host)]

    print(f"Dataset size before filtering for viruses with more than one hosts: {df.shape}")
    df = df[~df[VIRUS_NAME].isin(viruses_with_one_host)]
    print(f"Dataset size after filtering for viruses with more than one hosts: {df.shape}")

    df.to_csv(output_file_path, index=False)
    print(f"Output written to file {output_file_path}")

    filtered_df.to_csv(filtered_file_path, index=False)
    print(f"Filtered sequences written to file {filtered_file_path}")

    print("END: Remove sequences of viruses with one host.")


# Remove duplicate sequences
# Input: Dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# Output: Filtered dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
def remove_duplicate_sequences(input_file_path, output_file_path, filtered_file_path):
    print("START: Remove sequences with multiple hosts.")

    # Read input file
    df = pd.read_csv(input_file_path)

    df = df.set_index(UNIREF90_ID)
    filtered_df = df[df.index.duplicated(keep=False)]
    print(f"Dataset size before removing duplicates: {df.shape}")
    df = df[~df.index.duplicated(keep=False)]
    print(f"Dataset size after removing duplicates: {df.shape}")

    df.reset_index().to_csv(output_file_path, index=False)
    print(f"Output written to file {output_file_path}")

    filtered_df.reset_index().to_csv(filtered_file_path, index=False)
    print(f"Filtered sequences written to file {filtered_file_path}")

    print("END: Remove sequences with multiple hosts.")
