import pandas as pd
import os
import requests
import numpy as np
import argparse
import re
import pytaxonkit
from ast import literal_eval
from Bio import SeqIO
from multiprocessing import Pool
from itertools import repeat

# Filenames
UNIREF90_DATA_CSV_FILENAME = "uniref90_parsed.csv"
UNIREF90_DATA_W_HOSTS_FILENAME = "uniref90_w_hosts.csv"
UNIREF90_DATA_W_METADATA = "uniref90_w_metadata.csv"
UNIREF90_DATA_FILTERED_FILENAME = "uniref90_mammals_aves_virus.csv"
UNIREF90_DATA_FINAL_FILENAME = "uniref90_final.csv"

# Column names at various stages of dataset curation
UNIREF90_ID = "uniref90_id"
TAX_ID = "tax_id"
SEQUENCE = "seq"
HOST_TAX_IDS = "host_tax_ids"
HOST_COUNT = "host_count"
VIRUS_NAME = "virus_name"
VIRUS_TAXON_RANK = "virus_taxon_rank"
VIRUS_HOST_NAME = "virus_host_name"
VIRUS_HOST_TAXON_RANK = "virus_host_taxon_rank"

## Taxonomy columns
NAME = "Name"
RANK = "Rank"
NCBI_TAX_ID = "TaxID"


TAXONKIT_DIR = "~/dev/taxonkit"

# uniprot related constants
UNIPROT_REST_PROTS = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_REST_UNIREF90_QUERY_PARAM = "uniref_cluster_90:%s"
N_CPU = 6


def parse_args():
    parser = argparse.ArgumentParser(description='Parse and create UniRef Dataset')
    parser.add_argument('-if', '--input_file', required=True,
                        help="Absolute path to the input file in fasta format\n")
    parser.add_argument('-od', '--output_dir', required=True,
                        help="Absolute path to output directory to create intermediate file.\n")
    args = parser.parse_args()
    return args


# Parse fasta file
# input fasta file
# output: csv file with columns ["uniref90_id", "tax_id", "seq"]
def parse_fasta_file(fasta_file_path, output_directory):
    sequences = []
    i = 0
    no_id_count = 0
    print("START: Parsing fasta file")
    # parse fasta file to extract uniref90_id, tax_id of virus/organism, and protein sequence
    with open(fasta_file_path) as f:
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
    print(f"Writing to file {UNIREF90_DATA_CSV_FILENAME}")
    df.to_csv(os.path.join(output_directory, UNIREF90_DATA_CSV_FILENAME))


# Get hosts of virus from UniPROT using unireg90_id of protein sequences
# input: parsed csv file of all sequences
# output: csv file with hosts of virus. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
# Use multiprocessing to speed up the process
# Note: this method drops the sequence information to save memory.
# The sequences will be joined back and compiled into one dataset at a later stage
def get_virus_hosts(output_directory):
    # read the parsed uniref90_data csv file
    df = pd.read_csv(os.path.join(output_directory, UNIREF90_DATA_CSV_FILENAME))
    print(f"Uniref90 dataset size = {df.shape}")

    # retain only uniref90_ids to save memory
    df = df[[UNIREF90_ID, TAX_ID]]

    # read the existing output file to pick up from where the previous execution left.
    output_file_path = str(os.path.join(output_directory, UNIREF90_DATA_W_HOSTS_FILENAME))

    df_host = pd.read_csv(output_file_path, on_bad_lines=None, converters={2: literal_eval},
                          names=[UNIREF90_ID, TAX_ID, HOST_TAX_IDS])
    df_host = df_host[[TAX_ID, UNIREF90_ID]]
    print(f"Number of records already processed = {df_host.shape[0]}")

    # remove the uniref_ids which have already been processed in the previous executions.
    # no straightforward way to implement this filter
    # hack: 1. left join with indicator=True creates an additional column named '_merge' with values 'both' or 'left_only'
    #       2. retain only the rows with '_merge' column value == 'left_only'
    df = pd.merge(df, df_host, how="left", on=[UNIREF90_ID], indicator=True)
    df = df[df["_merge"] == "left_only"][[UNIREF90_ID, TAX_ID]]
    print(f"Number of records TO BE processed = {df.shape[0]}")

    # split into sub dfs for parallel processing
    dfs = np.array_split(df, N_CPU)
    print(f"Number of sub dfs = {len(dfs)}")
    print(f"Size of dfs[0] = {dfs[0].shape}")
    print(f"Size of dfs[1] = {dfs[1].shape}")
    print(f"Size of dfs[2] = {dfs[2].shape}")
    print(f"Size of dfs[3] = {dfs[3].shape}")
    print(f"Size of dfs[4] = {dfs[4].shape}")
    print(f"Size of dfs[5] = {dfs[5].shape}")
    # multiprocessing for parallelization
    cpu_pool = Pool(N_CPU)
    cpu_pool.starmap(get_virus_host, zip(dfs, repeat(output_file_path)))

    cpu_pool.close()
    cpu_pool.join()
    print(f"Written to file {UNIREF90_DATA_W_HOSTS_FILENAME}")


# call another method which will query UniProt to get hosts of the virus
# write the retrieved host ids to the output file
def get_virus_host(df, output_file_path):
    # get virus hosts
    for row in df.iterrows():
        # query uniprot
        uniref90_id = row[UNIREF90_ID]
        tax_id = row[TAX_ID]
        host_tax_ids = query_uniprot(uniref90_id)
        print(f"{uniref90_id}: {len(host_tax_ids)}")

        # write output to file
        f = open(output_file_path, mode="a")
        f.write(",".join([uniref90_id, tax_id, "\"" + str(host_tax_ids) + "\""]) + "\n")
        f.close()


# query UniProt for to get the host of the virus of the protein sequence
# input: uniref90_id
# output: list of host(s) of the virus
def query_uniprot(uniref90_id):
    response = requests.get(url=UNIPROT_REST_PROTS,
                            params={"query": UNIPROT_REST_UNIREF90_QUERY_PARAM % uniref90_id, "fields": "virus_hosts"})
    host_tax_ids = []
    try:
        data = response.json()["results"][0]
        org_hosts = data["organismHosts"]
        for org_host in org_hosts:
            host_tax_ids.append(org_host["taxonId"])
    except (KeyError, IndexError):
        pass
    return host_tax_ids


# remove sequences with no hosts of the virus from which the sequences were sampled
# input: Dataset in csv file containing sequences with host_tax_ids. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
# output: Dataframe with sequences containing atleast one host_tax_is. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
def remove_sequences_w_no_hosts(output_directory):
    print("\nRemoving Sequences with no hosts")
    df = pd.read_csv(os.path.join(output_directory, UNIREF90_DATA_W_HOSTS_FILENAME), on_bad_lines=None, converters={2: literal_eval},
                          names=[UNIREF90_ID, TAX_ID, HOST_TAX_IDS])
    # count the number of hosts for each sequence
    df[HOST_COUNT] = df.apply(lambda x: len(x[HOST_TAX_IDS]), axis=1)
    print(f"Dataset size = {df.shape}")

    # Filter for sequences with atleast one host
    df = df[df[HOST_COUNT] > 0]
    print(f"Dataset after excluding proteins with no virus host = {df.shape[0]}")
    # drop the host_count column
    df.drop(columns=[HOST_COUNT], inplace=True)
    return df


# Explode the host_tax_ids columns
# Create multiple entries (duplicate the uniref90id) for each virus host associated with a sequence uniref90_id
# input: Dataframe with sequence ids containing atleast one host_tax_is. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
# output: Dataframe with exploded host_tax_ids (one host_id per record in the host_tax_ids column). Columns = ["uniref90_id", "tax_id", "host_tax_ids]
def explode_virus_hosts(df):
    print("\nExploding virus hosts")
    print(f"Dataset size = {df.shape[0]}")
    df = df.explode(HOST_TAX_IDS)
    print(f"Dataset size after exploding {HOST_TAX_IDS} column = {df.shape[0]}")
    print(f"Number of unique protein sequences = {len(df[UNIREF90_ID].unique())}")
    print(f"Number of unique hosts  = {len(df[HOST_TAX_IDS].unique())}")
    return df


# Get taxonomy name and rank of the virus and its host for each sequence record.
# Input: Dataframe with exploded host_tax_ids. Columns = ["uniref90_id", "tax_id", "host_tax_ids]
# Output: Dataset with metadata. Columns = ["uniref90_id", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
def get_virus_metadata(df):
    print("\nRetrieving virus metadata")
    # Retrieve name and rank of all unique viruses in the dataset
    virus_tax_ids = df[TAX_ID].unique()
    print(f"Number of unique virus tax ids = {len(virus_tax_ids)}")
    virus_metadata_df = get_taxonomy_name_rank(virus_tax_ids)
    print(f"Size of virus metadata dataset = {virus_metadata_df.shape[0]}")

    # Retrieve name and rank of all unique viruses in the dataset
    virus_host_tax_ids = df[HOST_TAX_IDS].unique()
    print(f"Number of unique virus host tax ids = {len(virus_host_tax_ids)}")
    virus_host_metadata_df = get_taxonomy_name_rank(virus_host_tax_ids)
    print(f"Size of virus host metadata dataset = {virus_host_metadata_df.shape[0]}")

    # Merge df with virus_metadata_df to map metadata of viruses
    df_w_metadata = pd.merge(df, virus_metadata_df, left_on=TAX_ID, right_on=NCBI_TAX_ID, how="left")
    df_w_metadata.drop(columns=[NCBI_TAX_ID], inplace=True)
    df_w_metadata.rename(columns={NAME: VIRUS_NAME, RANK: VIRUS_TAXON_RANK}, inplace=True)
    print(f"Dataset size after merge with virus metadata = {df_w_metadata.shape}")

    df_w_metadata = pd.merge(df_w_metadata, virus_host_metadata_df, left_on=HOST_TAX_IDS, right_on=NCBI_TAX_ID, how="left")
    df_w_metadata.drop(columns=[NCBI_TAX_ID], inplace=True)
    df_w_metadata.rename(columns={NAME: VIRUS_HOST_NAME, RANK: VIRUS_HOST_TAXON_RANK}, inplace=True)
    print(f"Dataset size after merge with virus host metadata = {df_w_metadata.shape}")
    return df_w_metadata


# Get taxonomy names and ranks from ncbi using pytaxonkit for given list of tax_ids
# Input: list of tax_ids
# Output: Dataframe with columns: ["TaxID", "Name", "Rank"]
def get_taxonomy_name_rank(tax_ids):
    # There is no method with input parameter: taxid and output: scientific name and rank.
    # However, there is a method that takes in name and returns the taxid and rank
    # Hack:
    # 1. Get names of the tax_ids using name()
    # 2. Get ranks using the names from previous step using name2taxid()
    df = pytaxonkit.name(tax_ids)
    df_w_rank = pytaxonkit.name2taxid(df[NAME].values)
    # default datatype of TaxID column = int32
    # convert it to int64 for convenience in downstream analysis
    df_w_rank[NCBI_TAX_ID] = pd.to_numeric(df_w_rank[NCBI_TAX_ID], errors="coerce").fillna(0).astype("int64")
    return df_w_rank


# Join metadata dataset with sequence data using data from the parsed fasta file
# Input: Metadata dataset. Columns = ["uniref90_id", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# Output: Dataset written to csv file. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
def join_metadata_with_sequences_data(df, output_directory):
    print(f"Metadata dataset size = {df.shape}")
    sequence_data_df = pd.read_csv(os.path.join(output_directory, UNIREF90_DATA_CSV_FILENAME))
    print(f"Sequence dataset size = {sequence_data_df.shape}")
    uniref90_df = pd.merge(df, sequence_data_df[[UNIREF90_ID, SEQUENCE]], how="left", on=UNIREF90_ID)
    print(uniref90_df.head())
    print(f"Size of dataset after merge of metadata with sequence data = {uniref90_df.shape}")
    print(f"Writing to file {UNIREF90_DATA_W_METADATA}")
    uniref90_df.to_csv(os.path.join(output_directory, UNIREF90_DATA_W_METADATA), index=False)


# Filter for records with virus_name and virus_host_name at 'Species' level
# Input: Dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# Output: Filtered dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
def get_sequences_at_species_level(output_directory):
    df = pd.read_csv(os.path.join(output_directory, UNIREF90_DATA_W_METADATA))
    print(f"Dataset size before filter: {df.shape}")

    # Filter for virus rank == Species
    df = df[df[VIRUS_TAXON_RANK] == "species"]
    print(f"Dataframe size after virus at species level filter: {df.shape}")

    # Filter for virus_host rank == Species
    df = df[df[VIRUS_HOST_TAXON_RANK] == "species"]
    print(f"Dataframe size after virus_host at species level filter: {df.shape}")

    return df


# Filter for sequences with virus hosts belonging to the class of Mammals OR Aves (birds)
# Input: Dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# Output: Filtered dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
def get_sequences_from_mammals_aves_hosts(df):
    # Get all unique host tax ids
    host_tax_ids = df[HOST_TAX_IDS].unique()
    print(f"Number of unique host tax ids = {len(host_tax_ids)}")

    # Get taxids belonging to the class of mammals and aves
    mammals_aves_tax_ids = get_mammals_aves_tax_ids(host_tax_ids)
    # Filter
    print(f"Dataset size before filtering for mammals and aves: {df.shape}")
    df = df[df[HOST_TAX_IDS].isin(mammals_aves_tax_ids)]
    print(f"Dataset size after filtering for mammals and aves: {df.shape}")
    return df


# Get taxids belonging to the class of mammals and aves
# Input: list of tax_ids
# Output: list of tax_ids belonging to mammals and aves class
def get_mammals_aves_tax_ids(tax_ids):
    mammals_aves_tax_ids = []
    for i, tax_id in enumerate(tax_ids):
        tax_class = pytaxonkit.lineage([tax_id], formatstr='{c}')['Lineage'].iloc[0]
        print(f"{i}: {tax_id} = {tax_class}")
        if tax_class == "Mammalia" or tax_class == "Aves":
            mammals_aves_tax_ids.append(tax_id)
    return mammals_aves_tax_ids


# Remove sequences of virus with only one host
# Input: Dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# Output: Filtered dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
def remove_sequences_of_virus_with_one_host(output_directory, df):
    print("\nRemoving sequences with one host.")
    df = pd.read_csv(os.path.join(output_directory, UNIREF90_DATA_FILTERED_FILENAME))
    # group by virus name and count the number of unique hosts for each virus
    agg_df = df.groupby([VIRUS_NAME])[VIRUS_HOST_NAME].nunique()
    # list of viruses with only one unique host
    viruses_with_one_host = agg_df[agg_df == 1].index.tolist()

    print(f"Number of viruses with one host = {len(viruses_with_one_host)}")
    print(f"Dataset size before filtering for viruses with more than one hosts: {df.shape}")
    df = df[~df[VIRUS_NAME].isin(viruses_with_one_host)]
    print(f"Dataset size after filtering for viruses with more than one hosts: {df.shape}")
    return df


# Remove duplicate sequences
# Input: Dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
# Output: Filtered dataset with sequence and metadata. Columns = ["uniref90_id", "seq", "tax_id", "host_tax_ids", "virus_name", "virus_taxon_rank", "virus_host_name", "virus_host_taxon_rank"]
def remove_duplicate_sequences(df):
    df= df.set_index(UNIREF90_ID)
    print(f"Dataset size before removing duplicates: {df.shape}")
    df = df[~df.index.duplicated()]
    print(f"Dataset size after removing duplicates: {df.shape}")
    return df


def main():
    config = parse_args()
    input_file_path = config.input_file
    output_dir = config.output_dir

    # 1. Parse the Fasta file
    parse_fasta_file(input_file_path, output_dir)
    # 2. Get hosts of the virus from which the protein sequences were sampled
    get_virus_hosts(output_dir)
    # 3. Filter the dataset: Remove sequences with no hosts of the virus
    df = remove_sequences_w_no_hosts(output_dir)
    # 4. Explode the host column: Create multiple entries (duplicate the sequence) one for each host of the virus of the sequence
    df = explode_virus_hosts(df)
    # 5. Get metadata for each record: taxonomy name and rank of the virus and virus_hosts of the sequences
    df = get_virus_metadata(df)
    # Note: Data in steps 2, 3, 4, 5 do not contain the protein sequence. We dropped the sequence column in step 2 to save memory
    # 6. Rejoin the sequence data using the parsed fasta file output from step 1 and write to an intermediary dataset file
    join_metadata_with_sequences_data(df, output_dir)
    # 7. Filters
    # 7.1 Retain sequences with virus AND virus host with rank = "Species
    df = get_sequences_at_species_level(output_dir)
    # 7.2 Retain sequences with viruses hosts belonging to the class of Mammals OR Aves (birds)
    df = get_sequences_from_mammals_aves_hosts(df)
    # 7.3 Remove viruses with only one unique virus host
    df = remove_sequences_of_virus_with_one_host(output_dir, df)
    # 7.4 Remove duplicate sequences (same uniref90_id and sequence, but multiple hosts)
    df = remove_duplicate_sequences(df)
    # 8. Write the filtered dataset to a file
    print(f"Writing to file {UNIREF90_DATA_FINAL_FILENAME}")
    df.to_csv(os.path.join(output_dir, UNIREF90_DATA_FINAL_FILENAME), index=False)


if __name__ == '__main__':
    main()
