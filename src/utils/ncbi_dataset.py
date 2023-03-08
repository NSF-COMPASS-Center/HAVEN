import pandas as pd
import os
import requests
import numpy as np
import pytaxonkit
from ast import literal_eval

from multiprocessing import Pool


# Directory paths and Filenames
BENCHMARKS_DIR = '/home/grads/blessyantony/dev/git/protein_bert/protein_benchmarks'
UNIREF_DIR = "uniref"
UNIPROT_REST_PROTS = "https://rest.uniprot.org/uniprotkb/search"
UNIREF90_DATA_SEQ_FILENAME = "uniref_90_parsed.csv"
UNIREF90_DATA_WITH_HOSTS_FILENAME = "uniref90_id_hosts.csv"
UNIREF90_DATA_METADATA_FILENAME = "uniref90_id_hosts_metadata.csv"
UNIREF90_DATA_FILTERED_FILENAME = "uniref90_filtered.csv"
UNIREF90_DATA_FILENAME = "uniref90_mammals_aves_virus.csv"

TAXONKIT_DIR = "~/dev/taxonkit"

UNIPROT_REST_UNIREF90_QUERY_PARAM = "uniref_cluster_90:%s"
N_CPU = 6

def get_virus_host(uniref_id):
    response = requests.get(url=UNIPROT_REST_PROTS, params={"query": UNIPROT_REST_UNIREF90_QUERY_PARAM % uniref_id, "fields": "virus_hosts"})
    host_tax_ids = []
    try:
        data = response.json()["results"][0]
        org_hosts = data["organismHosts"]
        for org_host in org_hosts:
            host_tax_ids.append(org_host["taxonId"])
    except (KeyError, IndexError):
        pass
    return host_tax_ids


def get_metadata(df):
    output_file_path = str(os.path.join(BENCHMARKS_DIR, UNIREF_DIR))
    # get virus hosts
    for index, val in df.iteritems():
        uniref90_id = val
        host_tax_ids = get_virus_host(uniref90_id)
        print(f"[{index}] {uniref90_id}: {len(host_tax_ids)}")
        f = open(output_file_path, mode="a")
        f.write(",".join([uniref90_id, "\"" + str(host_tax_ids) + "\""]) + "\n")
        f.close()


def get_hosts():
    # read the downloaded, parsed uniref90 csv file
    df = pd.read_csv(os.path.join(BENCHMARKS_DIR, UNIREF_DIR, UNIREF90_DATA_SEQ_FILENAME))
    print(f"uniref90_parsed shape = {df.shape}")
    # retain only uniref90_ids to save memory
    df = df["uniref90_id"]
    print(f"df size = {df.shape}")
    print(f"df  = {df.head()}")

    # read the existing output file to pick up from where the previous execution left.
    output_file_path = str(os.path.join(BENCHMARKS_DIR, UNIREF_DIR, UNIREF90_DATA_WITH_HOSTS_FILENAME))

    df_host = pd.read_csv(output_file_path, on_bad_lines=None, converters={1: literal_eval}, names=["uniref90_id", "host_tax_ids"])
    df_host = df_host["uniref90_id"]
    print(f"df_host size = {df_host.shape}")
    print(f"df_host = {df_host.head()}")

    # remove the uniref_ids which have already been processed in the previous executions.
    # no straightforward way to implement this filter
    # hack: 1. left join with indicator=True creates an additional column named '_merge' with values 'both' or 'left_only'
    #       2. retain only the rows with '_merge' column value == 'left_only'
    df = pd.merge(df, df_host, how="left", on=["uniref90_id"], indicator=True)
    df = df[df["_merge"] == "left_only"]["uniref90_id"]
    print(f"df size after filter = {df.shape}")

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
    cpu_pool.map(get_metadata, dfs)

    cpu_pool.close()
    cpu_pool.join()


def get_mammals_aves(host_tax_ids):
    mammals_aves_ids = []
    for i, tax_id in enumerate(host_tax_ids):
        host_class = pytaxonkit.lineage([tax_id], formatstr='{c}')['Lineage'].iloc[0]
        print(f"{i}: {tax_id} = {host_class}")
        if host_class == "Mammalia" or host_class == "Aves":
            mammals_aves_ids.append(tax_id)
    return mammals_aves_ids


def filter_data():
    df = pd.read_csv(str(os.path.join(BENCHMARKS_DIR, UNIREF_DIR, UNIREF90_DATA_METADATA_FILENAME)))
    print(df.head())
    print(f"Dataframe size before filter: {df.shape}")
    df = df[df["virus_taxon_rank"] == "species"]
    print(f"Dataframe size after virus host at species level filter: {df.shape}")
    df = df[df["virus_host_taxon_rank"] == "species"]
    print(f"Dataframe size after virus at species level filter: {df.shape}")

    # viruses with hosts = Mammals or birds
    host_tax_ids = df["host_tax_ids"].unique()
    print(f"Number of unique host tax ids = {len(host_tax_ids)}")
    mammals_aves_ids = get_mammals_aves(host_tax_ids)
    df = df[df["host_tax_ids"].isin(mammals_aves_ids)]
    print(f"Dataframe size after filtering for mammals and aves: {df.shape}")
    df.to_csv(os.path.join(BENCHMARKS_DIR, UNIREF_DIR, UNIREF90_DATA_FILTERED_FILENAME), index=False)


def final_parse():
    df = pd.read_csv(os.path.join(BENCHMARKS_DIR, UNIREF_DIR, UNIREF90_DATA_FILTERED_FILENAME))
    print(df.shape)
    print(df.head())

    sequence_df = pd.read_csv(os.path.join(BENCHMARKS_DIR, UNIREF_DIR, UNIREF90_DATA_SEQ_FILENAME))
    print(sequence_df.shape)
    print(sequence_df.head())

    merged_df = pd.merge(df, sequence_df[["uniref90_id", "seq"]], how="left", on="uniref90_id")
    print(f"Merged df size = {merged_df.shape}")

    merged_df.drop(columns=["Unnamed: 0", "tax_id", "host_tax_ids", "virus_taxon_rank", "virus_host_taxon_rank", "host_count"], inplace=True)
    print(merged_df.head())
    print("unique hosts")
    print(merged_df["virus_host_name"].unique())
    print(merged_df["virus_host_name"].value_counts())
    merged_df.to_csv(os.path.join(BENCHMARKS_DIR, UNIREF_DIR, UNIREF90_DATA_FILENAME), index=False)


if __name__ == '__main__':
    final_parse()