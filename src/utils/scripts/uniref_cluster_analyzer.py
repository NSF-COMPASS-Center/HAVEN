import argparse
import os
from pathlib import Path
import pandas as pd
import requests

from utils import external_sources_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the UniRef cluster composition")
    parser.add_argument("-if", "--input_file", required=True,
                        help="File with sequences.\n")
    parser.add_argument("-id_col", "--id_col", required=True,
                        help="Name of the column containing the id,\n")
    parser.add_argument("-label_col", "--label_col", required=True,
                        help="Name of the column containing the label.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory.\n")
    args = parser.parse_args()
    return args


# query Uniprot for to get the host of the virus of the protein sequence
# input: uniprot_id
# output: list of host(s) of the virus
def query_uniprot(uniprot_id):
    UNIPROT_REST_API = "https://rest.uniprot.org/uniprotkb/search"
    response = requests.get(url=UNIPROT_REST_API,
                            params={"query": uniprot_id,
                                    "fields": ",".join(["virus_hosts", "xref_embl"])})
    embl_entry_id = None
    try:
        results = response.json()["results"]
        # ideally there should be only one matching primaryAccession entry for the seed uniprot_id
        data = results[0]

        # embl cross reference entry id
        cross_refs = data["uniProtKBCrossReferences"]
        embl_cross_ref_properties = \
            [cross_ref for cross_ref in cross_refs if cross_ref["database"] == "EMBL"][0]["properties"]
        embl_entry_id = \
            [property for property in embl_cross_ref_properties if property["key"] == "ProteinId"][0][
                "value"]

    except (KeyError, IndexError):
        pass
    return embl_entry_id


def analyze_cluster(uniref_id, output_dir):
    cluster_member_ids = external_sources_utils.get_uniref_cluster_members(uniref_id)
    cluster = []
    for cluster_member_id in cluster_member_ids:
        embl_ref_id = query_uniprot(cluster_member_id)
        cluster.append({
            "member_id": cluster_member_id,
            "embl_ref_id": embl_ref_id
        })
    cluster_df = pd.DataFrame(cluster)
    if cluster_df.shape[0] > 0:
        embl_ref_ids = list(cluster_df["embl_ref_id"].unique())
        embl_mapping = external_sources_utils.query_embl(embl_ref_ids, temp_dir=output_dir)

        cluster_df["embl_host_name"] = cluster_df["embl_ref_id"].apply(lambda x: embl_mapping[x])
    return cluster_df


def analyze_clusters(input_file, id_col, label_col, output_dir):
    df = pd.read_csv(input_file)
    print(f"Read input file {input_file}. Dataset size = {df.shape}")
    uniref_ids = list(df[id_col].unique())
    print(f"Number of unique cluster ids = {len(uniref_ids)}")
    if len(uniref_ids) != df.shape[0]:
        print("ERROR: Duplicate entries in input file. Exiting ...")
        return

    clusters = []
    for i, uniref_id in enumerate(uniref_ids):
        try:
            virus_host_name = df[df[id_col] == uniref_id][label_col].values[0]
            cluster_df = analyze_cluster(uniref_id, output_dir)
            cluster_df[id_col] = uniref_id
            cluster_df[label_col] = virus_host_name
            print(f"{i}. {uniref_id}: {cluster_df.shape[0]}")
            clusters.append(cluster_df)
        except:
            print("ERROR with uniref_id ", uniref_id)
            continue

    clusters_df = pd.concat(clusters)
    output_file_path = os.path.join(output_dir, Path(input_file).stem + "_members.csv")
    print(f"Writing output at {output_file_path}")
    clusters_df.to_csv(output_file_path, index=None)


def main():
    config = parse_args()
    input_file = config.input_file
    id_col = config.id_col
    label_col = config.label_col
    output_dir = config.output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    analyze_clusters(input_file, id_col, label_col, output_dir)


if __name__ == "__main__":
    main()
    exit(0)