# Util functions to query external sources  -
#  - PyTaxonKit: to get taxonomy
#  - UniProt: to get organism host and EMBL ids
#  - EMBL: to get organism host
import random

import requests
# import pytaxonkit
import pandas as pd
import os
from Bio import SeqIO

# UniProt keywords/contsant values
UNIPROT_REST_API = "https://rest.uniprot.org/uniprotkb/search"
UNIREF90_QUERY_PARAM = "uniref_cluster_90:%s"

# EMBL keywords/constant values
EMBL_REST_API = "https://www.ebi.ac.uk/Tools/dbfetch"

# NCBI Taxonomy keywords
NAME = "Name"
RANK = "Rank"
NCBI_TAX_ID = "TaxID"
TAXONKIT_DB = "TAXONKIT_DB"
MAMMALIA = "Mammalia"
AVES = "Aves"
VERTEBRATA_TAX_ID = "7742"


# query UniProt for to get the host of the virus of the protein sequence
# input: uniref90_id
# output: list of host(s) of the virus
def query_uniprot(uniref90_id):
    # split UniRef90_A0A023GZ41 and capture A0A023GZ41
    uniprot_id = uniref90_id.split("_")[1]
    response = requests.get(url=UNIPROT_REST_API,
                            params={"query": UNIREF90_QUERY_PARAM % uniref90_id,
                                    "fields": ",".join(["virus_hosts", "xref_embl"])})
    host_tax_ids = []
    embl_entry_id = None
    try:
        results = response.json()["results"]
        # ideally there should be only one matching primaryAccession entry for the seed uniprot_id
        data = [result for result in results if result["primaryAccession"] == uniprot_id][0]

        # embl cross reference entry id
        cross_refs = data["uniProtKBCrossReferences"]
        embl_cross_ref_properties = \
            [cross_ref for cross_ref in cross_refs if cross_ref["database"] == "EMBL"][0]["properties"]
        embl_entry_id = \
            [property for property in embl_cross_ref_properties if property["key"] == "ProteinId"][0][
                "value"]

        # organism hosts from uniprot
        org_hosts = data["organismHosts"]
        for org_host in org_hosts:
            host_tax_ids.append(org_host["taxonId"])

    except (KeyError, IndexError):
        # to differentiate between the absence of mapping for a given sequence and
        # a sequence with mapping but zero hosts
        host_tax_ids = None
        pass
    return host_tax_ids, embl_entry_id


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


# Get taxids belonging to the clade = Vertebrata
# Input: list of tax_ids
# Output: list of tax_ids belonging to Vertebrata clade
def get_vertebrata_tax_ids(tax_ids):
    vertebrata_tax_ids = []
    for i, tax_id in enumerate(tax_ids):
        # Issue: No placeholder formatter for rank=clade. Hence cannot use formatstr as for class {c}
        # Workaround: Get full lineage and filter for vertebrata Tax ID
        # example output from pytaxonkit.lineage([]):
        # '131567;2759;33154;33208;6072;33213;33511;7711;89593;7742;7776;117570;117571;8287;1338369;32523;32524;40674;32525;9347;1437010;314146;9443;376913;314293;9526;314295;9604;207598;9605;9606'
        # hence split by ";"
        full_lineage_tax_ids = pytaxonkit.lineage([tax_id])["FullLineageTaxIDs"].iloc[0].split(";")
        if VERTEBRATA_TAX_ID in full_lineage_tax_ids:
            vertebrata_tax_ids.append(tax_id)
    return vertebrata_tax_ids


# Get taxids belonging to the class of mammals and aves
# Input: list of tax_ids
# Output: list of tax_ids belonging to mammals and aves class
def get_mammals_aves_tax_ids(tax_ids):
    mammals_aves_tax_ids = []
    for i, tax_id in enumerate(tax_ids):
        tax_class = pytaxonkit.lineage([tax_id], formatstr="{c}")["Lineage"].iloc[0]
        print(f"{i}: {tax_id} = {tax_class}")
        if tax_class == MAMMALIA or tax_class == AVES:
            mammals_aves_tax_ids.append(tax_id)
    return mammals_aves_tax_ids


# query EMBL for to get the host of the virus of the protein sequence
# input: embl_ref_id
# output: host name(s) of the virus
def query_embl(embl_ref_ids, temp_dir):
    response = requests.get(url=EMBL_REST_API, params=dict(format="embl", style="raw", id=",".join(embl_ref_ids)))
    # BioPython's SeqIO only takes input in the form of files.
    # Hack:
    # 1. Write the REST API's response to a temporary file to be processed using BioPython
    # 2. Capture the host
    # 3. Delete the temp file at the end of processing

    temp_output_file_path = os.path.join(temp_dir, "temp_" + str(random.randint(0, 1e9)) + ".txt")
    with open(temp_output_file_path, "w") as f:
        f.write(response.text)
    embl_host_mapping = {}
    for record in SeqIO.parse(temp_output_file_path, "embl"):
        # find the source feature which contains the host information
        source_feature = None
        for feature in record.features:
            if feature.type == "source":
                source_feature = feature
                break

        host = None
        if source_feature and source_feature.qualifiers["host"]:
            host = source_feature.qualifiers["host"]
        embl_host_mapping[record.id] = host
    # delete the temporary file
    os.remove(temp_output_file_path)
    return embl_host_mapping
