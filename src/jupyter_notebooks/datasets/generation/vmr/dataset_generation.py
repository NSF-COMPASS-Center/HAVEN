import pandas as pd
from Bio import SeqIO, Entrez
# from Bio.SeqRecord import SeqRecord
import time
# from io import StringIO
#
# import warnings
import re
from tqdm import tqdm
from glob import glob
import os

# from concurrent.futures import ProcessPoolExecutor

# # Suppress the specific UserWarning for data validation removal
# warnings.filterwarnings("ignore", message=".*Data Validation extension is not supported.*")
#
try:
    # Try to import defusedxml for security
    import defusedxml
    from openpyxl import load_workbook

    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("Warning: openpyxl or defusedxml not installed. Hyperlink extraction will be limited.")
    print("Install with: pip install openpyxl defusedxml")

KEY = "684a897c2fd938d11f2fe9441aeef9e7af09"


def process_excel_file(excel_file, accessions_column, isolate_id_column=None, ictv_column=None, output_prefix="vmr_output", output_dir="", sheet_name=0):
    # Read Excel file
    print(f"Reading Excel file: {excel_file}")
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.to_csv(f"{output_dir}/{output_prefix}_full.csv", index=False)
    print(f"Removing {len(df[df['Genome coverage']=='No entry in Genbank'])} isolates without entries in GenBank")
    df = df[df["Genome coverage"]!="No entry in Genbank"]
    # print(df[df[accessions_column].str.contains(r"\)")])
    print(f"Separating isolates with multiple entries in GenBank")
    df = (
        df.assign(accession=df[accessions_column].str.split(";"))
        .explode("accession")
    )
    df["accession"] = (
        df["accession"]
        .str.split(":")
        .str[-1]
        .str.strip()
    )
    entries_with_coord = df[accessions_column].str.contains(r"\(", na=False).sum()
    print(f"{entries_with_coord} entries have coordinates")
    df["coords"] = df[accessions_column].str.extract(r"\((\d+\.\d+)\)")
    print(df[df[accessions_column].str.contains(r"\)")][accessions_column])
    df["accession"] = df["accession"].str.replace(r"\(.*\)", "", regex=True).str.strip()
    df.to_csv(f"{output_dir}/{output_prefix}_accessions.csv", index=False)
    return(pd.DataFrame(df[[isolate_id_column,ictv_column, "accession"]]))


def query_genbank(accession_ids, output_dir, output_prefix):
    proteins = []
    batch = 500
    failed_records = []
    protein_count = 0
    record_count = 0
    chunk_size = 500
    pbar = tqdm(total = len(accession_ids), desc = 'Querying GenBank', ncols = 120, colour = 'cyan')
    max_retries = 3

    for chunk_idx in range(0, len(accession_ids), chunk_size):
        chunk = accession_ids[chunk_idx:chunk_idx+chunk_size]
        for attempt in range(max_retries):
            try:
                handle = Entrez.efetch(db='nucleotide', id=chunk, rettype='gb',
                                       retmode='text', api_key=KEY, sleep_between_tries=True)
                chunk_proteins = []
                chunk_records_processed = 0

                for record in SeqIO.parse(handle, "genbank"):  # IncompleteRead thrown here
                    for feature in record.features:
                        if feature.type == "CDS":
                            new_row = {
                                'Accession ID': record.id,
                                'Seq Length': len(record.seq),
                                'Location': str(feature.location),
                                'Protein ID': feature.qualifiers.get('protein_id', [None])[0],
                                'Protein Sequence': feature.qualifiers.get('translation', [None])[0],
                            }
                            chunk_proteins.append(new_row)
                            protein_count += 1
                        record_count += 1

                    chunk_records_processed += 1
                    if record_count % batch == 0:
                        proteins.extend(chunk_proteins)
                        df_batch = pd.DataFrame(proteins)
                        df_batch.to_csv(
                            f"{output_dir}/{output_prefix}_checkpoint/checkpoint_{record_count}_proteins.csv",
                            index=False)
                        proteins = []
                        chunk_proteins = []

                    if record_count % 10 == 0:
                        pbar.set_postfix({'Rec': record_count, 'ID': record.id[:12], 'Prots': protein_count})
                    pbar.update(1)

                handle.close()
                proteins.extend(chunk_proteins)  # flush any remaining
                break  # success, exit retry loop

            except (IncompleteRead, Exception) as e:
                tqdm.write(f"Attempt {attempt + 1}/{max_retries} failed for chunk {chunk_idx}: {e}")
                try:
                    handle.close()
                except:
                    pass
                if attempt < max_retries - 1:
                    time.sleep(10 * (attempt + 1))  # back off: 10s, 20s, 30s
                else:
                    tqdm.write(f"Chunk {chunk_idx} failed after {max_retries} attempts, skipping.")
                    failed_records.extend(chunk)

        time.sleep(0.5) # Be nice to NCBI
    pbar.close()
    # Save remaining proteins
    if proteins:
        df_batch = pd.DataFrame(proteins)
        df_batch.to_csv(f"{output_dir}/{output_prefix}_checkpoint/checkpoint_remaining_proteins.csv", index=False)

def combine_protein_seq_files(output_dir, output_prefix):
    folder_path = f"{output_dir}/{output_prefix}_checkpoint"
    print("Combining Protein Checkpoint Files")
    csv_files = sorted(glob(os.path.join(folder_path, '*.csv')))
    first = pd.read_csv(csv_files[0])
    first.to_csv(f"{output_dir}/{output_prefix}_proteins.csv", index=False)
    for file in csv_files[1:]:
        df = pd.read_csv(file)
        df.to_csv(f"{output_dir}/{output_prefix}_proteins.csv", mode='a', header=False, index=False)
    print("Merging Protein Seq Files with Taxonomy")
    data = pd.read_csv(f"{output_dir}/{output_prefix}_proteins.csv")
    data['accession_clean'] = data['Accession ID'].str.split(".").str[0]
    df = pd.read_csv(f"{output_dir}/{output_prefix}_accessions.csv")
    merged_df = df.merge(
        data,
        left_on='accession',
        right_on ='accession_clean',
        how = 'left'
    )
    merged_df = merged_df.drop_duplicates(subset = ['accession_clean', "Protein ID"])
    initial = len(merged_df)
    print(f"Dropping Rows with missing Protein Sequences")
    merged_df = merged_df.dropna(subset = "Protein Sequence")
    final = len(merged_df)
    print(f"Dropped {initial - final} Rows with missing Protein Sequences")
    merged_df.to_csv(f"{output_dir}/{output_prefix}_proteins_taxonomy.csv", index=False)
    # Some entries specify coordinate locations, so need to update the sequences for those
    print("Updating the Entries with Locations")
    coords_split = merged_df["coords"].astype(str).str.split(r"\.", expand=True)
    merged_df[["genome_start", "genome_end"]] = coords_split.apply(pd.to_numeric, errors='coerce')
    def parse_protein(prot_str):
        # Ensure the input is a string (also handles NaN values)
        if isinstance(prot_str, str):
            match = re.match(r"\[(\d+):(\d+)\]\((\+|\-)\)", prot_str)
            if match:
                start, end, strand = match.groups()
                return {"prot_start": int(start), "prot_end": int(end), "strand": strand, "prot_str": prot_str}
        return None  # Return None if not a valid match or not a string

    merged_df["proteins_parsed"] = merged_df["Location"].apply(lambda p: parse_protein(p) if parse_protein(p) else None)
    def filter_by_coords(row):
        if pd.notna(row["genome_start"]) and pd.notna(row["genome_end"]) and row["proteins_parsed"]:
            p = row["proteins_parsed"]
            # Keep only proteins that overlap with genome coordinates
            if p["prot_end"] >= row["genome_start"] and p["prot_start"] <= row["genome_end"]:
                return p
        return None  # If protein doesn't overlap, remove it

    # Apply the filter only to rows with coordinates
    merged_df["filtered_protein"] = merged_df.apply(
        lambda row: filter_by_coords(row) if pd.notna(row["genome_start"]) and pd.notna(row["genome_end"]) else row[
            "proteins_parsed"], axis=1)

    # Now drop rows where proteins were not found after filtering (only for those with coordinates)
    df_filtered = merged_df.dropna(subset=["filtered_protein"]).reset_index(drop=True)

    # Normalize the filtered proteins (the ones that match the coordinates)
    df_filtered = pd.concat([
        df_filtered.drop(columns=["proteins_parsed"]),
        pd.json_normalize(df_filtered["filtered_protein"])
    ], axis=1)

    def get_final_seq(row):
        if (pd.notna(row["genome_start"]) and pd.notna(row["genome_end"]) and
                pd.notna(row["prot_start"]) and pd.notna(row["prot_end"])):
            if row["prot_end"] >= row["genome_start"] and row["prot_start"] <= row["genome_end"]:
                return row["Protein Sequence"][int(row["prot_start"]):int(row["prot_end"])]
            else:
                return None
        else:
            return row["Protein Sequence"]
    df_filtered["prot_seq"] = df_filtered.apply(get_final_seq, axis = 1)
    rows_before = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=["prot_seq"]).reset_index(drop=True)
    rows_after = len(df_filtered)
    print(
        f"Dropped {rows_before - rows_after} rows where coordinates were specified but no protein overlap was found ({rows_after} remaining)")
    df_filtered.to_csv(f"{output_dir}/{output_prefix}_proteins_taxonomy_final.csv", index=False)


if __name__ == "__main__":
    excel_file = "/Users/katelandsipe/Documents/Research/git/viral_tax_dataset/VMR_MSL40_v2_20251013.xlsx" #"/home/sipek/HAVEN/input/data/vmr"  # Your Excel file
    accessions_column = "Virus GENBANK accession"  # Column name with URLs/hyperlinks
    isolate_column = "Isolate ID"  # Column name with Isolate IDs
    ictv_column = "ICTV_ID" # Column name with ICTV IDs
    output_prefix = 'vmr_msl40_v2_20251013'
    output_dir = "/Users/katelandsipe/Documents/Research/git/viral_tax_dataset"
    Entrez.email = "sipek@vt.edu"

    # filtered_data = process_excel_file(
    #     excel_file=excel_file,
    #     accessions_column = accessions_column,
    #     isolate_id_column=isolate_column,
    #     ictv_column = ictv_column,
    #     output_prefix= output_prefix,
    #     output_dir = output_dir,
    #     sheet_name="VMR MSL40"  # Use 0 for first sheet, or "Sheet1" for sheet name
    # )
    # query_genbank(filtered_data['accession'].tolist(), output_dir, output_prefix)
    # combine_protein_seq_files(output_dir, output_prefix)
    df = pd.read_csv(f"{output_dir}/{output_prefix}_proteins_taxonomy_final.csv")
    # split into dataset with genera with only 1 genome vs >1 genome
    counts = df.groupby("Genus")["accession_clean"].transform("count")
    df_gt1 = df[counts > 1].reset_index(drop=True)
    df_gt1.to_csv(f"{output_dir}/{output_prefix}_proteins_taxonomy_final_gt1.csv", index=False)
    print(f"Genera with >1 genome: {len(df_gt1)}")
    print(f"Number of Species: {len(df_gt1['Species'].unique())}")
    df_eq1 = df[counts == 1].reset_index(drop=True)
    print(f"Genera with 1 genome: {len(df_eq1)}")
    print(f"Number of Species: {len(df_eq1['Species'].unique())}")
    df_eq1.to_csv(f"{output_dir}/{output_prefix}_proteins_taxonomy_final_eq1.csv", index=False)