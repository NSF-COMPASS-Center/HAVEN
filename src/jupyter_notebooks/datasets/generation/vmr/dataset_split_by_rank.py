import pandas as pd

df = pd.read_csv("/Users/katelandsipe/Documents/Research/git/viral_tax_dataset/vmr_msl40_v2_20251013_proteins_taxonomy_final_gt1.csv")
output_file_path = "/Users/katelandsipe/Documents/Research/git/viral_tax_dataset/splits/"
null_seqs = df["prot_seq"].isna().sum()
print(f"Number of NA Prot Sequences: {null_seqs}")

df = df.dropna(subset=["prot_seq"]).reset_index(drop=True)


for genus, group in df.groupby("Genus"):
    group.to_csv(f"{output_file_path}Genus/{genus}.csv", index=False)

for family, group in df.groupby("Family"):
    group.to_csv(f"{output_file_path}Family/{family}.csv", index=False)

for order, group in df.groupby("Order"):
    group.to_csv(f"{output_file_path}Order/{order}.csv", index=False)


for class_rank, group in df.groupby("Class"):
    group.to_csv(f"{output_file_path}Class/{class_rank}.csv", index=False)

for phylum, group in df.groupby("Phylum"):
    group.to_csv(f"{output_file_path}Phylum/{phylum}.csv", index=False)

for kingdom, group in df.groupby("Kingdom"):
    group.to_csv(f"{output_file_path}Kingdom/{kingdom}.csv", index=False)

for realm, group in df.groupby("Realm"):
    group.to_csv(f"{output_file_path}Realm/{realm}.csv", index=False)