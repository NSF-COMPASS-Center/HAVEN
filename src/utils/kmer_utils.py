from itertools import product
from collections import Counter
import pandas as pd


def compute_kmer_features(df, k, label_col):
    kmer_keys = get_kmer_keys(df, k)
    df["features"] = df.apply(lambda row: get_kmer_vector(row["sequence"], k, kmer_keys), axis=1)
    df.drop(columns=["sequence"], inplace=True)
    kmer_df = pd.DataFrame.from_records(df["features"].values, index=df.index)
    kmer_df_with_label = kmer_df.join(df[label_col], on="id", how="left")
    print(f"Size of kmer dataset with label = {kmer_df_with_label.shape}")
    print(f"Validation: First row in kmer dataset with label = \n{kmer_df_with_label.head(1)}")
    return kmer_df_with_label


def get_kmer_vector(x, k, kmer_keys):
    counter = initialize_kmer_counter(kmer_keys)
    n_kmers = len(x) - k + 1
    kmers = []
    for i in range(n_kmers):
        kmers.append(x[i:i+k])
    counter.update(kmers)
    return dict(counter)


def initialize_kmer_counter(kmer_keys):
    counter_map = {}
    for kmer in kmer_keys:
        counter_map[kmer] = 0
    counter = Counter(counter_map)
    return counter


def get_kmer_keys(dataset, k):
    sequences = dataset["sequence"].values
    print(f"Number of sequences = {len(sequences)}")
    unique_chars = list(set(''.join(sequences)))

    print(f"Unique characters in all sequences = {unique_chars}")
    print(f"Number of unique characters in all sequences = {len(unique_chars)}")

    kmer_keys = ["".join(p) for p in product("".join(unique_chars), repeat=k)]
    print(f"Number of kmer_keys = {len(kmer_keys)}")

    return kmer_keys