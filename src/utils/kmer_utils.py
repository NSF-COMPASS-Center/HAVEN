from itertools import product
from collections import Counter
import pandas as pd


def compute_kmer_based_dataset(df, k, label):
    kmer_keys = get_kmer_keys(df, k)
    df["features"] = df.apply(lambda row: get_kmer_vector(row["sequence"], k, kmer_keys), axis=1)
    df.drop(columns=["sequence"], inplace=True)
    transformed_df = pd.DataFrame.from_records(df["features"].values, index=df.index)
    transformed_df_with_label = transformed_df.join(df[label], on="id", how="left")
    # print(transformed_df_with_label)
    print(f"Size of dataset with label = {transformed_df_with_label.shape}")
    return transformed_df_with_label


def get_kmer_vector(x, k, kmer_keys):
    counter = initialize_kmer_counter(kmer_keys)
    n_kmers = len(x) - k + 1
    kmers = []
    for i in range(n_kmers):
        kmers.append(x[i:i+k])
    counter.update(kmers)
    return dict(counter)


def initialize_kmer_counter(kmers):
    counter_map = {}
    for kmer in kmers:
        counter_map[kmer] = 0
    counter = Counter(counter_map)
    # print(counter)
    return counter


def get_kmer_keys(dataset, k):
    sequences = dataset["sequence"].values
    print(f"number of sequences = {len(sequences)}")
    unique_chars = list(set(''.join(sequences)))
    print(f"unique characters in all sequences = {unique_chars}")
    print(f"number of unique characters in all sequences = {len(unique_chars)}")

    kmer_keys = ["".join(p) for p in product("".join(unique_chars), repeat=k)]
    print(f"Number of kmer_keys = {len(kmer_keys)}")

    return kmer_keys