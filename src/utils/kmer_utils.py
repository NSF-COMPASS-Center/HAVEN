from itertools import product
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_kmer_features(df, k, id_col, sequence_col, label_col, kmer_keys=None):
    print(f"Method:compute_kmer_features : df size = {df.shape}")
    if kmer_keys.any() and len(kmer_keys) > 0:
        print(f"Using provided {len(kmer_keys)} kmer_keys = {kmer_keys}")
    else:
        print("Computing kmer_keys")
        kmer_keys = get_kmer_keys(df, k, sequence_col)
    print(f"Number of kmer_keys = {len(kmer_keys)}")

    df["features"] = df.apply(lambda row: get_kmer_vector(row[sequence_col], k, kmer_keys), axis=1)
    df.drop(columns=[sequence_col], inplace=True)
    df.set_index(id_col, inplace=True)
    kmer_df = pd.DataFrame.from_records(df["features"].values, index=df.index)

    # retain only those columns (kmers) that occur at least once in the dataset i.e. sum across all rows > 0
    # kmer_df = kmer_df[kmer_df.columns[kmer_df.sum() > 0]]
    kmer_df_with_label = kmer_df.join(df[label_col], on=id_col, how="left")
    print(f"Size of kmer dataset with label = {kmer_df_with_label.shape}")
    print(f"Validation: First row in kmer dataset with label = \n{kmer_df_with_label.head(1)}")
    return kmer_df_with_label


def get_kmer_vector(x, k, kmer_keys):
    counter = initialize_kmer_counter(kmer_keys)
    n_kmers = len(x) - k + 1
    kmers = []
    for i in range(n_kmers):
        kmer = x[i:i+k]
        if kmer in kmer_keys:
            kmers.append(kmer)
    counter.update(kmers)
    return dict(counter)


def initialize_kmer_counter(kmer_keys):
    counter_map = {}
    for kmer in kmer_keys:
        counter_map[kmer] = 0
    counter = Counter(counter_map)
    return counter


def get_kmer_keys(dataset, k, sequence_col):
    sequences = dataset[sequence_col].values
    print(f"Number of sequences = {len(sequences)}")
    unique_chars = list(set(''.join(sequences)))

    print(f"Unique characters in all sequences = {unique_chars}")
    print(f"Number of unique characters in all sequences = {len(unique_chars)}")

    kmer_keys = ["".join(p) for p in product("".join(unique_chars), repeat=k)]
    print(f"Number of all possible kmer_keys = {len(kmer_keys)}")

    kmers_occurrence_count_map = {}
    for sequence in sequences:
        kmers_in_seq = set()
        n_kmers = len(sequence) - k + 1
        for i in range(n_kmers):
            kmer = sequence[i:i + k]
            if kmer not in kmers_in_seq:
                if kmer in kmers_occurrence_count_map:
                    kmers_occurrence_count_map[kmer] += 1
                else:
                    kmers_occurrence_count_map[kmer] = 1
                kmers_in_seq.add(kmer)

    filter_threshold = len(sequences) * 0.1  # 10% of the total number of sequences
    kmers_filtered = set()
    print(f"Number of kmer_keys BEFORE filtering for {filter_threshold} occurrences: {len(kmers_occurrence_count_map)}")
    for k, v in kmers_occurrence_count_map.items():
        if v > filter_threshold:
            kmers_filtered.add(k)
    print(f"Number of kmer_keys AFTER filtering for {filter_threshold} occurrences: {len(kmers_filtered)}")
    kmers_occurrence_count_map.clear()
    return list(kmers_filtered)