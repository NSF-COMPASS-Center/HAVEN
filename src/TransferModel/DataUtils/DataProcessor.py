import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


def split_seqs_dict(seqs, percentTrain=.8, seed=1):
    x_train, x_test, y_train, y_test = train_test_split(list(seqs.keys()), list(seqs.values()),
                                                        train_size=percentTrain, random_state=seed)

    train_seqs = {k: v for k, v in zip(x_train, y_train)}
    test_seqs = {k: v for k, v in zip(x_test, y_test)}

    return train_seqs, test_seqs


# Returns random train and test dataframes respectively
def split_df(df, percentTrain=.8, seed=1):
    if percentTrain == 0:
        df_train, df_test = pd.DataFrame(columns=df.columns), df.sample(frac=1, random_state=seed)
    elif percentTrain == 1:
        df_train, df_test = df.sample(frac=1, random_state=seed), pd.DataFrame(columns=df.columns)
    else:
        df_train, df_test = train_test_split(df, train_size=percentTrain, stratify=df['y'], random_state=seed)

    return df_train, df_test



# TODO: k-fold

# Expects a defined vocabulary map starting at 1, e.g {human : 1, dog : 2, ...}
# returns numerical representation of the df[key] with vocabulary, where 0 is unknown
def featurize_target(df, key, vocabulary):
    return df[key].apply(lambda x: vocabulary[x] if x in vocabulary else 0)


def featurize_host(target, vocabulary):
    assert (vocabulary)
    return vocabulary[target] if target in vocabulary else 0


def featurize_seqs_host(seq, vocabulary):
    # First two in vocabulary are paddings
    start_int = len(vocabulary) + 1

    # TODO: extensive comment
    # tldr since their model is 1 less, we chop off end_int.
    end_int = len(vocabulary) + 2

    return np.array([start_int] + [
        vocabulary[word] for word in seq
    ], dtype=np.int8)


def featurize_df(df, inputVocab, targetVocab, targetKey):
    """
    Args:
        df: Pandas df, with seq and targetVocab as a column
        inputVocab: Vocab for model inputs
        targetVocab: Target vocab for model outputs
        targetKey: Key for the column we're targeting

    Returns: The dataframe, but with two new columns, X, y with featurized results

    """
    df['X'] = df['seq'].apply(lambda x: featurize_seqs_host(x, inputVocab))
    df['y'] = df[targetKey].apply(lambda x: featurize_host(x, targetVocab))
    return df


def get_tokenizer(vocabulary, maxLen):
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize=None,
        output_mode="int",
        output_sequence_length=maxLen,
        split="character",
        vocabulary=vocabulary
    )
    return vectorize_layer


def toOneHot(vector, yVocab):
    vector = label_binarize(vector, classes=[x for x in yVocab.values()])
    if len(yVocab) == 2:
        return np.array([[item[0], 0 if item[0] else 1] for item in vector])
    return vector


def initilizeVocab(words):
    """
    Initializes a vocabulary mapping, starting from 1.
    Depending on your needs, you can use the zero index for anything,
    For example, for amino acid words, you can use the 0 index to represent padding
    For target prediction vocabulary, you can use the 0 index to represent unknown entity
    Args:
        words:

    Returns:

    """
    return {word: i for i, word in enumerate(sorted(words), start=1)}


def sparseToDense(y_pred):
    """
    Converts sparse to dense format.
    If y_pred is a one-hot-encoding, no information is lost when converting to dense
    If y_pred is a model prediction (e.g., softmax values), a prediction is made on the max's column index for dense representation
    Args:
        y_pred: Sparse numpy array, e.g [[.8, .2], [.1, .4], ...]

    Returns: Dense representation of data

    """
    return np.argmax(y_pred, axis=1)

def shortenSeqs(df, size):
    if df.shape[0] > 0 and "X" in df:
        return df["X"].apply(lambda x: x[0:size])
    return df

def mkdir_p(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)