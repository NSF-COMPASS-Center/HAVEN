import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.sequence import pad_sequences


def split_seqs_dict(seqs, percentTrain=.8, seed=1):
    x_train, x_test, y_train, y_test = train_test_split(list(seqs.keys()), list(seqs.values()),
                                                        train_size=percentTrain, random_state=seed)

    train_seqs = {k: v for k, v in zip(x_train, y_train)}
    test_seqs = {k: v for k, v in zip(x_test, y_test)}

    return train_seqs, test_seqs


# Returns train and test dataframes respectively
def split_df(df, percentTrain=.8):
    return np.split(df.sample(frac=1), [int(percentTrain * len(df))])


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



def featurize_df(df, seqLen, vocab, targetVocab, targetKey):
    df['seq_processed'] = df['seq'].apply(lambda x: featurize_seqs_host(x, vocab))
    df['target'] = df[targetKey].apply(lambda x: featurize_host(x, targetVocab))
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
