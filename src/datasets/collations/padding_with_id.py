from datasets.collations.padding import Padding


class PaddingWithID(Padding):
    def __init__(self, max_seq_length):
        super(PaddingWithID, self).__init__(max_seq_length)

    def __call__(self, batch):
        ids, sequences, labels = zip(*batch)
        padded_sequences, labels = super(PaddingWithID, self).__call__(list(zip(sequences, labels)))
        return ids, padded_sequences, labels
