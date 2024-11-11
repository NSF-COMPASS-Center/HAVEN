import torch

class ESM2CollateFunction:

    def __call__(self, batch):
        # batch is a list of ((name, sequence), label) tuples
        named_sequences = [(item[0][0], item[0][1]) for item in batch]  # (name, sequence)
        labels = torch.stack([item[1] for item in batch])  # labels

        return named_sequences, labels