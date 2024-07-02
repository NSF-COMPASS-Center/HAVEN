PAD_TOKEN_VAL = 0
CLS_TOKEN_VAL = 101
MASK_TOKEN_VAL = 102
AMINO_ACID_VOCABULARY = {"A": 1, "R": 2, "N": 3, "D": 4, "C": 5,
                         "Q": 6, "E": 7, "G": 8, "H": 9, "I": 10,
                         "L": 11, "K": 12, "M": 13, "F": 14, "P": 15,
                         "O": 16, "S": 17, "U": 18, "T": 19, "W": 20,
                         "Y": 21, "V": 22, "B": 23, "Z": 24, "X": 25,
                         "J": 26, "-": 27}
# the total size of the vocabulary. We need this hack because the token values serve as indices for the embedding look-up vocabulary in nn.Embedding.
# hence we need the maximum value and not just the total number of tokens
VOCAB_SIZE = max([max(AMINO_ACID_VOCABULARY.values()), PAD_TOKEN_VAL, CLS_TOKEN_VAL, MASK_TOKEN_VAL]) + 1
VOCAB_SIZE = 28