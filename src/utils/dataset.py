from torch.utils.data import Dataset


class HepDataset(Dataset):
    def __init__(self):
        super(HepDataset, self).__init__()
        self.init_static_vocab()

    def init_static_vocab(self):
        self.amino_map = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
                          'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
                          'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                          'O': 15, 'S': 16, 'U': 17, 'T': 18, 'W': 19,
                          'Y': 20, 'V': 21, 'B': 22, 'Z': 23, 'X': 24,
                          'J': 25}
        self.n_amino_tokens = 26

        def __getitem__(self, idx: int):
