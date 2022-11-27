import torch
from torch.utils.data import Dataset
import re


class WordDataset(Dataset):
    def __init__(self, data, block_size):
        words = re.split(r"\b", data)
        vocab = sorted(list(set(words)))
        data_size, vocab_size = len(words), len(vocab)
        print("data has %d words, %d unique." % (data_size, vocab_size))

        self.stoi = {word: i for i, word in enumerate(vocab)}
        self.itos = {i: word for i, word in enumerate(vocab)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = words

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        chunk = self.data[
            idx * self.block_size : (idx * self.block_size) + self.block_size + 1
        ]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
