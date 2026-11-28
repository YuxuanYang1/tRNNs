# basernn.py
import numpy as np
import random

class BaseCharRNN:
    def __init__(self, vocab_size, hidden_size=32, lr=0.01, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.char_to_idx = None
        self.idx_to_char = None

    @staticmethod
    def softmax(z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def forward(self, inputs, targets, h_prev):
        raise NotImplementedError

    def backward(self, inputs, targets, xs, hs, ps, h_prev):
        raise NotImplementedError

    def sample(self, seed_char, length, temperature=0.8):
        h = np.zeros(self.hidden_size)
        if seed_char not in self.char_to_idx:
            seed_char = random.choice(list(self.char_to_idx.keys()))
        idx = self.char_to_idx[seed_char]
        result = seed_char
        for _ in range(length):
            x = np.zeros(self.vocab_size)
            x[idx] = 1.0
            h = np.tanh(self.Wxh.T @ x + self.Whh.T @ h + self.bh)
            y = self.Why.T @ h + self.by
            y = y / temperature
            p = self.softmax(y)
            idx = np.random.choice(self.vocab_size, p=p)
            ch = self.idx_to_char[idx]
            result += ch
        return result

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
