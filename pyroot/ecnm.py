# ecnm.py
import numpy as np
import random
import os
import csv
from basernn import BaseCharRNN

class CharRNN(BaseCharRNN):
    def __init__(self, vocab_size, hidden_size=32, lr=0.01, seed=42):
        super().__init__(vocab_size, hidden_size, lr, seed)
        def rand_mat(r, c):
            std = np.sqrt(1.0 / r) * 0.1
            return np.random.randn(r, c) * std
        self.Wxh = rand_mat(vocab_size, hidden_size)
        self.Whh = rand_mat(hidden_size, hidden_size)
        self.Why = rand_mat(hidden_size, vocab_size)
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(vocab_size)

    def forward(self, inputs, targets, h_prev):
        xs, hs, ps = {}, {}, {}
        hs[-1] = h_prev.copy()
        loss = 0.0
        for t in range(len(inputs)):
            x = np.zeros(self.vocab_size)
            x[inputs[t]] = 1.0
            xs[t] = x
            h = np.tanh(self.Wxh.T @ x + self.Whh.T @ hs[t - 1] + self.bh)
            hs[t] = h
            y = self.Why.T @ h + self.by
            p = self.softmax(y)
            ps[t] = p
            loss += -np.log(max(p[targets[t]], 1e-8))
        return loss, xs, hs, ps

    def backward(self, inputs, targets, xs, hs, ps, h_prev):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros(self.hidden_size)
        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0
            dby += dy
            dWhy += np.outer(hs[t], dy)
            dh = self.Why @ dy + dhnext
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += np.outer(xs[t], dh_raw)
            dWhh += np.outer(hs[t - 1], dh_raw)
            dhnext = self.Whh @ dh_raw
        for d in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(d, -1.0, 1.0, out=d)
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

    def sample(self, seed_char, length=6, temperature=0.8):
        return super().sample(seed_char, length, temperature)

    def save(self, path):
        np.savez(path, Wxh=self.Wxh, Whh=self.Whh, Why=self.Why,
                 bh=self.bh, by=self.by)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.Wxh = data["Wxh"]
        self.Whh = data["Whh"]
        self.Why = data["Why"]
        self.bh = data["bh"]
        self.by = data["by"]
        self.vocab_size = self.Wxh.shape[0]
        self.hidden_size = self.Wxh.shape[1]

def train_names(
    names,
    hidden_size=32,
    lr=0.01,
    epochs=30000,
    model_path="names_model.npz",
    history_path="names_history.npz",
    history_csv="names_history.csv",
    resume=True,
    log_every=100,
    patience=30,
    min_improve=1e-5,
    target_loss=0.1,
):
    chars = sorted(set("".join(names)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    rnn = CharRNN(vocab_size, hidden_size=hidden_size, lr=lr, seed=42)
    rnn.char_to_idx = char_to_idx
    rnn.idx_to_char = idx_to_char
    start_epoch = 0
    best_loss = float("inf")
    wait = 0
    all_epochs = []
    all_losses = []
    if resume and os.path.exists(model_path) and os.path.exists(history_path):
        print("发现存档，正在恢复...")
        try:
            rnn.load(model_path)
            rnn.char_to_idx = char_to_idx
            rnn.idx_to_char = idx_to_char
            data = np.load(history_path, allow_pickle=True)
            all_epochs = list(data["epochs"])
            all_losses = list(data["losses"])
            start_epoch = len(all_epochs)
            best_loss = float(np.min(all_losses)) if all_losses else float("inf")
        except Exception:
            print("存档损坏，从头训练")
    else:
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "loss"])
    try:
        for epoch in range(start_epoch, epochs):
            total_loss = 0.0
            random.shuffle(names)
            for name in names:
                h_prev = np.zeros(rnn.hidden_size)
                inputs = [char_to_idx[c] for c in name[:-1]]
                targets = [char_to_idx[c] for c in name[1:]]
                loss, xs, hs, ps = rnn.forward(inputs, targets, h_prev)
                rnn.backward(inputs, targets, xs, hs, ps, h_prev)
                total_loss += loss
            avg_loss = total_loss / len(names)
            all_epochs.append(epoch)
            all_losses.append(float(avg_loss))
            if avg_loss < best_loss:
                best_loss = avg_loss
            if epoch % log_every == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:6d} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
                with open(history_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([epoch, f"{avg_loss:.8f}"])
            if avg_loss < target_loss:
                break
            if epoch % log_every == 0:
                improvement = best_loss - avg_loss
                if improvement < min_improve:
                    wait +=1
                    if wait >= patience:
                        break
                else:
                    wait = 0
    except KeyboardInterrupt:
        pass
    finally:
        rnn.save(model_path)
        np.savez(history_path, epochs=np.array(all_epochs), losses=np.array(all_losses))
    return rnn, all_epochs, all_losses
