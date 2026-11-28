# mte.py
import re
import os
import random
import numpy as np
from ecnm import CharRNN
from common import ensure_task_dir, find_unfinished, new_round_dir, mark_round_finished, save_config

def train_rnn(
    lines,
    hidden_size=64,
    lr=0.005,
    epochs=5000,
    log_every=100,
    seed=42,
    callback=None,
    stop_flag=None,
    round_dir=None,
    start_epoch=0,
    model_path=None,
    history_path=None,
    max_lines=None,
):
    if max_lines is not None:
        lines = lines[:max_lines]
    if round_dir is None:
        round_dir = "."
    if model_path is None:
        model_path = os.path.join(round_dir, "model.npz")
    if history_path is None:
        history_path = os.path.join(round_dir, "history.npz")
    config = {
        "hidden_size": hidden_size,
        "lr": lr,
        "epochs": epochs,
        "log_every": log_every,
        "max_lines": max_lines,
        "data_lines": len(lines),
        "seed": seed
    }
    save_config(round_dir, config)
    chars = sorted(set("".join(lines)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    rnn = CharRNN(vocab_size, hidden_size=hidden_size, lr=lr, seed=seed)
    rnn.char_to_idx = char_to_idx
    rnn.idx_to_char = idx_to_char
    all_losses = []
    if start_epoch > 0 and os.path.exists(model_path) and os.path.exists(history_path):
        try:
            rnn.load(model_path)
            rnn.char_to_idx = char_to_idx
            rnn.idx_to_char = idx_to_char
            data = np.load(history_path, allow_pickle=True)
            all_losses = list(data["losses"])
            if callback:
                callback(start_epoch, float("nan"))
        except Exception:
            pass
    finished = True
    for epoch in range(start_epoch, epochs):
        if stop_flag and stop_flag():
            finished = False
            break
        total_loss = 0.0
        random.shuffle(lines)
        for name in lines:
            if stop_flag and stop_flag():
                finished = False
                break
            h_prev = np.zeros(rnn.hidden_size)
            inputs = [char_to_idx[c] for c in name[:-1]]
            targets = [char_to_idx[c] for c in name[1:]]
            loss, xs, hs, ps = rnn.forward(inputs, targets, h_prev)
            rnn.backward(inputs, targets, xs, hs, ps, h_prev)
            total_loss += loss
        avg_loss = total_loss / len(lines)
        all_losses.append(avg_loss)
        if epoch % log_every == 0 and callback:
            callback(epoch, float(avg_loss))
    rnn.save(model_path)
    np.savez(history_path,
             epochs=np.arange(len(all_losses)),
             losses=np.array(all_losses))
    if finished:
        mark_round_finished(round_dir)
    return rnn, all_losses

def generate_samples(rnn, n=5, length=6, temperature=0.8):
    seeds = list(rnn.char_to_idx.keys())
    return [
        rnn.sample(random.choice(seeds), length=length, temperature=temperature)
        for _ in range(n)
    ]
