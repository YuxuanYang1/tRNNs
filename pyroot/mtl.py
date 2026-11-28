# mtl.py
import os
import numpy as np
from blom import XORNet
from common import save_config, mark_round_finished

GATES = {
    "and": np.array([[0], [0], [0], [1]], dtype=float),
    "or":  np.array([[0], [1], [1], [1]], dtype=float),
    "xor": np.array([[0], [1], [1], [0]], dtype=float),
}
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)


def train_xor(
    gate="xor",
    hidden_sizes=None,   # 新增：列表，如 [4] 或 [4,4] 或 []
    lr=0.5,
    epochs=5000,
    log_every=100,
    seed=42,
    callback=None,
    stop_flag=None,
    round_dir=None,
    start_epoch=0,
    model_path=None,
    history_path=None,
):
    gate = gate.lower()
    if gate not in GATES:
        raise ValueError(f"未知逻辑门: {gate}，可选 {list(GATES.keys())}")
    y = GATES[gate]

    # 兼容旧参数 hidden_size（单个数字）
    if hidden_sizes is None:
        hidden_sizes = [4]
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]

    if round_dir is None:
        round_dir = "."
    if model_path is None:
        model_path = os.path.join(round_dir, "model.npz")
    if history_path is None:
        history_path = os.path.join(round_dir, "history.npz")

    config = {
        "gate": gate,
        "hidden_sizes": hidden_sizes,
        "lr": lr,
        "epochs": epochs,
        "log_every": log_every,
        "seed": seed
    }
    save_config(round_dir, config)

    # 构建网络层：输入2 → 各隐藏层 → 输出1
    layers = [2] + list(hidden_sizes) + [1]
    net = XORNet(layers=layers, lr=lr, seed=seed)

    all_losses = []
    if start_epoch > 0 and os.path.exists(model_path) and os.path.exists(history_path):
        try:
            net.load(model_path)
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
        y_pred = net.forward(X)
        loss = net.compute_loss(y_pred, y)
        net.backward(X, y)
        all_losses.append(float(loss))
        if epoch % log_every == 0 and callback:
            callback(epoch, float(loss))

    net.save(model_path)
    np.savez(history_path,
             epochs=np.arange(len(all_losses)),
             losses=np.array(all_losses))
    if finished:
        mark_round_finished(round_dir)
    return net, all_losses


def predict_gate(net):
    prob = net.forward(X)
    pred = (prob > 0.5).astype(int)
    return pred, prob
