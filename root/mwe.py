# mwe.py
import os
import random
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QSpinBox,
    QDoubleSpinBox, QFormLayout, QGroupBox
)
from basethread import BaseTrainThread
from basewindow import BaseTrainWindow
from mte import train_rnn, generate_samples

class TrainThread(BaseTrainThread):
    def __init__(self, data_file, hidden, lr, epochs, log_every,
                round_dir, start_epoch, seed=42):
        super().__init__()
        self.data_file = data_file
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.log_every = log_every
        self.round_dir = round_dir
        self.start_epoch = start_epoch
        self.seed = seed

    def run_body(self) -> str:
        with open(self.data_file, encoding="utf-8") as f:
            names = [l.strip().lower() for l in f if l.strip()]
        names = [n for n in names if n.isalpha()]
        rnn, losses = train_rnn(
            names,
            hidden_size=self.hidden,
            lr=self.lr,
            epochs=self.epochs,
            log_every=self.log_every,
            round_dir=self.round_dir,
            start_epoch=self.start_epoch,
            seed=self.seed,
            callback=lambda e, l: self.progress.emit(e, l),
            stop_flag=lambda: not self._running,
        )
        samples = generate_samples(rnn, n=5)
        return f"训练完成！最终 loss: {losses[-1]:.4f}\n\n" + "\n".join(samples)

class NamesWindow(BaseTrainWindow):
    def __init__(self):
        super().__init__(task_key="ecnm")
        self.setWindowTitle("英文名字生成")
        self.setMinimumSize(560, 520)
        layout = QVBoxLayout()
        file_group = QGroupBox("数据文件")
        fl = QHBoxLayout()
        fl.addWidget(QLabel("names.txt："))
        self.file_input = QLineEdit("names.txt")
        fl.addWidget(self.file_input)
        file_group.setLayout(fl)
        layout.addWidget(file_group)
        param_group = QGroupBox("训练参数")
        form = QFormLayout()
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(16, 256)
        self.hidden_spin.setValue(64)
        form.addRow("隐藏层神经元：", self.hidden_spin)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.005)
        form.addRow("学习率：", self.lr_spin)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(100, 100000)
        self.epoch_spin.setValue(5000)
        form.addRow("训练轮数：", self.epoch_spin)
        self.log_spin = QSpinBox()
        self.log_spin.setRange(1, 1000)
        self.log_spin.setValue(100)
        form.addRow("日志频率：", self.log_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        form.addRow("随机种子：", self.seed_spin)
        param_group.setLayout(form)
        layout.addWidget(param_group)
        bl = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.eval_btn = QPushButton("运行模型评估(BPC/原创度)")
        self.eval_btn.clicked.connect(self.on_run_evaluate)
        bl.addWidget(self.start_btn)
        bl.addWidget(self.stop_btn)
        bl.addWidget(self.eval_btn)
        layout.addLayout(bl)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("训练日志...")
        layout.addWidget(self.log)
        self.setLayout(layout)

    @staticmethod
    def edit_distance_1(a, b):
        if abs(len(a) - len(b)) > 1:
            return False
        if len(a) == len(b):
            diff = sum(1 for x, y in zip(a, b) if x != y)
            return diff == 1
        if len(a) > len(b):
            a, b = b, a
        i = j = diff = 0
        while i < len(a) and j < len(b):
            if a[i] != b[j]:
                diff += 1
                j += 1
                if diff >1:
                    return False
            else:
                i +=1
                j +=1
        return True

    def compute_bpc(self, model, char_to_idx, names):
        total_loss = 0.0
        total_chars = 0
        for name in names:
            if len(name) <2:
                continue
            h_prev = np.zeros(model.hidden_size)
            inputs = [char_to_idx[c] for c in name[:-1]]
            targets = [char_to_idx[c] for c in name[1:]]
            xs, hs, ps = {}, {}, {}
            hs[-1] = h_prev.copy()
            loss = 0.0
            for t in range(len(inputs)):
                x = np.zeros(model.vocab_size)
                x[inputs[t]] =1.0
                xs[t]=x
                h = np.tanh(model.Wxh.T @ x + model.Whh.T @ hs[t-1] + model.bh)
                hs[t]=h
                y = model.Why.T @ h + model.by
                z = y - np.max(y)
                e = np.exp(z)
                p = e / np.sum(e)
                ps[t]=p
                loss += -np.log(max(p[targets[t]],1e-8))
            total_loss += loss
            total_chars += len(inputs)
        return (total_loss / total_chars) / np.log(2)

    def rule_hit_rate(self, model, char_to_idx, idx_to_char, n=100, temperature=0.8):
        vowels = set("aeiou")
        illegal_clusters = ["xk", "qz", "xz", "qk", "zx", "kq"]
        names = []
        for _ in range(n):
            seed = random.choice(list(char_to_idx.keys()))
            h = np.zeros(model.hidden_size)
            idx = char_to_idx[seed]
            res = seed
            for _ in range(6):
                x = np.zeros(model.vocab_size)
                x[idx]=1.0
                h = np.tanh(model.Wxh.T @ x + model.Whh.T @ h + model.bh)
                y = model.Why.T @ h + model.by
                y = y / temperature
                z = y - np.max(y)
                e = np.exp(z)
                p = e / np.sum(e)
                idx = np.random.choice(model.vocab_size, p=p)
                res += idx_to_char[idx]
            names.append(res)
        total = len(names)
        len_ok = sum(1 for x in names if 3 <= len(x) <=12)
        vowel_ok = sum(1 for x in names if any(c in vowels for c in x))
        illegal_ok = sum(1 for x in names if not any(cl in x for cl in illegal_clusters))
        return {
            "总数":total,
            "长度合规率(3~12)": len_ok/total,
            "元音存在率":vowel_ok/total,
            "无非法辅音簇率":illegal_ok/total
        }

    def originality_rate(self, model, char_to_idx, idx_to_char, train_set, n=100, temperature=0.8):
        generated = []
        for _ in range(n):
            seed = random.choice(list(char_to_idx.keys()))
            h = np.zeros(model.hidden_size)
            idx = char_to_idx[seed]
            res = seed
            for _ in range(6):
                x = np.zeros(model.vocab_size)
                x[idx]=1.0
                h = np.tanh(model.Wxh.T @ x + model.Whh.T @ h + model.bh)
                y = model.Why.T @ h + model.by
                y = y / temperature
                z = y - np.max(y)
                e = np.exp(z)
                p = e / np.sum(e)
                idx = np.random.choice(model.vocab_size, p=p)
                res += idx_to_char[idx]
            generated.append(res)
        exact = sum(1 for x in generated if x in train_set)
        variant = sum(1 for x in generated if x not in train_set and any(self.edit_distance_1(x,t) for t in train_set))
        return {
            "生成数":n,
            "完全记忆率": exact/n,
            "变体率(编辑距离1)": variant/n,
            "原创率(既非记忆也非变体)": (n-exact-variant)/n
        }

    def on_run_evaluate(self):
        data_path = self.file_input.text()
        if not os.path.exists(data_path):
            self.log_msg("❌ names.txt 文件不存在！")
            return
        round_dir, _ = self.resolve_round_dir()
        model_file = os.path.join(round_dir, "model.npz")
        if not os.path.exists(model_file):
            self.log_msg("❌ 当前Round目录下model.npz不存在，请先完成一轮训练！")
            return
        try:
            with open(data_path, encoding="utf-8") as f:
                all_names = [l.strip().lower() for l in f if l.strip()]
            all_names = [n for n in all_names if n.isalpha()]
            random.seed(42)
            random.shuffle(all_names)
            split = int(len(all_names)*0.8)
            train_names = all_names[:split]
            test_names = all_names[split:]
            chars = sorted(set("".join(all_names)))
            char_to_idx = {c:i for i,c in enumerate(chars)}
            idx_to_char = {i:c for i,c in char_to_idx.items()}
            from ecnm import CharRNN
            rnn = CharRNN(vocab_size=len(chars))
            rnn.load(model_file)
            rnn.char_to_idx = char_to_idx
            rnn.idx_to_char = idx_to_char

            self.log_msg("\n==========模型评估开始==========")
            train_bpc = self.compute_bpc(rnn, char_to_idx, train_names)
            test_bpc = self.compute_bpc(rnn, char_to_idx, test_names)
            self.log_msg(f"训练集BPC: {train_bpc:.4f}")
            self.log_msg(f"测试集BPC: {test_bpc:.4f}")
            self.log_msg(f"BPC差值: {test_bpc - train_bpc:+.4f}")
            if test_bpc - train_bpc > 0.3:
                self.log_msg("⚠ 测试BPC明显偏高，疑似过拟合")
            else:
                self.log_msg("✅ BPC泛化表现良好")

            hit = self.rule_hit_rate(rnn, char_to_idx, idx_to_char, n=100)
            self.log_msg("\n----规则命中率----")
            for k,v in hit.items():
                if isinstance(v,float):
                    self.log_msg(f"{k}: {v:.2%}")
                else:
                    self.log_msg(f"{k}: {v}")

            orig = self.originality_rate(rnn, char_to_idx, idx_to_char, set(train_names), n=100)
            self.log_msg("\n----原创性分析----")
            for k,v in orig.items():
                if isinstance(v,float):
                    self.log_msg(f"{k}: {v:.2%}")
                else:
                    self.log_msg(f"{k}: {v}")
            self.log_msg("==========评估结束==========\n")
        except Exception as e:
            self.log_msg(f"❌评估异常：{str(e)}")

    def start_training(self):
        if not os.path.exists(self.file_input.text()):
            self.log_msg("❌ 文件不存在")
            return
        round_dir, start_epoch = self.resolve_round_dir()
        if round_dir is None:
            self.log_msg("已取消训练")
            return
        self.log_msg("开始训练...\n")
        self.thread = TrainThread(
            data_file=self.file_input.text(),
            hidden=self.hidden_spin.value(),
            lr=self.lr_spin.value(),
            epochs=self.epoch_spin.value(),
            log_every=self.log_spin.value(),
            round_dir=round_dir,
            start_epoch=start_epoch,
            seed=self.seed_spin.value(),
        )
        self.thread.progress.connect(self.on_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.thread.start()
