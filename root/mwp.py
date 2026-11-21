# mwp.py
import os
import json
import re
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QSpinBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QFileDialog
)
from basethread import BaseTrainThread
from basewindow import BaseTrainWindow
from mte import train_rnn

class PoemsThread(BaseTrainThread):
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
            lines = [l.strip() for l in f if l.strip()]
        rnn, losses = train_rnn(
            lines,
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
        samples = []
        for seed in "床白春锄明黄夜花红孤千":
            if seed in rnn.char_to_idx:
                samples.append(f"{seed} → {rnn.sample(seed, length=10, temperature=0.8)}")
        return f"训练完成！最终 loss: {losses[-1]:.4f}\n\n" + "\n".join(samples)

class PoemsWindow(BaseTrainWindow):
    def __init__(self):
        super().__init__(task_key="pmm")
        self.setWindowTitle("古诗生成")
        self.setMinimumSize(560, 520)
        layout = QVBoxLayout()
        file_group = QGroupBox("数据文件")
        fl = QHBoxLayout()
        fl.addWidget(QLabel("poems.txt："))
        self.file_input = QLineEdit("poems.txt")
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
        self.json_import_btn = QPushButton("从唐诗JSON生成poems.txt")
        self.json_import_btn.clicked.connect(self.on_import_tang_json)
        bl.addWidget(self.start_btn)
        bl.addWidget(self.stop_btn)
        bl.addWidget(self.json_import_btn)
        layout.addLayout(bl)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("训练日志...")
        layout.addWidget(self.log)
        self.setLayout(layout)

    def on_import_tang_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择唐诗JSON文件", ".", "JSON(*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.log_msg(f"读入 {len(data)} 首诗")
            lines_out = []
            for poem in data:
                paras = poem.get("paragraphs") or poem.get("content") or []
                for para in paras:
                    parts = re.split(r"[，。！？；、]", para)
                    for p in parts:
                        p = p.strip()
                        if not p:
                            continue
                        if len(p) in (5,7):
                            lines_out.append(p + "<E>")
            lines_out = list(set(lines_out))
            out_path = "poems.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines_out))
            self.log_msg(f"✅ 已生成 {out_path}，共 {len(lines_out)} 句")
        except Exception as e:
            self.log_msg(f"❌ JSON处理失败: {str(e)}")

    def start_training(self):
        if not os.path.exists(self.file_input.text()):
            self.log_msg("❌ 文件不存在")
            return
        round_dir, start_epoch = self.resolve_round_dir()
        if round_dir is None:
            self.log_msg("已取消训练")
            return
        self.log_msg("开始训练...\n")
        self.thread = PoemsThread(
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
