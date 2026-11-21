# mwl.py
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QComboBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from basethread import BaseTrainThread
from basewindow import BaseTrainWindow
from mtl import train_xor, predict_gate, GATES, X


class LogicThread(BaseTrainThread):
    def __init__(self, gate, hidden_sizes, lr, epochs, log_every,
                round_dir, start_epoch, seed=42):
        super().__init__()
        self.gate = gate
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.log_every = log_every
        self.round_dir = round_dir
        self.start_epoch = start_epoch
        self.seed = seed

    def run_body(self) -> str:
        net, losses = train_xor(
            gate=self.gate,
            hidden_sizes=self.hidden_sizes,
            lr=self.lr,
            epochs=self.epochs,
            log_every=self.log_every,
            round_dir=self.round_dir,
            start_epoch=self.start_epoch,
            seed=self.seed,
            callback=lambda e, l: self.progress.emit(e, l),
            stop_flag=lambda: not self._running,
        )
        pred, prob = predict_gate(net)
        y_true = GATES[self.gate].astype(int)
        lines = [f"训练完成！最终 loss: {losses[-1]:.4f}\n"]
        lines.append(f"逻辑门: {self.gate.upper()}")
        lines.append(f"隐藏层结构: {self.hidden_sizes}\n")
        lines.append(f"{'输入':<12}{'真实':<8}{'预测':<8}{'概率':<10}")
        lines.append("-" * 40)
        for i in range(len(X)):
            inp = f"[{int(X[i][0])}, {int(X[i][1])}]"
            lines.append(f"{inp:<12}{int(y_true[i][0]):<8}{pred[i][0]:<8}{prob[i][0]:.4f}")
        return "\n".join(lines)


class LogicWindow(BaseTrainWindow):
    def __init__(self):
        super().__init__(task_key="blom")
        self.setWindowTitle("逻辑门实验")
        self.setMinimumSize(560, 580)
        layout = QVBoxLayout()

        gate_group = QGroupBox("逻辑门")
        gl = QHBoxLayout()
        gl.addWidget(QLabel("选择："))
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["AND", "OR", "XOR"])
        self.gate_combo.setCurrentText("XOR")
        gl.addWidget(self.gate_combo)
        gate_group.setLayout(gl)
        layout.addWidget(gate_group)

        param_group = QGroupBox("训练参数")
        form = QFormLayout()

        # 多层隐藏层输入：逗号分隔，如 "4" 或 "4,4"
        self.hidden_edit = QLineEdit("4")
        self.hidden_edit.setPlaceholderText('逗号分隔，如 "4" 或 "4,4"；留空=无隐藏层')
        form.addRow("隐藏层结构：", self.hidden_edit)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 2.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.05)
        self.lr_spin.setValue(0.5)
        form.addRow("学习率：", self.lr_spin)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(100, 100000)
        self.epoch_spin.setValue(5000)
        self.epoch_spin.setSingleStep(500)
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
        bl.addWidget(self.start_btn)
        bl.addWidget(self.stop_btn)
        layout.addLayout(bl)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("训练日志...")
        layout.addWidget(self.log)
        self.setLayout(layout)

    def parse_hidden_sizes(self):
        """解析输入框，返回隐藏层列表"""
        text = self.hidden_edit.text().strip()
        if not text:
            return []
        try:
            sizes = [int(x.strip()) for x in text.split(",") if x.strip()]
            return sizes
        except ValueError:
            return None

    def start_training(self):
        gate = self.gate_combo.currentText().lower()
        hidden_sizes = self.parse_hidden_sizes()
        if hidden_sizes is None:
            self.log_msg("❌ 隐藏层格式错误，请用逗号分隔数字，如 4 或 4,4")
            return
        if gate == "xor" and len(hidden_sizes) == 0:
            self.log_msg("⚠ XOR 线性不可分，无隐藏层通常学不会")

        round_dir, start_epoch = self.resolve_round_dir()
        if round_dir is None:
            self.log_msg("已取消训练")
            return
        self.log_msg(
            f"开始训练 {gate.upper()} | 隐藏层={hidden_sizes} | "
            f"lr={self.lr_spin.value()} | epochs={self.epoch_spin.value()}\n"
        )
        self.thread = LogicThread(
            gate=gate,
            hidden_sizes=hidden_sizes,
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
