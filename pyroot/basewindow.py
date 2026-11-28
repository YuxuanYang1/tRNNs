# basewindow.py
from PyQt5.QtWidgets import QWidget, QMessageBox, QInputDialog
from PyQt5.QtCore import Qt
import os
import re
import numpy as np
from common import ensure_task_dir, find_unfinished, new_round_dir, load_history_npz


class BaseTrainWindow(QWidget):
    def __init__(self, task_key: str):
        super().__init__()
        self.task_key = task_key
        self.thread = None

    def log_msg(self, m: str):
        self.log.append(m)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def resolve_round_dir(self):
        task_dir = ensure_task_dir(self.task_key)
        unfinished = find_unfinished(task_dir)
        if unfinished:
            round_dir = unfinished[-1][1]
            history = os.path.join(round_dir, "history.npz")
            start_epoch, _ = load_history_npz(history)
            self.log_msg(f"📦 检测到未完成训练：{os.path.basename(round_dir)}，自动接续")
            return round_dir, start_epoch

        existing = sorted([
            d for d in os.listdir(task_dir)
            if re.fullmatch(r"Round\d+", d) and os.path.exists(os.path.join(task_dir, d, "DONE"))
        ])
        if not existing:
            round_dir = new_round_dir(task_dir)
            self.log_msg(f"📁 新建训练目录：{os.path.basename(round_dir)}")
            return round_dir, 0

        msg = QMessageBox(self)
        msg.setWindowTitle("选择训练方式")
        msg.setText("检测到已完成的训练记录，要接续还是重开？")
        btn_resume = msg.addButton("接续", QMessageBox.AcceptRole)
        btn_new = msg.addButton("重开", QMessageBox.RejectRole)
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked == btn_resume:
            chosen, ok = QInputDialog.getItem(
                self, "选择 Round", "接续哪个：", existing, 0, False
            )
            if not ok or not chosen:
                return None, 0
            round_dir = os.path.join(task_dir, chosen)
            history = os.path.join(round_dir, "history.npz")
            start_epoch, _ = load_history_npz(history)
            self.log_msg(f"📦 接续训练：{chosen}（从 Epoch {start_epoch} 开始）")
            return round_dir, start_epoch
        elif clicked == btn_new:
            round_dir = new_round_dir(task_dir)
            self.log_msg(f"📁 新建训练目录：{os.path.basename(round_dir)}")
            return round_dir, 0
        else:
            self.log_msg("已取消训练")
            return None, 0

    def start_training(self):
        raise NotImplementedError

    def stop_training(self):
        if self.thread:
            self.thread.stop()
            self.log_msg("⏹ 正在停止...")

    def on_progress(self, e, l):
        if np.isnan(l):
            self.log_msg(f"📦 已从 Epoch {e} 恢复")
        else:
            self.log_msg(f"Epoch {e:6d} | Loss: {l:.4f}")

    def on_finished(self, msg):
        self.log_msg(f"\n✅ {msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_error(self, msg):
        self.log_msg(f"\n❌ {msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
