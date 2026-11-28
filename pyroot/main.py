# main.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from mwe import NamesWindow
from mwp import PoemsWindow
from mwl import LogicWindow
from cowi import CoanPlotWindow
from infos import rebuild_info


class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("神经网络研究项目")
        self.setMinimumSize(460, 440)
        self.child_windows = {}
        layout = QVBoxLayout()
        layout.setSpacing(18)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("神经网络实验平台")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        layout.addWidget(title)
        subtitle = QLabel("选择一个实验进入")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: gray;")
        layout.addWidget(subtitle)

        btn_names = QPushButton("英文名字生成")
        btn_names.setMinimumHeight(48)
        btn_names.clicked.connect(self.open_names)
        layout.addWidget(btn_names)

        btn_poems = QPushButton("古诗生成")
        btn_poems.setMinimumHeight(48)
        btn_poems.clicked.connect(self.open_poems)
        layout.addWidget(btn_poems)

        btn_logic = QPushButton("逻辑门实验")
        btn_logic.setMinimumHeight(48)
        btn_logic.clicked.connect(self.open_logic)
        layout.addWidget(btn_logic)

        btn_coan = QPushButton("损失对比绘图工具")
        btn_coan.setMinimumHeight(42)
        btn_coan.clicked.connect(self.open_coan_window)
        layout.addWidget(btn_coan)

        layout.addStretch()
        self.setLayout(layout)

    def _open(self, key, win_class):
        if key in self.child_windows and self.child_windows[key].isVisible():
            self.child_windows[key].activateWindow()
            self.child_windows[key].raise_()
            return
        w = win_class()
        self.child_windows[key] = w
        w.show()

    def open_names(self):
        self._open("names", NamesWindow)
    def open_poems(self):
        self._open("poems", PoemsWindow)
    def open_logic(self):
        self._open("logic", LogicWindow)
    def open_coan_window(self):
        self._open("coan", CoanPlotWindow)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        rebuild_info()
    except Exception as e:
        QMessageBox.warning(None, "info.txt 重建失败", str(e))
    launcher = Launcher()
    launcher.show()
    sys.exit(app.exec())
