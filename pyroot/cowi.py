# cowi.py
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit,
    QCheckBox, QMessageBox, QTextEdit, QLabel, QComboBox,
    QListWidgetItem, QListWidget
)
from PyQt5.QtCore import Qt, QMimeData

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

class CoanPlotWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("损失对比绘图工具")
        self.setMinimumSize(900, 760)
        self.setAcceptDrops(True)

        self.items = []

        main_layout = QVBoxLayout()

        # ---------- 参数表单 ----------
        form = QFormLayout()

        self.fig_width = QDoubleSpinBox(); self.fig_width.setRange(2, 20); self.fig_width.setSingleStep(0.5); self.fig_width.setValue(8)
        form.addRow("图宽度(inch):", self.fig_width)

        self.fig_height = QDoubleSpinBox(); self.fig_height.setRange(2, 15); self.fig_height.setSingleStep(0.5); self.fig_height.setValue(5)
        form.addRow("图高度(inch):", self.fig_height)

        self.dpi_sp = QSpinBox(); self.dpi_sp.setRange(50, 400); self.dpi_sp.setValue(150)
        form.addRow("DPI:", self.dpi_sp)

        self.title_edit = QLineEdit("训练损失收敛对比")
        form.addRow("图表标题:", self.title_edit)

        self.xlabel_edit = QLineEdit("epoch")
        form.addRow("X轴标签:", self.xlabel_edit)

        self.ylabel_edit = QLineEdit("loss")
        form.addRow("Y轴标签:", self.ylabel_edit)

        self.title_fs = QSpinBox(); self.title_fs.setRange(6, 30); self.title_fs.setValue(14)
        form.addRow("标题字号:", self.title_fs)

        self.label_fs = QSpinBox(); self.label_fs.setRange(6, 24); self.label_fs.setValue(11)
        form.addRow("轴标签字号:", self.label_fs)

        self.line_w = QDoubleSpinBox(); self.line_w.setRange(0.5, 5); self.line_w.setSingleStep(0.5); self.line_w.setValue(1.5)
        form.addRow("线宽:", self.line_w)

        self.legend_loc = QComboBox()
        self.legend_loc.addItems(["best", "upper right", "upper left", "lower right", "lower left",
                                  "center right", "center left", "upper center", "lower center"])
        form.addRow("图例位置:", self.legend_loc)

        self.grid_chk = QCheckBox(); self.grid_chk.setChecked(True)
        form.addRow("显示网格:", self.grid_chk)

        self.legend_chk = QCheckBox(); self.legend_chk.setChecked(True)
        form.addRow("显示图例:", self.legend_chk)

        # 坐标轴范围（留空 = 自动）
        range_row = QHBoxLayout()
        self.xmin_edit = QLineEdit(); self.xmin_edit.setPlaceholderText("自动"); self.xmin_edit.setMaximumWidth(60)
        self.xmax_edit = QLineEdit(); self.xmax_edit.setPlaceholderText("自动"); self.xmax_edit.setMaximumWidth(60)
        self.ymin_edit = QLineEdit(); self.ymin_edit.setPlaceholderText("自动"); self.ymin_edit.setMaximumWidth(60)
        self.ymax_edit = QLineEdit(); self.ymax_edit.setPlaceholderText("自动"); self.ymax_edit.setMaximumWidth(60)
        range_row.addWidget(QLabel("X:")); range_row.addWidget(self.xmin_edit)
        range_row.addWidget(QLabel("~")); range_row.addWidget(self.xmax_edit)
        range_row.addSpacing(15)
        range_row.addWidget(QLabel("Y:")); range_row.addWidget(self.ymin_edit)
        range_row.addWidget(QLabel("~")); range_row.addWidget(self.ymax_edit)
        range_row.addStretch()
        form.addRow("坐标轴范围:", range_row)

        main_layout.addLayout(form)

        # ---------- 拖拽提示 ----------
        tip = QLabel("把 history.npz 文件直接拖进来，可一次拖多个")
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet("color:#2266bb; padding:4px;")
        main_layout.addWidget(tip)

        # ---------- 文件列表 ----------
        self.file_text = QTextEdit()
        self.file_text.setReadOnly(True)
        self.file_text.setPlaceholderText("拖拽进来的 npz 文件会显示在这里...")
        self.file_text.setMaximumHeight(100)
        main_layout.addWidget(QLabel("已选择文件："))
        main_layout.addWidget(self.file_text)

        # ====== 新增：标签修改交互 ======
        from PyQt5.QtWidgets import QListWidget

        self.label_list = QListWidget()
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("选中条目，在这里修改图例名称")
        self.apply_label_btn = QPushButton("更新图例标签")

        label_row = QHBoxLayout()
        label_row.addWidget(self.label_edit)
        label_row.addWidget(self.apply_label_btn)

        main_layout.addWidget(QLabel("图例标签列表（可选中修改）："))
        main_layout.addWidget(self.label_list)
        main_layout.addLayout(label_row)

        # 绑定事件
        self.label_list.currentRowChanged.connect(self.on_select_item)
        self.apply_label_btn.clicked.connect(self.on_apply_label)


        # ---------- 按钮 ----------
        btn_row = QHBoxLayout()
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.clear_list)
        self.plot_btn = QPushButton("执行绘图")
        self.plot_btn.clicked.connect(self.do_plot)
        btn_row.addStretch(); btn_row.addWidget(self.clear_btn); btn_row.addWidget(self.plot_btn); btn_row.addStretch()
        main_layout.addLayout(btn_row)

        self.setLayout(main_layout)

    # ---------- 拖拽 ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        added = 0
        for url in e.mimeData().urls():
            if not url.isLocalFile():
                continue
            p = url.toLocalFile()
            if os.path.isfile(p) and p.lower().endswith(".npz"):
                if any(it["path"] == p for it in self.items):
                    continue
                default_label = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
                self.items.append({"path": p, "label": default_label})
                added += 1

        self.refresh_files()
        QMessageBox.information(self, "完成", f"添加了 {added} 个 npz 文件")
        e.acceptProposedAction()

    def refresh_files(self):
        self.sync_label_list()
        if not self.items:
            self.file_text.setPlainText("")
            return
        lines = []
        for i, it in enumerate(self.items):
            lines.append(f"[{i+1}] label:{it['label']}  ->  {it['path']}")
        self.file_text.setPlainText("\n".join(lines))

    def sync_label_list(self):
        self.label_list.clear()
        for it in self.items:
            QListWidgetItem(it["label"], self.label_list)


    def clear_list(self):
        self.items.clear()
        self.refresh_files()

    def on_select_item(self, row):
        if 0 <= row < len(self.items):
            self.label_edit.setText(self.items[row]["label"])

    def on_apply_label(self):
        row = self.label_list.currentRow()
        if 0 <= row < len(self.items):
            new_name = self.label_edit.text().strip()
            if new_name:
                self.items[row]["label"] = new_name
                self.sync_label_list()
                self.refresh_files()

    # ---------- 工具：解析范围输入 ----------
    @staticmethod
    def _parse_range(text):
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None



    # ---------- 绘图 ----------
    def do_plot(self):
        if not self.items:
            QMessageBox.warning(self, "提示", "请先拖拽 history.npz 文件")
            return

        w = self.fig_width.value()
        h = self.fig_height.value()
        dpi = self.dpi_sp.value()
        title = self.title_edit.text()
        xlab = self.xlabel_edit.text()
        ylab = self.ylabel_edit.text()
        tfs = self.title_fs.value()
        lfs = self.label_fs.value()
        lw = self.line_w.value()
        loc = self.legend_loc.currentText()
        show_grid = self.grid_chk.isChecked()
        show_legend = self.legend_chk.isChecked()

        # 新建figure，只用于保存图片，不显示在窗口
        fig = Figure(figsize=(w, h), dpi=dpi)
        ax = fig.add_subplot(111)

        ok = 0
        for item in self.items:
            fp = item["path"]
            try:
                data = np.load(fp, allow_pickle=True)
                if "epochs" in data.files and "losses" in data.files:
                    xs = data["epochs"]
                    ys = data["losses"]
                elif "losses" in data.files:
                    ys = data["losses"]
                    xs = np.arange(len(ys))
                else:
                    QMessageBox.warning(self, "跳过", f"{os.path.basename(fp)}\n找不到 epochs/losses")
                    continue
                label_text = item["label"]
                ax.plot(xs, ys, label=label_text, linewidth=lw)
                ok += 1
            except Exception as ex:
                QMessageBox.warning(self, "读取失败", f"{os.path.basename(fp)}\n{ex}")

        if ok == 0:
            QMessageBox.warning(self, "无有效数据", "没有成功加载任何曲线")
            return

        ax.set_title(title, fontsize=tfs)
        ax.set_xlabel(xlab, fontsize=lfs)
        ax.set_ylabel(ylab, fontsize=lfs)
        if show_grid:
            ax.grid(alpha=0.3)
        if show_legend:
            ax.legend(loc=loc)

        xmin = self._parse_range(self.xmin_edit.text())
        xmax = self._parse_range(self.xmax_edit.text())
        ymin = self._parse_range(self.ymin_edit.text())
        ymax = self._parse_range(self.ymax_edit.text())
        if xmin is not None or xmax is not None:
            ax.set_xlim(xmin, xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin, ymax)

        ax.relim()
        ax.autoscale_view()

        out_path = "plot_output.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        QMessageBox.information(self, "绘图完成", f"图片已保存到：\n{out_path}")
