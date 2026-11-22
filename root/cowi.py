# cowi.py
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit,
    QCheckBox, QMessageBox, QTextEdit, QLabel
)
from PyQt5.QtCore import Qt, QMimeData
# matplotlib嵌入qt
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
except ImportError:
    FigureCanvas = None
    NavigationToolbar = None
    Figure = None


class CoanPlotWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("损失对比绘图工具")
        self.setMinimumSize(900, 720)
        # 开启拖拽支持
        self.setAcceptDrops(True)

        self.npz_file_list = []
        self.figure = None
        self.canvas = None
        self.toolbar = None

        main_layout = QVBoxLayout()
        # 参数表单
        form_group = QFormLayout()
        self.fig_width_sp = QDoubleSpinBox()
        self.fig_width_sp.setRange(2.0, 20.0)
        self.fig_width_sp.setSingleStep(0.5)
        self.fig_width_sp.setValue(8.0)
        form_group.addRow("图宽度(inch):", self.fig_width_sp)

        self.fig_height_sp = QDoubleSpinBox()
        self.fig_height_sp.setRange(2.0, 15.0)
        self.fig_height_sp.setSingleStep(0.5)
        self.fig_height_sp.setValue(5.0)
        form_group.addRow("图高度(inch):", self.fig_height_sp)

        self.dpi_sp = QSpinBox()
        self.dpi_sp.setRange(50, 400)
        self.dpi_sp.setValue(150)
        form_group.addRow("DPI:", self.dpi_sp)

        self.title_edit = QLineEdit("训练损失收敛对比")
        form_group.addRow("图表标题:", self.title_edit)

        self.xlabel_edit = QLineEdit("epoch")
        form_group.addRow("X轴标签:", self.xlabel_edit)

        self.ylabel_edit = QLineEdit("loss")
        form_group.addRow("Y轴标签:", self.ylabel_edit)

        self.grid_check = QCheckBox()
        self.grid_check.setChecked(True)
        form_group.addRow("显示网格:", self.grid_check)

        self.legend_check = QCheckBox()
        self.legend_check.setChecked(True)
        form_group.addRow("显示图例:", self.legend_check)

        main_layout.addLayout(form_group)

        # 拖拽提示标签
        tip_label = QLabel("💡 请直接把 history.npz 文件拖拽到此窗口内，可以一次拖入多个")
        tip_label.setAlignment(Qt.AlignCenter)
        tip_label.setStyleSheet("color:#2266bb; padding:6px;")
        main_layout.addWidget(tip_label)

        # -------- 已选文件显示框 --------
        self.selected_files_text = QTextEdit()
        self.selected_files_text.setReadOnly(True)
        self.selected_files_text.setPlaceholderText("拖拽进来的npz文件列表会显示在这里...")
        self.selected_files_text.setMaximumHeight(120)
        main_layout.addWidget(QLabel("已选择文件列表："))
        main_layout.addWidget(self.selected_files_text)

        # 操作按钮行（移除文件对话框按钮，只保留清空、绘图）
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("清空选择列表")
        self.clear_btn.clicked.connect(self.clear_selection)
        self.plot_btn = QPushButton("🔨执行绘图")
        self.plot_btn.clicked.connect(self.do_plot)
        btn_layout.addStretch()
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.plot_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # matplotlib画布占位
        self.canvas_container = QVBoxLayout()
        main_layout.addLayout(self.canvas_container, stretch=10)

        self.setLayout(main_layout)

    def update_file_display(self):
        """刷新UI显示选中的文件"""
        if not self.npz_file_list:
            self.selected_files_text.setPlainText("")
            return
        lines = [f"[{i+1}] {fp}" for i, fp in enumerate(self.npz_file_list)]
        self.selected_files_text.setPlainText("\n".join(lines))

    def clear_selection(self):
        self.npz_file_list.clear()
        self.update_file_display()

    # ==========拖拽事件处理==========
    def dragEnterEvent(self, event):
        mime: QMimeData = event.mimeData()
        if mime.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime: QMimeData = event.mimeData()
        urls = mime.urls()
        added = 0
        for url in urls:
            if not url.isLocalFile():
                continue
            file_path = url.toLocalFile()
            # 只接受npz后缀，并且文件存在
            if os.path.isfile(file_path) and file_path.lower().endswith(".npz"):
                if file_path not in self.npz_file_list:
                    self.npz_file_list.append(file_path)
                    added +=1
        self.update_file_display()
        QMessageBox.information(self, "拖拽完成", f"成功添加 {added} 个npz文件")
        event.acceptProposedAction()

    def do_plot(self):
        if FigureCanvas is None:
            QMessageBox.critical(self, "缺少依赖", "未安装matplotlib\npip install matplotlib")
            return
        if not self.npz_file_list:
            QMessageBox.warning(self, "提示", "请拖拽history.npz文件到本窗口")
            return
        # 清理旧画布
        if self.canvas is not None:
            self.canvas_container.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.deleteLater()
        # 读取参数
        w = self.fig_width_sp.value()
        h = self.fig_height_sp.value()
        dpi = self.dpi_sp.value()
        title = self.title_edit.text()
        xlab = self.xlabel_edit.text()
        ylab = self.ylabel_edit.text()
        show_grid = self.grid_check.isChecked()
        show_legend = self.legend_check.isChecked()

        # 创建画布
        self.figure = Figure(figsize=(w, h), dpi=dpi)
        ax = self.figure.add_subplot(111)
        # 中文字体设置
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        ok_count = 0
        for fp in self.npz_file_list:
            try:
                data = np.load(fp, allow_pickle=True)
                epochs_arr = data["epochs"]
                losses_arr = data["losses"]
                label_text = os.path.basename(os.path.dirname(fp))
                ax.plot(epochs_arr, losses_arr, label=label_text)
                ok_count += 1
            except Exception as e:
                QMessageBox.warning(self, "读取失败", f"{os.path.basename(fp)} 读取异常:\n{str(e)}")

        if ok_count == 0:
            QMessageBox.critical(self, "错误", "没有可以绘制的有效数据")
            self.figure = None
            return

        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if show_grid:
            ax.grid(alpha=0.3)
        if show_legend:
            ax.legend()

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas_container.addWidget(self.toolbar)
        self.canvas_container.addWidget(self.canvas)
        self.canvas.draw()
