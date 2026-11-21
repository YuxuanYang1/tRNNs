# basethread.py
from PyQt5.QtCore import QThread, pyqtSignal


class BaseTrainThread(QThread):
    progress = pyqtSignal(int, float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            msg = self.run_body()
            self.finished.emit(msg)
        except Exception as e:
            self.error.emit(str(e))

    def run_body(self) -> str:
        raise NotImplementedError("必须实现run_body")
