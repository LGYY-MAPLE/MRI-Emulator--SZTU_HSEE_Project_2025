import sys
from PyQt5.QtWidgets import QApplication
from MRI_core import MRISimulatorModel
from MRI_UI import MRISimulatorWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 实例化逻辑模型
    model = MRISimulatorModel()
    # 实例化界面，并注入模型
    window = MRISimulatorWindow(model)
    window.show()
    sys.exit(app.exec_())