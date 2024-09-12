import sys
sys.path.append('/home/alan/AlanLiang/Projects/3D_Perception/AlanLiang/Projects/RoboViewer3D')
from windows import RoboMainWindow

from PyQt5 import QtWidgets

def main():

    app = QtWidgets.QApplication([])
    window = RoboMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':

    main()