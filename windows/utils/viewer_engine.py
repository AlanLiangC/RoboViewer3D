import pyqtgraph.opengl as gl
from PyQt5.QtCore import Qt

class AL_viewer(gl.GLViewWidget):
    
    def __init__(self, parent=None, devicePixelRatio=None, rotationMethod='euler'):
        super().__init__(parent, devicePixelRatio, rotationMethod)

        self.noRepeatKeys = [Qt.Key.Key_W, Qt.Key.Key_S, Qt.Key.Key_A, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E,
            Qt.Key.Key_Right, Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown]
        
        self.speed = 1
        
    def evalKeyState(self):
        vel_speed = 10 * self.speed 
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == Qt.Key.Key_Right:
                    self.orbit(azim=-self.speed, elev=0)
                elif key == Qt.Key.Key_Left:
                    self.orbit(azim=self.speed, elev=0)
                elif key == Qt.Key.Key_Up:
                    self.orbit(azim=0, elev=-self.speed)
                elif key == Qt.Key.Key_Down:
                    self.orbit(azim=0, elev=self.speed)
                elif key == Qt.Key.Key_A:
                    self.pan(vel_speed * self.speed, 0, 0, 'view-upright')
                elif key == Qt.Key.Key_D:
                    self.pan(-vel_speed, 0, 0, 'view-upright')
                elif key == Qt.Key.Key_W:
                    self.pan(0, vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_S:
                    self.pan(0, -vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_Q:
                    self.pan(0, 0, vel_speed, 'view-upright')
                elif key == Qt.Key.Key_E:
                    self.pan(0, 0, -vel_speed, 'view-upright')
                elif key == Qt.Key.Key_PageUp:
                    pass
                elif key == Qt.Key.Key_PageDown:
                    pass
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()