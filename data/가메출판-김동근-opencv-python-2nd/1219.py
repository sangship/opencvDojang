#1219
'''
https://gist.github.com/bsdnoobz/8464000
'''

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui     import QImage, QPixmap
from PyQt5.QtCore    import QTimer, Qt
import cv2
import sys
#import qimage2ndarray
            
class VideoWindow(QtWidgets.QWidget):
#1 
    def __init__(self, source=0):
        super().__init__()
        self.title ="OpenCV VideoCam: QTimer"
        self.mode = "RGB"
        self.cap = cv2.VideoCapture(source)
        self.initUI()
#2         
    def displayImage(self):
        ret, frame = self.cap.read()       
        if ret:
            H,W,C = frame.shape

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(image.data.tobytes(), W, H, W*C, QtGui.QImage.Format_RGB888)
            #image = qimage2ndarray.array2qimage(image)
            
            if self.mode =="GRAY":
                qimg = qimg.convertToFormat(QImage.Format_Grayscale8)
            
            self.label.setPixmap(QPixmap.fromImage(qimg))    
#3
    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(800, 600)
        
        self.label = QtWidgets.QLabel()
        self.label.resize(640, 480)
        self.label.setScaledContents(True)

        self.btn1 = QtWidgets.QPushButton("RGB")
        self.btn2 = QtWidgets.QPushButton("GRAY")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn1)
        layout.addWidget(self.btn2)
        self.setLayout(layout)
        
        self.btn1.clicked.connect(self.toRGB)
        self.btn2.clicked.connect(self.toGray)
        
        self.timer = QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.displayImage)
        self.timer.start()

#4        
    def toGray(self):
        self.mode = "GRAY"
        self.btn1.setEnabled(True)
        self.btn2.setEnabled(False)          

    def toRGB(self):
        self.mode = "RGB"
        self.btn1.setEnabled(False)
        self.btn2.setEnabled(True)

    def closeEvent(self, event):
            close = QtWidgets.QMessageBox.question(self,
                          "Msg", "Are you sure?",
                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if close == QtWidgets.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
#5            
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
