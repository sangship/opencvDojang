#1220
'''
ref: https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv

'''
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui  import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import cv2
import sys
 
#1
class VideoWorker(QThread):
#1-1
    send_qimage = pyqtSignal(QImage)
    
#1-2    
    def __init__(self, parent, source=0):
        super(VideoWorker, self).__init__(parent)
        self.parent = parent
        self.source = source
#1-3  
    def run(self):
        cap = cv2.VideoCapture(self.source)
        ret, frame = cap.read()
        if ret:
            self.stopped = False
        else:
            self.stopped = True
                    
        H, W, C = frame.shape
        bytesPerLine = W*C                     
        while not self.stopped:
            ret, frame = cap.read()            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(frame.data.tobytes(), W, H, bytesPerLine,
                                    QtGui.QImage.Format_RGB888)
                self.send_qimage.emit(qimg)
            else:
                self.stopped = True
        cap.release()
        #print("stop camera!")
        
#2             
class VideoWindow(QtWidgets.QWidget):
#2-1
    def __init__(self, source=0):
        super().__init__()
        self.title ="OpenCV VideoCam"
        self.mode = "RGB"
        self.source = source
        self.initUI()
#2-2
    @pyqtSlot(QImage)
    def displatImage(self, image):
        if self.mode =="GRAY":
            image = image.convertToFormat(QImage.Format_Grayscale8)
            
        self.image = image
        self.label.setPixmap(QPixmap.fromImage(image))       
#2-3
    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(800, 600)
        
        self.label = QtWidgets.QLabel()
        self.label.resize(640, 480)
        self.label.setScaledContents(True)


        self.btn1 = QtWidgets.QPushButton("RGB")
        self.btn2 = QtWidgets.QPushButton("GRAY")
        self.btn2.setEnabled(False) 

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.label)
        layout.addWidget(self.btn1)
        layout.addWidget(self.btn2)
        self.setLayout(layout)

        self.btn1.clicked.connect(self.toRGB)
        self.btn2.clicked.connect(self.toGray)
        
        self.thread  = VideoWorker(self, self.source)
        self.thread .send_qimage.connect(self.displatImage)
        self.thread .start()
#2-4        
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
            self.stopped = True
            self.close()
#3            
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
