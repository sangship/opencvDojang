#1216
'''
ref: https://github.com/hmeine/qimage2ndarray
'''
from PyQt5 import QtWidgets, QtGui
import cv2

#1
img = cv2.imread('./data/lena.jpg') #BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB
H,W,C = img.shape
print(f"W={W}, H={H}, C={C}")

#2
app = QtWidgets.QApplication([])
label = QtWidgets.QLabel()

#3: OpenCV's numpy array -> QImage
#3-1
qimg = QtGui.QImage(img.data, W, H, W*C, QtGui.QImage.Format_RGB888)

#3-2
##qimg = qimage2ndarray.array2qimage(img) # 32bit WImage

#3-3
##qimg = qimg.convertToFormat(QtGui.QImage.Format_Grayscale8)
##qimg.save('./data/1216.png')

#4
pixmap = QtGui.QPixmap.fromImage(qimg)
w, h = pixmap.width(), pixmap.height()
c = pixmap.depth()//8
print(f"w={w}, h={h}, c={c}")

#5
label.setPixmap(pixmap)
label.setScaledContents(True)
label.setWindowTitle("PyQt5: image")
label.show()
app.exec_()
