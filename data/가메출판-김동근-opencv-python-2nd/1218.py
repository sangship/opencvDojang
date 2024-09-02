#1218
'''
https://build-system.fman.io/pyqt5-tutorial
'''

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import   QMenuBar, QFileDialog, QAction, qApp
#import qimage2ndarray 

import numpy as np
import cv2
import sys

#1
#1-1
def toRGB(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    H,W,C = image.shape
    qimg = QtGui.QImage(image.data, W, H, W*C, QtGui.QImage.Format_RGB888)
    #qimg = qimage2ndarray.array2qimage(image)
     
    pixmap = QtGui.QPixmap.fromImage(qimg)
    label.setPixmap(pixmap)
    label.show()
    
    btn1.setEnabled(False) # btn1.setDisabled(True)
    btn2.setEnabled(True)
   
#1-2
def toGray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    H,W = gray.shape    
    qimg = QtGui.QImage(gray.data, W, H, W, QtGui.QImage.Format_Grayscale8)
##    qimg = qimage2ndarray.gray2qimage(gray)

    pixmap = QtGui.QPixmap.fromImage(qimg)
    label.setPixmap(pixmap)
    label.show()

    btn1.setEnabled(True)
    btn2.setEnabled(False)  
    
#2
#2-1
app = QtWidgets.QApplication([])
label= QtWidgets.QLabel()
#label.setScaledContents(True)
btn1 = QtWidgets.QPushButton("RGB")
btn2 = QtWidgets.QPushButton("GRAY")
menubar =  QMenuBar()

#2-2
layout  = QtWidgets.QVBoxLayout()
layout .addWidget(menubar)
layout .addWidget(label)
layout .addWidget(btn1)
layout .addWidget(btn2)

#2-3
window  = QtWidgets.QWidget()
window .setLayout(layout)
window .setWindowTitle('PyQt5: OpenCV Image')
window .show()

#3
cv_image = cv2.imread('./data/lena.jpg')
toRGB(cv_image) # initial display
btn1.clicked.connect(lambda: toRGB(cv_image))
btn2.clicked.connect(lambda: toGray(cv_image))

#4
def qimg2array(qimg):
    img_size = qimg.size()
    H, W, C = img_size.height(), img_size.width(), qimg.depth()//8
    #print(f'H={H}, W={W}, C={C}')

    n_bytes  = W * H * C   
    data = qimg.bits().asstring(n_bytes)
    arr = np.ndarray(shape = (H, W, C), buffer= data, dtype  = np.uint8)
    
    if qimg.isGrayscale():
        return arr[...,0] # GRAY
    return arr[...,:3]    # RGB

#5:
#5-1
def OnFileOpenDialog():
    global cv_image
    fname, _ = QFileDialog.getOpenFileName(window, 'Open file', './data',
                               "Images(*.png, *.jpg);;All files (*.*)")

    if fname: # fname !=''
        cv_image = cv2.imread(fname)
        toRGB(cv_image)

#5-2
def OnFileSaveDialog():
    fname, _ = QFileDialog.getSaveFileName(window, 'Save file', './data',
                                "PNG(*.png);; JPG(*.jpg);; Bmp(*.bmp)")
    if not fname:
        return
     
    pixmap = label.pixmap()   # QPixmap
    qimage = pixmap.toImage() # QImage
      
    #img = qimage2ndarray.rgb_view(qimage, byteorder='little') #BGR
    img = qimg2array(qimage)
    cv2.imwrite(fname, img)

#5-3
def OnExit():
    window .close()
    qApp.quit()
    
#6      
openFile = QAction(QIcon('./data/folder.png'),'Open')
openFile.setShortcut('Ctrl+O')
openFile.triggered.connect(OnFileOpenDialog)

saveFile = QAction(QIcon('./data/save.png'), 'Save')
saveFile.setShortcut('Ctrl+S')
saveFile.triggered.connect(OnFileSaveDialog)

exitFile = QAction(QIcon('./data/exit.png'), 'Exit')
exitFile.setShortcut('Ctrl+Q')
exitFile.triggered.connect(OnExit)

fileMenu = menubar.addMenu('&File')
fileMenu.addAction(openFile)
fileMenu.addAction(saveFile)
fileMenu.addAction(exitFile)

sys.exit(app.exec_())

