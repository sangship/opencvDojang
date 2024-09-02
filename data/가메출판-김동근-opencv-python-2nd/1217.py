#1217
'''
ref1: https://github.com/hmeine/qimage2ndarray
ref2: https://stackoverflow.com/questions/37552924/convert-qpixmap-to-numpy
'''
from PyQt5 import QtWidgets, QtGui
import numpy as np
import cv2
import qimage2ndarray # pip install qimage2ndarray

#1: 
def qimg2array(qimg): # QImage-> OpenCV's numpy array
#1-1
    img_size = qimg.size()
    H, W, C = img_size.height(), img_size.width(), qimg.depth()//8
    #print(f'H={H}, W={W}, C={C}')

    n_bytes  = W * H * C
    #print('n_bytes=', n_bytes)

#1-2    
    data = qimg.bits().asstring(n_bytes) # qtimg.constBits()  
    arr = np.ndarray(shape = (H, W, C), buffer= data, dtype  = np.uint8)           
##    arr = np.frombuffer(data, dtype=np.uint8).reshape(
##                img_size.height(), img_size.width(), -1)
#1-3    
    if qimg.isGrayscale():
        return arr[...,0] # GRAY
    return arr[...,:3]    # RGB

#2
app = QtWidgets.QApplication([])
label = QtWidgets.QLabel()
pixmap = QtGui.QPixmap('./data/lena.jpg') # QPixmap
label.setPixmap(pixmap)
label.setScaledContents(True)
label.show()
label.setWindowTitle("PyQt5: image")

#3
#3-1
qimg = pixmap.toImage() # QImage

#3-2
cv_img = qimage2ndarray.rgb_view(qimg)
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)#cv_img = cv_img[..., ::-1]
cv2.imshow('cv_img', cv_img)
print('cv_img.shape=', cv_img.shape )

#3-3
cv_img2 = qimg2array(qimg)   
cv2.imshow('cv_img2', cv_img2)
print('cv_img2.shape=', cv_img2.shape )

#3-4
qimg = qimg.convertToFormat(QtGui.QImage.Format_Grayscale8)
cv_img3 = qimg2array(qimg)
cv2.imshow('cv_img3', cv_img3)
print('cv_img3.shape=', cv_img3.shape )

#4
cv2.waitKey()
cv2.destroyAllWindows()
app.exec_()
