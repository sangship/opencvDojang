#1215
from PyQt5 import QtWidgets, QtGui
from PIL import Image, ImageQt

#1
app = QtWidgets.QApplication([])
label = QtWidgets.QLabel()

#2
pixmap = QtGui.QPixmap('./data/lena.jpg')

#3: PIL
##img = Image.open("./data/lena.png")
##qimage = ImageQt.ImageQt(img)
##pixmap = QtGui.QPixmap.fromImage(qimage)

#4
pixmap.save('./data/1215.png')
W, H = pixmap.width(), pixmap.height()
C = pixmap.depth()//8
print(f"W={W}, H={H}, C={C}")

#5
label.setPixmap(pixmap)
label.setScaledContents(True)
label.setWindowTitle("PyQt5:image")
label.show()
app.exec_() #loop
