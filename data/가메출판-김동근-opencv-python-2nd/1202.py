#1202py
import cv2
import numpy as np
from PIL import Image

#1
img = Image.open("./data/lena.jpg")
img_rgb = np.array(img) # RGB

#2
cv_bgr=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

#3
##r, g, b = cv2.split(img_rgb)
##cv_bgr= cv2.merge([b, g, r])

cv2.imshow('cv_bgr', cv_bgr)
cv2.waitKey()
cv2.destroyAllWindows()
