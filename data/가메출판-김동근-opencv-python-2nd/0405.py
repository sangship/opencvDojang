# 0405.py
import cv2
##import numpy as np

img = cv2.imread('./data/lena.jpg')

##for y in range(100, 400):
##    for x in range(200, 300):
##        img[y, x, 0] = 255      
        
img[100:400, 200:300, 0] = 255 
img[100:400, 300:400, 1] = 255
img[100:400, 400:500, 2] = 255

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
