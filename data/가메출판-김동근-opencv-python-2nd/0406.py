# 0406.py
import cv2
import numpy as np

src = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)
dst = np.zeros(src.shape, dtype=src.dtype)

N = 4 # 8, 32, 64
height, width = src.shape
##height, width, channel = src.shape

h = height // N
w = width  // N
for i in range(N):
    for j in range(N):
        y = i*h
        x = j*w       
        roi = src[y:y+h, x:x+w]
        dst[y:y+h, x:x+w] = cv2.mean(roi)[0]
##        dst[y:y+h, x:x+w] = cv2.mean(roi)[0:3]
        
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
