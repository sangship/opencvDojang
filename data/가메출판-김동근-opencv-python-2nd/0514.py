# 0514.py
import cv2
import numpy as np

src = np.array([[2, 2, 2, 2, 0,   0,   0,   0],
                [2, 1, 1, 2, 0,   0,   0,   0],
                [2, 1, 1, 2, 0,   0,   0,   0],
                [2, 2, 2, 2, 0,   0,   0,   0],
                [0, 0, 0, 0, 255, 255, 255, 255],
                [0, 0, 0, 0, 255, 1,   1,   255],
                [0, 0, 0, 0, 255, 1,   1,   255],
                [0, 0, 0, 0, 255, 255, 255, 255]], dtype=np.uint8)

#1
clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(1,1))
dst = clahe.apply(src)
print("dst=\n", dst)

#2
clahe2 = cv2.createCLAHE(clipLimit=40, tileGridSize=(2,2))
dst2 = clahe2.apply(src)
print("dst2=\n", dst2)
