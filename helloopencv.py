

import cv2
import sys

# print(cv2.__version__)



print('Hello OpenCV', cv2.__version__)

img_gray = cv2.imread('data/lenna.bmp', cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.imread('data/lenna.bmp')



if img_gray is None or img_bgr is None:
    print('Image load failed!')
    sys.exit()

cv2.namedWindow('image_gray')
cv2.namedWindow('image_bgr')


cv2.imshow('image_gray', img_gray)
cv2.imshow('image_bgr', img_bgr)


cv2.waitKey()
cv2.destroyAllWindows()

