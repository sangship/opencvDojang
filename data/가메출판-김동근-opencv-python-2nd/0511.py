# 0511.py
import cv2
import numpy as np

src = np.array([[2, 2, 4, 4],
                [2, 2, 4, 4],
                [4, 4, 4, 4],
                [4, 4, 4, 4]
              ], dtype=np.uint8)
#1
dst = cv2.equalizeHist(src)
print('dst =', dst)

#2
'''
ref: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
'''
##hist = cv2.calcHist(images = [src], channels = [0], mask = None,
##                    histSize = [256], ranges = [0, 256])
hist,bins = np.histogram(src.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf, 0) # cdf에서 0을 True 마스킹  
T = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
T = np.ma.filled(T, 0).astype('uint8') # 마스킹을 0으로 채우기 
dst2 = T[src] # dst2 == dst
#print('dst2 =', dst2)
