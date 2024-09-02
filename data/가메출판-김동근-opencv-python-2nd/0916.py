# 0916.py
import cv2
from   matplotlib import pyplot as plt

src = cv2.imread('./data/people1.png')

#1: HOG in color image
hog1 = cv2.HOGDescriptor()
des1 = hog1.compute(src)
print("HOG feature size = ",  hog1.getDescriptorSize())
print('des1.shape=', des1.shape)
##print('des1=', des1)


#2: HOG in color image
hog2 = cv2.HOGDescriptor(_winSize=(64, 128),
                         _blockSize=(16,16),
                         _blockStride=(8,8),
                         _cellSize=(8,8),
                         _nbins=9,
                         _derivAperture=1,
                         _winSigma= -1,
                         _histogramNormType=0,
                         _L2HysThreshold=0.2,
                         _gammaCorrection=True,
                         _nlevels=64,
                         _signedGradient=False)

des2 = hog2.compute(src)
print('des2.shape=', des2.shape)
##print('des2=', des2)

#3: 
hog3 = cv2.HOGDescriptor(_winSize=(64, 128),
                         _blockSize=(16,16),
                         _blockStride=(8,8),
                         _cellSize=(8,8),
                         _nbins=9)   # _gammaCorrection=False
des3 = hog3.compute(src)
print('des3.shape=', des3.shape)
##print('des3=', des3)

#4 HOG in grayscale image
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
des4 = hog3.compute(gray)
print('des4.shape=', des4.shape)
##print('des4=', des4)

#5
plt.title('HOGDescriptor')
plt.plot(des1[::36], color='b',linewidth=4,label='des1')
plt.plot(des2[::36], color='g',linewidth=4,label='des2')
plt.plot(des3[::36], color='r',linewidth=2,label='des3')
plt.plot(des4[::36], color='y',linewidth=1,label='des4')
plt.legend(loc='best')
plt.show()
