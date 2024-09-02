import cv2
import sys
from matplotlib import pyplot as plt


fileName = 'cat.jpg'

imgGray = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

print(imgGray.shape)

plt.axis('off')
plt.imshow(imgGray, cmap='gray', interpolation = 'bicubic')

plt.show()


