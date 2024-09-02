# 0205.py
import cv2
from   matplotlib import pyplot as plt

imageFile = './data/lena.jpg'
imgGray = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
# plt.imshow(imgGray, cmap = 'gray')
plt.imshow(imgGray)


plt.axis('off')
plt.savefig('./data/0205.png')
plt.show()
