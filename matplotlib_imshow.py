
#opencv패키지 특성과 matplotlib 패키지 특성 차이



import cv2
import sys
from matplotlib import pyplot as plt

filename = 'cat.jpg'

img = cv2.imread(filename)

if img is None:
    sys.exit("image load is failed.")
    
#채널 순서, 컬러 스페이스를 바꿔주는 함수

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(imgRGB)
plt.axis('off')
plt.show()

