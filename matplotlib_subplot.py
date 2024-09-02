
import cv2
from matplotlib import pyplot as plt
import sys




imgBGR1 = cv2.imread('lena.jpg')
imgBGR2 = cv2.imread('orange.jpg')
imgBGR3 = cv2.imread('apple.jpg')
imgBGR4 = cv2.imread('cat.jpg')


if imgBGR1 is None or imgBGR2 is None \
    or imgBGR3 is None or imgBGR4 is None:
        
        sys.exit("image load is failed.")
        
        
imgRGB1 = cv2.cvtColor(imgBGR1, cv2.COLOR_BGR2RGB)
imgRGB2 = cv2.cvtColor(imgBGR2, cv2.COLOR_BGR2RGB)
imgRGB3 = cv2.cvtColor(imgBGR3, cv2.COLOR_BGR2RGB)
imgRGB4 = cv2.cvtColor(imgBGR4, cv2.COLOR_BGR2RGB)

figsize = (10,10)
fig, ax = plt.subplots(2, 2, figsize=figsize)
ax[0][0].axis('off')
ax[0][1].axis('off')
ax[1][0].axis('off')
ax[1][1].axis('off')



ax[0][0].imshow(imgRGB1)
ax[0][1].imshow(imgRGB3)
ax[1][0].imshow(imgRGB2)
ax[1][1].imshow(imgRGB4)

fig.canvas.manager.set_window_title('심플 샘플')



plt.show()







