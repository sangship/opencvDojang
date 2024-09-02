#1203py
import cv2
import numpy as np
from PIL import Image

#1
img  = cv2.imread("./data/lena.jpg") 
img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

##b, g, r = cv2.split(img)
##img_rgb= cv2.merge([r, g, b])

#2
pil_img=Image.fromarray(img_rgb)
pil_img.show()

