#1205.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

#1
img = Image.open("./data/lena.jpg")

#1-1
#img0 = img.resize(size=(480, 320)) # resample=PIL.Image.NEAREST
img0 = img.resize(size=(480, 320), box=(0, 0, 256, 512))

#1-2
img1 = img.filter(ImageFilter.GaussianBlur(radius=10))

#1-3
img2 = img.convert(mode="L") # # grayscale image

#1-4
img3 = img2.filter(ImageFilter.FIND_EDGES)
img3 = img3.point(lambda i: i > 50)

#2
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
fig.canvas.manager.set_window_title("IPL: Image processing")

ax[0][0].set_title("resize=(480, 320)", fontsize=10)
ax[0][0].axis("off")
ax[0][0].imshow(img0) # aspect = "auto"

ax[0][1].set_title("GaussianBlur", fontsize=10)
ax[0][1].axis("off")
ax[0][1].imshow(img1)

ax[1][0].set_title("Grayscale", fontsize=10)
ax[1][0].axis("off")
ax[1][0].imshow(img2, cmap=plt.cm.gray)

ax[1][1].set_title("Edge", fontsize=10)
ax[1][1].axis("off")
ax[1][1].imshow(img3, cmap=plt.cm.gray)

plt.savefig("./data/1205.jpg", bbox_inches='tight')
plt.show()
