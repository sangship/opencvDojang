#1204.py

from PIL import Image
import matplotlib.pyplot as plt

#1
img = Image.open( "./data/lena.jpg")
R, G, B = img.split()
print(R.size, R.mode)

#2
fig, ax = plt.subplots(2, 2, figsize=(10,10))
fig.canvas.manager.set_window_title("lena.jpg")

ax[0][0].set_title("RGB", fontsize=10)
ax[0][0].axis("off")
ax[0][0].imshow(img, aspect = "auto")

ax[0][1].set_title("R", fontsize=10)
ax[0][1].axis("off")
ax[0][1].imshow(R, aspect = "auto", cmap= plt.cm.gray)

ax[1][0].set_title("G", fontsize=10)
ax[1][0].axis("off")
ax[1][0].imshow(G, aspect = "auto", cmap=plt.cm.gray)

ax[1][1].set_title("B", fontsize=10)
ax[1][1].axis("off")
ax[1][1].imshow(B, aspect = "auto", cmap=plt.cm.gray)

plt.subplots_adjust(left=0,bottom=0,right=1,top=0.98,wspace=0.05, hspace=0.05)
plt.savefig("./data/1204.png", bbox_inches='tight')
plt.show()
