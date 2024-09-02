#1201.py
#1
from PIL import Image
img = Image.open( "./data/lena.jpg")
img.save("./data/1201-1.png")
img.show()
print(img.format, img.size, img.mode)

#2
img2 = Image.new("RGB", (200, 200), (0, 0, 255))
img.paste(img2, (100, 100))
img.save("./data/1201-2.png")
img.show()
img.close()
