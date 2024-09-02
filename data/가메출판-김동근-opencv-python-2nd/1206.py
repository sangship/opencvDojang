#1206.py

from PIL import ImageGrab
import numpy as np
import cv2

#1
image = ImageGrab.grab() 
width, height=image.size 
print(f"width={width}, height={height}")

#2
H, W = 1024, 768
fourcc= cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('./data/screen.mp4',fourcc, 20.0, (H, W))

#3
#cv2.namedWindow("Screen")
while True:
    rgb = ImageGrab.grab()
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR) 
    bgr = cv2.resize(bgr, (H, W))

    video.write(bgr)
    cv2.imshow ("Screen", bgr)
    if cv2.waitKey(50) ==27:
        break
    
video.release ()
cv2.destroyAllWindows()
