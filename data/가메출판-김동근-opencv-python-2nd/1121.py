# 1121.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)


#1
select = 1 # 2
if select == 1:
    nx, ny  = 3, 3 # "charuco_6x6_250.png"
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # "charuco_6x6.png"
else:
    nx, ny  = 4, 7 # "charuco_5x5_1000.png"
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000) # "charuco_5x5_1000.png"

#2
board = cv2.aruco.CharucoBoard_create(squaresX=nx, squaresY=ny,         
                                      squareLength=0.04, markerLength=0.02, 
                                      dictionary=aruco_dict)

#3 an image from the board
if select == 1:
    img = board.draw(outSize=(600, 600), marginSize=50)
    cv2.imwrite("./data/charuco_6x6_250.png", img)
else:
    img = board.draw(outSize=(600, 800), marginSize=50)
    cv2.imwrite("./data/charuco_5x5_1000.png", img)
    
cv2.imshow('img', img)
cv2.waitKey(0)    
cv2.destroyAllWindows()
 





