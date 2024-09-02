# 1117.py
import cv2
import numpy as np

#1 
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250) #cv2.aruco.DICT_5X5_250
 
#2
nx = 2 
ny = 2  
board = cv2.aruco.GridBoard_create(nx, ny,
                                   markerLength=1.0,
                                   markerSeparation= 0.1,
                                   dictionary = aruco_dict, firstMarker=0)
#3 
img = board.draw(outSize=(600, 600), marginSize=50)
#img = cv2.aruco.drawPlanarBoard(board, outSize=(600, 600), marginSize=50)

#4
cv2.imwrite("./data/aruco_6x6.png", img)
cv2.imshow('img', img) 
cv2.waitKey(0)    
cv2.destroyAllWindows()
