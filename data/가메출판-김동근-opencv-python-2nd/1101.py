# 1101.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1
img1 = cv2.imread('./data/image1.jpg')
img2 = cv2.imread('./data/image2.jpg')

def FindCornerPoints(src_img, patternSize):
    found, corners = cv2.findChessboardCorners(src_img, patternSize)
    if not found:
        return found, corners
    
    term_crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term_crit)
    corners = corners[::-1] # reverse order, in this example, to set origin to (left-upper)
    return found, corners
    #print('corners1.shape=', corners1.shape)

patternSize = (6,3)       
found1, corners1 = FindCornerPoints(img1, patternSize)
print('corners1.shape=', corners1.shape)

found2, corners2 = FindCornerPoints(img2, patternSize)
print('corners2.shape=', corners2.shape)

#2
method = 0 # cv2.RANSAC, cv2.LMEDS,
if method == 0: # least square method
    H, mask = cv2.findHomography(corners1, corners2, method) 
else: 
    H, mask = cv2.findHomography(corners1, corners2, method, 2.0) 

mask_matches = list(mask.flatten())
print("H=\n", H)

#3: perspective projections using 4-corners
pts = cv2.perspectiveTransform(corners1, H)  # pts = corners1*H

index = [0, 5, 17, 12]
p1 = corners1[index]
p2 = pts[index]

#4
cv2.drawChessboardCorners(img1, patternSize, corners1, found1)
img1 = cv2.polylines(img1,[np.int32(p1)],True,(255,0, 0), 2)

#5
#cv2.drawChessboardCorners(img2, patternSize, corners2, found2)
#img2 = cv2.polylines(img2,[np.int32(corners2[index])],True,(0,255,255), 2)
img2 = cv2.polylines(img2,[np.int32(p2)],True,(0,0,255), 2)

cv2.imshow('img1',  img1)
cv2.imshow('img2',  img2)
cv2.waitKey()
cv2.destroyAllWindows()
