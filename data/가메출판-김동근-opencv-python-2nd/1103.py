# 1103.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1: open video 
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./data/chess1.wmv')
if (not cap.isOpened()): 
     print('Error opening video')
     import sys
     sys.exit()
     
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

#2
patternSize = (6, 3)
def FindCornerPoints(src_img, patternSize):
    found, corners = cv2.findChessboardCorners(src_img, patternSize)
    if not found:
        return found, corners
    
    term_crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term_crit)
    corners = corners[::-1] # reverse order, in this example, to set origin to (left-upper)
    return found, corners


#3: set world(object) coordinates to Z = 0
xN, yN = patternSize # (6, 3)
mW = np.zeros((xN*yN, 3), np.float32) # (18, 3)
mW[:, :2] = np.mgrid[0:xN, 0:yN].T.reshape(-1, 2) # mW points on Z = 0
mW[:, :2]+= 1 # (1, 1, 0): coord of the start corner point in the pattern
#mW *= 3.8

# for perspective projections 
mW = mW[:, :2].reshape(-1, 1, 2) # mW.shape:(18, 3)->(18, 1, 2) 
pW = mW[[0, 5, 17, 12]] # 4-corners

#4: find corners1 in 1st frame, H1: mW->corners1
#4-1: capture 1st frame
while True:
    ret, frame1 = cap.read()
    cv2.imshow('frame',frame1)
    key = cv2.waitKey(20)
    if key == 27:  break
    if ret:
        found, corners1 = FindCornerPoints(frame1, patternSize)
        if found:
            break  
#print('corners1.shape=', corners1.shape)

#4-2
method = cv2.LMEDS # cv2.RANSAC
H1, mask = cv2.findHomography(mW, corners1, method, 2.0) 
#print("H1=\n", H1)

#5
while True:
#5-1
    ret, frame = cap.read()
    if not ret:
        break
    found, corners = FindCornerPoints(frame, patternSize)
    if not found:
        cv2.imshow('frame',frame)
        key = cv2.waitKey(20)
        if key == 27:  break
        continue
    H, mask = cv2.findHomography(corners1, corners, method, 2.0)
    
#5-2
    H1 = np.dot(H, H1)
    p2 = cv2.perspectiveTransform(pW, H1)
    
#5-3
    cv2.polylines(frame,[np.int32(p2)],True,(255,0, 0), 3)
    #pt = np.int32(corners1[0].flatten())
    #cv2.circle(frame, pt, 5, (255,255, 0), 2)
    
    corners1 = corners.copy()

    cv2.imshow('frame',frame)  
    key = cv2.waitKey(20)
    if key == 27:
        break
#6
if cap.isOpened(): cap.release()
cv2.destroyAllWindows()
 





