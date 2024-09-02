# 1111.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1: open video 
##cap = cv2.VideoCapture(0)
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

#4: load camera matrix K
with np.load('./data/calib_1104.npz') as X: # './data/calib_1109.npz'
    K = X['K']
    #dists = X['dists']
dists = np.zeros(5)
print("K=\n", K)
print("dists=", dists)

#5: decompose H into R(rvec) and T(tvec)
def decomposeH2RT(H):
    H = H/cv2.norm(H[:, 0]) # normalization ||c1|| = 1
    c1 = H[:, 0]
    c2 = H[:, 1]
    c3 = np.cross(c1, c2)

    tvec = H[:, 2]
    Q = np.stack([c1, c2, c3], axis = 1)
    U, s, VT = np.linalg.svd(Q)
    R = np.dot(U, VT)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec

#6: pose estimation from H, project, and re-projection errors
index = [0, 5, 17, 12] # 4-corner index
axis3d = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
method = cv2.LMEDS # cv2.RANSAC

t = 1 # frame counter
bInit = True
while True:
#6-1
    ret, frame = cap.read()
    if not ret:
        break
    found, corners = FindCornerPoints(frame, patternSize)
    #cv2.drawChessboardCorners(frame, patternSize, corners, found)

    if not found:
        cv2.imshow('frame',frame)
        key = cv2.waitKey(20)
        if key == 27:  break
        bInit = True
        continue
#6-2: 
    curr_corners = cv2.undistortPoints(corners, K, dists)
    if bInit: # find H1: mW->corners, in 1st frame
        print("Initialize.......")
        prev_corners = curr_corners.copy()
        H1, mask = cv2.findHomography(mW, prev_corners, method, 2.0)
        bInit = False
        #continue
#6-3:        
    #pose estimation from H1 between mW and corners
    H, mask = cv2.findHomography(prev_corners, curr_corners, method, 2.0)
    H1 = np.dot(H, H1)
    
    rvec, tvec = decomposeH2RT(H1)
    prev_corners = curr_corners.copy() # for next frame
    
#6-4: display axis and cube
    axis_2d, _ = cv2.projectPoints(axis3d, rvec, tvec, K, dists)
    axis_2d    = np.int32(axis_2d).reshape(-1,2)
    cv2.line(frame, tuple(axis_2d[0]), tuple(axis_2d[1]),(255, 0, 0),3)
    cv2.line(frame, tuple(axis_2d[0]), tuple(axis_2d[2]),(0, 255, 0),3)
    cv2.line(frame, tuple(axis_2d[0]), tuple(axis_2d[3]),(0, 0, 255),3)
    
    #display pW on Z = 0
    pW = mW[index]  # 4-corners' coord (x, y, 0)
    p1, _ = cv2.projectPoints(pW, rvec, tvec, K, dists)
    p1    = np.int32(p1)
    
    cv2.drawContours(frame, [p1],-1,(0,255,255), -1)
    cv2.polylines(frame,[p1],True,(0,255,0), 2)

    #display pW on Z = -2    
    pW[:, 2] = -2 # 4-corners' coord (x, y, -2)   
    p2, _ = cv2.projectPoints(pW, rvec, tvec, K, dists)
    p2    = np.int32(p2)
    cv2.polylines(frame,[p2],True,(0,0,255), 2)

    #display edges between two rectangles 
    for j in range(4):
        x1, y1 = p1[j][0] # Z = 0
        x2, y2 = p2[j][0] # Z = -2
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
#6-5: re-projection errors
    pts, _ = cv2.projectPoints(mW, rvec, tvec, K, dists) 
    errs = cv2.norm(corners, np.float32(pts))
    print("errs[{}]={:.2f}".format(t, errs))
    t += 1
    
    cv2.imshow('frame',frame)  
    key = cv2.waitKey(20)
    if key == 27:
        break     
#7
if cap.isOpened(): cap.release()
cv2.destroyAllWindows()
 





