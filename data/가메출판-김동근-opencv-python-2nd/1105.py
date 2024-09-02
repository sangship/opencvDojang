# 1105.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1
img1 = cv2.imread('./data/image1.jpg')
img2 = cv2.imread('./data/image2.jpg')
imageSize= (img1.shape[1], img1.shape[0]) # (width, height)

def FindCornerPoints(src_img, patternSize):
    found, corners = cv2.findChessboardCorners(src_img, patternSize)
    if not found:
        return found, corners
    
    term_crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)  
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term_crit)
    corners = corners[::-1] # reverse order, in this example, to set origin to (left-upper)
    return found, corners

patternSize = (6, 3)       
found1, corners1 = FindCornerPoints(img1, patternSize)
found2, corners2 = FindCornerPoints(img2, patternSize)

#2: set world(object) coordinates to Z = 0
xN, yN = patternSize # (6, 3)
mW = np.zeros((xN*yN, 3), np.float32) # (18, 3)
mW[:, :2] = np.mgrid[0:xN, 0:yN].T.reshape(-1, 2) # mW points on Z = 0
mW[:, :2] += 1

#3: camera calibration
#3-1: intrinc parameters
K = cv2.initCameraMatrix2D([mW, mW], [corners1, corners2], imageSize) #(640, 480)
print("initial K=\n", K)

#3-2: extrinc parameters(camera pose) by cv2.solvePnP()
dists = None  # dists = np.zeros(5)
ret1, rvec1, tvec1 = cv2.solvePnP(mW, corners1, K, dists) 
ret2, rvec2, tvec2 = cv2.solvePnP(mW, corners2, K, dists)
rvecs = [rvec1, rvec2]
tvecs = [tvec1, tvec2]

#4: load the calibrated camera parameters
##with np.load('./data/calib_1104.npz') as X:
##    K, dists, rvecs, tvecs = [X[i] for i in ('K','dists','rvecs','tvecs')]

#5: display, project, and re-projection errors
index = [0, 5, 17, 12] # 4-corner index
axis3d = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

for i in range(2):
#5-1
    print("tvec[{}]={}".format(i, tvecs[i].T))
    print("rvec[{}]={}".format(i, rvecs[i].T)) # rvecs[i].shape = (3, 1)

    R, _ = cv2.Rodrigues(rvecs[i]) # R.shape = (3, 3)
    print("R[{}]=\n{}".format(i, R))
    
#5-2
    if i==0:
        img = img1
    else:
        img = img2

#5-3: display axis
    axis_2d, _ = cv2.projectPoints(axis3d, rvecs[i], tvecs[i], K, dists)
    axis_2d    = np.int32(axis_2d).reshape(-1,2)
    cv2.line(img, tuple(axis_2d[0]), tuple(axis_2d[1]),(255, 0, 0),3)
    cv2.line(img, tuple(axis_2d[0]), tuple(axis_2d[2]),(0, 255, 0),3)
    cv2.line(img, tuple(axis_2d[0]), tuple(axis_2d[3]),(0, 0, 255),3)
    
#5-4: display pW on Z = 0
    pW = mW[index]  # 4-corners' coord (x, y, 0)
    p1, _ = cv2.projectPoints(pW, rvecs[i], tvecs[i], K, dists)
    p1    = np.int32(p1)
    
    cv2.drawContours(img, [p1],-1,(0,255,255), -1)
    cv2.polylines(img,[p1],True,(0,255,0), 2)

#5-5: display pW on Z = -2    
    pW[:, 2] = -2 # 4-corners' coord (x, y, -2)  
    p2, _ = cv2.projectPoints(pW, rvecs[i], tvecs[i], K, dists)
    p2    = np.int32(p2)
    cv2.polylines(img,[p2],True,(0,0,255), 2)

#5-6: display edges between two rectangles 
    for j in range(4):
        x1, y1 = p1[j][0] # Z = 0
        x2, y2 = p2[j][0] # Z = -2
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
#5-7: re-projection errors
    pts, _ = cv2.projectPoints(mW, rvecs[i], tvecs[i], K, dists) 
    if i==0:
        errs = cv2.norm(corners1, np.float32(pts))
    else:
        errs = cv2.norm(corners2, np.float32(pts))
    print("errs[{}]={}".format(i, errs))        
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.waitKey()
cv2.destroyAllWindows()
        




