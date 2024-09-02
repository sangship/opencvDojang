# 1106.py
# https://docs.opencv.org/master/d9/dab/tutorial_homography.html#projective_transformations
# https://team.inria.fr/lagadic/camera_localization/tutorial-pose-dlt-planar-opencv.html
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

#3: intrinc parameters
K = cv2.initCameraMatrix2D([mW, mW], [corners1, corners2], imageSize) #(640, 480) 
print("K=\n", K)  
 
#4: decompose H into R(rvec) and T(tvec)
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

#5: pose estimation from H, project, and re-projection errors
dists = None  # dists = np.zeros(5)
index = [0, 5, 17, 12] # 4-corner index
axis3d = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

for i in range(2):
    print("----- i=", i)

#5-1
    if i==0:
        img = img1       
        corners = cv2.undistortPoints(corners1, K, dists)
        
        # for checking
        #ret1, rvec1, tvec1 = cv2.solvePnP(mW, corners1, K, dists) 
        #print("rvec1=", rvec1.T)
        #print("tvec1=", tvec1.T)

    else:
        img = img2
        corners = cv2.undistortPoints(corners2, K, dists)

        #for checking
        #ret2, rvec2, tvec2 = cv2.solvePnP(mW, corners2, K, dists) 
        #print("rvec2=", rvec2.T)
        #print("tvec2=", tvec2.T)
        
#5-2: pose estimation from H
    H, mask = cv2.findHomography(mW, corners, cv2.LMEDS, 2.0)
    print("H=", H)
    rvec, tvec = decomposeH2RT(H)
    print("rvec=", rvec.T)
    print("tvec=", tvec.T)

#5-3: display axis
    axis_2d, _ = cv2.projectPoints(axis3d, rvec, tvec, K, dists)
    axis_2d    = np.int32(axis_2d).reshape(-1,2)
    cv2.line(img, tuple(axis_2d[0]), tuple(axis_2d[1]),(255, 0, 0),3)
    cv2.line(img, tuple(axis_2d[0]), tuple(axis_2d[2]),(0, 255, 0),3)
    cv2.line(img, tuple(axis_2d[0]), tuple(axis_2d[3]),(0, 0, 255),3)
    
#5-4: display pW on Z = 0
    pW = mW[index]  # 4-corners' coord (x, y, 0) on Z = 0
    p1, _ = cv2.projectPoints(pW, rvec, tvec, K, dists)
    p1    = np.int32(p1)   
    cv2.drawContours(img, [p1],-1,(0,255,255), -1)
    cv2.polylines(img,[p1],True,(0,255,0), 2)

#5-5: display pW on Z = -2    
    pW[:, 2] = -2 # 4-corners' coord (x, y, -2)    
    p2, _ = cv2.projectPoints(pW, rvec, tvec, K, dists)
    p2    = np.int32(p2)
    cv2.polylines(img,[p2],True,(0,0,255), 2)

#5-6: display edges between two rectangles 
    for j in range(4):
        x1, y1 = p1[j][0] # Z = 0
        x2, y2 = p2[j][0] # Z = -2
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

#5-7: re-projection errors
    pts, _ = cv2.projectPoints(mW, rvec, tvec, K, dists) 
    if i==0:
        errs = cv2.norm(corners1, np.float32(pts))
    else:
        errs = cv2.norm(corners2, np.float32(pts))
    print("errs[{}]={}".format(i, errs))  
      
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.waitKey()
cv2.destroyAllWindows()



