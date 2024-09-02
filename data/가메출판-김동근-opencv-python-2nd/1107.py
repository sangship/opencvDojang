# 1107.py
# https://docs.opencv.org/master/d9/dab/tutorial_homography.html#projective_transformations
# https://hal.inria.fr/inria-00174036v3/document
#       Deeper understanding of the homography decomposition for vision-based control

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
#print("K=\n", K)  
 
#4: pose estimation on camera 1 and camera 2
dists = None  # dists = np.zeros(5)
ret1, rvec1, tvec1 = cv2.solvePnP(mW, corners1, K, dists) 
ret2, rvec2, tvec2 = cv2.solvePnP(mW, corners2, K, dists)
print("rvec1=", rvec1.T)
print("tvec1=", tvec1.T)
print("rvec2=", rvec2.T)
print("tvec2=", tvec2.T)

def computeC2MC1(rvec1, t1, rvec2, t2):
    R1 = cv2.Rodrigues(rvec1)[0] # vector to matrix
    R2 = cv2.Rodrigues(rvec2)[0]

    R_1to2 = np.dot(R2, R1.T)
    r_1to2 = cv2.Rodrigues(R_1to2)[0]

    t_1to2 = np.dot(R2, np.dot(-R1.T, t1)) + t2
    return r_1to2, t_1to2

rvec_1to2, tvec_1to2 = computeC2MC1(rvec1, tvec1, rvec2, tvec2)
#print("rvec_1to2=",  rvec_1to2.T)
#print("tvec_1to2=",  tvec_1to2.T)

#6: just for check, rvec==rvec2, tvec == tvec2
rvec, tvec = cv2.composeRT(rvec1, tvec1, rvec_1to2, tvec_1to2)[:2]
print("rvec=",  rvec.T)
print("tvec=",  tvec.T)

#7: homography from the camera displacement

#7-1: the plane normal at camera 1
normal = np.array([0., 0., 1.]).reshape(3, 1) # +Z
R1 = cv2.Rodrigues(rvec1)[0] # vector to matrix
normal1 = np.dot(R1, normal)

#7-2: the origin as a point on the plane at camera 1
origin = np.array([0., 0., 0.]).reshape(3, 1)
origin1 = np.dot(R1, origin) + tvec1

#7-3: the plane distance to the camera 1
#     as the dot product between the plane normal and a point on the plane
d1 = np.sum(normal1*origin1)
print("normal1=", normal1.T)
print("origin1=", origin1.T)
print("d1=", d1)

#7-4: homography from camera displacement
def computeHomography(rvec_1to2, tvec_1to2, d, normal):
    R_1to2 = cv2.Rodrigues(rvec_1to2)[0] # vector to matrix    
    homography = R_1to2 + np.dot(tvec_1to2, normal.T)/d
    return homography

homography_euclidean = computeHomography(rvec_1to2, tvec_1to2, d1, normal1)
homography = np.dot(np.dot(K, homography_euclidean), np.linalg.inv(K))
homography /= homography[2,2]
homography_euclidean /= homography_euclidean[2,2]
print("homography=",homography)

#8: same but using absolute camera poses, just for check
def computeHomography2(rvec1, tvec1, rvec2, tvec2, d, normal):
    R1 = cv2.Rodrigues(rvec1)[0] # vector to matrix
    R2 = cv2.Rodrigues(rvec2)[0] # vector to matrix
    
    homography = np.dot(R2, R1.T) + np.dot((np.dot(np.dot(-R2,R1.T),tvec1)+ tvec2), normal.T)/d
    return homography

homography_euclidean2 = computeHomography2(rvec1, tvec1, rvec2, tvec2, d1, normal1)
homography2 = np.dot(np.dot(K, homography_euclidean2), np.linalg.inv(K))
homography2 /= homography2[2,2]
homography_euclidean2 /= homography_euclidean2[2,2]
print("homography2=",homography2)

#9: homography from image points
H, mask = cv2.findHomography(corners1, corners2, cv2.LMEDS, 2.0)
print("H=",H) 

#10: perspective projections using homography
index = [0, 5, 17, 12] # 4-corner index
img1 = cv2.polylines(img1,[np.int32(corners1[index])],True,(255,0,0), 2)

#pts = cv2.perspectiveTransform(corners1, H)           # pts = corners1*H
#pts = cv2.perspectiveTransform(corners1, homography2) # pts = corners1*homography2
pts = cv2.perspectiveTransform(corners1, homography)   # pts = corners1*homography

img2 = cv2.polylines(img2,[np.int32(pts[index])],True,(0,0,255), 2) 
#cv2.imshow('img1',  img1)
cv2.imshow('img2',  img2)
cv2.waitKey()
cv2.destroyAllWindows()
