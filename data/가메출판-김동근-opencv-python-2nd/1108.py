# 1108.py
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
#print("normal1=", normal1.T)
#print("origin1=", origin1.T)
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
#print("homography2=",homography2)

#9: homography from image points
H, mask = cv2.findHomography(corners1, corners2, cv2.LMEDS, 2.0)
print("H=",H) 

#10: decompose the homography to a set of rotations, translations and plane normals
#10-1 
ret, Rs_decomp, ts_decomp, ns_decomp = cv2.decomposeHomographyMat(homography, K) # homography2
#ret, Rs_decomp, ts_decomp, ns_decomp = cv2.decomposeHomographyMat(H, K)

#10-2
solutions = cv2.filterHomographyDecompByVisibleRefpoints(Rs_decomp, ns_decomp, corners1, corners2)
solutions = solutions.flatten()
print("solutions=", solutions)

#10-3: for check, the same as rvec_decomp, tvec_decomp, nvec_decomp in #10-4
print("rvec_1to2=",  rvec_1to2.T)
print("tvec_1to2=",  tvec_1to2.T)
print("normal1=",    normal1.T)

#10-4: find a solution with minimum re-projection errors
min_errors = 1.0E5
for i in solutions:
    print("----- solutions, i=", i)

    rvec_decomp = cv2.Rodrigues(Rs_decomp[i])[0]
    tvec_decomp = ts_decomp[i]*d1 # scale by the plane distance to the camera 1
    #print("rvec_decomp=", rvec_decomp.T)
    #print("tvec_decomp=", tvec_decomp.T)
    #print("ns_decomp=", ns_decomp[i].T)
    
    # re-projection errors
    rvec3, tvec3 = cv2.composeRT(rvec1, tvec1, rvec_decomp, tvec_decomp)[:2]
    pts, _ = cv2.projectPoints(mW, rvec3, tvec3, K, dists) 
    errs = cv2.norm(corners2, np.float32(pts))
    print("errs[{}]={}".format(i, errs))
    if errs < min_errors:
        min_errors = errs
        min_i = i

print("min_errors[{}]={}".format(min_i, min_errors))
rvec_decomp = cv2.Rodrigues(Rs_decomp[min_i])[0]    
tvec_decomp = ts_decomp[min_i]*d1 # scale by the plane distance to the camera 1
nvec_decomp = ns_decomp[min_i]
print("rvec_decomp=", rvec_decomp.T)
print("tvec_decomp=", tvec_decomp.T)
print("nvec_decomp=", nvec_decomp.T)
      
#11: project and display
#11-1: compose camera1 + decomposition( or displacement)
rvec, tvec = cv2.composeRT(rvec1, tvec1, rvec_decomp, tvec_decomp)[:2]
print("rvec=", rvec.T)
print("tvec=", tvec.T)

#11-2: display axis in img2
index = [0, 5, 17, 12] # 4-corner index
axis3d = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
 
axis_2d, _ = cv2.projectPoints(axis3d, rvec, tvec, K, dists)
axis_2d    = np.int32(axis_2d).reshape(-1,2)
cv2.line(img2, tuple(axis_2d[0]), tuple(axis_2d[1]),(255, 0, 0),3)
cv2.line(img2, tuple(axis_2d[0]), tuple(axis_2d[2]),(0, 255, 0),3)
cv2.line(img2, tuple(axis_2d[0]), tuple(axis_2d[3]),(0, 0, 255),3)
        
#11-3: display pW on Z = 0
pW = mW[index]  # 4-corners' coord (x, y, 0) on Z = 0
p1, _ = cv2.projectPoints(pW, rvec, tvec, K, dists)
p1    = np.int32(p1)
        
cv2.drawContours(img2, [p1],-1,(0,255,255), -1)
cv2.polylines(img2,[p1],True,(0,255,0), 2)

#11-4: display pW on Z = -2    
pW[:, 2] = -2 # 4-corners' coord (x, y, -2) 
p2, _ = cv2.projectPoints(pW, rvec, tvec, K, dists
                                  )
p2    = np.int32(p2)
cv2.polylines(img2,[p2],True,(0,0,255), 2)

#11-5: display edges between two rectangles 
for j in range(4):
    x1, y1 = p1[j][0] # Z = 0
    x2, y2 = p2[j][0] # Z = -2
    cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
               
cv2.imshow('img2',img2)
cv2.waitKey()
cv2.destroyAllWindows()    
