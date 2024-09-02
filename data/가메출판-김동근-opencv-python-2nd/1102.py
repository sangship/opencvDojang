# 1102.py
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


#2: set world(object) coordinates to Z = 0
xN, yN = patternSize # (6, 3)
mW = np.zeros((xN*yN, 3)) # (18, 3)
mW[:, :2]= np.mgrid[0:xN, 0:yN].T.reshape(-1, 2) # mW points on Z = 0
mW[:, :2]+= 1 # (1, 1, 0): coord of the start corner point in the pattern
#mW *= 3.8 # grid size 

#3: calulate homography
method = cv2.LMEDS # cv2.RANSAC
H1, mask = cv2.findHomography(mW, corners1, method, 2.0) 
H2, mask = cv2.findHomography(mW, corners2, method, 2.0)
H, mask = cv2.findHomography(corners1, corners2, method, 2.0)
#mask_matches = list(mask.flatten())

print("H=\n", H)
print("H1=\n", H1)
print("H2=\n", H2)

#4: perspective projections 
mW = mW[:, :2].reshape(-1, 1, 2) # mW.shape:(18, 3)->(18, 1, 2) 
pW = mW[[0, 5, 17, 12]] # 4-corners

p1 = cv2.perspectiveTransform(pW, H1)
p2 = cv2.perspectiveTransform(pW, H2)
#print("p1=",p1)
#print("p2=",p2)

#5:  
H3 = np.dot(H, H1) # H3 == H2 with some errors
print("H3=\n", H3)

#pW = cv2.convertPointsFromHomogeneous(pW.T)
p3 = cv2.perspectiveTransform(pW, H3)
#print("p3=",p3) # p3 = H3*pW = H*H1*pW= p2 with some errors


#6: perspective projections using matrix multiplications
#6-1: p4 rows are p1's homogeneous coords
pW = pW.reshape(-1, 2) # rows are points
pW = cv2.convertPointsToHomogeneous(pW) # shape:(4, 2)->(4, 3)
p4 = np.dot(pW, H1.T)
p4 = p4.reshape(-1, 3)/p4[:,:,2]
#print("p4=",p4) # shape=(4, 3): rows are points

#6-2: p5 columns are p1's homogeneous coords
pW = pW.reshape(-1, 3).T  # shape=(3, 4) 
p5 = np.dot(H1, pW)
p5 = p5/p5[2] 
#print("p5=",p5) # shape=(3,4): columns are points

#7: display points
#7-1: start corner point in the pattern: (1, 1) in this example
x, y = corners1[0][0] #p1[0][0]
cv2.circle(img1, (int(x), int(y)), 10, (255,255, 0), -1)

x, y = corners2[0][0] #p2[0][0]
cv2.circle(img2, (int(x), int(y)), 10, (255,255, 0), -1)

#7-2: p1 = H1*pW
cv2.drawChessboardCorners(img1, patternSize, corners1, found1)
img1 = cv2.polylines(img1,[np.int32(p1)],True,(255,0, 0), 3)

#7-3: p2 = H2*pW
cv2.drawChessboardCorners(img2, patternSize, corners2, found2)
img2 = cv2.polylines(img2,[np.int32(p2)],True,(0,0, 255), 4)

#7-4: p3 = H3*pW
img2 = cv2.polylines(img2,[np.int32(p3)],True,(0,255, 255), 1)

cv2.imshow('img1',  img1)
cv2.imshow('img2',  img2)
cv2.waitKey()
cv2.destroyAllWindows()



