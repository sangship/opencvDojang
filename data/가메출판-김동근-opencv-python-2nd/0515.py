#0515.py
'''
ref1: https://github.com/opencv/opencv/blob/master/modules/imgproc/src/clahe.cpp#L157
ref2:http://www.realtimerendering.com/resources/GraphicsGems/gemsiv/clahe.c
ref3:https://gist.github.com/sadimanna/52c320ce5c49e200ce398f800d39a2c1#file-clahe-py

'''
import cv2
import numpy as np

#1
src = np.array([[2, 2, 2, 2, 0,   0,   0,   0],
                [2, 1, 1, 2, 0,   0,   0,   0],
                [2, 1, 1, 2, 0,   0,   0,   0],
                [2, 2, 2, 2, 0,   0,   0,   0],
                [0, 0, 0, 0, 255, 255, 255, 255],
                [0, 0, 0, 0, 255, 1,   1,   255],
                [0, 0, 0, 0, 255, 1,   1,   255],
                [0, 0, 0, 0, 255, 255, 255, 255]], dtype=np.uint8)

#2
def interpolate(sub_image, UL,UR,BL,BR):
    dst = np.zeros(sub_image.shape)
    sY, sX = sub_image.shape
    area = sX*sY
    #print("sX={}, sY={}".format(sX, sY))

    for y in range(sY):
        invY = sY-y
        for x in range(sX):
            invX = sX-x
            val = sub_image[y, x].astype(int)
            dst[y,x] = np.floor((invY*(invX*UL[val] + x*UR[val])+\
                                    y*(invX*BL[val] + x*BR[val]) )/area)          
    return dst

#3
def CLAHE(src, clipLimit = 40.0, tileX = 8, tileY = 8):

#3-1
    histSize = 256    
    tileSizeX = src.shape[1]//tileX
    tileSizeY = src.shape[0]//tileY
    tileArea  = tileSizeX*tileSizeY
    clipLimit = max(clipLimit*tileArea/histSize, 1)
    lutScale = (histSize - 1) / tileArea
    print("clipLimit=", clipLimit)

    LUT = np.zeros((tileY, tileX, histSize))
    dst = np.zeros_like(src)
    #print("tileX={}, tileY={}".format(tileX, tileY))

#3-2: sublocks, tiles
    for iy in range(tileY):
        for ix in range(tileX):
#3-2-1
            y = iy*tileSizeY
            x = ix*tileSizeX
            roi = src[y:y+tileSizeY, x:x+tileSizeX] # tile
            
            tileHist, bins = np.histogram(roi, histSize,[0,256])
            #tileHist=cv2.calcHist([roi],[0],None,[histSize],[0,256]).astype(np.int)
            #tileHist = tileHist.flatten()                                           
            #print("tileHist[{},{}]=\n{}".format(iy, ix, tileHist))

#3-2-2                  
            if clipLimit > 0: # clip histogram
                clipped = 0
                for i in range(histSize):
                    if tileHist[i]>clipLimit:
                        clipped += tileHist[i] - clipLimit
                        tileHist[i] = clipLimit
        
                # redistribute clipped pixels    
                redistBatch = int(clipped/ histSize)
                residual = clipped - redistBatch * histSize
                
                for i in range(histSize):
                    tileHist[i] += redistBatch
                if residual != 0:
                    residualStep = max(int(histSize/residual), 1)
                    for i in range(0, histSize, residualStep):
                        if residual> 0:
                            tileHist[i] += 1
                            residual -= 1                            
            #print("redistributed[{},{}]=\n{}".format(iy, ix, tileHist))
            
#3-2-3:     calculate Lookup table for equalizing
            cdf = tileHist.cumsum()            
            tileLut = np.round(cdf*lutScale)
            LUT[iy, ix] = tileLut          
#3-3            
    # bilinear interpolation 
    y = 0
    for i in range(tileY+1):
        if i==0:  # top row
            subY = int(tileSizeY/2)
            yU = yB = 0
        elif i==tileY: # bottom row 
            subY = int(tileSizeY/2)
            yU= yB = tileY-1
        else:
            subY = tileSizeY
            yU = i-1
            yB = i
        #print("i={}, yU={}, yB={}, subY={}".format(i, yU, yB, subY))
        
        x = 0
        for j in range(tileX+1):
            if j==0: # left column
                subX = tileSizeX//2
                xL = xR = 0
            elif j==tileX: # right column
                subX = tileSizeX//2
                xL = xR = tileX-1
            else:
                subX = tileSizeX
                xL = j-1
                xR = j
            #print(" j={}, xL={}, xR={}, subX={}".format(j, xL, xR, subX))
            
            UL = LUT[yU,xL]
            UR = LUT[yU,xR]
            BL = LUT[yB,xL]
            BR = LUT[yB,xR]
            
            roi = src[y:y+subY, x:x+subX] 
            dst[y:y+subY, x:x+subX] = interpolate(roi,UL,UR,BL,BR)
            x += subX
        y += subY        
    return  dst

#4 
##dst = CLAHE(src, clipLimit= 40.0, tileX= 1, tileY= 1)
##print("dst=", dst)
dst2 = CLAHE(src, clipLimit= 40.0, tileX= 2, tileY= 2)
print("dst=\n", dst2)
