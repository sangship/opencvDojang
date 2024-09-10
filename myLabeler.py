import cv2, sys
import numpy as np
from glob import glob
import os


# 0. 파일 목록 읽기(data폴더) *.jpg ---> 리스트
# 1. 이미지 불러오기
# 2. 마우스 콜백함수 생성
# 3. 콜백함수안에서 박스를 그리고, 박스 좌표를 뽑아낸다. (마우스 좌표 2개)
#    참고로 YOLO에서는 박스의 중심좌표(x, y), w, h
# 4. 이미지 파일명과 동일한 파일명으로(확장자만 떼고) txt파일 생성
# 추가 기능0 : 박스를 잘못 쳤을때 's'를 누르면 현재파일의 박스 내용 초기화
# 추가 기능1 : 화살표(->)를 누르면 다음 이미지 로딩되고(1~4 반복)
# 추가 기능2 : 화살표(<-)를 눌렀을 때, txt파일이 있다면 박스를 이미지 위에 띄워주면




# def getImageList():

#     # 현재 작업 디렉토리 확인
#     basePath = os.getcwd()
#     dataPath = os.path.join(basePath, 'images')
#     #print(dataPath)
#     fileNames = glob(os.path.join(dataPath, '*.jpg'))
#     #print(fileNames)

#     return fileNames


# # corners : 좌표(startPt, endPt)
# # 2개 좌표를 이용해서 직사각형 그리기

# def drawROI(img, corners):
#     #박스를 그릴 레이어를 생성 : cpy
#     cpy = img.copy()
#     line_c = (128, 128, 255) #직선의 색상
#     lineWidth = 2
#     cv2.rectangle(cpy, corners[0], corners[1], color=line_c, thickness=lineWidth)
#     disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)
    
#     return disp


# # 마우스 콜백 함수 정의
# def onMouse(event, x, y, flags, param):
    
#     global startPt, img, ptList
#     cpy = img.copy()
    
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         endPt=(x,y)
        
#     elif event == cv2.EVENT_LBUTTONUP:
#         ptList = [startPt, (x,y)]
#         cpy = drawROI(img, ptList)
#         startPt = None
#         cv2.imshow('Label', cpy)
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if startPt:
#             ptList=[startPt, (x,y)]
#             cpy = drawROI(img, ptList)
        
#     cv2.imshow('Label', cpy)


# # 마우스가 눌리지 않으면 좌표값은 없음
# ptList = []


# fileNames = getImageList()

# img = cv2.imread(fileNames[0])

# cv2.namedWindow('Label')
# cv2.setMouseCallback('Label', onMouse, [img])
# cv2.imshow('Label', img)

# while True:
#     key = cv2.waitKey()
#     if key == 27:
#         break

        
        

# cv2.destroyAllWindows()

##############################################33


# 0. 파일 목록 읽기(data폴더) *.jpg -> 리스트 
# 1. 이미지 불러오기
# 2. 마우스 콜백함수 생성
# 3. 콜백함수안에서 박스를 그리고, 박스 좌표를 뽑아낸다. (마우스 좌표 2개)
#    참고로 YOLO에서는 박스의 중심좌표(x,y), w,h
# 4. 이미지파일명과 동일한 파일명으로(확장자만 떼고) txt파일 생성
# 추가 기능0 : 박스를 잘못 쳤을때 'c'를 누르면 현재파일의 박스 내용 초기화
# 추가 기능1 : 화살표(->)를 누르면 다음 이미지 로딩되고(1~4)
# 추가 기능2 : 화살표(<-)를 눌렀을때 txt파일이 있다면 박스를 이미지 위에 띄워주면


# import cv2, sys
# import numpy as np
# from glob import glob
# import os

# def getImageList():
#     # 현재 작업 디렉토리 확인
#     basePath = os.getcwd()
#     dataPath = os.path.join(basePath,'images')
#     fileNames = glob(os.path.join(dataPath,'*.jpg'))
    
#     return fileNames

# # corners : 좌표(startPt, endPt)
# # 2개 좌표를 이용해서 직사각형 그리기
# def drawROI(img, corners):
#     # 박스를 그릴 레이어를 생성 : cpy
#     cpy = img.copy()
#     line_c = (128,128,255) #직선의 색상
#     lineWidth = 2
#     print(corners)
#     cv2.rectangle(cpy, tuple(corners[0]), tuple(corners[1]), color=line_c, thickness=lineWidth)

#     # alpha=0.3, beta=0.7, gamma=0
#     disp = cv2.addWeighted(img,0.3,cpy,0.7,0)
#     return disp


# #  마우스 콜백 함수 정의
# def onMouse(event, x, y, flags, param):
#     global startPt, img, ptList, cpy, txtWrData
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         startPt=(x,y)
#     elif event == cv2.EVENT_LBUTTONUP:
#         ptList = [startPt,(x,y)]
#         txtWrData = str(ptList)
#         cpy = drawROI(img, ptList)
#         startPt = None
#         cv2.imshow('label',cpy)
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if startPt: 
#             ptList=[startPt, (x,y)]
#             cpy = drawROI(img, ptList)
#             cv2.imshow('label',cpy)


# ptList=[]
# startPt=None
# cpy=[]
# txtWrData = None



# fileNames = getImageList()

# img = cv2.imread(fileNames[0])


# cv2.namedWindow('label')
# cv2.setMouseCallback('label',onMouse,[img])
# cv2.imshow('label', img)

# while True:
#     key = cv2.waitKey()
#     if key==27:
#         break
#     elif key == ord('s'):
#         fileName, ext = os.path.splitext(fileNames[0])
#         txtfileName = fileName + '.txt'

#         f = open(txtfileName, 'w')
#         f.write(txtWrData)
#         f.close()
        
#         # print('before write txt: {}'.format(txtWrData))
        
        
        
# cv2.destroyAllWindows()


################################################################################

import cv2
import numpy as np
from glob import glob
import os

def getImageList():
    # 현재 작업 디렉토리 확인
    basePath = os.getcwd()
    dataPath = os.path.join(basePath,'images')
    fileNames = glob(os.path.join(dataPath,'*.jpg'))
    
    return fileNames

# 직사각형을 그리는 함수
def drawROI(img, rectangles):
    cpy = img.copy()
    line_c = (128, 128, 255)  # 직선의 색상
    lineWidth = 1

    for rect in rectangles:
        cv2.rectangle(cpy, tuple(rect[0]), tuple(rect[1]), color=line_c, thickness=lineWidth)

    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)
    return disp

# 마우스 콜백 함수 정의
def onMouse(event, x, y, flags, param):
    global startPt, img, rectangles, cpy  # rectangles를 전역 변수로 설정

    if event == cv2.EVENT_LBUTTONDOWN:
        startPt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        endPt = (x, y)
        rectangles.append([startPt, endPt])  # 직사각형 추가
        cpy = drawROI(img, rectangles)
        startPt = None
        cv2.imshow('label', cpy)

    elif event == cv2.EVENT_MOUSEMOVE:
        if startPt:
            temp_rectangles = rectangles + [[startPt, (x, y)]]
            cpy = drawROI(img, temp_rectangles)
            cv2.imshow('label', cpy)

def saveRectanglesToFile(fileName, rectangles):
    baseName, _ = os.path.splitext(fileName)
    txtFileName = baseName + '.txt'

    with open(txtFileName, 'w') as f:
        for rect in rectangles:
            f.write(f"{rect}\n")
    print(f"Saved rectangles to {txtFileName}")

# 초기 변수 설정
rectangles = []  # 여러 개의 직사각형을 저장할 리스트
startPt = None
cpy = None

fileNames = getImageList()
img = cv2.imread(fileNames[0])

cv2.namedWindow('label')
cv2.setMouseCallback('label', onMouse, [img])
cv2.imshow('label', img)

while True:
    key = cv2.waitKey()
    if key == 27:  # ESC 키로 종료
        break
    elif key == ord('s'):  # 's' 키로 직사각형 정보 저장
        saveRectanglesToFile(fileNames[0], rectangles)

cv2.destroyAllWindows()
