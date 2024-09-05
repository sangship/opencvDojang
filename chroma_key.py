
import sys, numpy as np, cv2

#동영상 불러오기
fileName1 = 'data2/woman.mp4'
fileName2 = 'data2/raining.mp4'


#영상 불러오기
cap1 = cv2.VideoCapture(fileName1)
cap2 = cv2.VideoCapture(fileName2)


if not cap1.isOpened():
    sys.exit('video1 open failed')


if not cap2.isOpened():
    sys.exit('video2 open failed')



#fps확인
fps1 = int(cap1.get(cv2.CAP_PROP_FPS))

fps2 = int(cap2.get(cv2.CAP_PROP_FPS))


delay = int(1000/fps1)
#delay = int(1000/fps2)

do_composite = True
# frame1 = None  # frame1 변수를 초기화합니다.
# frame2 = None  # frame2 변수를 초기화합니다.

while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        break
    
    # frame1의 크기
    h, w = frame1.shape[:2]
    
    if do_composite:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
    
    
    # # frame2의 크기를 frame1과 동일하게 조정
    # frame2 = cv2.resize(frame2, (w, h))
        
    # hsv 색공간에서 영역을 검출해서 합성
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    # h: 50~70, s: 150~255, v:0~255
    mask = cv2.inRange(hsv,(50,150,0),(70,255,255))
    
    # # 마스크 크기와 frame2 크기 일치 확인
    # if mask.shape[:2] == frame2.shape[:2]:
    #     cv2.copyTo(frame2, mask, frame1)
    
    cv2.copyTo(frame2, mask, frame1)
    
    # 결과 확인
    cv2.imshow('frame', frame1)
    key = cv2.waitKey(delay)

    # 스페이스 바를 눌렀을 때 do_composite를 반전
    if key == ord(' '):
        do_composite = not do_composite

    # ESC키를 입력하면 종료
    elif key == 27:
        break



cap1.release()
cap2.release()
cv2.destroyAllWindows()




