
# import cv2
# import numpy as np

# canvas_size = (512, 512, 3)
# canvas_color = (255, 255, 255)  # 흰색
# canvas = np.ones(canvas_size, dtype=np.uint8) * 255
# points = []  # 다각형을 그리기 위한 점들을 저장

# # 마우스 콜백 함수 정의
# def draw_shape(event, x, y, flags, param):
#     global points
    
#     thick = 1
    
#     if event == cv2.EVENT_LBUTTONDOWN:
        
#         if flags & cv2.EVENT_FLAG_SHIFTKEY:
#             # Shift + 왼쪽 버튼 클릭 시 다각형 그리기
#             if len(points) > 2:  # 최소 3개의 점이 있어야 다각형을 그릴 수 있음
#                 cv2.polylines(canvas, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=thick)
#             points = []  # 점 초기화
#         else:
#             # Shift 없이 왼쪽 버튼 클릭 시 점 추가
#             points.append((x, y))
#             cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         # 오른쪽 버튼 클릭 시 원 그리기
#         radius = 90
#         color = (0, 0, 255)  # 빨간색
        
#         cv2.circle(canvas, (x, y), radius, color, thick)

# cv2.namedWindow('Canvas')
# cv2.setMouseCallback('Canvas', draw_shape)

# while True:
#     cv2.imshow('Canvas', canvas)
#     key = cv2.waitKey(1) & 0xFF
#     if  key == 27:  # ESC 키로 종료
#         break

#     elif key == ord('r'): # r키를 누르면 리사이즈
#         resized = 128   # 256
#         blurred_canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
#         resized_canvas_area = cv2.resize(blurred_canvas, (resized, resized), interpolation=cv2.INTER_AREA)
#         resized_canvas_linear = cv2.resize(canvas, (resized, resized), interpolation=cv2.INTER_LINEAR)
        
#         cv2.imshow('Resized Canvas (INTER_AREA)', resized_canvas_area)
#         cv2.imshow('Resized Canvas (INTER_LINEAR)', resized_canvas_linear)
        
        
    
# cv2.destroyAllWindows()


######################################################################################################################

import cv2
import numpy as np

canvas_size = (512, 512, 3)
canvas_color = (255, 255, 255)  # 흰색
canvas = np.ones(canvas_size, dtype=np.uint8) * 255
points = []  # 다각형을 그리기 위한 점들을 저장

# 텍스트를 추가하는 함수
def add_canvas_name(canvas, text, position=(10, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2
    color = (0, 0, 0)  # 검정색
    thickness = 1
    cv2.putText(canvas, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

# 마우스 콜백 함수 정의
def draw_shape(event, x, y, flags, param):
    global points
    
    thick = 1
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            # Shift + 왼쪽 버튼 클릭 시 다각형 그리기
            if len(points) > 2:  # 최소 3개의 점이 있어야 다각형을 그릴 수 있음
                cv2.polylines(canvas, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=thick)
            points = []  # 점 초기화
        else:
            # Shift 없이 왼쪽 버튼 클릭 시 점 추가
            points.append((x, y))
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 오른쪽 버튼 클릭 시 원 그리기
        radius = 90
        color = (0, 0, 255)  # 빨간색
        
        cv2.circle(canvas, (x, y), radius, color, thick)

cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas', draw_shape)

while True:
    # 캔버스에 이름 추가
    canvas_copy = canvas.copy()
    add_canvas_name(canvas_copy, 'Original Canvas')
    
    cv2.imshow('Canvas', canvas_copy)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키로 종료
        break
    elif key == ord('r'):  # r키를 누르면 리사이즈
        resized = 128  # 리사이즈 크기
        blurred_canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
        resized_canvas_area = cv2.resize(blurred_canvas, (resized, resized), interpolation=cv2.INTER_AREA)
        resized_canvas_linear = cv2.resize(canvas, (resized, resized), interpolation=cv2.INTER_LINEAR)
        
        # 리사이즈된 캔버스에 이름 추가
        add_canvas_name(resized_canvas_area, 'Resized Canvas (INTER_AREA)')
        add_canvas_name(resized_canvas_linear, 'Resized Canvas (INTER_LINEAR)')
        
        cv2.imshow('Resized Canvas (INTER_AREA)', resized_canvas_area)
        cv2.imshow('Resized Canvas (INTER_LINEAR)', resized_canvas_linear)

cv2.destroyAllWindows()

