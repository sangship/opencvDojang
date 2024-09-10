import cv2
import numpy as np
import os
from glob import glob

# 전역 변수 설정
drawing = False  # 드래그 상태 플래그
ix, iy = -1, -1  # 시작 좌표 초기화
rectangles = []  # 그려진 직사각형 목록
current_image = None  # 현재 이미지
file_index = 0  # 현재 파일 인덱스
image_files = []  # 이미지 파일 목록

def load_images(image_dir='images'):
    """이미지 파일 목록을 가져옵니다."""
    # 지원하는 이미지 확장자 목록
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    
    image_files = []
    # 각 확장자에 대해 파일 목록을 가져와 합침
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, ext)))
    
    return image_files


def load_image(index):
    """인덱스에 해당하는 이미지를 로드하고 화면에 표시합니다."""
    global current_image
    img = cv2.imread(image_files[index])
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return None
    current_image = img.copy()  # 원본 이미지 복사
    draw_existing_rectangles(current_image)  # 이미 그린 직사각형을 그립니다.
    cv2.imshow('Image', current_image)
    return img

def draw_existing_rectangles(img):
    """이미지에 기존의 직사각형들을 그립니다."""
    for rect in rectangles:
        cv2.rectangle(img, rect[0], rect[1], (0, 255, 0), 2)

def save_annotations(file_name, rectangles, img_shape):
    """YOLO 형식으로 주석을 저장합니다."""
    txt_file_name = os.path.splitext(file_name)[0] + '.txt'
    with open(txt_file_name, 'w') as f:
        for rect in rectangles:
            x_center = (rect[0][0] + rect[1][0]) / 2.0 / img_shape[1]
            y_center = (rect[0][1] + rect[1][1]) / 2.0 / img_shape[0]
            width = abs(rect[1][0] - rect[0][0]) / img_shape[1]
            height = abs(rect[1][1] - rect[0][1]) / img_shape[0]
            f.write(f"0 {x_center} {y_center} {width} {height}\n")
    print(f"주석이 {txt_file_name} 파일로 저장되었습니다.")

def load_annotations(file_name, img_shape):
    """주석 파일을 불러와 YOLO 형식 또는 좌표 리스트 형식에 따라 처리합니다."""
    txt_file_name = os.path.splitext(file_name)[0] + '.txt'
    if not os.path.exists(txt_file_name):
        return []
    
    loaded_rectangles = []
    with open(txt_file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            
            # YOLO 형식으로 저장된 경우 처리
            if line[0].isdigit():  # 첫 문자가 숫자라면 YOLO 형식으로 가정
                try:
                    # YOLO 형식: "0 x_center y_center width height"
                    label, x_center, y_center, width, height = map(float, line.split())
                    x1 = int((x_center - width / 2) * img_shape[1])
                    y1 = int((y_center - height / 2) * img_shape[0])
                    x2 = int((x_center + width / 2) * img_shape[1])
                    y2 = int((y_center + height / 2) * img_shape[0])
                    loaded_rectangles.append(((x1, y1), (x2, y2)))
                except ValueError as e:
                    print(f"Error parsing YOLO format: {e}")
            else:
                # 좌표 리스트 형식으로 저장된 경우 처리
                try:
                    rect = eval(line)  # 좌표 리스트를 튜플로 변환
                    if isinstance(rect, tuple) and len(rect) == 2 and isinstance(rect[0], tuple) and isinstance(rect[1], tuple):
                        loaded_rectangles.append(rect)
                except (SyntaxError, ValueError) as e:
                    print(f"Invalid format in annotation file: {line}. Error: {e}")
    
    return loaded_rectangles

def mouse_callback(event, x, y, flags, param):
    """마우스 이벤트를 처리하여 직사각형을 그리고 저장합니다."""
    global ix, iy, drawing, current_image, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭
        drawing = True
        ix, iy = x, y
        print(f"시작점: ({ix}, {iy})")

    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if drawing:
            img_copy = current_image.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 뗌
        drawing = False
        rectangles.append(((ix, iy), (x, y)))  # 직사각형 추가
        print(f"끝점: ({x}, {y})")
        cv2.rectangle(current_image, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('Image', current_image)


def handle_keypress():
    """키 입력을 처리하여 기능을 수행합니다."""
    global file_index, rectangles, current_image

    # 방향키에 대한 키 코드 상수 정의 (cv2.waitKeyEx() 사용)
    LEFT_ARROW_KEY = 2424832  # 왼쪽 화살표 키 코드
    RIGHT_ARROW_KEY = 2555904  # 오른쪽 화살표 키 코드
    ESC_KEY = 27

    while True:
        key = cv2.waitKeyEx(0)  # cv2.waitKey() 대신 cv2.waitKeyEx() 사용
        if key == ESC_KEY:  # ESC 키 - 프로그램 종료
            break
        elif key == ord('c'):  # 'C' 키 - 직사각형 초기화
            rectangles.clear()
            load_image(file_index)
        elif key == RIGHT_ARROW_KEY:  # 오른쪽 화살표 - 다음 이미지
            if file_index < len(image_files) - 1:  # 마지막 파일이 아닌 경우에만 증가
                file_index += 1
                current_image = load_image(file_index)  # 다음 이미지 로드 후 업데이트
                rectangles = load_annotations(image_files[file_index], current_image.shape)  # 다음 이미지의 주석 로드
                draw_existing_rectangles(current_image)  # 직사각형 그리기
                cv2.imshow('Image', current_image)
            else:
                print("마지막 이미지입니다.")  # 끝에 도달했을 때 메시지 출력
        elif key == LEFT_ARROW_KEY:  # 왼쪽 화살표 - 이전 이미지
            if file_index > 0:  # 첫 번째 파일이 아닌 경우에만 감소
                file_index -= 1
                current_image = load_image(file_index)  # 이전 이미지 로드 후 업데이트
                rectangles = load_annotations(image_files[file_index], current_image.shape)  # 이전 이미지의 주석 로드
                draw_existing_rectangles(current_image)  # 직사각형 그리기
                cv2.imshow('Image', current_image)
            else:
                print("첫 번째 이미지입니다.")  # 처음에 도달했을 때 메시지 출력
        elif key == ord('s'):  # 'S' 키 - 주석 저장
            save_annotations(image_files[file_index], rectangles, current_image.shape)  # 현재 이미지의 주석 저장
            print(f"{image_files[file_index]}의 주석이 저장되었습니다.")

        
def main():
    """메인 함수: 이미지 목록을 불러오고 마우스 콜백 함수를 설정합니다."""
    global image_files, current_image, rectangles

    image_files = load_images()  # 이미지 목록 가져오기
    if not image_files:
        print("이미지를 찾을 수 없습니다.")
        return

    # 첫 번째 이미지 로드 및 기존 주석 로드
    current_image = load_image(file_index)
    if current_image is None:
        return

    # 첫 번째 이미지의 주석 로드
    rectangles = load_annotations(image_files[file_index], current_image.shape)
    draw_existing_rectangles(current_image)  # 첫 번째 이미지의 직사각형 그리기
    cv2.imshow('Image', current_image)  # 화면에 표시

    # 추가할 부분: 모든 이미지 파일 목록 출력
    print("불러온 이미지 파일 목록:")
    for i, file in enumerate(image_files):
        print(f"{i + 1}: {file}")
    
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)  # 마우스 콜백 함수 설정

    handle_keypress()  # 키 입력 처리

    cv2.destroyAllWindows()  # 모든 창 닫기



if __name__ == '__main__':
    main()
