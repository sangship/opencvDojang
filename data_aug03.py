import cv2
import os
import sys
import numpy as np
import shutil  # 파일 복사에 필요한 라이브러리 추가

# 데이터 경로 설정
dataPath = os.path.join(os.getcwd(), 'DataAug')  # 현재 작업 디렉터리에 'DataAug' 폴더 생성
dataOrg = os.path.join(dataPath, 'org')  # 원본 이미지를 위한 디렉터리
dataAug = os.path.join(dataPath, 'augmented')  # 증강된 이미지를 저장할 디렉터리
os.makedirs(dataAug, exist_ok=True)  # 증강된 이미지 디렉터리 생성

fileName = os.path.join(dataOrg, 'mousepad_wood.jpg')  # 예제 이미지 파일 경로

# 새로운 폴더 구조 설정 (추가된 부분)
class_folders = {
    'adapter': ['adapter_white.jpg', 'adapter_wood.jpg'],
    'mousepad': ['mousepad_white.jpg', 'mousepad_wood.jpg'],
    'toy': ['toy_white.jpg', 'toy_wood.jpg']
}

# 1단계: 폴더 구조 생성 및 파일 복사 (수정된 부분)
for class_name, files in class_folders.items():
    # 클래스별 폴더 생성
    class_dir = os.path.join(dataAug, f'class_{class_name}')
    os.makedirs(class_dir, exist_ok=True)  # 클래스 폴더 생성
    print(f"클래스 폴더 생성됨: {class_dir}")  # 폴더 생성 확인 메시지

    # 배경별 폴더 생성 및 원본 파일 복사
    for file in files:
        if 'white' in file:
            background_dir = os.path.join(class_dir, f'class_{class_name}_white')
        else:
            background_dir = os.path.join(class_dir, f'class_{class_name}_wood')
        
        os.makedirs(background_dir, exist_ok=True)  # 배경별 폴더 생성
        print(f"배경별 폴더 생성됨: {background_dir}")  # 폴더 생성 확인 메시지

        # 원본 파일 복사
        src_file = os.path.join(dataOrg, file)  # 원본 파일 경로
        if os.path.exists(src_file):  # 파일이 존재하는지 확인
            dst_file = os.path.join(background_dir, file)  # 복사될 파일 경로
            shutil.copy(src_file, dst_file)  # 원본 파일 복사
            print(f"파일 복사됨: {src_file} -> {dst_file}")  # 복사 확인용 메시지
        else:
            print(f"원본 파일을 찾을 수 없습니다: {src_file}")  # 파일이 없을 때 메시지

# 2단계: 폴더 및 파일 상태 확인 (수정된 부분)
def check_folders():
    print("\n폴더 구조 확인:")
    for class_name in class_folders:
        class_dir = os.path.join(dataAug, f'class_{class_name}')
        if os.path.exists(class_dir):
            print(f"클래스 폴더 확인됨: {class_dir}")
            for subdir in os.listdir(class_dir):
                subdir_path = os.path.join(class_dir, subdir)
                print(f"  하위 폴더: {subdir_path}, 파일: {os.listdir(subdir_path)}")
        else:
            print(f"클래스 폴더를 찾을 수 없습니다: {class_dir}")
            
# 이미지 불러오기 함수 (이전 단계와 동일)
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)
    return img

# 수정된 이미지 리사이즈 함수
def resize_image(img, size=(224, 224), interpolation=cv2.INTER_AREA):
    """
    이미지를 주어진 크기로 리사이즈하고 보간법을 적용합니다.
    """
    img_resized = cv2.resize(img, size, interpolation=interpolation)
    return img_resized

# 이미지 축소 함수 (이전 단계와 동일)
def scale_image_to_screen(img, max_width=800, max_height=600):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        img_scaled = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img_scaled
    return img

# 1. 이미지 회전 함수 수정 (보간법 추가)
def rotate_image(img, angle, interpolation=cv2.INTER_AREA):
    """
    이미지를 지정된 각도로 회전하고 보간법을 적용합니다.
    """
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)  # 이미지의 중심 계산
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 회전 변환 행렬 생성
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=interpolation)
    return rotated_img

# 3. 이미지 크롭 함수 수정 (보간법 추가)
def crop_image(img, x, y, w, h, interpolation=cv2.INTER_CUBIC):
    """
    이미지를 지정된 위치에서 크롭하고 보간법을 적용하여 리사이즈합니다.
    """
    cropped_img = img[y:y+h, x:x+w]
    cropped_img_resized = resize_image(cropped_img, (w, h), interpolation=interpolation)
    return cropped_img_resized

# 2. 이미지 좌우 및 상하 반전 함수 (변경 없음)
def flip_image(img, flip_code):
    flipped_img = cv2.flip(img, flip_code)
    return flipped_img

# 추가된 부분: 이미지 저장 함수 구현
def save_augmented_image(image, save_dir, base_name, suffix):
    """
    증강된 이미지를 파일로 저장합니다.
    """
    save_path = os.path.join(save_dir, f"{base_name}_{suffix}.jpg")
    cv2.imwrite(save_path, image)
    print(f"이미지 저장됨: {save_path}")
    return save_path

# 이미지 디스플레이 함수 (창 위치 및 크기 조절 추가)
def display_images(original_img, transformed_img, title="Transformed Image"):
    """
    원본 이미지와 변환된 이미지를 화면에 표시합니다.
    """
    original_img_scaled = scale_image_to_screen(original_img)

    # 창 이름 설정
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    # 창 크기 및 위치 설정
    cv2.resizeWindow("Original Image", 400, 400)  # 원본 이미지 창 크기 조절
    cv2.resizeWindow(title, 400, 400)  # 변환된 이미지 창 크기 조절
    cv2.moveWindow("Original Image", 100, 100)  # 원본 이미지 창 위치 조절 (100, 100)
    cv2.moveWindow(title, 550, 100)  # 변환된 이미지 창 위치 조절 (550, 100)

    cv2.imshow("Original Image", original_img_scaled)  # 원본 이미지 창
    cv2.imshow(title, transformed_img)    # 변환된 이미지 창
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 새로운 추가 부분: 이미지 증강 조합 파이프라인 함수 구현
def augment_image_pipeline(image, pipeline):
    """
    이미지에 대한 증강 조합을 적용하는 파이프라인 함수입니다.
    """
    augmented_image = image.copy()  # 원본 이미지 복사
    for step in pipeline:
        method = step["method"]
        params = step["params"]
        augmented_image = method(augmented_image, **params)
    return augmented_image

# 테스트 함수 수정 (증강 데이터 저장 포함)
def test_augmentation_functions():
    img = load_image(fileName)  # 원본 이미지 불러오기
    img_resized = resize_image(img, (224, 224))  # 원본 이미지를 리사이즈

    # 1. 리사이즈된 이미지 회전 테스트 (10도)
    rotated_img = rotate_image(img_resized, 10)  # 보간법 기본값 사용 (INTER_AREA)
    display_images(img_resized, rotated_img, "Rotated Image (10 degrees)")
    save_augmented_image(rotated_img, dataAug, "mousepad_wood", "rotated_10")  # 이미지 저장 추가

    # 2. 리사이즈된 이미지 좌우 반전 테스트
    flipped_img_h = flip_image(img_resized, 1)  # 좌우 반전
    display_images(img_resized, flipped_img_h, "Horizontally Flipped Image")
    save_augmented_image(flipped_img_h, dataAug, "mousepad_wood", "flipped_horizontal")  # 이미지 저장 추가

    # 3. 리사이즈된 이미지 상하 반전 테스트
    flipped_img_v = flip_image(img_resized, 0)  # 상하 반전
    display_images(img_resized, flipped_img_v, "Vertically Flipped Image")
    save_augmented_image(flipped_img_v, dataAug, "mousepad_wood", "flipped_vertical")  # 이미지 저장 추가

    # 4. 리사이즈된 이미지 크롭 테스트
    cropped_img = crop_image(img_resized, 50, 50, 100, 100)  # 보간법 기본값 사용 (INTER_AREA)
    display_images(img_resized, cropped_img, "Cropped Image")
    save_augmented_image(cropped_img, dataAug, "mousepad_wood", "cropped_50_50_100_100")  # 이미지 저장 추가

# 증강 조합 테스트 함수 수정 (증강 데이터 저장 포함)
def test_augmentation_combinations():
    img = load_image(fileName)  # 원본 이미지 불러오기
    img_resized = resize_image(img, (224, 224))  # 원본 이미지를 리사이즈

    # 증강 조합 설정
    augmentation_pipeline = [
        {"method": rotate_image, "params": {"angle": 15}},  # 15도 회전, 보간법 설정 (기본값 INTER_AREA)
        {"method": flip_image, "params": {"flip_code": 1}},  # 좌우 반전
        {"method": crop_image, "params": {"x": 30, "y": 30, "w": 150, "h": 150}}  # (30, 30) 위치에서 150x150 크기로 크롭, 보간법 설정 (기본값 INTER_AREA)
    ]

    # 증강 조합 적용
    augmented_img = augment_image_pipeline(img_resized, augmentation_pipeline)

    # 증강된 이미지 표시
    display_images(img_resized, augmented_img, "Augmented Image Pipeline")
    save_augmented_image(augmented_img, dataAug, "mousepad_wood", "pipeline_augmented")  # 이미지 저장 추가

# 추가된 부분: 클래스 폴더에서 이미지 증강 작업 수행 함수 구현
def augment_images_in_class_folder(class_name):
    """
    주어진 클래스 폴더 내의 각 하위 폴더에 대해 이미지 증강 작업을 수행하는 함수입니다.
    """
    class_dir = os.path.join(dataAug, f'class_{class_name}')  # 클래스 폴더 경로
    for subdir in os.listdir(class_dir):  # 하위 폴더 탐색
        subdir_path = os.path.join(class_dir, subdir)
        for file in os.listdir(subdir_path):  # 각 이미지 파일에 대해
            file_path = os.path.join(subdir_path, file)
            img = load_image(file_path)  # 이미지 불러오기
            img_resized = resize_image(img, (224, 224))  # 이미지 리사이즈

            # 1. 리사이즈된 이미지 회전 테스트 (10도)
            rotated_img = rotate_image(img_resized, 10)  # 보간법 기본값 사용 (INTER_AREA)
            save_augmented_image(rotated_img, subdir_path, file.split('.')[0], "rotated_10")  # 이미지 저장

            # 2. 리사이즈된 이미지 좌우 반전 테스트
            flipped_img_h = flip_image(img_resized, 1)  # 좌우 반전
            save_augmented_image(flipped_img_h, subdir_path, file.split('.')[0], "flipped_horizontal")  # 이미지 저장

            # 3. 리사이즈된 이미지 상하 반전 테스트
            flipped_img_v = flip_image(img_resized, 0)  # 상하 반전
            save_augmented_image(flipped_img_v, subdir_path, file.split('.')[0], "flipped_vertical")  # 이미지 저장

            # 4. 리사이즈된 이미지 크롭 테스트
            cropped_img = crop_image(img_resized, 50, 50, 100, 100)  # 보간법 기본값 사용 (INTER_AREA)
            save_augmented_image(cropped_img, subdir_path, file.split('.')[0], "cropped_50_50_100_100")  # 이미지 저장

# 모든 클래스 폴더에 대해 증강 작업 수행하는 메인 함수
def main():
    for class_name in class_folders.keys():
        augment_images_in_class_folder(class_name)  # 각 클래스 폴더에 대해 이미지 증강 작업 수행

# 테스트 실행
#test_augmentation_functions()  # 원본 이미지에 대한 증강 테스트
main()  # 모든 클래스 폴더에 대해 증강 작업 수행
# check_folders()  # 폴더 및 파일 상태 확인 함수 호출
