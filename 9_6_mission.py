import cv2
import numpy as np

# 이미지 불러오기
image_path = 'mission/01.png'  # 첫 번째 이미지 (노이즈가 있는 이미지)
src = cv2.imread(image_path)

if src is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 1. 노이즈 제거 (Median Blur)
# 커널 크기는 3, 5, 7 등 홀수로 설정 가능
denoised = cv2.medianBlur(src, 5)

# 2. 명암 대비 향상 (MinMaxLoc, Normalize)
# 이미지를 LAB 색 공간으로 변환하여 명암 대비를 조정
lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# 밝기 채널(L)에서 최소 및 최대 값 찾기
min_val, max_val, _, _ = cv2.minMaxLoc(l)

# 밝기 채널을 0~255 사이로 정규화
l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)

# 조정된 L 채널을 다시 병합하고 BGR로 변환
lab_adjusted = cv2.merge((l_norm, a, b))
contrast_enhanced = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

# 3. 컬러 보정 (컬러 조정 그대로 적용)
# LAB 색 공간에서 조정한 이미지로 작업 계속 진행

# 4. 밝기 조정 (AddWeighted)
# 두 이미지를 가중합으로 밝기 조절
alpha = 1.2  # 밝기 조정 정도
beta = 50    # 추가적인 밝기 증가
bright_adjusted = cv2.convertScaleAbs(contrast_enhanced, alpha=alpha, beta=beta)

# 5. 세부 정보 보강 (Sharpening)
# 샤프닝 커널 정의
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])

# 필터를 사용하여 이미지 샤프닝 적용
sharpened_image = cv2.filter2D(bright_adjusted, -1, sharpening_kernel)


# 결과 출력 (원본과 보정된 이미지)
cv2.imshow('Original Image', src)
cv2.imshow('Denoised and Enhanced Image', sharpened_image)


# 결과를 저장
cv2.imwrite('mission/enhanced_image.png', sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

