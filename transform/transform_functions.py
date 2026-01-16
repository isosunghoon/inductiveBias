import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from config import *

def extract_boundary(image_path):
    # 1. 이미지 로드 (그레이스케일)
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 이미지가 112x112가 아닐 경우를 대비해 크기 확인 (선택 사항)
    # img = cv2.resize(img, (112, 112))

    # 2. 이진화 확인 (이미 binary라면 생략 가능하지만 안전을 위해 수행)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 3. 커널 생성 (3x3 크기)
    kernel = np.ones((3, 3), np.uint8)

    # 4. 침식 연산 (Erosion): 도형을 1픽셀씩 깎아냅니다.
    eroded_img = cv2.erode(binary_img, kernel, iterations=1)

    # 5. 경계 추출: 원본 - 침식 이미지
    transformed_img = cv2.subtract(binary_img, eroded_img) 

    # 6. 경계 이미지 저장
    name = "boundary"
    saved_path = TRANSFORM_DATA_DIR / (image_path.split("/")[-1].split(".")[0] + "_" + name + ".png")
    cv2.imwrite(saved_path, transformed_img )

    return saved_path 

def extract_inverted(image_path):
    # 1. 이미지 로드 (그레이스케일)
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. 이미지 반전
    transformed_img = cv2.bitwise_not(img) 

    name = "inverted"
    saved_path = TRANSFORM_DATA_DIR / (image_path.split("/")[-1].split(".")[0] + "_" + name + ".png")
    cv2.imwrite(saved_path, transformed_img)
    
    return saved_path 

def extract_rotated(image_path):
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 예: 45도 회전
    transformed_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    name = "rotated"
    saved_path = TRANSFORM_DATA_DIR / (image_path.split("/")[-1].split(".")[0] + "_" + name + ".png") 
    cv2.imwrite(str(saved_path), transformed_img)
    
    return saved_path

def extract_noise(image_path, noise_ratio = 0.02):
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이미지가 이미 0과 255로 구성되어 있다고 가정
    noisy_img = img.copy()
    
    # 무작위 행렬 생성 (0~1 사이)
    random_matrix = np.random.random(img.shape)
    
    noisy_img[random_matrix < noise_ratio] = 255 - noisy_img[random_matrix < noise_ratio]
    
    name = f"noise_{noise_ratio:.2f}"
    saved_path = TRANSFORM_DATA_DIR / (image_path.split("/")[-1].split(".")[0] + "_" + name + ".png")
    cv2.imwrite(saved_path, noisy_img)
    
    return saved_path 

def extract_noise_internal(image_path, noise_ratio = 0.05):
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1. 하얀색(255) 픽셀의 좌표를 모두 찾음
    white_pixels = np.argwhere(img == 255)
    num_white = len(white_pixels)
    
    # 2. 노이즈를 넣을 픽셀 개수 계산 (하얀색 면적 대비)
    num_noise = int(num_white * noise_ratio)
    
    # 3. 하얀색 픽셀 인덱스 중 무작위로 num_noise개 선택
    noise_indices = np.random.choice(num_white, num_noise, replace=False)
        
    # 4. 선택된 좌표의 값을 0(검은색)으로 변경
    for idx in noise_indices:
        y, x = white_pixels[idx]
        img[y, x] = 0
    
    name = f"noise_internal_{noise_ratio:.2f}"
    saved_path = TRANSFORM_DATA_DIR / (image_path.split("/")[-1].split(".")[0] + "_" + name + ".png")
    cv2.imwrite(saved_path, img)
    
    return saved_path

def extract_blurred(image_path, ksize=5):
    """
    이미지에 가우시안 블러를 적용하여 경계를 흐릿하게 만듭니다.
    - ksize: 커널 크기 (홀수여야 함. 숫자가 클수록 더 많이 흐려짐)
    """
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 가우시안 블러 적용
    blurred_img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 저장 경로 생성
    name = f"blurred_k{ksize}"
    saved_path = TRANSFORM_DATA_DIR / (image_path.split("/")[-1].split(".")[0] + "_" + name + ".png")
    cv2.imwrite(saved_path, blurred_img)
    
    return saved_path

def make_circle(ratio):
    """
    이미지 한 변(112px) 대비 직경 비율을 입력받아 원을 생성합니다.
    - ratio: 0.0 ~ 1.0 사이의 실수 (예: 0.5면 직경이 56픽셀)
    - 원이 이미지 경계 밖으로 나가지 않도록 랜덤 위치를 선정합니다.
    """
    N = 112
    
    # 1. 픽셀 단위 직경 및 반지름 계산
    diameter_px = ratio * N
    radius_px = diameter_px / 2.0
    
    # 2. 이미지 생성 (검은색 배경)
    img = np.zeros((N, N), dtype=np.uint8)
    
    # 3. 랜덤 중심 좌표 (cx, cy) 결정
    # 인덱스는 0~111까지이므로, 원이 잘리지 않으려면 
    # [반지름]과 [111 - 반지름] 사이에서 중심이 결정되어야 합니다.
    low = int(np.ceil(radius_px))
    high = int(np.floor((N - 1) - radius_px))
    
    if low >= high:
        # 반지름이 너무 커서 여유 공간이 없을 경우 중앙 배치
        cx, cy = N // 2, N // 2
    else:
        cx = random.randint(low, high)
        cy = random.randint(low, high)
    
    # 4. 원 그리기 (하얀색, 내부 채움)
    # 두께를 -1로 설정하면 내부가 채워진 도형이 그려집니다.
    cv2.circle(img, (cx, cy), int(round(radius_px)), 255, -1)
    
    # 5. 직경(Diameter) 계산 (Normalized Coordinate 기준)
    # 전체 112px = 2.0 단위이므로, ratio 1.0일 때 diameter는 2.0입니다.
    diameter = ratio * 2.0
    
    # 6. 이미지 저장
    saved_path = TRANSFORM_DATA_DIR / f"circle_ratio_{ratio:.2f}.png"
    cv2.imwrite(str(saved_path), img)
    
    return saved_path, diameter


functions_dict = {
    "boundary": extract_boundary,
    "inverted": extract_inverted,
    "rotated": extract_rotated,
    "circle": make_circle,
    "noise": extract_noise,
    "noise_internal": extract_noise_internal,
    "blurred": extract_blurred,
}

if __name__ == "__main__":
    # 실행 및 시각화
    image_path = "dataset/images/1.png"  # 파일 경로를 입력하세요
    transformed_path = functions_dict["noise_internal"](image_path)
    # transformed_path, diameter = functions_dict["circle"](0.5)

    # 결과 확인
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Binary Image")
    plt.imshow(cv2.imread(image_path), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Extracted Boundary")
    plt.imshow(cv2.imread(transformed_path), cmap='gray')

    plt.show()

    # 필요 시 이미지 저장
    # cv2.imwrite('boundary_result.png', edge)