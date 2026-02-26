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
    N = IMG_SIZE
    
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

def make_diverse_shapes(target_area=1500):
    """
    동일 면적, 다른 직경을 가진 6종의 도형을 생성합니다.
    """
    N = IMG_SIZE
    results = []
    
    # 도형 설정: (이름, 이미지 생성 함수)
    shape_configs = [
        ("circle", _create_circle_img),
        ("square", _create_square_img),
        ("rect_long", lambda n, a: _create_rect_img(n, a, ratio=3.5)),
        ("ellipse", lambda n, a: _create_ellipse_img(n, a, ratio=2.0)),
        ("triangle", _create_triangle_img),
        ("diamond", _create_diamond_img)
    ]

    for name, create_func in shape_configs:
        # 1. 이미지 생성 및 픽셀 직경(Max Span) 계산
        img, diameter_px = create_func(N, target_area)
        
        # 2. 정규화된 직경 계산 (112px = 2.0)
        diameter = (diameter_px / N) * 2.0
        
        # 3. 이미지 저장
        saved_path = TRANSFORM_DATA_DIR / f"{name}_area_{target_area}.png"
        cv2.imwrite(str(saved_path), img)
        
        results.append((str(saved_path), diameter))
        
    return results

# --- 도형별 생성 보조 함수 ---

def _create_circle_img(N, area):
    img = np.zeros((N, N), dtype=np.uint8)
    radius_px = np.sqrt(area / np.pi)
    cv2.circle(img, (N // 2, N // 2), int(round(radius_px)), 255, -1)
    return img, radius_px * 2

def _create_square_img(N, area):
    img = np.zeros((N, N), dtype=np.uint8)
    side = int(np.sqrt(area))
    start = (N // 2) - (side // 2)
    cv2.rectangle(img, (start, start), (start + side, start + side), 255, -1)
    return img, side * (2 ** 0.5)

def _create_rect_img(N, area, ratio):
    img = np.zeros((N, N), dtype=np.uint8)
    w = int(np.sqrt(area * ratio))
    h = int(area / w)
    start_x, start_y = (N // 2) - (w // 2), (N // 2) - (h // 2)
    cv2.rectangle(img, (start_x, start_y), (start_x + w, start_y + h), 255, -1)
    return img, (w ** 2 + h ** 2) ** 0.5

def _create_ellipse_img(N, area, ratio):
    img = np.zeros((N, N), dtype=np.uint8)
    # Area = pi * a * b, a = b * ratio
    b = np.sqrt(area / (np.pi * ratio))
    a = b * ratio
    cv2.ellipse(img, (N // 2, N // 2), (int(a), int(b)), 0, 0, 360, 255, -1)
    return img, a * 2

def _create_triangle_img(N, area):
    img = np.zeros((N, N), dtype=np.uint8)
    # 정삼각형 면적 S = (sqrt(3)/4) * s^2
    side = np.sqrt((4 * area) / np.sqrt(3))
    height = (np.sqrt(3) / 2) * side
    
    # 중앙 배치를 위한 정점 계산
    pts = np.array([
        [N // 2, int(N // 2 - (2/3 * height))],
        [int(N // 2 - side // 2), int(N // 2 + (1/3 * height))],
        [int(N // 2 + side // 2), int(N // 2 + (1/3 * height))]
    ], np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img, side

def _create_diamond_img(N, area):
    img = np.zeros((N, N), dtype=np.uint8)
    # 마름모 면적 S = (d1 * d2) / 2. d1 = d2 * 1.5 (약간 길쭉하게)
    d2 = np.sqrt((2 * area) / 1.5)
    d1 = d2 * 1.5
    
    pts = np.array([
        [N // 2, int(N // 2 - d1 // 2)], # 상
        [int(N // 2 + d2 // 2), N // 2], # 우
        [N // 2, int(N // 2 + d1 // 2)], # 하
        [int(N // 2 - d2 // 2), N // 2]  # 좌
    ], np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img, d1

def make_fixed_diagonal_shapes(D=80):
    """
    모든 도형의 직경(최대 Span, 대각선)을 D로 고정하고 면적을 다양하게 생성합니다.
    """
    N = IMG_SIZE
    results = []
    
    # 도형 설정: (이름, 생성 함수)
    shape_configs = [
        ("circle", _draw_circle_diag),             # 원 (지름 = D)
        ("square", _draw_square_diag),             # 정사각형 (대각선 = D)
        ("rect_mid", lambda n, d: _draw_rect_diag(n, d, h_ratio=0.4)), # 중간 직사각형
        ("rect_thin", lambda n, d: _draw_rect_diag(n, d, h_ratio=0.2)), # 얇은 직사각형
        ("rect_extreme", lambda n, d: _draw_rect_diag(n, d, h_px=4)),  # 극단적 직사각형
        ("diamond", _draw_diamond_diag)            # 마름모 (긴 대각선 = D)
    ]

    for name, create_func in shape_configs:
        img = create_func(N, D)
        
        area = np.sum(img == 255)
        diameter_norm = (D / N) * 2.0  # 정규화된 직경 (2.0 기준)
        
        saved_path = TRANSFORM_DATA_DIR / f"{name}_diag_{D}_area_{area}.png"
        cv2.imwrite(str(saved_path), img)
        
        results.append((str(saved_path), diameter_norm))
        
    return results

# --- 보조 함수 (모든 도형은 중앙 배치 및 내부 채움) ---

def _draw_circle_diag(N, D):
    # 원은 지름이 곧 직경
    img = np.zeros((N, N), dtype=np.uint8)
    cv2.circle(img, (N // 2, N // 2), int(D // 2), 255, -1)
    return img

def _draw_square_diag(N, D):
    # 정사각형 대각선 D = sqrt(2) * side  => side = D / sqrt(2)
    img = np.zeros((N, N), dtype=np.uint8)
    side = int(D / np.sqrt(2))
    start = (N // 2) - (side // 2)
    cv2.rectangle(img, (start, start), (start + side, start + side), 255, -1)
    return img

def _draw_rect_diag(N, D, h_ratio=None, h_px=None):
    # 직사각형 대각선 D^2 = w^2 + h^2  => w = sqrt(D^2 - h^2)
    img = np.zeros((N, N), dtype=np.uint8)
    
    if h_px is not None:
        h = h_px
    else:
        # D의 일정 비율을 높이로 설정 (단, D보다 작아야 함)
        h = int(D * h_ratio)
    
    w = int(np.sqrt(D**2 - h**2))
    
    start_x, start_y = (N // 2) - (w // 2), (N // 2) - (h // 2)
    cv2.rectangle(img, (start_x, start_y), (start_x + w, start_y + h), 255, -1)
    return img

def _draw_diamond_diag(N, D):
    # 마름모의 경우 두 대각선 중 긴 쪽을 D로 설정
    img = np.zeros((N, N), dtype=np.uint8)
    d1 = D
    d2 = int(D * 0.5) # 짧은 대각선
    
    pts = np.array([
        [N // 2, N // 2 - d1 // 2], [N // 2 + d2 // 2, N // 2],
        [N // 2, N // 2 + d1 // 2], [N // 2 - d2 // 2, N // 2]
    ], np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img 

def make_two_circles(target_diameters, radius = 5):
    """
    입력받은 직경 목록에 맞춰 서로 떨어진 두 개의 작은 원 이미지를 생성합니다.
    - target_diameters: 생성할 목표 직경(px)들의 리스트 (예: [30, 50, 80])
    - 정의된 직경 = 두 원의 중심 사이 거리 + 10px (반지름이 5px이므로)
    """
    N = IMG_SIZE
    RADIUS = radius
    results = []
    
    # 원이 이미지 밖으로 나가지 않기 위한 안전 영역 설정
    # 여유 있게 반지름 + 2px 정도 안쪽에 중심이 오도록 함
    margin = RADIUS + 2
    low = margin
    high = N - margin

    for target_D in target_diameters:
        # 1. 목표 직경 달성을 위한 중심 간 거리(d) 역산
        # target_D = distance + 10  => distance = target_D - 10
        required_dist = target_D - (RADIUS * 2)
        
        if required_dist <= 0:
            print(f"경고: 직경 {target_D}px은 두 원을 분리하기에 너무 작습니다. 건너뜁니다.")
            continue
            
        img = np.zeros((N, N), dtype=np.uint8)
        
        # 2. 랜덤 위치 선정 (Rejection Sampling)
        # 첫 번째 원을 놓고, 두 번째 원이 이미지 안에 들어올 때까지 시도
        max_attempts = 1000
        placed = False
        
        for _ in range(max_attempts):
            # 첫 번째 원의 중심 (cx1, cy1) 랜덤 선택
            cx1 = random.randint(low, high)
            cy1 = random.randint(low, high)
            
            # 랜덤 각도 선택 (0 ~ 2pi)
            angle = random.uniform(0, 2 * np.pi)
            
            # 거리와 각도를 이용해 두 번째 원의 중심 (cx2, cy2) 계산
            cx2 = int(cx1 + required_dist * np.cos(angle))
            cy2 = int(cy1 + required_dist * np.sin(angle))
            
            # 두 번째 원이 안전 영역 안에 있는지 확인
            if low <= cx2 <= high and low <= cy2 <= high:
                # 배치 성공! 그리기 수행
                cv2.circle(img, (cx1, cy1), RADIUS, 255, -1)
                cv2.circle(img, (cx2, cy2), RADIUS, 255, -1)
                placed = True
                break
        
        if not placed:
            print(f"경고: 직경 {target_D}px 배치를 {max_attempts}번 시도했으나 실패했습니다.")
            continue

        # 3. 정규화된 직경 계산 (112px = 2.0 기준)
        diameter_norm = (target_D / N) * 2.0
        
        # 4. 이미지 저장
        # 파일명에 중심거리(dist)도 같이 기록
        saved_path = TRANSFORM_DATA_DIR / f"two_circles_diam_{target_D}_dist_{int(required_dist)}.png"
        cv2.imwrite(str(saved_path), img)
        
        results.append((str(saved_path), diameter_norm))
        
    return results


functions_dict = {
    "boundary": extract_boundary,
    "inverted": extract_inverted,
    "rotated": extract_rotated,
    "circle": make_circle,
    "noise": extract_noise,
    "noise_internal": extract_noise_internal,
    "blurred": extract_blurred,
    "fixed_area": make_diverse_shapes,
    "fixed_diagonal": make_fixed_diagonal_shapes,
    "two_circles": make_two_circles,
}

if __name__ == "__main__":
    # 실행 및 시각화
    image_path = "dataset/images/1.png"  # 파일 경로를 입력하세요
    transformed_path = functions_dict["noise_internal"](image_path)
    # # transformed_path, diameter = functions_dict["circle"](0.5)

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