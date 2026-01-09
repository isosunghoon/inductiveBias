import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
import timm
from timm.layers import set_fused_attn
from vit_baseline.visualization_attention import visualize_attention 
from config import * 

# 1. 환경 설정 및 데이터 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224  # 사전 학습된 ViT는 보통 224를 사용하므로 리사이즈합니다.

# 시각화를 위해 Fused Attention 기능을 끕니다 (필수)
set_fused_attn(False)

# 2. 모델 준비 (회귀용 ViT)
def get_model():
    # 학생 수준에서 가벼운 'vit_tiny_patch16_224' 추천
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

    # 입력 채널 수정 (Grayscale 1채널 -> RGB 3채널로 변환해 쓸 것이므로 그대로 둬도 됨)
    # 회귀 문제이므로 마지막 출력을 1로 변경
    model.head = nn.Linear(model.head.in_features, 1)
    return model.to(device).eval()

model = get_model()

# 3. 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 4. 데이터 시각화 실행 
try:
    df = pd.read_csv(LABEL_FILE)
    sample_row = df.iloc[:5]
    for _, row in sample_row.iterrows():
        visualize_attention(model, DATA_DIR / row['filename'], transform, row['diameter'])
except FileNotFoundError:
    print("데이터셋이 없습니다. gen_dataset.py를 먼저 실행해 주세요.")
