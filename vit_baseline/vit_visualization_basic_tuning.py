import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import timm
import os 
from PIL import Image
from torchvision import transforms
from timm.layers import set_fused_attn
from vit_baseline.visualization_attention import visualize_attention 
from config import * 

# 1. 설정 및 하이퍼파라미터
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

# 시각화를 위해 Fused Attention 비활성화
set_fused_attn(False)

# 2. 커스텀 데이터셋 정의 
class PolygonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 3. 모델 준비 (ViT-Tiny)
def get_regression_model():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 1)
    return model.to(device)

loaded_model = get_regression_model()  # 1. 모델 구조(껍데기) 생성
loaded_model.load_state_dict(torch.load(CKPT_DIR / 'vit_polygon_model.pth')) # 2. 가중치 주입

# 4. 데이터 시각화 실행 
try:
    df = pd.read_csv(LABEL_FILE)
    sample_row = df.iloc[:5]
    for _, row in sample_row.iterrows():
        visualize_attention(loaded_model, DATA_DIR / row['filename'], transform, row['diameter'])
except FileNotFoundError:
    print("데이터셋이 없습니다. gen_dataset.py를 먼저 실행해 주세요.")