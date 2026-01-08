import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
import timm
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm 
from config import * 


# 1. 설정 및 하이퍼파라미터
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 15
LR = 2e-4

# 2. 커스텀 데이터셋 정의 (기존과 동일)
class PolygonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 데이터셋 접근 시 인덱스 에러 방지를 위해 iloc 사용 주의 (Subset 사용 시 인덱스 매핑됨)
        # 여기서는 DataFrame 전체를 로드하고 인덱싱하므로 문제 없음
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

model = get_regression_model()

# 4. 학습 과정
if not os.path.exists(LABEL_FILE):
    print("데이터셋이 없습니다. gen_dataset.py를 실행하여 데이터를 먼저 생성하세요.")
    exit()

# 전체 데이터셋 로드
# full_dataset = PolygonDataset(LABEL_FILE, DATA_DIR, transform=transform)
full_dataset = PolygonDataset("/content/labels.csv", "/content/images", transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_data, test_data = random_split(full_dataset, [train_size, test_size])

print(f"데이터셋 분할 완료: Train {len(train_data)}장, Test {len(test_data)}장")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2, pin_memory = True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 2, pin_memory = True) # Test용 로더

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

scheduler = OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy='cos'
)

train_losses = [] # 그래프를 그리기 위한 Loss 기록용 리스트
test_losses = [] # 그래프를 그리기 위한 Test Loss 기록용 리스트

print(f"학습 시작 (Device: {device})...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # tqdm으로 학습 진행바 시각화
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")
    
    # 에포크별 평균 Loss 저장
    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    
    # Validation (Test) Loop
    model.eval() # 평가 모드로 전환 (Dropout, BatchNorm 등 고정)
    running_test_loss = 0.0
    
    with torch.no_grad(): # 기울기 계산 끔 (메모리 절약, 속도 향상)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
    
    epoch_test_loss = running_test_loss / len(test_loader)
    test_losses.append(epoch_test_loss)

    # Epoch 결과 출력
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")

# 학습 종료 후 Loss 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss', marker='o')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# 모델 저장 
torch.save(model.state_dict(), CKPT_DIR / 'vit_polygon_model.pth')
print(f"모델 저장 완료: {CKPT_DIR / 'vit_polygon_model.pth'}")