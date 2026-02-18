# Created via ChatGPT, temporary

import os
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# =========================
# 설정
# =========================
DATASET_DIR = Path("./dataset")
IMG_DIR = DATASET_DIR / "images"
LABEL_FILE = DATASET_DIR / "labels.csv"

IMAGE_SIZE = 50
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
TRAIN_RATIO = 0.8
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =========================
# Dataset
# =========================
class PolygonDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.img_dir = self.dataset_dir / "images"

        self.samples = []
        with open(self.dataset_dir / "labels.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(
                    (row["filename"], float(row["diameter"]))
                )

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (1,H,W), float32 in [0,1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, y = self.samples[idx]
        img = Image.open(self.img_dir / fname).convert("L")
        x = self.transform(img)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# =========================
# Model
# =========================
class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)  # (B,64,1,1)
        )

        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x.squeeze(1)

# =========================
# Load dataset & split
# =========================
dataset = PolygonDataset(DATASET_DIR)

train_size = int(TRAIN_RATIO * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE)

# =========================
# Init model
# =========================
model = CNNRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# =========================
# Train / Eval
# =========================
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- test ----
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            test_loss += loss.item() * x.size(0)

    test_loss /= len(test_loader.dataset)

    print(
        f"[Epoch {epoch:02d}] "
        f"Train MSE: {train_loss:.6f} | "
        f"Test MSE: {test_loss:.6f}"
    )

# =========================
# Done
# =========================
print("Training finished.")
