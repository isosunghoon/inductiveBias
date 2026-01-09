import torch
import torch.nn as nn
import pandas as pd
import cv2
import numpy as np
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from config import * 
from vit_explainability.prior_research.ViT_LRP import vit_tiny_patch16_224 as vit_LRP
from vit_explainability.prior_research.ViT_explanation_generator import LRP

# 설정 및 하이퍼파라미터
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
use_thresholding = False

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# initialize ViT pretrained
model = vit_LRP(pretrained=True).to(device)
model.load_state_dict(torch.load(CKPT_DIR / 'vit_polygon_model.pth'))
model.eval()
attribution_generator = LRP(model)

def generate_visualization(original_image, device, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).to(device), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
      transformer_attribution = transformer_attribution * 255
      transformer_attribution = transformer_attribution.astype(np.uint8)
      ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      transformer_attribution[transformer_attribution == 255] = 1

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def visualize_attention_explainability(model, img_path, transform, target_diameter):
    raw_image = Image.open(img_path).convert('RGB')
    original_image = transform(raw_image)
    vis = generate_visualization(original_image, device)

    prediction = model(original_image.unsqueeze(0).to(device))
    
        # 6. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: 원본 이미지
    axes[0].imshow(raw_image.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title(f"Original (True Dia: {target_diameter:.2f})")
    axes[0].axis('off')

    # 오른쪽: Attention Overlay
    axes[1].imshow(vis)
    axes[1].set_title(f"Attention Map (Pred: {prediction.item():.2f})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(LABEL_FILE)
    sample_row = df.iloc[:5]
    for _, row in sample_row.iterrows():
        visualize_attention_explainability(model, DATA_DIR / row['filename'], transform, row['diameter'])