import torch 
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import timm
import torch.nn as nn
from timm.layers import set_fused_attn
from config import *
# Attention Map을 가로채기 위한 Hook 설정
DEPTH = 12 # ViT-Tiny의 Block 수
IMG_SIZE = 224 # ViT에서 다루는 이미지 크기
attention_weights = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

set_fused_attn(False)

def get_vit_basic_model():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 1)
    model.load_state_dict(torch.load(CKPT_DIR / 'vit_polygon_model.pth', map_location=device))
    return model.to(device), transform

def hook_fn(module, input, output):
    global attention_weights
    # output 구조: (batch, num_heads, query_tokens, key_tokens)
    attention_weights = output
    
def visualize_attention(model, img_path, transform, target_diameter):
    global attention_weights
    # 이미지 로드 및 변환
    raw_image = Image.open(img_path).convert('RGB')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 마지막 블록의 attention 레이어에 hook 등록
    model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

    # 추론
    with torch.no_grad():
        prediction = model(input_tensor)

    # 5. Attention Map 처리
    # attention_weights: [1, num_heads, 197, 197] (197 = 14*14 patches + 1 CLS token)

    # [CLS] 토큰(index 0)이 나머지 패치들(index 1~)을 보는 점수만 추출
    # 모든 헤드의 평균을 냅니다.
    attentions = attention_weights[0, :, 0, 1:]
    attentions = attentions.mean(dim=0) # [196]

    # 14x14 그리드로 재배열
    w_featmap = input_tensor.shape[-2] // model.patch_embed.patch_size[0]
    h_featmap = input_tensor.shape[-1] // model.patch_embed.patch_size[1]
    attentions = attentions.reshape(w_featmap, h_featmap).detach().cpu().numpy()

    # 원래 이미지 크기로 확대
    attentions = cv2.resize(attentions, (IMG_SIZE, IMG_SIZE))

    # 6. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: 원본 이미지
    axes[0].imshow(raw_image.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title(f"Original (True Dia: {target_diameter:.2f})")
    axes[0].axis('off')

    # 오른쪽: Attention Overlay
    axes[1].imshow(raw_image.resize((IMG_SIZE, IMG_SIZE)))
    axes[1].imshow(attentions, cmap='jet', alpha=0.5) # 히트맵 오버레이
    axes[1].set_title(f"Attention Map (Pred: {prediction.item():.2f})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# 훅 함수: 인덱스 고민 없이 그냥 리스트에 추가
def hook_fn_consecutive(module, input, output):
    global attention_weights_list
    # output을 CPU로 옮겨서 저장 (GPU 메모리 절약)
    attention_weights_list.append(output.detach().cpu())

def visualize_attention_consecutive(model, img_path, transform, target_diameter):
    global attention_weights_list
    
    # 1. 리스트 초기화 (이거 안 하면 이전 실행 결과랑 섞임)
    attention_weights_list = []
    
    # 2. 이미지 로드
    raw_image = Image.open(img_path).convert('RGB')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 3. 모든 블록에 훅 등록
    hooks = []
    for block in model.blocks:
        # timm 버전에 따라 attn.attn_drop 또는 attn_drop 등 위치 확인 필요
        # 보통 output을 뱉는 곳에 거는 것이 안전
        h = block.attn.attn_drop.register_forward_hook(hook_fn_consecutive)
        hooks.append(h)

    # 4. 추론 (이때 훅이 순서대로 실행되며 list에 쌓임)
    with torch.no_grad():
        prediction = model(input_tensor)

    # 5. 훅 제거 (중요! 안 지우면 계속 중복 등록됨)
    for h in hooks:
        h.remove()

    # 6. 시각화 처리
    # attention_weights_list에는 [Layer0_out, Layer1_out, ..., Layer11_out] 순서로 들어있음
    
    # 시각화용 캔버스 설정
    # Layer 개수만큼 옆으로 긺 (DEPTH+1: 원본 포함)
    fig, axes = plt.subplots(1, DEPTH + 1, figsize=(3 * (DEPTH + 1), 3))

    # (1) 맨 왼쪽: 원본 이미지
    axes[0].imshow(raw_image.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title(f"Original\n({target_diameter:.2f})")
    axes[0].axis('off')

    # (2) 나머지: 각 레이어별 Attention Map
    for i in range(DEPTH):
        # i번째 레이어의 Attention 가져오기
        att = attention_weights_list[i] # [1, num_heads, tokens, tokens]
        
        # [CLS] 토큰(0번)이 나머지 패치들(1번~)을 보는 점수 추출
        # (Batch, Heads, CLS, Patches) -> (Heads, Patches)
        att = att[0, :, 0, 1:] 
        
        # 모든 헤드 평균
        att = att.mean(dim=0) # [196]
        
        # 2D 그리드로 변환 (14x14)
        w_featmap = input_tensor.shape[-2] // model.patch_embed.patch_size[0]
        h_featmap = input_tensor.shape[-1] // model.patch_embed.patch_size[1]
        
        att = att.reshape(w_featmap, h_featmap).numpy()
        
        # 이미지 크기로 확대
        att = cv2.resize(att, (IMG_SIZE, IMG_SIZE))

        # 그리기
        axes[i+1].imshow(raw_image.resize((IMG_SIZE, IMG_SIZE)))
        axes[i+1].imshow(att, cmap='jet', alpha=0.5)
        axes[i+1].set_title(f"Layer {i+1}")
        axes[i+1].axis('off')

    plt.suptitle(f"Prediction: {prediction.item():.2f}", fontsize=16)
    plt.tight_layout()
    plt.show()