import torch
import torch.nn as nn
import pandas as pd
import timm
from torchvision import transforms
from timm.layers import set_fused_attn
from vit_baseline.visualization_attention import get_vit_basic_model, visualize_attention 
from config import * 

loaded_model, transform = get_vit_basic_model()

# 데이터 시각화 실행 
try:
    df = pd.read_csv(LABEL_FILE)
    sample_row = df.iloc[:5]
    for _, row in sample_row.iterrows():
        visualize_attention(loaded_model, DATA_DIR / row['filename'], transform, row['diameter'])
except FileNotFoundError:
    print("데이터셋이 없습니다. gen_dataset.py를 먼저 실행해 주세요.")