from pathlib import Path

BASE_DIR = Path(".") # 현재 디렉토리 기준으로 사용
DATA_DIR = BASE_DIR / "dataset/images"
CKPT_DIR = BASE_DIR / "ckpt"
LABEL_FILE = BASE_DIR / "dataset/labels.csv"
TRANSFORM_DATA_DIR = BASE_DIR / "dataset/transform"

IMG_SIZE = 112