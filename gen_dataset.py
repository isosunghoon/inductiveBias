import numpy as np
from pathlib import Path
from PIL import Image
import csv

N = 100
NUM_SAMPLES = 10000
POLY_RANGE = [4, 10] # 몇각형인지
CENTER_MARGIN, CENTER_STD = 0.2, 0.3

OUT_DIR = Path("./dataset")
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def generate_polygon():
  m = np.random.randint(*POLY_RANGE)

  cx, cy = 1 + 1e6, 1 + 1e6
  while abs(cx) > 1 - CENTER_MARGIN or abs(cy) > 1 - CENTER_MARGIN:
    cx, cy = np.random.normal(scale = CENTER_STD, size=2)
  
  angles = np.sort(np.random.uniform(0, 2*np.pi, m))
  max_r = min(1-abs(cx), 1-abs(cy))
  min_r = 0.5*max_r # hmm...
  r = np.random.uniform(min_r, max_r, m)

  return np.stack([cx + r*np.cos(angles), cy + r*np.sin(angles)], axis=1)


def pixel_to_coord(i, j, n):
    x = 2.0 * (i + 0.5) / n - 1.0
    y = 2.0 * (j + 0.5) / n - 1.0
    return x, y

def point_in_poly(x, y, poly): # 다각형 외부의 어떤 점과 (x,y)를 이은 선분과 다각형이 홀수번 만나면 내부 점임 (ray casting)
  inside = False
  m = len(poly)

  # 외부 점을 (inf, y)라고 하자
  for i in range(m):
    x1, y1 = poly[i]
    x2, y2 = poly[(i + 1) % m]
    if (y1 <= y < y2) or (y2 < y <= y1): # 꼭짓점 처리 (한쪽만 등호포함)
      x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
      if x <= x_intersect:
        inside = not inside
  
  return inside

def polygon_to_img(poly, n):
  minx, maxx = poly[:,0].min(), poly[:,0].max()
  miny, maxy = poly[:,1].min(), poly[:,1].max()
  minx, maxx = max(int(np.floor((minx + 1) * 0.5 * n)), 0), min(int(np.ceil ((maxx + 1) * 0.5 * n)), n - 1)
  miny, maxy = max(int(np.floor((miny + 1) * 0.5 * n)), 0), min(int(np.ceil ((maxy + 1) * 0.5 * n)), n - 1)

  img = np.zeros((n, n), dtype=np.uint8)

  for i in range(minx, maxx + 1):
    for j in range(miny, maxy + 1):
      x, y = pixel_to_coord(i, j, n)
      if point_in_poly(x, y, poly):
        img[i, j] = 1
  
  return img

def calc_diameter(poly):
  ans = 0
  m = len(poly)
  for i in range(m):
    for j in range(i+1, m):
      ans = max(ans, np.linalg.norm(poly[i] - poly[j]))
  return ans

labels = []

for i in range(1, NUM_SAMPLES + 1):
  poly = generate_polygon()
  img = Image.fromarray(polygon_to_img(poly, N) * 255)
  diameter = calc_diameter(poly)
  filename = f"{i}.png"
  img.save(IMG_DIR / filename)
  labels.append((filename, diameter))

  if i / NUM_SAxMPLES
  print(f"sample {i} / {NUM_SAMPLES} Done!")

f = open(OUT_DIR / "labels.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["filename", "diameter"])
writer.writerows(labels)