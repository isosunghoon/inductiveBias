# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'
import argparse
import os
import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns
#   Set figure parameters
large = 24; med = 24; small = 24
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
# plt.rc('font', **{'family': 'Times New Roman'})
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
from analysis.erf import make_erf

def heatmap(data, camp='RdYlGn', figsize=(10, 10.75), ax=None, save_path=None):
    plt.figure(figsize=figsize, dpi=40)

    # axes_grid1 / custom colorbar 없이, seaborn 기본 colorbar만 사용 (cbar=True)
    ax = sns.heatmap(data, xticklabels=False, yticklabels=False, cmap=camp, center=0,
        annot=False, ax=ax, cbar=True, annot_kws={"size": 24}, fmt='.2f',)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)


def get_rectangle(data, thresh):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h // 2):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
    return None


def analyze_erf(npy_path, save_image_path):
    data = np.load(npy_path)
    print(np.max(data))
    print(np.min(data))
    data = np.log10(data + 1)       #   the scores differ in magnitude. take the logarithm for better readability
    data = data / np.max(data)      #   rescale to [0,1] for the comparability among models
    print('======================= the high-contribution area ratio =====================')
    for thresh in [0.2, 0.3, 0.5, 0.99]:
        result = get_rectangle(data, thresh)
        if result is None:
            print('thresh, rectangle side length, area ratio: ', thresh, 'N/A (no rectangle exceeded threshold)')
        else:
            side_length, area_ratio = result
            print('thresh, rectangle side length, area ratio: ', thresh, side_length, area_ratio)
    heatmap(data, save_path=save_image_path)
    print('heatmap saved at ', save_image_path)


def visualize_erf(save_path):
    """
    save_path: 모델 체크포인트와 yaml 들이 있는 폴더 경로.
               이 폴더 안에 'erf.npy'와 'erf.png'를 생성한다.
    """
    os.makedirs(save_path, exist_ok=True)
    npy_path = make_erf(save_path)
    png_path = os.path.join(save_path, "erf.png")
    analyze_erf(npy_path, png_path)


def main():
    """
    args.project: output 폴더 내의 프로젝트 이름 (예: 'base_model_exp')
                  실제 경로는 'output/{project}' 로 간주하고,
                  그 안의 모든 하위 폴더에 대해 visualize_erf를 실행한다.
    """
    parser = argparse.ArgumentParser('Script for making & visualizing the ERF for all runs in a project', add_help=False)
    parser.add_argument('--project', type=str, required=True, help="output 폴더 내의 프로젝트 이름 (예: 'base_model_exp')",)
    args = parser.parse_args()

    project_root = os.path.join('output', args.project)
    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"Project directory not found: {project_root}")

    # project_root 아래의 모든 디렉터리에 대해 visualize_erf 실행 (에러 시 해당 폴더만 건너뛰고 계속)
    for name in sorted(os.listdir(project_root)):
        run_dir = os.path.join(project_root, name)
        if not os.path.isdir(run_dir):
            continue
        print(f"=== Visualizing ERF for run: {run_dir} ===")
        try:
            visualize_erf(run_dir)
        except Exception as e:
            print(f"[Error] Failed for folder: {run_dir}\n  {type(e).__name__}: {e}")
            continue

if __name__ == '__main__':
    main()