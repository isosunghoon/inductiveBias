import os
import numpy as np
import matplotlib.pyplot as plt


# Explicitly fixed result paths (no auto-discovery/search)
LANDSCAPE_FILES = {
    "ViT": "output/vit/loss_landscape_eigenvalues.npy",
    "PoolFormer": "output/poolformer/loss_landscape_eigenvalues.npy",
}

SAVE_PATH = "output/loss_landscape_comparison.png"
HISTOGRAM_PATH = "output/loss_landscape_eigenvalue_histogram.png"


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    all_eigvals = {}

    for name, path in LANDSCAPE_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        eigvals = np.load(path)
        if eigvals.ndim != 2 or eigvals.shape[1] < 1:
            raise ValueError(f"Unexpected shape {eigvals.shape} in {path}")

        all_eigvals[name] = eigvals.flatten()

        steps = np.arange(1, eigvals.shape[0] + 1)
        top1 = eigvals[:, 0]
        top5_mean = eigvals[:, :5].mean(axis=1) if eigvals.shape[1] >= 5 else eigvals.mean(axis=1)

        axes[0].plot(steps, top1, label=name, linewidth=1.8)
        axes[1].plot(steps, top5_mean, label=name, linewidth=1.8)

    axes[0].set_title("Top-1 Hessian Eigenvalue")
    axes[0].set_xlabel("Subset Batch Step")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Mean of Top-5 Hessian Eigenvalues")
    axes[1].set_xlabel("Subset Batch Step")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=220)
    print(f"Saved figure to {SAVE_PATH}")

    # Eigenvalue 도수분포표 (x축: eigenvalue, y축: 빈도) — 모델별 겹쳐서 비교
    all_flat = np.concatenate(list(all_eigvals.values()))
    bins = np.linspace(all_flat.min(), all_flat.max(), 51)

    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    for name, flat_eig in all_eigvals.items():
        ax_hist.hist(flat_eig, bins=bins, label=name, alpha=0.6, edgecolor="black", linewidth=0.5)
    ax_hist.set_title("Eigenvalue 도수분포 (모델 비교)")
    ax_hist.set_xlabel("Eigenvalue")
    ax_hist.set_ylabel("빈도 (도수)")
    ax_hist.legend()
    ax_hist.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PATH, dpi=220)
    plt.close(fig_hist)
    print(f"Saved eigenvalue histogram to {HISTOGRAM_PATH}")


if __name__ == "__main__":
    main()
