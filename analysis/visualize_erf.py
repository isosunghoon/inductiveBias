import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_NPY_DIR = "analysis_output/erf_layers/model_resized_all_200"


def display_name_from_erf_layers_file(path: str) -> str:
    stem = os.path.basename(path).replace("_erf_layers.npy", "")
    stem = re.sub(r"-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", "", stem)
    labels = {
        "vit": "ViT",
        "localvit": "local ViT",
        "localvit_w5": "local ViT w5",
        "localvit_w7": "local ViT w7",
        "denseformer": "MLP mixer",
        "convformer": "Convformer",
        "convformer_w5": "Convformer w5",
        "convformer_w7": "Convformer w7",
        "identity": "Identity",
    }
    return labels.get(stem, stem)


def discover_erf_layer_files(npy_dir: str, exclude=None) -> dict:
    exclude = [item.lower() for item in (exclude or [])]
    files = sorted(
        os.path.join(npy_dir, name)
        for name in os.listdir(npy_dir)
        if name.endswith("_erf_layers.npy")
    )
    if not files:
        raise FileNotFoundError(f"No *_erf_layers.npy files found in {npy_dir}")

    name_to_path = {}
    for path in files:
        name = display_name_from_erf_layers_file(path)
        searchable = f"{name} {os.path.basename(path)}".lower()
        if any(pattern in searchable for pattern in exclude):
            continue
        name_to_path[name] = path

    if not name_to_path:
        raise FileNotFoundError(
            f"No *_erf_layers.npy files left in {npy_dir} after excludes: {exclude}"
        )

    preferred_order = [
        "Identity",
        "local ViT",
        "local ViT w5",
        "local ViT w7",
        "Convformer",
        "Convformer w5",
        "Convformer w7",
        "MLP mixer",
        "ViT",
    ]
    ordered = {name: name_to_path[name] for name in preferred_order if name in name_to_path}
    ordered.update({name: path for name, path in name_to_path.items() if name not in ordered})
    return ordered


def input_to_output_erd_curve(path: str, max_output_index: int = 13):
    matrix = np.load(path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{path} must be a square ERD matrix, got shape {matrix.shape}")

    input_row = 0
    first_output_col = 1
    last_output_col = min(matrix.shape[1] - 1, first_output_col + max_output_index)
    values = matrix[input_row, first_output_col:last_output_col + 1]
    output_indices = np.arange(values.shape[0])
    return output_indices, values


def make_input_to_output_erd_plot(npy_files: dict, max_output_index: int = 13):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    max_seen_index = 0

    for model_name, path in npy_files.items():
        output_indices, erd_values = input_to_output_erd_curve(
            path, max_output_index=max_output_index
        )
        if len(output_indices) == 0:
            continue
        max_seen_index = max(max_seen_index, int(output_indices[-1]))
        ax.plot(output_indices, erd_values, marker="o", linewidth=2, label=model_name)

    ax.set_title("Input-to-Output ERD by Layer", fontsize=15)
    ax.set_xlabel("Output layer index", fontsize=13)
    ax.set_ylabel("ERD from input", fontsize=13)
    ax.set_xticks(np.arange(max_seen_index + 1))
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    return fig


def save_input_to_output_erd_csv(npy_files: dict, save_path: str, max_output_index: int = 13):
    rows = ["model,output_layer_index,erd,path"]
    for model_name, path in npy_files.items():
        output_indices, erd_values = input_to_output_erd_curve(
            path, max_output_index=max_output_index
        )
        for idx, value in zip(output_indices, erd_values):
            rows.append(f"{model_name},{int(idx)},{float(value):.10f},{path}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ERD from input to each output layer using *_erf_layers.npy matrices."
    )
    parser.add_argument(
        "--npy_dir",
        type=str,
        default=DEFAULT_NPY_DIR,
        help="Directory containing *_erf_layers.npy files.",
    )
    parser.add_argument(
        "--max_output_index",
        type=int,
        default=13,
        help="Maximum output layer index to plot. Index 0 maps to matrix column 1.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Case-insensitive substrings to exclude from display names or filenames.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    npy_files = discover_erf_layer_files(args.npy_dir, exclude=args.exclude)

    save_png = os.path.join(args.npy_dir, "input_to_output_erd.png")
    save_csv = os.path.join(args.npy_dir, "input_to_output_erd.csv")

    fig = make_input_to_output_erd_plot(
        npy_files,
        max_output_index=args.max_output_index,
    )
    fig.savefig(save_png, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)

    save_input_to_output_erd_csv(
        npy_files,
        save_csv,
        max_output_index=args.max_output_index,
    )

    print(f"Loaded {len(npy_files)} ERF layer matrices from {args.npy_dir}")
    print(f"Saved input-to-output ERD plot to {save_png}")
    print(f"Saved input-to-output ERD data to {save_csv}")
