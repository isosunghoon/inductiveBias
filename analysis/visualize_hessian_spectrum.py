from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ckpt_tag(ckpt_name: str) -> str:
    tag = ckpt_name.replace(os.sep, "__")
    return os.path.splitext(tag)[0]


def _pretty_ckpt_label(ckpt_name: str) -> str:
    stem = Path(ckpt_name).stem
    label = stem.replace("_", " ")
    label = label.replace("epoch warmup end", "Warmup End")
    label = label.replace("best", "Best")
    return label.title() if label == stem.replace("_", " ") else label


def _pretty_model_label(model_name: str) -> str:
    prefix = model_name.split('-', 1)[0]
    label_map = {
        'localvit': 'local ViT',
        'convformer': 'Convformer',
        'denseformer': 'MLP mixer',
        'vit': 'ViT',
        'pretrained_vit': 'pretrained ViT',
    }
    return label_map.get(prefix, prefix)


def _candidate_ckpt_dirs(analysis_root: str, project_name: str, ckpt_name: str) -> list[Path]:
    root = Path(analysis_root)
    ckpt_tag = _ckpt_tag(ckpt_name)
    return [
        root / ckpt_tag / 'hessian_spectrum' / project_name,
        root / 'hessian_spectrum' / ckpt_tag / 'hessian_spectrum' / project_name,
        root / 'hessian_spectrum' / ckpt_tag / project_name,
    ]


def _load_extreme_eigenvalues(analysis_root: str, project_name: str, ckpt_name: str) -> dict[str, np.ndarray]:
    candidate_dirs = _candidate_ckpt_dirs(analysis_root, project_name, ckpt_name)
    ckpt_dir = next((path for path in candidate_dirs if path.exists()), None)
    if ckpt_dir is None:
        checked = '\n  '.join(str(path) for path in candidate_dirs)
        raise FileNotFoundError(f'Hessian spectrum directory not found. Checked:\n  {checked}')

    results = {}
    suffix = '_hessian_spectrum_extreme_eigenvalues.npy'
    for path in sorted(ckpt_dir.glob(f'*{suffix}')):
        model_name = path.name[:-len(suffix)]
        results[model_name] = np.load(path)

    if not results:
        raise FileNotFoundError(f'No hessian spectrum arrays found under: {ckpt_dir}')
    return results


def _gaussian_kernel(size: int = 9, sigma: float = 1.6) -> np.ndarray:
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def _smooth_density(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    kernel = _gaussian_kernel()
    smooth = np.convolve(hist, kernel, mode='same')
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, smooth


def _finite_column(arrays: list[np.ndarray], column: int) -> np.ndarray:
    chunks = []
    for arr in arrays:
        values = np.asarray(arr)[:, column]
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(finite)
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)


def _trim_percentile(values: np.ndarray, trim_percent: float) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or trim_percent <= 0.0:
        return finite

    trim_percent = min(trim_percent, 0.49)
    lo = float(np.quantile(finite, trim_percent))
    hi = float(np.quantile(finite, 1.0 - trim_percent))
    trimmed = finite[(finite >= lo) & (finite <= hi)]
    return trimmed if trimmed.size else finite


def save_hessian_spectrum_comparison(
    analysis_root: str,
    project_name: str,
    ckpt_names: list[str],
    save_path: str,
    selected_models: list[str] | None = None,
    max_models: int | None = None,
    trim_percent: float = 0.10,
) -> str:
    by_ckpt = {
        ckpt_name: _load_extreme_eigenvalues(analysis_root, project_name, ckpt_name)
        for ckpt_name in ckpt_names
    }

    model_names = sorted(set.intersection(*(set(results.keys()) for results in by_ckpt.values())))
    if selected_models is not None:
        wanted = set(selected_models)
        model_names = [name for name in model_names if name in wanted]
    if max_models is not None:
        model_names = model_names[:max_models]
    if not model_names:
        raise ValueError('No overlapping model names found across checkpoints for comparison.')

    palette = ['#7e2f8e', '#ff9f68', '#d62828', '#2563eb', '#1b9e77', '#6b7280']
    colors = {name: palette[idx % len(palette)] for idx, name in enumerate(model_names)}

    all_arrays = [by_ckpt[ckpt][model] for ckpt in ckpt_names for model in model_names]
    max_values = _trim_percentile(_finite_column(all_arrays, 0), trim_percent)
    min_values = _trim_percentile(_finite_column(all_arrays, 1), trim_percent)
    if max_values.size == 0 or min_values.size == 0:
        raise ValueError('Need finite min/max eigenvalues to draw comparison plot.')

    def build_bins(values: np.ndarray) -> np.ndarray:
        lo = float(values.min())
        hi = float(values.max())
        if np.isclose(lo, hi):
            span = max(1.0, abs(lo) * 0.1 + 1.0)
            lo -= span
            hi += span
        return np.linspace(lo, hi, 80)

    bins_max = build_bins(max_values)
    bins_min = build_bins(min_values)

    n_rows = len(ckpt_names)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13.5, 4.2 * n_rows), sharex='col', sharey='col')
    if n_rows == 1:
        axes = np.asarray([axes])

    for row_idx, ckpt_name in enumerate(ckpt_names):
        label = _pretty_ckpt_label(ckpt_name)
        row_results = by_ckpt[ckpt_name]

        for model_name in model_names:
            ev_pairs = row_results[model_name]
            max_ev = _trim_percentile(np.asarray(ev_pairs)[:, 0], trim_percent)
            min_ev = _trim_percentile(np.asarray(ev_pairs)[:, 1], trim_percent)
            if min_ev.size:
                x, y = _smooth_density(min_ev, bins_min)
                axes[row_idx, 0].plot(
                    x,
                    y,
                    color=colors[model_name],
                    linewidth=2.3,
                    label=_pretty_model_label(model_name),
                )
            if max_ev.size:
                x, y = _smooth_density(max_ev, bins_max)
                axes[row_idx, 1].plot(
                    x,
                    y,
                    color=colors[model_name],
                    linewidth=2.3,
                    label=_pretty_model_label(model_name),
                )

        axes[row_idx, 0].set_title(f'{label} - Min eigenvalue')
        axes[row_idx, 1].set_title(f'{label} - Max eigenvalue')
        axes[row_idx, 0].set_ylabel('Density')

        for col in range(2):
            axes[row_idx, col].grid(alpha=0.22)
            axes[row_idx, col].spines['top'].set_visible(False)
            axes[row_idx, col].spines['right'].set_visible(False)

    axes[-1, 0].set_xlabel('Min Eigenvalue')
    axes[-1, 1].set_xlabel('Max Eigenvalue')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=min(len(model_names), 5),
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
        )

    if len(ckpt_names) == 1:
        title = f'Hessian spectrum comparison across token mixers - {project_name} ({_pretty_ckpt_label(ckpt_names[0])})'
    else:
        title = f'Hessian spectrum comparison across token mixers - {project_name}'
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return str(save_path)


def _default_save_path(analysis_root: str, project_name: str, ckpt_names: list[str]) -> str:
    comparison_dir = Path(analysis_root) / 'comparison' / 'hessian_spectrum' / project_name
    if len(ckpt_names) == 1:
        filename = f'{_ckpt_tag(ckpt_names[0])}_model_comparison.png'
    else:
        filename = 'checkpoint_comparison.png'
    return str(comparison_dir / filename)


def _parse_cli():
    parser = argparse.ArgumentParser(
        description='Compare Hessian spectrum outputs across models within a checkpoint.'
    )
    parser.add_argument('--analysis_root', default='analysis_output')
    parser.add_argument('--project_name', required=True)
    parser.add_argument(
        '--ckpt_name',
        default='best.pt',
        help='Single checkpoint to compare models within. Ignored when --ckpt_names is provided.',
    )
    parser.add_argument(
        '--ckpt_names',
        nargs='+',
        default=None,
        help='Optional checkpoint list. With one checkpoint, compares models inside it.',
    )
    parser.add_argument(
        '--save_path',
        default=None,
        help='Output PNG path. Defaults under <analysis_root>/comparison/hessian_spectrum/<project_name>/.',
    )
    parser.add_argument('--models', nargs='*', default=None)
    parser.add_argument('--max_models', type=int, default=None)
    parser.add_argument('--trim_percent', type=float, default=0.10)
    return parser.parse_args()


if __name__ == '__main__':
    cli = _parse_cli()
    ckpt_names = cli.ckpt_names or [cli.ckpt_name]
    save_path = cli.save_path or _default_save_path(cli.analysis_root, cli.project_name, ckpt_names)
    saved = save_hessian_spectrum_comparison(
        analysis_root=cli.analysis_root,
        project_name=cli.project_name,
        ckpt_names=ckpt_names,
        save_path=save_path,
        selected_models=cli.models,
        max_models=cli.max_models,
        trim_percent=cli.trim_percent,
    )
    print(f'Saved comparison plot to {saved}')
