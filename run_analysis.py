"""
Entry point for running analyses on saved checkpoints.

Usage
-----
    # Run everything (default)
    python run_analysis.py

    # Override project / checkpoint
    python run_analysis.py --project_path output/exp2 --ckpt_name last.pt

    # Run the same analyses for multiple checkpoints
    python run_analysis.py --ckpt_names epoch_warmup_end.pt best.pt --analyses hessian_spectrum

    # Run only specific analyses (key names defined in ANALYSIS_FNS / ANALYSIS_FNS_PAIR)
    python run_analysis.py --analyses erf
    python run_analysis.py --analyses erf loss_landscape cka
"""

import argparse
import os

from analysis.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Import analysis functions here as they are created
# ---------------------------------------------------------------------------
from analysis.erf import analyze_erf, analyze_erf_layers
from analysis.loss_hessian import analyze_loss_landscape
from analysis.hessian_spectrum import analyze_hessian_spectrum
from analysis.cka import analyze_cka
from analysis.calc_param import analyze_params
# from analysis.dis_occ_fn import analyze_dis_occ


# ---------------------------------------------------------------------------
# CONFIGURATION  (CLI args override project_path and ckpt_name)
# ---------------------------------------------------------------------------

PROJECT_PATH = "output/model_resized_pretrained_vit"
CKPT_NAME    = "best.pt"
OUTPUT_ROOT  = "analysis_output"

# ---------------------------------------------------------------------------
# Single-model analyses  (n_models=1)
# fn signature: (args, model, **kwargs) -> list[AnalysisResult]
# ---------------------------------------------------------------------------
ANALYSIS_FNS: dict = {
    # "erf":            analyze_erf,
    # "erf_layers":     analyze_erf_layers,
    # "loss_landscape": analyze_loss_landscape,
    # "params":         analyze_params,
    # "dis_occ":      analyze_dis_occ,
    "hessian_spectrum": analyze_hessian_spectrum,
}

ANALYSIS_KWARGS = {
    # erf
    "num_images":      500,
    "anchor_mode":     "all",
    "num_anchors":     3,
    "distance_metric": "taxi",
    "average":         True,
    "custom_x_values": [0, 7, 3],
    "custom_y_values": [0, 7, 9],
    # hessian_spectrum
    "batch_size":      16,
    "lanczos_steps":   30,
    "num_batches":     800,
    # loss_landscape / erf shared
    "ratio":           1,
    "top_n":           5,
}

# ---------------------------------------------------------------------------
# Two-model (pairwise) analyses  (n_models=2)
# fn signature: (args1, model1, args2, model2, **kwargs) -> list[AnalysisResult]
# ---------------------------------------------------------------------------
ANALYSIS_FNS_PAIR: dict = {
    # "cka": analyze_cka,
}

ANALYSIS_KWARGS_PAIR = {
    "max_samples": 4096,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_cli():
    all_keys = list(ANALYSIS_FNS.keys()) + list(ANALYSIS_FNS_PAIR.keys())
    parser = argparse.ArgumentParser(description="Run analysis pipeline on saved checkpoints.")
    parser.add_argument(
        "--project_path", "-p",
        default=PROJECT_PATH,
        help=f"Project output folder (default: {PROJECT_PATH})",
    )
    parser.add_argument(
        "--ckpt_name", "-k",
        default=CKPT_NAME,
        help=f"Checkpoint filename inside each model dir (default: {CKPT_NAME})",
    )
    parser.add_argument(
        "--ckpt_names",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Optional list of checkpoint filenames. When provided, analyses are run once per "
            "checkpoint and saved under checkpoint-specific output folders."
        ),
    )
    parser.add_argument(
        "--output_root", "-o",
        default=OUTPUT_ROOT,
        help=f"Root directory for analysis outputs (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--analyses", "-a",
        nargs="+",
        choices=all_keys,
        default=None,
        metavar="NAME",
        help=(
            f"Analyses to run. Choices: {all_keys}. "
            "Pass one or more names. Omit to run all."
        ),
    )
    return parser.parse_args()


def _ckpt_tag(ckpt_name: str) -> str:
    tag = ckpt_name.replace(os.sep, "__")
    return os.path.splitext(tag)[0]


def _analysis_output_root(base_output_root: str, analysis_names: list[str], ckpt_name: str) -> str:
    """
    Place single-analysis runs under:
        <output_root>/<analysis_name>/<ckpt_tag>
    so final files land in:
        <output_root>/<analysis_name>/<ckpt_tag>/<project_name>/...

    When multiple analyses are requested together, keep the previous layout and
    only split by checkpoint at the top level:
        <output_root>/<ckpt_tag>/<analysis_name>/<project_name>/...
    """
    ckpt_tag = _ckpt_tag(ckpt_name)
    if len(analysis_names) == 1:
        return os.path.join(base_output_root, analysis_names[0], ckpt_tag)
    return os.path.join(base_output_root, ckpt_tag)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli = _parse_cli()
    selected = set(cli.analyses) if cli.analyses else None
    ckpt_names = cli.ckpt_names or [cli.ckpt_name]

    fns_1 = {k: v for k, v in ANALYSIS_FNS.items()
              if selected is None or k in selected}
    fns_2 = {k: v for k, v in ANALYSIS_FNS_PAIR.items()
              if selected is None or k in selected}
    analysis_names_1 = list(fns_1.keys())
    analysis_names_2 = list(fns_2.keys())

    for ckpt_name in ckpt_names:
        if fns_1:
            ckpt_output_root = _analysis_output_root(cli.output_root, analysis_names_1, ckpt_name)
            run_pipeline(
                project_path=cli.project_path,
                analysis_fns=fns_1,
                ckpt_name=ckpt_name,
                output_root=ckpt_output_root,
                n_models=1,
                **ANALYSIS_KWARGS,
            )

        if fns_2:
            ckpt_output_root = _analysis_output_root(cli.output_root, analysis_names_2, ckpt_name)
            run_pipeline(
                project_path=cli.project_path,
                analysis_fns=fns_2,
                ckpt_name=ckpt_name,
                output_root=ckpt_output_root,
                n_models=2,
                **ANALYSIS_KWARGS_PAIR,
            )
