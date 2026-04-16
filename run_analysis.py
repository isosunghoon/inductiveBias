"""
Entry point for running analyses on saved checkpoints.

Usage
-----
    # Run everything (default)
    python run_analysis.py

    # Override project / checkpoint
    python run_analysis.py --project_path output/exp2 --ckpt_name last.pt

    # Run only specific analyses (key names defined in ANALYSIS_FNS / ANALYSIS_FNS_PAIR)
    python run_analysis.py --analyses erf
    python run_analysis.py --analyses erf loss_landscape cka
"""

import argparse

from analysis.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Import analysis functions here as they are created
# ---------------------------------------------------------------------------
from analysis.erf import analyze_erf
from analysis.loss_hessian import analyze_loss_landscape
from analysis.cka import analyze_cka
from analysis.calc_param import analyze_params
# from analysis.dis_occ_fn import analyze_dis_occ


# ---------------------------------------------------------------------------
# CONFIGURATION  (CLI args override project_path and ckpt_name)
# ---------------------------------------------------------------------------

PROJECT_PATH = "output/final1"
CKPT_NAME    = "best.pt"
OUTPUT_ROOT  = "analysis_output"

# ---------------------------------------------------------------------------
# Single-model analyses  (n_models=1)
# fn signature: (args, model, **kwargs) -> list[AnalysisResult]
# ---------------------------------------------------------------------------
ANALYSIS_FNS: dict = {
    "erf":            analyze_erf,
    # "loss_landscape": analyze_loss_landscape,
    # "params":         analyze_params,
    # "dis_occ":      analyze_dis_occ,
}

ANALYSIS_KWARGS = {
    # erf
    "num_images":      300,
    "anchor_mode":     "custom",
    "num_anchors":     3,
    "distance_metric": "taxi",
    "average":         False,
    "custom_x_values": [0, 6, 4],
    "custom_y_values": [0, 3, 4],
    # loss_landscape
    "batch_size":      16,
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
    "max_samples": 1024,
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


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli = _parse_cli()
    selected = set(cli.analyses) if cli.analyses else None

    fns_1 = {k: v for k, v in ANALYSIS_FNS.items()
              if selected is None or k in selected}
    fns_2 = {k: v for k, v in ANALYSIS_FNS_PAIR.items()
              if selected is None or k in selected}

    if fns_1:
        run_pipeline(
            project_path=cli.project_path,
            analysis_fns=fns_1,
            ckpt_name=cli.ckpt_name,
            output_root=OUTPUT_ROOT,
            n_models=1,
            **ANALYSIS_KWARGS,
        )

    if fns_2:
        run_pipeline(
            project_path=cli.project_path,
            analysis_fns=fns_2,
            ckpt_name=cli.ckpt_name,
            output_root=OUTPUT_ROOT,
            n_models=2,
            **ANALYSIS_KWARGS_PAIR,
        )
