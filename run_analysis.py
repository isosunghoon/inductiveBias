"""
Entry point for running the report's analyses on saved checkpoints.

Every analysis in the registry below is available from the command line — you should
never need to edit this file to switch analyses on or off.

Usage
-----
    # List the available analyses and exit
    python run_analysis.py --list

    # Effective Receptive Field / ERD (report Section 5.1)
    python run_analysis.py -p output/model_resized -a erf

    # Loss landscape Hessian spectrum (report Section 5.3)
    python run_analysis.py -p output/model_resized -a hessian_spectrum

    # CKA (report Section 5.2) — pairwise, runs over every pair of runs in the project
    python run_analysis.py -p output/model_resized -a cka

    # Several at once, and over several checkpoints
    python run_analysis.py -p output/model_resized -a erf params \
        --ckpt_names epoch_warmup_end.pt best.pt

Each analysis runs over every subdirectory of `--project_path` that contains a
`config.yaml` (i.e. every training run), loading `--ckpt_name` from it. Results land
under `--output_root`; see analysis/pipeline.py for the exact layout.

Analysis-specific options (number of images, Lanczos steps, ...) are exposed as CLI
flags below and default to the settings used for the report.
"""

import argparse
import importlib
import os

from analysis.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Registry. Every entry is selectable via --analyses; nothing here is commented out.
#   n_models=1 -> fn(args, model, **kwargs)
#   n_models=2 -> fn(args1, model1, args2, model2, **kwargs), run over all run pairs
#
# Analysis functions are imported on demand so that a missing optional dependency
# (e.g. pyhessian, which is only needed for the Hessian analyses) does not stop the
# other analyses — or --help / --list — from working.
# ---------------------------------------------------------------------------

ANALYSES = {
    "erf": {
        "module": "analysis.erf", "fn": "analyze_erf", "n_models": 1,
        "help": "Effective Receptive Field + ERD (report Section 5.1, Fig 4/5)",
    },
    "erf_layers": {
        "module": "analysis.erf", "fn": "analyze_erf_layers", "n_models": 1,
        "help": "Layer-wise ERF / ERD (report Appendix D)",
    },
    "hessian_spectrum": {
        "module": "analysis.hessian_spectrum", "fn": "analyze_hessian_spectrum", "n_models": 1,
        "help": "Loss landscape Hessian min/max eigenvalue spectrum (Section 5.3, Fig 8)",
    },
    "loss_landscape": {
        "module": "analysis.loss_hessian", "fn": "analyze_loss_landscape", "n_models": 1,
        "help": "Top-n Hessian eigenvalues per batch (needs pyhessian)",
    },
    "params": {
        "module": "analysis.calc_param", "fn": "analyze_params", "n_models": 1,
        "help": "Trainable parameter count (report Table 2)",
    },
    "cka": {
        "module": "analysis.cka", "fn": "analyze_cka", "n_models": 2,
        "help": "Centered Kernel Alignment between every pair of runs (Section 5.2, Fig 6/7)",
    },
}


def _load(name):
    """Import an analysis function on demand."""
    spec = ANALYSES[name]
    try:
        module = importlib.import_module(spec["module"])
    except ImportError as exc:
        raise SystemExit(
            f"Analysis '{name}' needs a dependency that is not installed: {exc}.\n"
            f"Install the extras in requirements.txt, or pick a different --analyses."
        ) from exc
    return getattr(module, spec["fn"])

DEFAULT_PROJECT_PATH = "output/model_resized"
DEFAULT_CKPT_NAME = "best.pt"
DEFAULT_OUTPUT_ROOT = "analysis_output"


def _parse_cli():
    parser = argparse.ArgumentParser(
        description="Run analyses on saved checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="available analyses:\n" + "\n".join(
            f"  {name:18s} {spec['help']}" for name, spec in ANALYSES.items()
        ),
    )
    parser.add_argument(
        "--project_path", "-p", default=DEFAULT_PROJECT_PATH,
        help=f"Project output folder holding one subdirectory per run (default: {DEFAULT_PROJECT_PATH})",
    )
    parser.add_argument(
        "--ckpt_name", "-k", default=DEFAULT_CKPT_NAME,
        help=f"Checkpoint filename inside each run dir (default: {DEFAULT_CKPT_NAME})",
    )
    parser.add_argument(
        "--ckpt_names", nargs="+", default=None, metavar="NAME",
        help="Run the analyses once per checkpoint, into checkpoint-specific output folders.",
    )
    parser.add_argument(
        "--output_root", "-o", default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for analysis outputs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--analyses", "-a", nargs="+", choices=list(ANALYSES), default=None, metavar="NAME",
        help="Analyses to run (default: all). See the list at the bottom of this help.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the available analyses and exit.",
    )

    erf = parser.add_argument_group("erf / erf_layers")
    erf.add_argument("--num_images", type=int, default=500, help="images to average the ERF over")
    erf.add_argument("--anchor_mode", default="all", help="'all' or 'custom' anchor patches")
    erf.add_argument("--num_anchors", type=int, default=3, help="anchors used when anchor_mode='custom'")
    erf.add_argument("--custom_x_values", type=int, nargs="+", default=[0, 7, 3], help="anchor x coords")
    erf.add_argument("--custom_y_values", type=int, nargs="+", default=[0, 7, 9], help="anchor y coords")
    erf.add_argument("--distance_metric", default="taxi", choices=["taxi", "euclid"],
                     help="distance used for ERD (the report uses L1 / taxi)")
    erf.add_argument("--ratio", type=float, default=1, help="fraction of the dataset to sample from")

    hess = parser.add_argument_group("hessian_spectrum / loss_landscape")
    hess.add_argument("--batch_size", type=int, default=16, help="mini-batch size per eigenvalue estimate")
    hess.add_argument("--lanczos_steps", type=int, default=30, help="Lanczos iterations per estimate")
    hess.add_argument("--num_batches", type=int, default=800, help="batches to accumulate the spectrum over")
    hess.add_argument("--top_n", type=int, default=5, help="top-n eigenvalues for loss_landscape")

    cka = parser.add_argument_group("cka")
    cka.add_argument("--max_samples", type=int, default=4096, help="samples used to build the Gram matrices")

    return parser.parse_args()


def _analysis_kwargs(cli, n_models):
    """Collect the analysis-specific flags. Analyses ignore what they do not need."""
    if n_models == 2:
        return {"max_samples": cli.max_samples}
    return {
        "num_images": cli.num_images,
        "anchor_mode": cli.anchor_mode,
        "num_anchors": cli.num_anchors,
        "custom_x_values": cli.custom_x_values,
        "custom_y_values": cli.custom_y_values,
        "distance_metric": cli.distance_metric,
        "average": True,
        "ratio": cli.ratio,
        "batch_size": cli.batch_size,
        "lanczos_steps": cli.lanczos_steps,
        "num_batches": cli.num_batches,
        "top_n": cli.top_n,
    }


def _ckpt_tag(ckpt_name):
    return os.path.splitext(ckpt_name.replace(os.sep, "__"))[0]


def _analysis_output_root(base_output_root, analysis_names, ckpt_name):
    """
    One analysis  -> <output_root>/<analysis_name>/<ckpt_tag>/<project_name>/...
    Several       -> <output_root>/<ckpt_tag>/<analysis_name>/<project_name>/...
    """
    ckpt_tag = _ckpt_tag(ckpt_name)
    if len(analysis_names) == 1:
        return os.path.join(base_output_root, analysis_names[0], ckpt_tag)
    return os.path.join(base_output_root, ckpt_tag)


def main():
    cli = _parse_cli()

    if cli.list:
        print("available analyses:")
        for name, spec in ANALYSES.items():
            models = "pairwise" if spec["n_models"] == 2 else "per-run"
            print(f"  {name:18s} [{models}]  {spec['help']}")
        return

    selected = cli.analyses or list(ANALYSES)
    ckpt_names = cli.ckpt_names or [cli.ckpt_name]

    for n_models in (1, 2):
        names = [n for n in selected if ANALYSES[n]["n_models"] == n_models]
        if not names:
            continue
        fns = {n: _load(n) for n in names}

        for ckpt_name in ckpt_names:
            run_pipeline(
                project_path=cli.project_path,
                analysis_fns=fns,
                ckpt_name=ckpt_name,
                output_root=_analysis_output_root(cli.output_root, names, ckpt_name),
                n_models=n_models,
                **_analysis_kwargs(cli, n_models),
            )


if __name__ == "__main__":
    main()
