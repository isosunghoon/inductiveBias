"""
Unified analysis pipeline.

Usage
-----
Define analysis functions with the signature::

    def analyze_something(args, model, **kwargs) -> list[AnalysisResult]:
        ...
        return [AnalysisResult("erf", arr, ".npy"), AnalysisResult("erf", fig, ".png")]

Then call run_pipeline() from a top-level script (e.g. run_analysis.py)::

    from analysis.pipeline import run_pipeline
    from analysis.erf_fn import analyze_erf

    run_pipeline(
        project_path="output/base_model_exp",
        analysis_fns=[analyze_erf],
        ckpt_name="best.pt",
        experiment_name="erf",
    )

Output layout
-------------
    analysis_output/
        {experiment}/
            {project_name}/
                {model_dir}_{suffix}.npy
                {model_dir}_{suffix}.png
                ...

AnalysisResult
--------------
    suffix   : str   – appended after model name, e.g. "erf"
    data     : Any   – numpy array / matplotlib Figure / object / str
    fmt      : str   – one of ".npy", ".png", ".pickle", ".txt"
"""

from __future__ import annotations

import itertools
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Callable

from utils.config import parse_args
from utils.build_model import build_model


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """
    Return this from every analysis function.

    Attributes
    ----------
    suffix : str
        Short name appended to the model directory name when saving.
        e.g. "erf"  →  saved as  "{model_dir}_erf.npy"
    data : Any
        - ".npy"    : numpy ndarray
        - ".png"    : matplotlib Figure  (pipeline calls fig.savefig + plt.close)
        - ".pickle" : any picklable object
        - ".txt"    : str
    fmt : str
        File extension including dot: ".npy" | ".png" | ".pickle" | ".txt"
    """
    suffix: str
    data: Any
    fmt: str


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _save_result(result: AnalysisResult, save_dir: str, model_name: str) -> str:
    """Save a single AnalysisResult. Returns the saved file path."""
    filename = f"{model_name}_{result.suffix}{result.fmt}"
    path = os.path.join(save_dir, filename)

    if result.fmt == ".npy":
        np.save(path, result.data)
    elif result.fmt == ".png":
        result.data.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(result.data)
    elif result.fmt == ".pickle":
        with open(path, "wb") as f:
            pickle.dump(result.data, f)
    elif result.fmt == ".txt":
        with open(path, "w", encoding="utf-8") as f:
            f.write(result.data)
    else:
        raise ValueError(f"Unsupported format: {result.fmt!r}. Use .npy / .png / .pickle / .txt")

    return path


# ---------------------------------------------------------------------------
# Model discovery helpers
# ---------------------------------------------------------------------------

def _iter_model_dirs(project_path: str):
    """
    Yield (model_dir_name, model_dir_path) for every subdirectory under
    project_path that looks like a saved run (has config.yaml).
    """
    for name in sorted(os.listdir(project_path)):
        full = os.path.join(project_path, name)
        if not os.path.isdir(full):
            continue
        if not os.path.exists(os.path.join(full, "config.yaml")):
            continue
        yield name, full


def _iter_model_pairs(project_path: str):
    """
    Yield ((name1, path1), (name2, path2)) for every combination of two model
    directories under project_path (each must have config.yaml).
    """
    dirs = list(_iter_model_dirs(project_path))
    for (name1, path1), (name2, path2) in itertools.combinations(dirs, 2):
        yield (name1, path1), (name2, path2)


def _load_args(model_dir_path: str) -> Any:
    """Parse args from the saved config.yaml inside a model directory."""
    config_yaml = os.path.join(model_dir_path, "config.yaml")
    return parse_args([
        "--config", config_yaml,
        "--output_path", model_dir_path,
    ])


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    project_path: str,
    analysis_fns: dict[str, Callable],
    ckpt_name: str = "best.pt",
    output_root: str = "analysis_output",
    n_models: int = 1,
    **kwargs,
) -> None:
    """
    Run all analysis_fns on every model (or model pair) under project_path.

    Parameters
    ----------
    project_path : str
        Path to the project output folder (e.g. "output/base_model_exp").
        Every subdirectory with a config.yaml is treated as one model run.
    analysis_fns : dict[str, Callable]
        Mapping of experiment_name -> analysis function.
        Each key becomes the top-level folder under analysis_output/.

        When n_models=1, each function must follow::

            def analyze_X(args, model, **kwargs) -> list[AnalysisResult]

        When n_models=2, each function must follow::

            def analyze_X(args1, model1, args2, model2, **kwargs) -> list[AnalysisResult]

        Example::

            {"erf": analyze_erf, "cka": analyze_cka}

    ckpt_name : str
        Checkpoint filename to load from each model directory (default "best.pt").
    output_root : str
        Root directory for all analysis outputs (default "analysis_output").
    n_models : int
        Number of models each analysis function receives (1 or 2, default 1).
        When 2, the pipeline iterates over all pairs of model directories and
        saves results under "{model1_name}_vs_{model2_name}".
    **kwargs
        Forwarded verbatim to every analysis function.
    """
    if n_models not in (1, 2):
        raise ValueError(f"n_models must be 1 or 2, got {n_models}")

    project_name = os.path.basename(os.path.normpath(project_path))
    model_dirs = list(_iter_model_dirs(project_path))
    if not model_dirs:
        raise RuntimeError(f"No model directories with config.yaml found in: {project_path}")

    print(f"[pipeline] project    : {project_name}  ({len(model_dirs)} models)")
    print(f"[pipeline] checkpoint : {ckpt_name}")
    print(f"[pipeline] analyses   : {list(analysis_fns.keys())}")
    print(f"[pipeline] n_models   : {n_models}")
    print()

    _SEP = "[pipeline] " + "─" * 60

    if n_models == 1:
        total = len(model_dirs)
        for idx, (model_name, model_dir_path) in enumerate(model_dirs, 1):
            print(_SEP)
            print(f"[pipeline] model {idx}/{total}: {model_name}")
            print(_SEP)

            args = _load_args(model_dir_path)
            model = build_model(args, ckpt_name=ckpt_name)

            for experiment_name, fn in analysis_fns.items():
                save_dir = os.path.join(output_root, experiment_name, project_name)
                os.makedirs(save_dir, exist_ok=True)

                print(f"[pipeline]    [{experiment_name}] running {fn.__name__} ...", flush=True)
                results = fn(args, model, **kwargs)

                for result in results:
                    saved_path = _save_result(result, save_dir, model_name)
                    print(f"[pipeline]    [{experiment_name}] saved  → {saved_path}")

            print()

    else:  # n_models == 2
        pairs = list(_iter_model_pairs(project_path))
        if not pairs:
            raise RuntimeError(
                f"Need at least 2 model directories for n_models=2, found: {len(model_dirs)}"
            )
        print(f"[pipeline] pairs      : {len(pairs)}")
        print()

        total = len(pairs)
        for idx, ((name1, path1), (name2, path2)) in enumerate(pairs, 1):
            pair_name = f"{name1}_vs_{name2}"
            print(_SEP)
            print(f"[pipeline] pair {idx}/{total}: {pair_name}")
            print(_SEP)

            args1 = _load_args(path1)
            model1 = build_model(args1, ckpt_name=ckpt_name)
            args2 = _load_args(path2)
            model2 = build_model(args2, ckpt_name=ckpt_name)

            for experiment_name, fn in analysis_fns.items():
                save_dir = os.path.join(output_root, experiment_name, project_name)
                os.makedirs(save_dir, exist_ok=True)

                print(f"[pipeline]    [{experiment_name}] running {fn.__name__} ...", flush=True)
                results = fn(args1, model1, args2, model2, **kwargs)

                for result in results:
                    saved_path = _save_result(result, save_dir, pair_name)
                    print(f"[pipeline]    [{experiment_name}] saved  → {saved_path}")

            print()

    print(_SEP)
    print(f"[pipeline] done. results in {output_root}/")
