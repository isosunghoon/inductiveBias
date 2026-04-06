"""
Entry point for running analyses on saved checkpoints.

Edit the CONFIGURATION section below, then run:
    python run_analysis.py
"""

from analysis.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Import analysis functions here as they are created
# ---------------------------------------------------------------------------
from analysis.erf_once import analyze_erf
# from analysis.cka_fn    import analyze_cka
# from analysis.dis_occ_fn import analyze_dis_occ


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_PATH = "output/for_analysis"   # folder containing model run dirs
CKPT_NAME    = "best.pt"
OUTPUT_ROOT  = "analysis_output"

# experiment name → analysis function
# each key becomes a top-level folder under analysis_output/
ANALYSIS_FNS = {
    "erf":     analyze_erf,
    # "dis_occ": analyze_dis_occ,
}

# Extra keyword args forwarded to every analysis function
ANALYSIS_KWARGS = {}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(
        project_path=PROJECT_PATH,
        analysis_fns=ANALYSIS_FNS,
        ckpt_name=CKPT_NAME,
        output_root=OUTPUT_ROOT,
        **ANALYSIS_KWARGS,
    )
