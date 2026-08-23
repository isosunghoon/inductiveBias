"""Orchestrator: python -m theory.run_all [--panels P0,P1,...] (spec 4/8 priority order)."""
import argparse
import json
import os
import sys
import time

from . import panels, plots
from .core import RESULTS_DIR, nstar_writer, raw_writer

PRIORITY = ["P0", "P1", "P2", "P4", "P4C", "P3", "P5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="P0,P1,P2,P4,P3,P5")
    ap.add_argument("--budget-hours", type=float, default=12.0)
    ap.add_argument("--skip-gate", action="store_true",
                    help="run later panels even if P0 gate fails (debug only)")
    ap.add_argument("--p0-seeds", type=int, default=3)
    ap.add_argument("--p0-ratio", type=float, default=2 ** 0.5)
    ap.add_argument("--p4-models", default="")
    ap.add_argument("--p4-ds", default="")
    ap.add_argument("--p4-time-limit", type=float, default=900)
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    todo = {p.strip().upper() for p in args.panels.split(",") if p.strip()}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw, nst = raw_writer(), nstar_writer()
    slopes_path = os.path.join(RESULTS_DIR, "slopes.json")
    slopes = json.load(open(slopes_path)) if os.path.exists(slopes_path) else {}

    def dump_slopes():
        json.dump(slopes, open(slopes_path, "w"), indent=2)

    for panel in [p for p in PRIORITY if p in todo]:
        log(f"=== {panel} start ===")
        tp = time.time()
        if panel == "P0":
            gate, s0 = panels.run_p0(raw, nst, log, seeds=args.p0_seeds, ratio=args.p0_ratio)
            slopes["P0"] = s0
            dump_slopes()
            json.dump(gate, open(os.path.join(RESULTS_DIR, "p0_gate.json"), "w"), indent=2)
            log(f"P0 gate: {'PASS' if gate['pass'] else 'FAIL'}")
            if not gate["pass"] and not args.skip_gate:
                log("P0 gate FAILED — aborting all subsequent panels (spec 4/8).")
                plots.make_all(log)
                return 1
        elif panel == "P1":
            _, s1 = panels.run_p1(raw, nst, log)
            slopes["P1"] = s1
            dump_slopes()
        elif panel == "P2":
            summary = panels.run_p2(raw, log)
            json.dump(summary, open(os.path.join(RESULTS_DIR, "p2_summary.json"), "w"), indent=2)
        elif panel == "P4":
            s4 = panels.run_p4(
                raw, nst, log,
                only_models=[m for m in args.p4_models.split(",") if m],
                only_ds=[int(x) for x in args.p4_ds.split(",") if x],
                time_limit=args.p4_time_limit)
            if s4:
                slopes.setdefault("P4", {}).update(s4)
                dump_slopes()
        elif panel == "P4C":
            rec = panels.run_p4_constants(raw, log)
            rec.update(panels.run_p4_constants(raw, log, model="untied", ds=(48,), seeds=60))
            json.dump(rec, open(os.path.join(RESULTS_DIR, "p4_constants.json"), "w"), indent=2)
        elif panel == "P3":
            s3 = panels.run_p3(raw, nst, log)
            if s3:
                slopes["P3"] = s3
                dump_slopes()
        elif panel == "P5":
            if time.time() - t0 > args.budget_hours * 3600 * 0.66:
                log("P5 skipped: not enough budget left (optional panel).")
                continue
            panels.run_p5(raw, nst, log)
        log(f"=== {panel} done in {time.time() - tp:.0f}s ===")

    plots.make_all(log)
    log("all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
