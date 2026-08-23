"""Panel runners P0-P5 (spec 4)."""
import time

import numpy as np

from .core import (CAP, C_tied, C_untied, EVAL_M, LAMBDAS, fit_slope, gen_batch,
                   nstar_for_regime, nstar_from_curve, one_step, predict, risk_pred,
                   risk_theory, rng_for, sweep)
from .ridge import make_ridge_compute, ridge_risks_from_data
from .train import estimate_L, run_gd, run_sgd, run_softmax_adam, run_untied_adam

P0_BASE = {"tied": {6: 46, 8: 64, 12: 150, 16: 281},
           "untied": {6: 305, 8: 684, 12: 2059, 16: 4388}}
P0_SLOPE_RANGE = {"tied": (1.7, 2.1), "untied": (2.5, 2.9)}
ONESTEP_CAP = 16384


def n_start(N):
    return 4 * N * np.log(N)


def make_onestep_compute(panel, model, N, d):
    evals = {}
    tied = model == "tied"

    def compute(n, seed):
        rng = rng_for(panel, model, N, d, n, seed)
        K, v, q, y = gen_batch(n, N, d, rng)
        A = one_step(K, v, q, y, tied)
        if seed not in evals:
            evals[seed] = gen_batch(EVAL_M, N, d, rng_for(panel, model, N, d, "eval", seed))
        Ke, ve, qe, ye = evals[seed]
        return {"onestep": risk_pred(predict(A, Ke, ve, qe, tied), ye)}

    return compute


def run_p0(raw, nst, log, seeds=3, ratio=None):
    d = 48
    ratio = ratio if ratio is not None else 2 ** 0.5
    res = {}
    for model in ("tied", "untied"):
        for N in (6, 8, 12, 16):
            comp = make_onestep_compute("P0", model, N, d)
            ns, _, med, trunc = sweep(
                comp, n_start(N), ONESTEP_CAP, seeds=seeds, ratio=ratio, time_limit=1800,
                on_row=lambda n, s, reg, r, m=model, NN=N: raw.row("P0", m, NN, d, n, s, reg, f"{r:.6g}"))
            nstar, method = nstar_for_regime(ns, med, "onestep", ONESTEP_CAP, trunc)
            nst.row("P0", model, N, d, "onestep", f"{nstar:.6g}", method)
            res[(model, N)] = nstar
            log(f"P0 {model} N={N}: n*={nstar:.1f} ({method}), grid {ns[0]}..{ns[-1]}")
    gate = {"pass": True, "seeds": seeds, "ratio": ratio, "cells": {}, "slopes": {}}
    slopes = {}
    for model in ("tied", "untied"):
        Ns = [6, 8, 12, 16]
        for N in Ns:
            base = P0_BASE[model][N]
            got = res[(model, N)]
            ok = bool(np.isfinite(got) and 0.7 * base <= got <= 1.3 * base)
            gate["cells"][f"{model},N={N}"] = {"base": base, "measured": got, "ok": ok}
            gate["pass"] = gate["pass"] and ok
        vals = [res[(model, N)] for N in Ns]
        if all(np.isfinite(v) for v in vals):
            s, se = fit_slope(np.log(Ns), np.log(vals))
        else:
            s, se = float("nan"), float("nan")
        lo, hi = P0_SLOPE_RANGE[model]
        ok = bool(np.isfinite(s) and lo <= s <= hi)
        gate["slopes"][model] = {"slope": s, "stderr": se, "range": [lo, hi], "ok": ok}
        gate["pass"] = gate["pass"] and ok
        slopes[model] = {"onestep": {"slope": s, "stderr": se, "points": len(Ns)}}
    return gate, slopes


P1_NS = [6, 8, 12, 16, 24]


def run_p1(raw, nst, log):
    d = 48
    out = {}
    regimes = [f"lam={lam:g}" for lam in LAMBDAS] + ["best"]
    for model in ("tied", "untied"):
        cap = CAP[model]
        for N in P1_NS:
            t0 = time.time()
            comp = make_ridge_compute("P1", model, N, d)
            # spec 8's 10-min limit is per (model,N,n,seed) cell; the largest single
            # eigh here is ~150s, so the whole sweep gets a proportionally larger budget.
            ns, _, med, trunc = sweep(
                comp, n_start(N), cap, seeds=3, time_limit=2400,
                on_row=lambda n, s, reg, r, m=model, NN=N: raw.row("P1", m, NN, d, n, s, reg, f"{r:.6g}"))
            for n in ns:
                med[n]["best"] = min(med[n][f"lam={lam:g}"] for lam in LAMBDAS)
            for reg in regimes:
                nstar, method = nstar_for_regime(ns, med, reg, cap, trunc)
                nst.row("P1", model, N, d, reg, f"{nstar:.6g}", method)
                out.setdefault(model, {}).setdefault(N, {})[reg] = (nstar, method)
            b = out[model][N]["best"]
            log(f"P1 {model} N={N}: best n*={b[0]:.1f} ({b[1]}), grid {ns[0]}..{ns[-1]}, "
                f"{time.time()-t0:.0f}s{' TRUNCATED' if trunc else ''}")
    slopes = {}
    for model in ("tied", "untied"):
        slopes[model] = {}
        for reg in regimes:
            pts = [(N, out[model][N][reg][0]) for N in P1_NS
                   if out[model][N][reg][1] == "interp"]
            if len(pts) >= 2:
                s, se = fit_slope(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]))
                slopes[model][reg] = {"slope": s, "stderr": se, "points": len(pts)}
            else:
                slopes[model][reg] = {"slope": None, "stderr": None, "points": len(pts)}
    return out, slopes


P2_LOG_EPOCHS = (1, 4, 16, 64, 256, 1024, 4096)
P2_SGD_LOG = (1, 4, 16, 64, 256, 1024)


def run_p2(raw, log):
    d = 48
    summary = {}
    for (N, n) in ((8, 1024), (16, 4096)):
        for model in ("tied", "untied"):
            for seed in range(3):
                rng = rng_for("P2", model, N, d, n, seed)
                train = gen_batch(n, N, d, rng)
                eval_ = gen_batch(EVAL_M, N, d, rng_for("P2", model, N, d, "eval", seed))
                lr, gdr = run_gd(model, train, eval_, set(P2_LOG_EPOCHS),
                                 rng=rng_for("P2", "pow", model, N, d, n, seed))
                for t, r in sorted(gdr.items()):
                    raw.row("P2", model, N, d, n, seed, f"gd_epoch={t}", f"{r:.6g}")
                lam_effs = [1.0 / (lr * t) for t in P2_LOG_EPOCHS]
                rr = ridge_risks_from_data(model, train, eval_, n, lambdas=lam_effs)
                devs, late = [], []
                for t, lam in zip(P2_LOG_EPOCHS, lam_effs):
                    raw.row("P2", model, N, d, n, seed, f"ridge_epoch={t}", f"{rr[lam]:.6g}")
                    if rr[lam] > 1e-12:
                        dev = abs(gdr[t] - rr[lam]) / rr[lam]
                        devs.append(dev)
                        if t >= 16:
                            late.append(dev)
                # what P1 actually rests on: does the GD path reach the ridge path's best risk?
                best_gd = min(gdr.values())
                best_ridge = min(ridge_risks_from_data(model, train, eval_, n).values())
                summary[f"{model},N={N},n={n},seed={seed}"] = {
                    "lr": lr, "max_rel_dev": float(max(devs)),
                    "max_rel_dev_t>=16": float(max(late)),
                    "best_risk_gd": best_gd, "best_risk_ridge": best_ridge,
                    "best_rel_dev": abs(best_gd - best_ridge) / max(best_ridge, 1e-12)}
                log(f"P2 {model} N={N} n={n} seed={seed}: lr={lr:.3g}, "
                    f"max dev={max(devs)*100:.1f}% (t>=16: {max(late)*100:.1f}%), "
                    f"best risk gd={best_gd:.4f} vs ridge={best_ridge:.4f}")
    # SGD spot: untied, N=16, n=4096, batch 64, 3 lr values (seed 0)
    N, n, model, seed = 16, 4096, "untied", 0
    rng = rng_for("P2", model, N, d, n, seed)
    train = gen_batch(n, N, d, rng)
    eval_ = gen_batch(EVAL_M, N, d, rng_for("P2", model, N, d, "eval", seed))
    # batch-64 curvature, not full-batch: the mini-batch Hessian is far larger, so
    # reusing the full-batch step size would just diverge.
    sub = rng_for("P2", "sgdsub", model, N, d, n, seed).choice(n, 64, replace=False)
    L_b = estimate_L(model, train[0][sub], train[1][sub], train[2][sub],
                     rng_for("P2", "pow", "sgd", model, N, d, n, seed))
    lr_base = 0.9 * 2.0 / L_b
    for mult in (0.25, 0.5, 1.0):
        lr = mult * lr_base
        risks = run_sgd(model, train, eval_, set(P2_SGD_LOG), lr, 64,
                        rng_for("P2", "sgd", model, N, d, n, seed, mult))
        for t, r in sorted(risks.items()):
            raw.row("P2", model, N, d, n, seed, f"sgd_lr={lr:.3g}_epoch={t}", f"{r:.6g}")
        summary[f"sgd,lr={lr:.3g}"] = {"final_risk": risks[max(P2_SGD_LOG)]}
        log(f"P2 SGD lr={lr:.3g}: final risk {risks[max(P2_SGD_LOG)]:.4f}")
    return summary


P4_DS = (24, 48, 96, 192)


def run_p4(raw, nst, log, only_models=None, only_ds=None, time_limit=900):
    N = 8
    slopes = {}
    for model, seeds in (("tied", 20), ("untied", 10)):
        if only_models and model not in only_models:
            continue
        nstars = []
        for d in (only_ds or P4_DS):
            comp = make_onestep_compute("P4", model, N, d)
            local = {}

            def on_row(n, s, reg, r, m=model, dd=d):
                raw.row("P4", m, N, dd, n, s, reg, f"{r:.6g}")
                local.setdefault(s, {})[n] = r

            ns, _, med, trunc = sweep(comp, n_start(N), ONESTEP_CAP, seeds=seeds,
                                      ratio=2 ** 0.25, on_row=on_row,
                                      down_check="all_seeds", time_limit=time_limit)
            nstar, method = nstar_for_regime(ns, med, "onestep", ONESTEP_CAP, trunc)
            nst.row("P4", model, N, d, "onestep", f"{nstar:.6g}", method)
            per_seed = []
            for s in range(seeds):
                curve = [local[s][n] for n in ns]
                v, m_ = nstar_from_curve(ns, curve, truncated=trunc,
                                         reached_cap=(ns[-1] >= ONESTEP_CAP))
                if m_ in ("interp", "below_floor"):
                    per_seed.append(v)
            nstars.append(nstar)
            log(f"P4 {model} d={d}: median n*={nstar:.1f} ({method}), "
                f"per-seed mean={np.mean(per_seed):.1f} sd={np.std(per_seed):.1f} "
                f"[{len(per_seed)}/{seeds} seeds]")
        if len(nstars) == len(P4_DS) and all(np.isfinite(v) for v in nstars):
            s, se = fit_slope(np.log(P4_DS), np.log(nstars))
            slopes[model] = {"onestep_vs_d": {"slope": s, "stderr": se, "points": len(P4_DS)}}
    return slopes


def run_p4_constants(raw, log, model="tied", ds=(48, 192), seeds=200):
    """P4 deliverable (ii): which constant does R(n) sit on, at fixed wide n range.

    The main P4 sweep stops as soon as it brackets R=0.5, which leaves too few
    points to separate hypotheses that differ by ~6% in C. Here the grid is fixed
    to 0.3C..4C and the seed count raised until the seed-median resolves them.
    """
    N = 8
    out = {}
    for d in ds:
        C = C_tied(N, d) if model == "tied" else C_untied(N, d)
        ns = [int(round(C * f)) for f in (0.3, 0.45, 0.65, 0.9, 1.2, 1.6, 2.2, 3.0, 4.0)]
        meds, sems = [], []
        for n in ns:
            comp = make_onestep_compute("P4c", model, N, d)
            rs = [comp(n, s)["onestep"] for s in range(seeds)]
            for s, r in enumerate(rs):
                raw.row("P4c", model, N, d, n, s, "onestep", f"{r:.6g}")
            meds.append(float(np.median(rs)))
            sems.append(float(np.std(rs) / np.sqrt(seeds)) * 1.253)  # median SEM
        ns_a, meds_a, sems_a = np.array(ns), np.array(meds), np.array(sems)
        cands = ({"leading N(N+2)": N * (N + 2), "C_t=N(N+2)+5": N * (N + 2) + 5,
                  "C_t(d) exact": C} if model == "tied" else
                 {"leading N^2(N+2)": N * N * (N + 2),
                  "C_u=N^2(N+2)+6N-1": N * N * (N + 2) + 6 * N - 1, "C_u(d) exact": C})
        rec = {}
        for name, Cc in cands.items():
            resid = meds_a - risk_theory(ns_a, Cc)
            chi2 = float(np.sum((resid / sems_a) ** 2)) / len(ns_a)
            rec[name] = {"C": float(Cc), "rms": float(np.sqrt(np.mean(resid ** 2))),
                         "chi2_per_pt": chi2}
            log(f"P4c {model} d={d} {name:22s} C={Cc:7.1f} rms={rec[name]['rms']:.4f} "
                f"chi2/pt={chi2:8.1f}")
        out[f"{model},d={d}"] = {"n": ns, "median_risk": meds, "sem": sems, "candidates": rec}
    return out


def run_p3(raw, nst, log):
    d = 48
    slopes = {}
    pts = []
    for N in (8, 16):
        evals = {}

        def compute(n, seed, NN=N):
            rng = rng_for("P3", "softmax", NN, d, n, seed)
            train = gen_batch(n, NN, d, rng)
            if seed not in evals:
                evals[seed] = gen_batch(EVAL_M, NN, d, rng_for("P3", "softmax", NN, d, "eval", seed))
            return {"adam2000": run_softmax_adam(train, evals[seed], d)}

        ns, _, med, trunc = sweep(
            compute, n_start(N), 8192, seeds=3, ratio=2.0,
            on_row=lambda n, s, reg, r, NN=N: raw.row("P3", "softmax", NN, d, n, s, reg, f"{r:.6g}"))
        nstar, method = nstar_for_regime(ns, med, "adam2000", 8192, trunc)
        nst.row("P3", "softmax", N, d, "adam2000", f"{nstar:.6g}", method)
        log(f"P3 softmax N={N}: n*={nstar:.1f} ({method})")
        if method == "interp":
            pts.append((N, nstar))
    if len(pts) >= 2:
        s, se = fit_slope(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]))
        slopes["softmax"] = {"adam2000": {"slope": s, "stderr": se, "points": len(pts)}}
    return slopes


def run_p5(raw, nst, log):
    d = 48
    regimes = [f"adam_lr={lr:g}" for lr in (1e-2, 1e-3)]
    for N in (8, 16):
        evals = {}

        def compute(n, seed, NN=N):
            rng = rng_for("P5", "untied", NN, d, n, seed)
            train = gen_batch(n, NN, d, rng)
            if seed not in evals:
                evals[seed] = gen_batch(EVAL_M, NN, d, rng_for("P5", "untied", NN, d, "eval", seed))
            return {f"adam_lr={lr:g}": run_untied_adam(train, evals[seed], lr)
                    for lr in (1e-2, 1e-3)}

        ns, _, med, trunc = sweep(
            compute, n_start(N), 8192, seeds=3, ratio=2.0,
            on_row=lambda n, s, reg, r, NN=N: raw.row("P5", "untied", NN, d, n, s, reg, f"{r:.6g}"))
        for reg in regimes:
            nstar, method = nstar_for_regime(ns, med, reg, 8192, trunc)
            nst.row("P5", "untied", N, d, reg, f"{nstar:.6g}", method)
            log(f"P5 untied N={N} {reg}: n*={nstar:.1f} ({method})")
