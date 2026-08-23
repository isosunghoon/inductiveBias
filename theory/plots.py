"""Plot generation (spec 7) — reads results/{raw,nstar}.csv, writes results/plots/."""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import (C_tied, C_untied, LAMBDAS, RESULTS_DIR, read_csv, risk_theory)
from .panels import P1_NS, P2_LOG_EPOCHS, P2_SGD_LOG, P4_DS
from .core import nstar_from_curve

PLOTS = os.path.join(RESULTS_DIR, "plots")
LAM_REGIMES = [f"lam={lam:g}" for lam in LAMBDAS]


def load():
    raw = read_csv(os.path.join(RESULTS_DIR, "raw.csv"))
    for r in raw:
        r["N"] = int(r["N"]); r["d"] = int(r["d"]); r["n"] = int(r["n"])
        r["seed"] = int(r["seed"]); r["risk"] = float(r["risk"])
    nst = read_csv(os.path.join(RESULTS_DIR, "nstar.csv"))
    for r in nst:
        r["N"] = int(r["N"]); r["d"] = int(r["d"]); r["nstar"] = float(r["nstar"])
    return raw, nst


def median_curve(raw, panel, model, N, d, regime):
    by_n = {}
    for r in raw:
        if (r["panel"] == panel and r["model"] == model and r["N"] == N
                and r["d"] == d and r["regime"] == regime):
            by_n.setdefault(r["n"], []).append(r["risk"])
    ns = sorted(by_n)
    return ns, [float(np.median(by_n[n])) for n in ns]


def best_curve(raw, panel, model, N, d):
    """min over lambda regimes of the seed-median risk, per n."""
    meds = {}
    for reg in LAM_REGIMES:
        ns, ms = median_curve(raw, panel, model, N, d, reg)
        for n, m in zip(ns, ms):
            meds.setdefault(n, []).append(m)
    ns = sorted(meds)
    return ns, [min(meds[n]) for n in ns]


def _save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def p0_validation(raw, nst):
    from .panels import P0_BASE
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    Nfine = np.linspace(5, 18, 50)
    for model, color in (("tied", "C0"), ("untied", "C1")):
        Ns, stars = [], []
        for r in nst:
            if r["panel"] == "P0" and r["model"] == model and r["method"] == "interp":
                Ns.append(r["N"]); stars.append(r["nstar"])
        ax.loglog(Ns, stars, "o-", color=color, label=f"{model} measured")
        base = P0_BASE[model]
        ax.loglog(list(base), list(base.values()), "x", color=color, ms=10,
                  label=f"{model} baseline")
        C = C_tied(Nfine, 48) if model == "tied" else C_untied(Nfine, 48)
        ax.loglog(Nfine, C, "--", color=color, alpha=0.5, label=f"{model} theory n*=C")
    ax.set_xlabel("N"); ax.set_ylabel("n* (R=0.5)")
    ax.set_title("P0 validation: one-step n*(N), d=48")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)
    _save(fig, "p0_validation.png")


def p1_nstar_loglog(nst):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, ("tied", "untied")):
        for i, reg in enumerate(LAM_REGIMES + ["best"]):
            Ns, stars = [], []
            for r in nst:
                if (r["panel"] == "P1" and r["model"] == model
                        and r["regime"] == reg and r["method"] == "interp"):
                    Ns.append(r["N"]); stars.append(r["nstar"])
            if not Ns:
                continue
            if reg == "best":
                ax.loglog(Ns, stars, "k-o", lw=2.5, ms=6, label="best-lambda")
            else:
                ax.loglog(Ns, stars, "o--", ms=4, alpha=0.8, color=f"C{i}", label=reg)
        Nfine = np.linspace(5, 26, 50)
        C = C_tied(Nfine, 48) if model == "tied" else C_untied(Nfine, 48)
        ax.loglog(Nfine, C, ":", color="gray", label="one-step theory n*=C")
        ax.set_xlabel("N"); ax.set_title(model)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("n* (R=0.5)")
    fig.suptitle("P1: ridge-path n*(N), d=48")
    _save(fig, "p1_nstar_loglog.png")


def _heatmap(raw, which, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, ("tied", "untied")):
        cols = {}
        for N in P1_NS:
            if which == "lam0":
                ns, ms = median_curve(raw, "P1", model, N, 48, "lam=0")
            else:
                ns, ms = best_curve(raw, "P1", model, N, 48)
            cols[N] = dict(zip(ns, ms))
        # each N was swept on its own adaptive grid; interpolate onto a common
        # log grid so the R=0.5 contour is drawable
        allns = [n for c in cols.values() for n in c]
        if not allns:
            continue
        ngrid = np.geomspace(min(allns), max(allns), 120)
        Z = np.full((len(ngrid), len(P1_NS)), np.nan)
        for j, N in enumerate(P1_NS):
            ns = np.array(sorted(cols[N]))
            if len(ns) < 2:
                continue
            rs = np.array([cols[N][n] for n in ns])
            inside = (ngrid >= ns[0]) & (ngrid <= ns[-1])
            Z[inside, j] = np.interp(np.log(ngrid[inside]), np.log(ns), rs)
        Zm = np.ma.masked_invalid(Z)
        pc = ax.pcolormesh(P1_NS, ngrid, Zm, shading="nearest",
                           vmin=0, vmax=1, cmap="viridis")
        try:
            ax.contour(P1_NS, ngrid, Zm, levels=[0.5], colors="red", linewidths=2)
        except Exception:
            pass
        n_union = ngrid
        Nfine = np.linspace(min(P1_NS), max(P1_NS), 50)
        C = C_tied(Nfine, 48) if model == "tied" else C_untied(Nfine, 48)
        ax.plot(Nfine, C, "w--", lw=1.5, label="one-step theory R=0.5 (n=C)")
        ax.set_yscale("log")
        ax.set_ylim(min(n_union), max(n_union))
        ax.set_xlabel("N"); ax.set_ylabel("n"); ax.set_title(model)
        ax.legend(fontsize=7, loc="lower right")
        fig.colorbar(pc, ax=ax, label="median risk")
    fig.suptitle(f"P1 risk(n, N) heatmap — {'lambda=0' if which == 'lam0' else 'best-lambda'}"
                 " (red: measured R=0.5)")
    _save(fig, fname)


def p1_heatmaps(raw):
    _heatmap(raw, "lam0", "p1_heatmap_lam0.png")
    _heatmap(raw, "best", "p1_heatmap_best.png")


def p2_gd_vs_ridge(raw):
    configs = [("tied", 8, 1024), ("untied", 8, 1024), ("tied", 16, 4096), ("untied", 16, 4096)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (model, N, n) in zip(axes.flat, configs):
        for kind, style in (("gd", "o-"), ("ridge", "s--")):
            xs, ys = [], []
            for t in P2_LOG_EPOCHS:
                ns_, ms = median_curve(raw, "P2", model, N, 48, f"{kind}_epoch={t}")
                if ns_:
                    xs.append(t); ys.append(ms[0])
            ax.loglog(xs, ys, style, label=f"{kind} (median 3 seeds)")
        ax.set_title(f"{model}, N={N}, n={n}")
        ax.set_xlabel("epoch t  (ridge at lambda=1/(lr t))"); ax.set_ylabel("risk")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    ax = axes.flat[4]
    lrs = sorted({m.group(1) for r in raw if r["panel"] == "P2"
                  for m in [re.match(r"sgd_lr=([^_]+)_epoch=", r["regime"])] if m})
    for lr in lrs:
        xs, ys = [], []
        for t in P2_SGD_LOG:
            ns_, ms = median_curve(raw, "P2", "untied", 16, 48, f"sgd_lr={lr}_epoch={t}")
            if ns_:
                xs.append(t); ys.append(ms[0])
        ax.loglog(xs, ys, "^-", label=f"SGD lr={lr}")
    xs, ys = [], []
    for t in P2_SGD_LOG:
        ns_, ms = median_curve(raw, "P2", "untied", 16, 48, f"gd_epoch={t}")
        if ns_:
            xs.append(t); ys.append(ms[0])
    ax.loglog(xs, ys, "k--", label="full-batch GD")
    ax.set_title("SGD spot: untied, N=16, n=4096, batch 64")
    ax.set_xlabel("epoch"); ax.set_ylabel("risk")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    axes.flat[5].axis("off")
    fig.suptitle("P2: literal GD vs ridge equivalence")
    _save(fig, "p2_gd_vs_ridge.png")


def _per_seed_nstars(raw, panel, model, N, d, regime, seeds):
    by_seed = {}
    for r in raw:
        if (r["panel"] == panel and r["model"] == model and r["N"] == N
                and r["d"] == d and r["regime"] == regime):
            by_seed.setdefault(r["seed"], {})[r["n"]] = r["risk"]
    out = []
    for s in range(seeds):
        if s not in by_seed:
            continue
        ns = sorted(by_seed[s])
        v, m = nstar_from_curve(ns, [by_seed[s][n] for n in ns])
        if m in ("interp", "below_floor"):
            out.append(v)
    return out


def p4_dflat(raw):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for model, seeds, color in (("tied", 20, "C0"), ("untied", 10, "C1")):
        means, sds, ds = [], [], []
        for d in P4_DS:
            stars = _per_seed_nstars(raw, "P4", model, 8, d, "onestep", seeds)
            if stars:
                ds.append(d); means.append(np.mean(stars)); sds.append(np.std(stars))
        ax.errorbar(ds, means, yerr=sds, fmt="o-", color=color, capsize=4,
                    label=f"{model} (per-seed mean±sd)")
        dfine = np.linspace(min(P4_DS), max(P4_DS), 50)
        C = C_tied(8, dfine) if model == "tied" else C_untied(8, dfine)
        ax.plot(dfine, C, "--", color=color, alpha=0.5, label=f"{model} theory C(d)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("d"); ax.set_ylabel("n* (R=0.5)")
    ax.set_title("P4: d-flatness of one-step n*, N=8")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    _save(fig, "p4_dflat.png")


def p4_overlay(raw):
    N = 8
    fig, axes = plt.subplots(2, len(P4_DS), figsize=(4 * len(P4_DS), 7), sharey=True)
    for row, model in enumerate(("tied", "untied")):
        if model == "tied":
            C_lead, C_full = N * (N + 2), N * (N + 2) + 5
            lab_lead, lab_full = "C=N(N+2)", "C_t=N(N+2)+5"
        else:
            C_lead, C_full = N * N * (N + 2), N * N * (N + 2) + 6 * N - 1
            lab_lead, lab_full = "C=N^2(N+2)", "C_u=N^2(N+2)+6N-1"
        for col, d in enumerate(P4_DS):
            ax = axes[row, col]
            ns, ms = median_curve(raw, "P4", model, N, d, "onestep")
            if ns:
                ax.semilogx(ns, ms, "o", ms=4, label="measured (median)")
                nf = np.geomspace(min(ns), max(ns), 100)
                ax.semilogx(nf, risk_theory(nf, C_lead), "--", label=lab_lead)
                ax.semilogx(nf, risk_theory(nf, C_full), "-", alpha=0.7, label=lab_full)
            ax.axhline(0.5, color="gray", lw=0.5)
            ax.set_title(f"{model}, d={d}", fontsize=9)
            ax.set_xlabel("n")
            if col == 0:
                ax.set_ylabel("risk")
                ax.legend(fontsize=7)
    fig.suptitle("P4: R(n) vs theory C/(n+C), N=8 (no fitted constants)")
    _save(fig, "p4_overlay.png")


def p3_softmax(raw):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, N in zip(axes, (8, 16)):
        ns, ms = median_curve(raw, "P3", "softmax", N, 48, "adam2000")
        if ns:
            ax.semilogx(ns, ms, "o-", label="softmax + Adam (2000 steps)")
        bns, bms = best_curve(raw, "P1", "tied", N, 48)
        if bns:
            ax.semilogx(bns, bms, "s--", label="tied linear, best-lambda (P1)")
        allns = (ns or []) + (bns or [])
        if allns:
            nf = np.geomspace(min(allns), max(allns), 100)
            ax.semilogx(nf, risk_theory(nf, C_tied(N, 48)), ":", color="gray",
                        label="tied one-step theory")
        ax.axhline(1 - 1 / N, color="brown", lw=1, ls="-.", label="mixer floor 1-1/N")
        ax.axhline(0.5, color="gray", lw=0.5)
        ax.set_title(f"N={N}"); ax.set_xlabel("n")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("risk")
    fig.suptitle("P3: softmax attention reference vs tied-linear, d=48")
    _save(fig, "p3_softmax.png")


def make_all(log=print):
    raw, nst = load()
    for fn in (lambda: p0_validation(raw, nst), lambda: p1_nstar_loglog(nst),
               lambda: p1_heatmaps(raw), lambda: p2_gd_vs_ridge(raw),
               lambda: p4_dflat(raw), lambda: p4_overlay(raw),
               lambda: p3_softmax(raw)):
        try:
            fn()
        except Exception as e:  # a missing panel shouldn't kill the others
            log(f"plot failed: {e!r}")
    log(f"plots written to {PLOTS}")
