"""Shared primitives for the tied/untied associative-recall toy experiment (spec v1.1)."""
import csv
import hashlib
import os
import time

import numpy as np

SQRT2 = 2.0 ** 0.5
LAMBDAS = [1e2, 1.0, 1e-2, 1e-4, 1e-6, 0.0]
EVAL_M = 4000
CAP = {"tied": 12288, "untied": 8192}
CELL_TIME_LIMIT = 600.0
RESULTS_DIR = os.environ.get("TOY_RESULTS", "results")


# spec 8: hash() is process-salted for strings, which would break reproducibility;
# md5 keeps the same (panel, model, N, d, n, seed) -> seed pattern but stable.
def stable_seed(*key):
    h = hashlib.md5(repr(key).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


def rng_for(*key):
    return np.random.default_rng(stable_seed(*key))


# ---- spec 5: verified snippets (verbatim) ----
def gen_batch(B, N, d, rng):
    G = rng.standard_normal((B, d, N)).astype(np.float32)
    K, _ = np.linalg.qr(G)
    v = rng.standard_normal((B, N)).astype(np.float32)
    i = rng.integers(0, N, size=B)
    q = K[np.arange(B), :, i]
    y = v[np.arange(B), i]
    return K, v, q, y


def one_step(K, v, q, y, tied):
    B = K.shape[0]
    if tied:
        u = np.einsum('bdn,bn->bd', K, v)
        return np.einsum('b,bd,be->de', y, q, u) / B
    w = v * y[:, None]
    return np.einsum('bn,bd,ben->nde', w, q, K) / B


def risk_pred(f, y):
    f = np.asarray(f, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        return 1.0
    num = float(f @ y) ** 2
    den = float(f @ f) * float(y @ y)
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return 1.0
    return 1.0 - num / den


def predict(A, K, v, q, tied):
    if tied:
        u = np.einsum('bdn,bn->bd', K, v)
        return np.einsum('bd,de,be->b', q, A, u)
    return np.einsum('bd,nde,ben,bn->b', q, A, K, v)


# ---- spec 2: theory formulas (no fitted constants) ----
def C_tied(N, d):
    Phi = (N + 8) / (d * (d + 2)) + (N - 1) * ((d + 1) * (N + 2) - 6) / (d * (d + 2) * (d - 1))
    return Phi * d * d - 1


def C_untied(N, d):
    Psi = (1 / N) * (9 / (d * (d + 2)) + (N - 1) * (d + 1) / (d * (d + 2) * (d - 1))) \
        + (1 - 1 / N) * (4 / (d * (d + 2)) + (N - 2) * (d + 1) / (d * (d + 2) * (d - 1)))
    return Psi * (N * d) ** 2 - 1


def risk_theory(n, C):
    return C / (np.asarray(n, dtype=np.float64) + C)


# ---- n* machinery ----
def sweep(compute, n0, cap, seeds, ratio=SQRT2, time_limit=CELL_TIME_LIMIT,
          r_stop=0.45, n_floor=4, on_row=None, down_check="median"):
    """Adaptive geometric sweep over n.

    compute(n, seed) -> {regime: risk}. Extends the grid upward until every
    regime's seed-median risk < r_stop (or cap / time limit), and downward until
    the R=0.5 crossing is bracketed at the smallest n (median criterion, or
    every seed if down_check="all_seeds").
    Returns (ns, vals, med, truncated); vals[n][regime] is the per-seed risk
    list, med[n][regime] the seed-median.
    """
    t0 = time.time()
    vals = {}
    med = {}

    def eval_n(n):
        per = {}
        for s in range(seeds):
            out = compute(n, s)
            for reg, val in out.items():
                per.setdefault(reg, []).append(val)
                if on_row is not None:
                    on_row(n, s, reg, val)
        vals[n] = per
        med[n] = {reg: float(np.median(v)) for reg, v in per.items()}

    def need_down(n):
        if down_check == "all_seeds":
            return min(min(v) for v in vals[n].values()) < 0.5
        return min(med[n].values()) < 0.5

    ns = [max(n_floor, int(round(n0)))]
    eval_n(ns[0])
    truncated = False
    while max(med[ns[-1]].values()) >= r_stop and ns[-1] < cap:
        if time.time() - t0 > time_limit:
            truncated = True
            break
        nxt = min(cap, max(ns[-1] + 1, int(round(ns[-1] * ratio))))
        ns.append(nxt)
        eval_n(nxt)
    while need_down(ns[0]) and ns[0] > n_floor:
        if time.time() - t0 > time_limit:
            truncated = True
            break
        prv = max(n_floor, int(round(ns[0] / ratio)))
        if prv >= ns[0]:
            break
        ns.insert(0, prv)
        eval_n(prv)
    return ns, vals, med, truncated


def nstar_from_curve(ns, rs, truncated=False, reached_cap=False):
    """First crossing of R=0.5 while sweeping n upward; log-linear interpolation."""
    for i in range(1, len(ns)):
        if rs[i - 1] >= 0.5 and rs[i] < 0.5:
            t = (rs[i - 1] - 0.5) / (rs[i - 1] - rs[i])
            ln = np.log(ns[i - 1]) + t * (np.log(ns[i]) - np.log(ns[i - 1]))
            return float(np.exp(ln)), "interp"
    if rs and rs[0] < 0.5:
        return float(ns[0]), "below_floor"
    if reached_cap and not truncated:
        return float("nan"), ">cap"
    return float("nan"), "skipped" if truncated else ">cap"


def nstar_for_regime(ns, med, regime, cap, truncated):
    rs = [med[n][regime] for n in ns]
    return nstar_from_curve(ns, rs, truncated=truncated, reached_cap=(ns[-1] >= cap))


def fit_slope(logN, lognstar):
    logN = np.asarray(logN, dtype=np.float64)
    lognstar = np.asarray(lognstar, dtype=np.float64)
    A = np.vstack([logN, np.ones_like(logN)]).T
    coef, *_ = np.linalg.lstsq(A, lognstar, rcond=None)
    yhat = A @ coef
    dof = max(len(logN) - 2, 1)
    s2 = float(((lognstar - yhat) ** 2).sum()) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


# ---- results IO (spec 7 schema) ----
class Csv:
    def __init__(self, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if fresh:
            self.w.writerow(header)
            self.f.flush()

    def row(self, *vals):
        self.w.writerow(vals)
        self.f.flush()

    def close(self):
        self.f.close()


def raw_writer():
    return Csv(os.path.join(RESULTS_DIR, "raw.csv"),
               ["panel", "model", "N", "d", "n", "seed", "regime", "risk"])


def nstar_writer():
    return Csv(os.path.join(RESULTS_DIR, "nstar.csv"),
               ["panel", "model", "N", "d", "regime", "nstar", "method"])


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
