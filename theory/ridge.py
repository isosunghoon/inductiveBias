"""Kernel ridge path (spec 3): exact Gram computation + one eigh per (model,N,n,seed).

The untied kernel k(a,b) = (q_a.q_b) * sum_i v_ai v_bi (k_ai.k_bi) equals
(q_a.q_b) * <T_a, T_b>_F with T = K diag(v), so the Gram is two BLAS matmuls
on flattened T — algebraically identical to the spec's chunked einsum, cheaper.
"""
import numpy as np

from .core import EVAL_M, LAMBDAS, gen_batch, rng_for, risk_pred


def features(model, K, v, q):
    q64 = q.astype(np.float64)
    if model == "tied":
        u = np.einsum('bdn,bn->bd', K, v).astype(np.float64)
        return q64, u
    T = (K * v[:, None, :]).reshape(K.shape[0], -1).astype(np.float64)
    return q64, T


def gram(Fa, Fb):
    (qa, ta), (qb, tb) = Fa, Fb
    return (qa @ qb.T) * (ta @ tb.T)


def ridge_risks_from_data(model, train, eval_, n, lambdas=LAMBDAS, rcond=1e-10):
    """train/eval_ = (K, v, q, y). Returns {lam: risk} using one eigh."""
    K, v, q, y = train
    Ke, ve, qe, ye = eval_
    Ft = features(model, K, v, q)
    Fe = features(model, Ke, ve, qe)
    G = gram(Ft, Ft)
    Cx = gram(Fe, Ft)
    w, V = np.linalg.eigh(G)
    b = V.T @ y.astype(np.float64)
    wmax = max(float(w.max()), 1e-300)
    out = {}
    for lam in lambdas:
        if lam == 0.0:
            inv = np.where(w > rcond * wmax, 1.0 / np.maximum(w, 1e-300), 0.0)
            alpha = V @ (inv * b)
        else:
            alpha = V @ (b / (w + n * lam))
        f = Cx @ alpha
        out[lam] = risk_pred(f, ye)
    return out


def make_ridge_compute(panel, model, N, d, lambdas=LAMBDAS):
    """compute(n, seed) -> {"lam=...": risk}; eval set fixed per seed."""
    evals = {}

    def compute(n, seed):
        rng = rng_for(panel, model, N, d, n, seed)
        train = gen_batch(n, N, d, rng)
        if seed not in evals:
            evals[seed] = gen_batch(EVAL_M, N, d, rng_for(panel, model, N, d, "eval", seed))
        risks = ridge_risks_from_data(model, train, evals[seed], n, lambdas)
        return {f"lam={lam:g}": r for lam, r in risks.items()}

    return compute
