"""Literal iterative training (numpy): full-batch GD, SGD, Adam, softmax model."""
import numpy as np

from .core import risk_pred


# ---- linear models: BLAS-friendly forward/grad ----
def fwd_tied(A, Q, U):
    return ((Q @ A) * U).sum(1)


def grad_tied(r, Q, U, n):
    return Q.T @ (r[:, None] * U) / n


def fwd_untied(A, Q, Kt, V):
    M = np.empty((Q.shape[0], A.shape[0]), dtype=np.float64)
    for i in range(A.shape[0]):
        M[:, i] = ((Q @ A[i]) * Kt[:, :, i]).sum(1)
    return (M * V).sum(1)


def grad_untied(r, Q, Kt, V, n):
    N, d = V.shape[1], Q.shape[1]
    g = np.empty((N, d, d), dtype=np.float64)
    w = r[:, None] * V
    for i in range(N):
        g[i] = Q.T @ (w[:, i:i + 1] * Kt[:, :, i]) / n
    return g


def estimate_L(model, K, v_, q, rng, iters=200):
    """Smoothness L = lambda_max(J^T J / n) for loss ||f-y||^2/(2n).

    J J^T is exactly the kernel Gram, so lambda_max(G)/n gives L. Power iteration
    converges from below, so too few iterations underestimates L and makes the
    spec's lr = 0.9*2/L exceed the stability limit — hence the generous iters.
    """
    from .ridge import features, gram
    n = K.shape[0]
    G = gram(features(model, K, v_, q), features(model, K, v_, q))
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(iters):
        y = G @ x
        ny = float(np.linalg.norm(y))
        if ny == 0:
            return 0.0
        x = y / ny
        lam = ny
    return lam / n


def run_gd(model, train, eval_, log_epochs, lr=None, rng=None):
    """Full-batch GD from 0. Returns (lr, {epoch: eval_risk})."""
    K, v, q, y = train
    Ke, ve, qe, ye = eval_
    Q = q.astype(np.float64)
    Qe = qe.astype(np.float64)
    n = Q.shape[0]
    if model == "tied":
        U = np.einsum('bdn,bn->bd', K, v).astype(np.float64)
        Ue = np.einsum('bdn,bn->bd', Ke, ve).astype(np.float64)
        aux, aux_e = U, Ue
        A = np.zeros((Q.shape[1], Q.shape[1]))
        fwd = lambda A, Qx, ax: fwd_tied(A, Qx, ax)
        grad = lambda r: grad_tied(r, Q, U, n)
    else:
        Kt = K.astype(np.float64)
        Kte = Ke.astype(np.float64)
        V = v.astype(np.float64)
        Ve = ve.astype(np.float64)
        aux, aux_e = (Kt, V), (Kte, Ve)
        A = np.zeros((V.shape[1], Q.shape[1], Q.shape[1]))
        fwd = lambda A, Qx, ax: fwd_untied(A, Qx, ax[0], ax[1])
        grad = lambda r: grad_untied(r, Q, Kt, V, n)
    if lr is None:
        lr = 0.9 * 2.0 / estimate_L(model, K, v, q, rng)
    risks = {}
    T = max(log_epochs)
    for t in range(1, T + 1):
        r = fwd(A, Q, aux) - y
        A -= lr * grad(r)
        if t in log_epochs:
            risks[t] = risk_pred(fwd(A, Qe, aux_e), ye)
    return lr, risks


def run_sgd(model, train, eval_, log_epochs, lr, batch, rng):
    """Mini-batch SGD from 0 (untied or tied)."""
    K, v, q, y = train
    Ke, ve, qe, ye = eval_
    Q = q.astype(np.float64)
    Qe = qe.astype(np.float64)
    n = Q.shape[0]
    Kt = K.astype(np.float64)
    V = v.astype(np.float64)
    Kte = Ke.astype(np.float64)
    Ve = ve.astype(np.float64)
    if model == "tied":
        U = np.einsum('bdn,bn->bd', K, v).astype(np.float64)
        Ue = np.einsum('bdn,bn->bd', Ke, ve).astype(np.float64)
        A = np.zeros((Q.shape[1], Q.shape[1]))
    else:
        A = np.zeros((V.shape[1], Q.shape[1], Q.shape[1]))
    risks = {}
    T = max(log_epochs)
    for t in range(1, T + 1):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            if model == "tied":
                r = fwd_tied(A, Q[idx], U[idx]) - y[idx]
                A -= lr * grad_tied(r, Q[idx], U[idx], len(idx))
            else:
                r = fwd_untied(A, Q[idx], Kt[idx], V[idx]) - y[idx]
                A -= lr * grad_untied(r, Q[idx], Kt[idx], V[idx], len(idx))
        if t in log_epochs:
            if model == "tied":
                risks[t] = risk_pred(fwd_tied(A, Qe, Ue), ye)
            else:
                risks[t] = risk_pred(fwd_untied(A, Qe, Kte, Ve), ye)
    return risks


class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, param, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        param -= self.lr * mh / (np.sqrt(vh) + self.eps)


def softmax_forward(W, Q, Kt, V, sqd):
    z = np.einsum('be,bei->bi', Q @ W, Kt) / sqd
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p, (p * V).sum(1)


def run_softmax_adam(train, eval_, d, steps=2000, lr=1e-2):
    """f = sum_i softmax(q^T W k / sqrt(d))_i v_i, W0 = 0, full-batch Adam."""
    K, v, q, y = train
    Ke, ve, qe, ye = eval_
    Q = q.astype(np.float64)
    Kt = K.astype(np.float64)
    V = v.astype(np.float64)
    Qe = qe.astype(np.float64)
    Kte = Ke.astype(np.float64)
    Ve = ve.astype(np.float64)
    n = Q.shape[0]
    sqd = np.sqrt(d)
    W = np.zeros((d, d))
    opt = Adam(W.shape, lr)
    for _ in range(steps):
        p, f = softmax_forward(W, Q, Kt, V, sqd)
        r = f - y
        coef = (r[:, None] * p * (V - f[:, None])) / (sqd * n)
        g = Q.T @ np.einsum('bi,bei->be', coef, Kt)
        opt.step(W, g)
    _, fe = softmax_forward(W, Qe, Kte, Ve, sqd)
    return risk_pred(fe, ye)


def run_untied_adam(train, eval_, lr, max_steps=5000, tol=1e-7):
    """P5: untied linear model trained with full-batch Adam to convergence."""
    K, v, q, y = train
    Ke, ve, qe, ye = eval_
    Q = q.astype(np.float64)
    Kt = K.astype(np.float64)
    V = v.astype(np.float64)
    Qe = qe.astype(np.float64)
    Kte = Ke.astype(np.float64)
    Ve = ve.astype(np.float64)
    n = Q.shape[0]
    A = np.zeros((V.shape[1], Q.shape[1], Q.shape[1]))
    opt = Adam(A.shape, lr)
    prev = np.inf
    check = 100
    for t in range(1, max_steps + 1):
        r = fwd_untied(A, Q, Kt, V) - y
        opt.step(A, grad_untied(r, Q, Kt, V, n))
        if t % check == 0:
            loss = float(r @ r) / (2 * n)
            if prev - loss < tol:
                break
            prev = loss
    return risk_pred(fwd_untied(A, Qe, Kte, Ve), ye)
