"""Non-negative least squares (Lawson-Hanson), pure numpy.

Replaces scipy.optimize.nnls (scipy/sklearn won't build from source on the
python/3.12 module here). Solves  min_x ||A x - b||_2  s.t.  x >= 0.
Standard active-set algorithm; converges in finite steps for full-rank A.
"""

from __future__ import annotations

import numpy as np


def nnls(A, b, max_iter=None, tol=1e-10):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape
    if max_iter is None:
        max_iter = 3 * n
    x = np.zeros(n)
    P = np.zeros(n, dtype=bool)        # passive (free, >0) set
    w = A.T @ (b - A @ x)              # gradient
    it = 0
    while (~P).any() and np.max(w[~P]) > tol and it < max_iter * 10:
        it += 1
        # add the most-correlated currently-zero coordinate to the passive set
        cand = np.where(~P, w, -np.inf)
        j = int(np.argmax(cand))
        P[j] = True
        # inner loop: solve unconstrained LS on passive set, fix negatives
        while True:
            Ap = A[:, P]
            s_p, *_ = np.linalg.lstsq(Ap, b, rcond=None)
            if (s_p > 0).all():
                x[P] = s_p
                x[~P] = 0.0
                break
            # move toward s as far as feasibility allows
            s = np.zeros(n)
            s[P] = s_p
            mask = P & (s <= 0)
            alpha = np.min(x[mask] / (x[mask] - s[mask]))
            x = x + alpha * (s - x)
            P[x <= tol] = False
        w = A.T @ (b - A @ x)
    return x, float(np.linalg.norm(A @ x - b))
