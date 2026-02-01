from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

ValueGrad = Callable[[np.ndarray], Tuple[float, np.ndarray]]


@dataclass
class BFGSResult:
    x: np.ndarray
    f: float
    g: np.ndarray
    n_iter: int
    n_feval: int
    converged: bool
    history: Dict[str, Any]


def strong_wolfe_line_search(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 25,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Strong Wolfe line search:
      f(x+αp) <= f(x) + c1 α g^T p
      |∇f(x+αp)^T p| <= c2 |g^T p|
    Returns (alpha, f_new, g_new, n_feval_inc).
    """
    gTp0 = float(np.dot(g, p))
    if not np.isfinite(gTp0) or gTp0 >= 0.0:
        p = -g
        gTp0 = float(np.dot(g, p))
    if gTp0 == 0.0 or not np.isfinite(gTp0):
        return 0.0, f, g, 0

    nfe = 0
    alpha_prev = 0.0
    f_prev = f

    def phi(a: float):
        nonlocal nfe
        xa = x + a * p
        fa, ga = f_and_g(xa)
        nfe += 1
        return float(fa), np.asarray(ga, dtype=np.float64)

    def zoom(alo: float, ahi: float, flo: float):
        for _ in range(max_iter):
            aj = 0.5 * (alo + ahi)
            fj, gj = phi(aj)
            gTpj = float(np.dot(gj, p))

            if (fj > f + c1 * aj * gTp0) or (fj >= flo):
                ahi = aj
            else:
                if abs(gTpj) <= c2 * abs(gTp0):
                    return aj, fj, gj
                if gTpj * (ahi - alo) >= 0:
                    ahi = alo
                alo, flo = aj, fj

            if abs(ahi - alo) < 1e-16:
                break

        fj, gj = phi(alo)
        return alo, fj, gj

    alpha = float(alpha0)

    for it in range(max_iter):
        f_new, g_new = phi(alpha)

        if (f_new > f + c1 * alpha * gTp0) or (it > 0 and f_new >= f_prev):
            a, fn, gn = zoom(alpha_prev, alpha, f_prev)
            return a, fn, gn, nfe

        gTp = float(np.dot(g_new, p))
        if abs(gTp) <= c2 * abs(gTp0):
            return alpha, f_new, g_new, nfe

        if gTp >= 0.0:
            a, fn, gn = zoom(alpha, alpha_prev, f_new)
            return a, fn, gn, nfe

        alpha_prev = alpha
        f_prev = f_new
        alpha *= 2.0
        if alpha > 1e6:
            break

    return 0.0, f, g, nfe


def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Algorithm 6.3-style BFGS:
      p_k = -H_k g_k
      alpha_k from Strong Wolfe
      inverse-Hessian update with curvature check
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    n_feval = 1

    n = x.size
    H = np.eye(n, dtype=np.float64)

    hist: Dict[str, Any] = {"f": [f], "gnorm": [float(np.linalg.norm(g))], "alpha": []}
    min_curv = 1e-12

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval, converged=True, history=hist)

        p = -H @ g
        if float(np.dot(g, p)) >= 0.0 or not np.all(np.isfinite(p)):
            p = -g

        alpha, f_new, g_new, inc = strong_wolfe_line_search(
            f_and_g, x, f, g, p, alpha0=alpha0, c1=1e-4, c2=0.9
        )
        n_feval += inc
        hist["alpha"].append(float(alpha))

        if alpha == 0.0:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval, converged=False, history=hist)

        s = alpha * p
        x_new = x + s
        y = g_new - g

        x = x_new
        f = float(f_new)
        g = np.asarray(g_new, dtype=np.float64)

        hist["f"].append(f)
        hist["gnorm"].append(float(np.linalg.norm(g)))

        yTs = float(np.dot(y, s))
        if np.isfinite(yTs) and yTs > min_curv:
            rho = 1.0 / yTs
            I = np.eye(n, dtype=np.float64)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
            H = 0.5 * (H + H.T)
        else:
            H = np.eye(n, dtype=np.float64)

    return BFGSResult(x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval, converged=False, history=hist)
