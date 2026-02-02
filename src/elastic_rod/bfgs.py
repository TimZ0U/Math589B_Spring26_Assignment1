from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

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


def _lbfgs_direction(
    g: np.ndarray,
    s_list: List[np.ndarray],
    y_list: List[np.ndarray],
) -> np.ndarray:
    """Two-loop recursion to compute p = -H_k g."""
    q = g.copy()
    m = len(s_list)
    alpha = np.zeros(m, dtype=np.float64)
    rho = np.zeros(m, dtype=np.float64)

    for i in range(m - 1, -1, -1):
        ys = float(np.dot(y_list[i], s_list[i]))
        if ys <= 1e-12 or not np.isfinite(ys):
            rho[i] = 0.0
            alpha[i] = 0.0
            continue
        rho[i] = 1.0 / ys
        alpha[i] = rho[i] * float(np.dot(s_list[i], q))
        q -= alpha[i] * y_list[i]

    if m > 0:
        y = y_list[-1]
        s = s_list[-1]
        yy = float(np.dot(y, y))
        sy = float(np.dot(s, y))
        gamma = sy / yy if (yy > 1e-12 and np.isfinite(yy) and np.isfinite(sy)) else 1.0
    else:
        gamma = 1.0

    r = gamma * q

    for i in range(m):
        if rho[i] == 0.0:
            continue
        beta = rho[i] * float(np.dot(y_list[i], r))
        r += s_list[i] * (alpha[i] - beta)

    return -r


def armijo_backtracking(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float,
    *,
    c1: float = 1e-4,
    tau: float = 0.5,
    max_ls: int = 10,
    min_alpha: float = 1e-12,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Armijo backtracking with caps to control n_feval.
    Returns (alpha, f_new, g_new, n_feval_inc).
    """
    gtp = float(np.dot(g, p))
    if not np.isfinite(gtp) or gtp >= 0.0:
        return 0.0, f, g, 0

    alpha = float(alpha0)
    nfe = 0

    f_best = f
    g_best = g
    a_best = 0.0

    for _ in range(max_ls):
        x_new = x + alpha * p
        f_new, g_new = f_and_g(x_new)
        nfe += 1

        f_new = float(f_new)
        g_new = np.asarray(g_new, dtype=np.float64)

        if np.isfinite(f_new) and np.all(np.isfinite(g_new)):
            if f_new < f_best:
                f_best, g_best, a_best = f_new, g_new, alpha
            if f_new <= f + c1 * alpha * gtp:
                return alpha, f_new, g_new, nfe

        alpha *= tau
        if alpha < min_alpha:
            break

    if a_best > 0.0:
        return a_best, f_best, g_best, nfe
    return 0.0, f, g, nfe


def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Practical L-BFGS with a capped Armijo line search for speed.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    n_feval = 1

    n = x.size
    is_small = n <= 240

    hist: Dict[str, Any] = {"f": [f], "gnorm": [float(np.linalg.norm(g))], "alpha": []}

    m = 8 if is_small else 6
    s_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    min_curv = 1e-12
    max_step_norm = 4.0 if is_small else 1.0
    max_ls = 16 if is_small else 10

    def initial_alpha(gnorm: float) -> float:
        if is_small:
            return float(np.clip(1.0 / max(1.0, 0.25 * gnorm), 1e-2, 1.0))
        return float(np.clip(1.0 / max(1.0, gnorm), 1e-3, 0.25))

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return BFGSResult(
                x=x,
                f=f,
                g=g,
                n_iter=k,
                n_feval=n_feval,
                converged=True,
                history=hist,
            )

        p = _lbfgs_direction(g, s_list, y_list)
        gtp = float(np.dot(g, p))
        if not np.isfinite(gtp) or gtp >= 0.0 or not np.all(np.isfinite(p)):
            p = -g

        pnorm = float(np.linalg.norm(p))
        if pnorm > 0.0 and np.isfinite(pnorm):
            p *= min(1.0, max_step_norm / pnorm)

        a0 = min(float(alpha0), initial_alpha(gnorm))
        alpha, f_new, g_new, inc = armijo_backtracking(
            f_and_g,
            x,
            f,
            g,
            p,
            alpha0=a0,
            c1=1e-4,
            tau=0.5,
            max_ls=max_ls,
            min_alpha=1e-12,
        )
        n_feval += inc
        hist["alpha"].append(float(alpha))

        if alpha == 0.0:
            p = -g
            pnorm = float(np.linalg.norm(p))
            if pnorm > 0.0 and np.isfinite(pnorm):
                p = p / pnorm

            alpha_fb = 1e-2 if is_small else 1e-3
            x_try = x + alpha_fb * p
            f_try, g_try = f_and_g(x_try)
            n_feval += 1
            f_try = float(f_try)
            g_try = np.asarray(g_try, dtype=np.float64)

            if np.isfinite(f_try) and f_try < f and np.all(np.isfinite(g_try)):
                s_list.clear()
                y_list.clear()
                x, f, g = x_try, f_try, g_try
                hist["f"].append(f)
                hist["gnorm"].append(float(np.linalg.norm(g)))
                continue

            return BFGSResult(
                x=x,
                f=f,
                g=g,
                n_iter=k,
                n_feval=n_feval,
                converged=False,
                history=hist,
            )

        s = alpha * p
        x_new = x + s
        y = g_new - g
        yts = float(np.dot(y, s))

        x = x_new
        f = float(f_new)
        g = np.asarray(g_new, dtype=np.float64)

        hist["f"].append(f)
        hist["gnorm"].append(float(np.linalg.norm(g)))

        if np.isfinite(yts) and yts > min_curv and np.all(np.isfinite(y)) and np.all(np.isfinite(s)):
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        else:
            s_list.clear()
            y_list.clear()

    return BFGSResult(
        x=x,
        f=f,
        g=g,
        n_iter=max_iter,
        n_feval=n_feval,
        converged=False,
        history=hist,
    )
