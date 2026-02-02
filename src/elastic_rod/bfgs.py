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

def backtracking_line_search(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    tau: float = 0.5,
    max_steps: int = 30,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Simple Armijo backtracking line search.
    Returns (alpha, f_new, g_new, n_feval_increment).
    """
    gtp = float(g @ p)
    alpha = float(alpha0)
    n_feval_inc = 0
    f_new = f
    g_new = g
    for _ in range(max_steps):
        x_new = x + alpha * p
        f_new, g_new = f_and_g(x_new)
        n_feval_inc += 1
        if f_new <= f + c1 * alpha * gtp:
            break
        alpha *= tau
    return alpha, f_new, g_new, n_feval_inc

def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Minimize f(x) with BFGS.

    You should:
    - maintain an approximation H_k to the inverse Hessian
    - compute p_k = -H_k g_k
    - perform a line search to get step alpha_k
    - update x, f, g
    - update H via the BFGS formula (with curvature checks)

    Return BFGSResult with a small iteration history useful for plotting.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    n_feval = 1

    n = x.size
    H = np.eye(n)  # inverse Hessian approximation

    hist = {"f": [f], "gnorm": [np.linalg.norm(g)], "alpha": []}

    for k in range(max_iter):
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval,
                              converged=True, history=hist)

        # Search direction
        p = -H @ g

        # Line search
        alpha, f_new, g_new, inc = backtracking_line_search(
            f_and_g, x, f, g, p, alpha0=alpha0
        )
        n_feval += inc
        x_new = x + alpha * p
        hist["alpha"].append(alpha)

        # Update step
        s = x_new - x
        y = g_new - g
        ys = float(y @ s)
        if ys > 0.0:
            rho = 1.0 / ys
            Hy = H @ y
            yHy = float(y @ Hy)
            H += (1.0 + yHy * rho) * rho * np.outer(s, s) - rho * (np.outer(Hy, s) + np.outer(s, Hy))

        x = x_new
        f = f_new
        g = g_new
        hist["f"].append(f)
        hist["gnorm"].append(np.linalg.norm(g))

    return BFGSResult(x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval,
                      converged=False, history=hist)
