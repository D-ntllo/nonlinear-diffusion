import numpy as np
from typing import Optional, Tuple, Dict, Any

from scipy.optimize import minimize_scalar

from check_k2 import check_k2_consistent


def _k2_boundary_loss(F, SO, D, Dp, Dpp, k2: float) -> float:
    """
    Loss = (residual_numerical)^2 from check_k2_consistent.
    Returns +inf if the residual is not finite.
    """
    res = check_k2_consistent(F, SO, D, Dp, Dpp, k2)
    val = float(res.get("residual_numerical", np.nan))
    if not np.isfinite(val):
        return float("inf")
    return val * val


def scan_k2_boundary_loss(
    F,
    SO,
    D,
    Dp,
    Dpp,
    low: float,
    high: float,
    n: int = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (k2_grid, loss_grid) for the squared boundary residual on [low, high].
    """
    if high <= low:
        raise ValueError("scan bounds must satisfy high > low.")
    if n < 3:
        raise ValueError("scan n must be >= 3.")
    grid = np.linspace(float(low), float(high), int(n))
    losses = np.array([_k2_boundary_loss(F, SO, D, Dp, Dpp, x) for x in grid], dtype=float)
    return grid, losses


def optimize_k2_boundary_loss(
    F,
    SO,
    D,
    Dp,
    Dpp,
    *,
    k2_guess: Optional[float] = None,
    bounds: Optional[Tuple[float, float]] = None,
    scan: Optional[Tuple[float, float, int]] = None,
    tol: float = 1e-8,
    maxiter: int = 200,
    expand: bool = True,
    expand_factor: float = 2.0,
    max_expand: int = 8,
    print_residual: bool = True,
    return_diag: bool = False,
) -> float:
    """
    Minimize the squared boundary residual (residual_numerical**2) with respect to K2.

    Parameters
    ----------
    k2_guess : optional float
        Used to center a default scan/bounds if none are provided.
    bounds : optional (low, high)
        If provided, uses bounded minimization on this interval.
    scan : optional (low, high, n)
        Coarse scan to pick a reasonable interval if bounds is not provided.
    tol : float
        Tolerance passed to the scalar minimizer.
    maxiter : int
        Maximum iterations for the scalar minimizer.

    Returns
    -------
    float
        K2 value that minimizes the squared boundary residual.
    """
    diag: Dict[str, Any] = {}

    if bounds is None:
        if scan is None:
            if k2_guess is not None:
                scan = (k2_guess - 1.0, k2_guess + 1.0, 101)
            else:
                scan = (-1.0, 1.0, 101)

        lo, hi, n = scan
        scan_hist = []
        for _ in range(max(1, int(max_expand) + 1)):
            grid, losses = scan_k2_boundary_loss(F, SO, D, Dp, Dpp, lo, hi, int(n))
            if not np.any(np.isfinite(losses)):
                raise RuntimeError("All losses were non-finite; adjust scan/bounds.")

            best_idx = int(np.nanargmin(losses))
            scan_hist.append(
                dict(
                    bounds=(float(lo), float(hi)),
                    best_idx=best_idx,
                    best_k2=float(grid[best_idx]),
                    best_loss=float(losses[best_idx]),
                )
            )

            # If the minimum is internal, bracket it locally for a unimodal search.
            if 0 < best_idx < len(grid) - 1:
                bounds = (float(grid[best_idx - 1]), float(grid[best_idx + 1]))
                break

            if not expand:
                bounds = (float(grid[0]), float(grid[-1]))
                break

            # Expand the interval outward in the direction of the boundary minimum.
            width = float(hi - lo)
            if best_idx == 0:
                lo = float(lo - expand_factor * width)
            else:
                hi = float(hi + expand_factor * width)
        else:
            bounds = (float(lo), float(hi))

        diag.update(scan_history=scan_hist)

    if bounds[1] <= bounds[0]:
        raise ValueError("bounds must satisfy high > low.")

    result = minimize_scalar(
        lambda x: _k2_boundary_loss(F, SO, D, Dp, Dpp, x),
        bounds=bounds,
        method="bounded",
        options={"xatol": tol, "maxiter": int(maxiter)},
    )

    k2_opt = float(result.x)
    if print_residual:
        res = check_k2_consistent(F, SO, D, Dp, Dpp, k2_opt)
        res_num = float(res.get("residual_numerical", float("nan")))
        print(f"k2_opt={k2_opt:.12g} residual_numerical={res_num:.12g} loss={res_num*res_num:.12g}")
    if return_diag:
        diag.update(bounds_used=(float(bounds[0]), float(bounds[1])), opt_loss=float(result.fun))
        return k2_opt, diag
    return k2_opt
