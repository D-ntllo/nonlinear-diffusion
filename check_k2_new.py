import numpy as np
import math
from typing import Tuple

from scipy.special import j1, y1, jvp, yvp
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d


# Solve (L_1 + κ^2) σ = f on [0;R0], regular at 0, with Dirichlet σ(R0)=target
def _solve_sigma_numerical(
    kappa: float,
    R0: float,
    f: np.ndarray,
    r: np.ndarray,
    target: float
) -> Tuple[np.ndarray, float]:
    """
    Solve (L_1 + kappa^2) sigma = f on [0, R0], regular at r=0, with Dirichlet sigma(R0)=target.

    Assumes the n=1 radial operator
        L_1[sigma] = sigma'' + (1/r) sigma' - (1/r^2) sigma.

    Regular at r=0 for n=1 implies sigma(0)=0 (and sigma ~ a r near 0).

    Returns
    -------
    sigma_vals : np.ndarray
        sigma evaluated at the input grid r (same shape as r).
    sigma_r_R0 : float
        derivative sigma'(R0).
    """
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    if r.ndim != 1 or f.ndim != 1 or r.shape != f.shape:
        raise ValueError("r and f must be 1D arrays of the same shape.")
    if R0 <= 0:
        raise ValueError("R0 must be positive.")
    if kappa < 0:
        raise ValueError("kappa must be nonnegative (use abs if needed).")

    # Sort by r
    idx = np.argsort(r)
    r_sorted = r[idx]
    f_sorted = f[idx]

    # Build an interpolant for f(r)
    f_of_r = interp1d(
        r_sorted, f_sorted,
        kind="cubic" if r_sorted.size >= 4 else "linear",
        fill_value="extrapolate",
        bounds_error=False,
        assume_sorted=True,
    )

    # Build the mesh for solve_bvp on [0, R0]
    # Use the user's r-grid where possible, but ensure endpoints exist.
    x = r_sorted.copy()

    # If user's r-grid doesn't exactly match [0,R0], augment and clip.
    # (We still evaluate the final solution back on the original r.)
    x = x[(x >= 0.0) & (x <= R0)]
    if x.size == 0:
        x = np.linspace(0.0, R0, 200)

    if x[0] > 0.0:
        x = np.concatenate(([0.0], x))
    if x[-1] < R0:
        x = np.concatenate((x, [R0]))

    # Ensure strictly increasing mesh for solve_bvp
    x = np.unique(x)
    if x.size < 5:
        x = np.linspace(0.0, R0, 200)

    # First-order system: y0 = sigma, y1 = sigma'
    def fun(xi, y):
        # y shape: (2, m)
        sigma = y[0]
        sigmap = y[1]
        rhs = f_of_r(xi)

        # Avoid the r=0 singular terms by using the regular-limit value at 0.
        term = np.zeros_like(xi)
        m = xi > 0
        # sigma'' = f - kappa^2*sigma - (1/r)*sigma' + (1/r^2)*sigma
        term[m] = (-sigmap[m] / xi[m]) + (sigma[m] / (xi[m] ** 2))

        sigma_pp = rhs - (kappa ** 2) * sigma + term
        return np.vstack((sigmap, sigma_pp))

    # Boundary conditions: regular at 0 => sigma(0)=0; Dirichlet at R0 => sigma(R0)=target
    def bc(ya, yb):
        return np.array([ya[0], yb[0] - target], dtype=float)

    # Initial guess: linear profile sigma ~ (target/R0) r
    y0_guess = target * (x / R0)
    y1_guess = np.full_like(x, target / R0)
    y_guess = np.vstack((y0_guess, y1_guess))

    sol = solve_bvp(
        fun, bc, x, y_guess,
        tol=1e-6,
        max_nodes=20000,
        verbose=0,
    )

    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")

    # Evaluate on the original r grid (in original order)
    sigma_vals_sorted = sol.sol(r_sorted)[0]
    sigma_vals = np.empty_like(sigma_vals_sorted)
    sigma_vals[idx] = sigma_vals_sorted  # unsort back to match input r

    sigma_r_R0 = float(sol.sol(R0)[1])
    return sigma_vals, sigma_r_R0
