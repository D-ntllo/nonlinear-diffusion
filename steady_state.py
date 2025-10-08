import numpy as np
from scipy import optimize, special
from typing import Callable, Dict, Tuple

J1   = lambda x: special.jv(1, x)
J1p  = lambda x: special.jvp(1, x, 1)
J1pp = lambda x: special.jvp(1, x, 2)


def solve_R0(P: float, gamma: float, rmax: float = 100.0) -> float:
    """Largest positive root of: 0 = -γ/R + 1 - πR^2 - P/(πR^2).  (eq. 13)"""
    def F(R):
        return -gamma/R + 1.0 - np.pi*R**2 - P/(np.pi*R**2)
    xs = np.geomspace(1e-4, rmax, 1200)
    vals = F(xs)
    roots = []
    for a,b,fa,fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa*fb < 0:
            roots.append(optimize.brentq(F, a, b))
    if not roots:
        raise RuntimeError("Failed to find R0; widen rmax or check (P, gamma).")
    return max(roots)

def solve_Khat0(
    P: float, Z: float, R0: float, m0: float,
    *,
    root_index: int = 0,
    Khat0_max: float = 5e3,
    n_samples: int = 20000,
) -> Tuple[float, float]:
    """
    Solve for Khat0 = K0 / D(m0) from (14)-(15) without D:
        G(Khat0) = P*m0 - (1/Khat0) * [ J1(α) / (α J1'(α)) ] = 0,
        α = (R0/√Z)*sqrt(P*m0*Khat0 - 1).

    Returns:
        (Khat0, alpha) for the chosen root_index (0 = principal).

    Notes:
    - Domain: Khat0 > 1/(P*m0) to make α real.
    - Skips singularities where J1'(α)=0 by ignoring NaN intervals.
    """
    if min(P, Z, R0, m0) <= 0:
        raise ValueError("P, Z, R0, m0 must be positive.")

    Kmin = (1.0 / (P * m0)) * (1 + 1e-12)  # tiny offset above threshold

    def alpha_of(Khat0: float) -> float:
        return (R0/np.sqrt(Z)) * np.sqrt(max(P*m0*Khat0 - 1.0, 0.0))

    def G(Khat0: float) -> float:
        if Khat0 <= Kmin:
            return np.nan
        a = alpha_of(Khat0)
        j1p = J1p(a)
        if not np.isfinite(a) or a == 0.0 or not np.isfinite(j1p) or j1p == 0.0:
            return np.nan  # singular; skip this point
        return P*m0 - (J1(a) / (a * j1p)) / Khat0

    # Scan to find sign-change brackets between singularities
    K_grid = np.linspace(Kmin*(1+1e-6), Khat0_max, n_samples)
    Gvals  = np.array([G(K) for K in K_grid])

    brackets = []
    i = 0
    while i < len(K_grid) - 1:
        if not np.isfinite(Gvals[i]):
            i += 1; continue
        j = i + 1
        while j < len(K_grid) and not np.isfinite(Gvals[j]):
            j += 1
        if j >= len(K_grid):
            break
        if Gvals[i] * Gvals[j] < 0:
            brackets.append((K_grid[i], K_grid[j]))
        i = j

    if not brackets:
        raise RuntimeError("No sign change found for G(Khat0). Increase Khat0_max or n_samples.")

    if root_index < 0 or root_index >= len(brackets):
        raise RuntimeError(f"root_index={root_index} out of range (found {len(brackets)} bracket(s)).")

    a, b = brackets[root_index]
    sol = optimize.root_scalar(G, bracket=(a, b), method="brentq", maxiter=500)
    if not sol.converged:
        raise RuntimeError(f"root_scalar failed: {sol.flag}")

    Khat0 = float(sol.root)
    alpha = alpha_of(Khat0)
    return Khat0, alpha
