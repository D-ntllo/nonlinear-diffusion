
import numpy as np
import math
from typing import Dict, Any, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt

# Reuse your solvers
from second_order import _Ln_inverse_neumann, _cumtrapz_from_zero
from scipy.special import j1, y1, jvp, yvp
from check_k2_new import _solve_sigma_numerical

# -------------------- utils --------------------

def _deriv_central(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(x)
    dy = np.zeros_like(y, dtype=float)
    if n == 1:
        return dy
    dy[1:-1] = (y[2:] - y[:-2])/(x[2:] - x[:-2])
    dy[0]    = (y[1] - y[0])/(x[1] - x[0])
    dy[-1]   = (y[-1] - y[-2])/(x[-1] - x[-2])
    return dy

def _cumtrapz_to_R0(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 1:
        return np.zeros_like(x, dtype=float)
    dx = np.diff(x)
    areas = 0.5 * (y[:-1] + y[1:]) * dx
    B = np.zeros(n, dtype=float)
    B[:-1] = np.cumsum(areas[::-1])[::-1]
    return B

# Solve (L_1 + κ^2) σ = f, regular at 0, with Dirichlet σ(R0)=target

def _solve_sigma_dirichlet(
    kappa: float,
    f: np.ndarray,
    r: np.ndarray,
    R0: float,
    sR0_target: float,
) -> Tuple[np.ndarray, float]:
    """
    Solve (L_1 + kappa^2) s = f on (0, R0),
        L_1 s = s'' + (1/r) s' - (1/r**2) s,
    with:
        - regularity at r = 0 (no Y1 component),
        - Dirichlet boundary condition s(R0) = sR0_target.

    Uses variation-of-parameters integral representation:

        W(r) = J1(kappa r) Y1'(kappa r) - J1'(kappa r) Y1(kappa r)
             = -2 / (pi r)

        I1(r) = ∫_0^r J1(kappa rho) f(rho) / W(rho) d rho
        I2(r) = ∫_0^r Y1(kappa rho) f(rho) / W(rho) d rho

        s_p(r) = J1(kappa r) I2(r) - Y1(kappa r) I1(r)

        s(r) = s_p(r) + a_reg J1(kappa r),
        a_reg chosen so that s(R0) = sR0_target.

    Also returns s'(R0), computed analytically from the same representation.

    Parameters
    ----------
    kappa : float
        Positive parameter (assumes kappa > 0).
    f : np.ndarray
        Values of the right-hand side f(r) on the grid r.
    r : np.ndarray
        1D grid of radii, strictly increasing, with r[0] > 0 and r[-1] = R0.
    R0 : float
        Outer radius; must match r[-1].
    sR0_target : float
        Dirichlet boundary value s(R0).

    Returns
    -------
    s : np.ndarray
        Approximate solution s(r) on the grid r.
    s_prime_R0 : float
        Approximate derivative s'(R0).
    """

    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)

    if r.ndim != 1 or f.ndim != 1:
        raise ValueError("r and f must be 1D arrays.")
    if r.shape != f.shape:
        raise ValueError("r and f must have the same shape.")
    if not np.all(np.diff(r) > 0):
        raise ValueError("r must be strictly increasing.")
    if not np.isclose(r[-1], R0):
        raise ValueError("R0 must equal r[-1].")
    if r[0] <= 0.0:
        raise ValueError("r[0] must be > 0 (use a small positive cutoff).")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive.")

    # Fundamental solutions
    kr = kappa * r
    J1_vals = j1(kr)
    Y1_vals = y1(kr)

    # Derivatives w.r.t. r:
    # d/dr J1(k r) = k * d/dx J1(x) |_{x=k r}, similarly for Y1.
    J1p_arg = jvp(1, kr)   # derivative wrt argument
    Y1p_arg = yvp(1, kr)
    dJ1_dr = kappa * J1p_arg
    dY1_dr = kappa * Y1p_arg

    J1_R0 = J1_vals[-1]
    if np.isclose(J1_R0, 0.0):
        raise RuntimeError("J1(kappa*R0) ≈ 0; Dirichlet problem is resonant.")

    # Wronskian: W(r) = -2/(pi r)
    W_vals = -2.0 / (math.pi * r)

    # Integrands for I1, I2
    # I1(r) = ∫_0^r J1(kappa rho) f(rho) / W(rho) d rho
    # I2(r) = ∫_0^r Y1(kappa rho) f(rho) / W(rho) d rho
    integrand_I1 = J1_vals * f / W_vals
    integrand_I2 = Y1_vals * f / W_vals

    N = len(r)
    I1 = np.zeros_like(r)
    I2 = np.zeros_like(r)

    # Cumulative trapezoidal integration from 0 to each r[i]
    for i in range(1, N):
        dr_i = r[i] - r[i - 1]
        I1[i] = I1[i - 1] + 0.5 * dr_i * (integrand_I1[i] + integrand_I1[i - 1])
        I2[i] = I2[i - 1] + 0.5 * dr_i * (integrand_I2[i] + integrand_I2[i - 1])

    # Particular solution s_p (regular at 0)
    s_particular = J1_vals * I2 - Y1_vals * I1

    # Impose Dirichlet at R0
    a_reg = (sR0_target - s_particular[-1]) / J1_R0

    s = s_particular + a_reg * J1_vals

    # Derivative of s_p:
    # s_p = J1 I2 - Y1 I1
    # s_p' = (dJ1/dr) I2 + J1 I2' - (dY1/dr) I1 - Y1 I1'
    # and by construction I1'(r) = integrand_I1, I2'(r) = integrand_I2
    s_particular_prime = (
        dJ1_dr * I2
        + J1_vals * integrand_I2
        - dY1_dr * I1
        - Y1_vals * integrand_I1
    )

    # Total derivative: s' = s_p' + a_reg * dJ1/dr
    s_prime = s_particular_prime + a_reg * dJ1_dr

    s_prime_R0 = float(s_prime[-1])

    return s, s_prime_R0

# -------------------- forcings with A/B weighting like build_sigma_m --------------------

@dataclass
class K2LoT:
    r: np.ndarray
    q31: np.ndarray
    sR0_target: float
    bracket_c: float
    bracket_s: float

def make_forcings_ms31(F, SO, D, Dp, Dpp, K2: float) -> K2LoT:
    """
    Build q31 and boundary targets using the SAME A/B mixing as build_sigma_m:
      X_tot = (1/D(m0)^2) * X_A + (Dp(m0)/D(m0)^3) * X_B,  for X ∈ {σ_20,σ_22,m_20,m_22,ρ_20,ρ_22}.
    """
    m0 = float(F.m0)
    
    D0   = float(D(m0))
    Dp0  = float(Dp(m0))
    Dpp0 = float(Dpp(m0))

    R0 = float(F.R0); K0 = D0*float(F.Khat0);
    r  = np.asarray(SO.A.r, dtype=float); rr = r

    # first-order

    s11  =  1.0 / D0 * F.s11(r)
    s11p =  1.0 / D0 * F.s11p(r)
    s11pp=  1.0 / D0 * F.s11pp(r)

    m11  = 1.0 / D0 * F.m11(r)
    m11p = 1.0 / D0 * F.m11p(r)
    m11pp= 1.0 / D0 * F.m11pp(r)

    # A/B arrays
    s0A, s2A, m0A, m2A = np.asarray(SO.A.s0), np.asarray(SO.A.s2), np.asarray(SO.A.m0), np.asarray(SO.A.m2)
    s0B, s2B, m0B, m2B = np.asarray(SO.B.s0), np.asarray(SO.B.s2), np.asarray(SO.B.m0), np.asarray(SO.B.m2)

    # Consistent weighting
    wa, wb = 1.0/(D0**2), Dp0/(D0**3)
    s20 = wa*s0A + wb*s0B
    s22 = wa*s2A + wb*s2B
    m20 = wa*m0A + wb*m0B
    m22 = wa*m2A + wb*m2B

    # rho weighting too
    rho20 = wa*float(SO.rho20A) + wb*float(SO.rho20B)
    rho22 = wa*float(SO.rho22A) + wb*float(SO.rho22B)

    # derivatives of arrays
    def d1(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.gradient(y, r, edge_order=2)

    def d2(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        g1 = np.gradient(y, r, edge_order=2)
        return np.gradient(g1, r, edge_order=2)

    s20p, s20pp = d1(s20), d2(s20)
    s22p, s22pp = d1(s22), d2(s22)
    m20p, m20pp = d1(m20), d2(m20)
    m22p, m22pp = d1(m22), d2(m22)

    # Long f(r)
    f = (
    (Dpp0 / (8.0 * r**2)) * (
        - m11**3
        + 6.0 * r**2 * m11 * (m11p**2)
        + 3.0 * r * (m11**2) * (m11p + r * m11pp)
    )
    + (Dp0 / (8.0 * r**2)) * (
        - 8.0 * m11 * m20
        - 4.0 * m11 * m22
        + 8.0 * r * m20 * m11p
        + 4.0 * r * m22 * m11p
        + 8.0 * r * m11 * m20p
        + 16.0 * r**2 * m11p * m20p
        + 4.0 * r * m11 * m22p
        + 8.0 * r**2 * m11p * m22p
        + 8.0 * r**2 * m20 * m11pp
        + 4.0 * r**2 * m22 * m11pp
        + 8.0 * r**2 * m11 * m20pp
        + 4.0 * r**2 * m11 * m22pp
    )
    + (1.0 / (8.0 * r**2)) * (
        8.0 * K0 * m20 * s11
        - 4.0 * K0 * m22 * s11
        + 8.0 * K0 * m11 * s22
        + 8.0 * r**3 * m20p
        + 4.0 * r**3 * m22p
        - 8.0 * r * K0 * m20 * s11p
        - 4.0 * r * K0 * m22 * s11p
        - 8.0 * r**2 * K0 * m20p * s11p
        - 4.0 * r**2 * K0 * m22p * s11p
        - 8.0 * r * K0 * m11 * s20p
        - 4.0 * r * K0 * m11 * s22p
        - 4.0 * r**2 * K0 * m11p * s22p
        - 8.0 * r**2 * K0 * m20 * s11pp
        - 4.0 * r**2 * K0 * m22 * s11pp
        - 8.0 * r**2 * K0 * m11p * s20p
        - 8.0 * r**2 * K0 * m11 * s20pp
        - 4.0 * r**2 * K0 * m11 * s22pp
    )
)

    # g(r) for the K2 term
    g_r = -(s11 / (rr**2)) - (s11p / rr) + s11pp

    # q31: L1 u = q31. u = m31-K0m0/D0*s31
    q31 = (K2 * m0 / D0) * g_r - f / D0

    # Dirichlet target for σ31 and bracket for m31' BC
    sR0_target = (rho20 + 0.5*rho22) * float(F.s11p(R0))
    bracket_c  = (rho20 + 0.5*rho22) * float(F.m11pp(R0)) + (rho22 / (R0**2)) * float(F.m11(R0))
    bracket_s  = (rho20 + 0.5*rho22) * float(F.s11pp(R0)) + (rho22 / (R0**2)) * float(F.s11(R0)) + K2/K0 * F.s11p(R0)

    return K2LoT(r=r, q31=q31, sR0_target=sR0_target, bracket_c=bracket_c, bracket_s= bracket_s)

# -------------------- main checker --------------------

def check_k2_consistent(F, SO, D, Dp, Dpp, K2: float) -> Dict[str, Any]:
    """
    Use the same A/B mixing as build_sigma_m for second-order fields *and* rho,
    then solve the (3,1) system via: u = L1^{-1} q31 (regular), (L1 - κ^2)σ = (P/Z)u with Dirichlet at R0,
    determine the homogeneous A from m31' BC, and return the compatibility residual in (ms31_e).
    """
    R0 = float(F.R0); Z = float(F.Z); P = float(F.P); m0 = float(F.m0); Khat0 = float(F.Khat0)

    # diffusion at m0
    D0 = float(D(m0)); Dp0 = float(Dp(m0))

    K0 = D0*Khat0

    lot = make_forcings_ms31(F, SO, D, Dp, Dpp, K2)
    r = lot.r

    r_step = r[1:]-r[:-1]

    # Step 1: u = L1^{-1} q31 (regular; we will add A r from BC)
    q_reg = _Ln_inverse_neumann(1, lambda rr: np.interp(rr, r, lot.q31), r, R0, neuman_bc= -lot.bracket_c+K0*m0/D0*lot.bracket_s)

    # Step 2: (L1 + κ^2) σ = -(P/Z) q, with κ^2 = (1 + (P K0 m0)/D0) / Z
    kappa2 = (-1.0 + (P*K0*m0)/D0) / Z
    if kappa2 <= 0:
        raise RuntimeError(f"κ^2 must be positive; got {kappa2}")
    kappa = float(np.sqrt(kappa2))

    # σ with zero Dirichlet at R0 for u_reg, and for u_hom = r
    sigma31, sig31p_R0 = _solve_sigma_dirichlet(kappa, -(P/Z)*q_reg, r, R0, sR0_target=-lot.sR0_target)
    sigma31n, sig31p_R0n = _solve_sigma_numerical(kappa, R0, f=-(P/Z)*q_reg, r=r, target=-lot.sR0_target)
    
    target = np.zeros_like(sigma31n)
    target[-1] = -lot.bracket_s 

    plt.plot(r, sigma31n)
    plt.draw(r,target)
    plt.show()



    res = sig31p_R0+lot.bracket_s #K2*float(F.s11p(R0)) + K0*(sig31p_R0 + lot.bracket_s)
    res_n = sig31p_R0n+lot.bracket_s 

    return dict(
        residual_bc=float(abs(res)),
        residual_numerical = float(res_n),
        signed_residual_bc=float(res),
        regularised_loss = float(res)/np.dot(sigma31[:-1]**2, r_step),
        #A=float(A),
        sigma_R0=float(sigma31[-1]),
        sigma_p_R0=float(sig31p_R0),
    )
