
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Reuse your solvers
from second_order import _Ln_inverse_neumann, _cumtrapz_from_zero
from scipy.special import iv as I, kv as K, ivp as Ip

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

# Solve (L_1 - κ^2) σ = f, regular at 0, with Dirichlet σ(R0)=target
def _solve_sigma_modified_dirichlet(kappa: float, f: np.ndarray, r: np.ndarray, R0: float, sR0_target: float) -> Tuple[np.ndarray, float, float]:
    rr  = np.asarray(r, dtype=float)
    fv  = np.asarray(f, dtype=float)
    kr  = kappa * rr
    kr0 = kappa * R0

    # Dirichlet Green using ψ_D = K_1 + cD I_1 with ψ_D(R0)=0
    cD   = - K(1, kr0) / I(1, kr0)
    psiD = K(1, kr) + cD * I(1, kr)

    Iphi = _cumtrapz_from_zero(rr * I(1, kr) * fv, rr)      # ∫_0^r s I_1(κs) f(s) ds
    Bpsi = _cumtrapz_to_R0(rr * psiD * fv, rr)              # ∫_r^{R0} s ψ_D(κs) f(s) ds
    sigma0 = psiD * Iphi + I(1, kr) * Bpsi                  # has σ0(R0)=0

    # add homogeneous to meet Dirichlet
    phiR0 = float(I(1, kr0))
    sigma = sigma0 + (sR0_target/phiR0) * I(1, kr)
    sigma_p_R0 = float(_deriv_central(sigma, rr)[-1])
    return sigma, float(sR0_target), sigma_p_R0

# -------------------- forcings with A/B weighting like build_sigma_m --------------------

@dataclass
class K2LoT:
    r: np.ndarray
    q31: np.ndarray
    sR0_target: float
    bracket_c: float

def make_forcings_ms31_consistent(F, SO, D, Dp, Dpp, K2: float) -> K2LoT:
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

  

    # first-order callables
    scale = 1.0 / D0

    s11  = lambda x: scale * F.s11(x)
    sp11 = lambda x: scale * F.s11p(x)
    spp11= lambda x: scale * F.s11pp(x)

    m11  = lambda x: scale * F.m11(x)
    mp11 = lambda x: scale * F.m11p(x)
    mpp11= lambda x: scale * F.m11pp(x)

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
    term1 = Dpp0 * (m11(rr)**3)
    term2 = -3*Dpp0 * rr * (m11(rr)**2) * (mp11(rr) + rr * mpp11(rr))
    term3 = -2*rr * ( 2*rr**2 * m20p + 8*Dp0 * rr * mp11(rr) * m20p
                      + rr**2 * m22p + 4*Dp0 * rr * mp11(rr) * m22p
                      + 4*Dp0 * m20 * (mp11(rr) + rr * mpp11(rr))
                      + 2*Dp0 * m22 * (mp11(rr) + rr * mpp11(rr)) )
    term4 = 2*m11(rr) * ( 4*Dp0*m20 + 2*Dp0*m22 - 2*K0*s22
                          - 3*Dpp0 * rr**2 * (mp11(rr)**2)
                          - 4*Dp0 * rr * m20p - 2*Dp0 * rr * m22p
                          + 2*rr*K0*s20p + rr*K0*s22p
                          - 4*Dp0 * rr**2 * m20pp - 2*Dp0 * rr**2 * m22pp
                          + 2*rr**2 * K0 * s20pp + rr**2 * K0 * s22pp )
    term5 = 2*K0 * ( 2*rr**2 * m20p * sp11(rr) + rr**2 * m22p * sp11(rr)
                     + 2*rr**2 * mp11(rr) * s20p + rr**2 * mp11(rr) * s22p
                     + m22 * ( s11(rr) + rr*(sp11(rr) + rr*spp11(rr)) )
                     + m20 * ( -2*s11(rr) + 2*rr*(sp11(rr) + rr*spp11(rr)) ) )
    f_r = - (term1 + term2 + term3 + term4 + term5) / (4*rr**2)

    # g(r) for the K2 term
    g_r = (mpp11(rr) / (rr**2)) - (mp11(rr) / rr) + mpp11(rr)

    # q31: L1 u = q31
    q31 = (K2 * m0 / D0) * g_r - f_r / D0

    # Dirichlet target for σ31 and bracket for m31' BC
    sR0_target = - (rho20 + 0.5*rho22) * float(sp11(R0))
    bracket_c  = (rho20 + 0.5*rho22) * float(mpp11(R0)) + (rho22 / (R0**2)) * float(m11(R0))

    return K2LoT(r=r, q31=q31, sR0_target=sR0_target, bracket_c=bracket_c)

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

    lot = make_forcings_ms31_consistent(F, SO, D, Dp, Dpp, K2)
    r = lot.r

    # Step 1: u = L1^{-1} q31 (regular; we will add A r from BC)
    u_reg = _Ln_inverse_neumann(1, lambda rr: np.interp(rr, r, lot.q31), r, R0, neuman_bc= lot.bracket_c)

    # Step 2: (L1 - κ^2) σ = (P/Z) u, with κ^2 = (1 + (P K0 m0)/D0) / Z
    kappa2 = (1.0 + (P*K0*m0)/D0) / Z
    if kappa2 <= 0:
        raise RuntimeError(f"κ^2 must be positive; got {kappa2}")
    kappa = float(np.sqrt(kappa2))

    # σ with zero Dirichlet at R0 for u_reg, and for u_hom = r
    sig_part, _, sigp_part_R0 = _solve_sigma_modified_dirichlet(kappa, (P/Z)*u_reg, r, R0, sR0_target=0.0)
    sig_hom,  _, sigp_hom_R0  = _solve_sigma_modified_dirichlet(kappa, (P/Z)*r,     r, R0, sR0_target=0.0)

    # Add Dirichlet lift σ_D to hit σ(R0)=sR0_target
    phiR0   = float(I(1, kappa*R0))
    sigp_lift_R0 = (kappa * float(Ip(1, kappa*R0))) * (lot.sR0_target / phiR0)

    # choose A from (ms31_c): m31' = u' + (K0 m0 / D0) σ' ⇒ m31'(R0)+bracket_c=0
    up_reg_R0 = float(_deriv_central(u_reg, r)[-1])
    coeff_A = 1.0 + (K0*m0/D0) * sigp_hom_R0
    rhs     = - ( up_reg_R0 + (K0*m0/D0)*( sigp_part_R0 + sigp_lift_R0 ) + lot.bracket_c )
    A = rhs / coeff_A

    # Build fields
    u_tot   = u_reg + A * r
    sigma31 = sig_part + A * sig_hom + (lot.sR0_target/phiR0) * I(1, kappa*r)
    m31     = u_tot + (K0*m0/D0) * sigma31

    # Compatibility residual (ms31_e)
    s31p_R0 = float(_deriv_central(sigma31, r)[-1])
    rho20 = None; rho22 = None  # already used inside lot to build targets
    res = K2*float(F.s11p(R0)) + K0*( s31p_R0 + lot.bracket_c )

    return dict(
        residual_bc=float(abs(res)),
        signed_residual_bc=float(res),
        A=float(A),
        sigma_R0=float(sigma31[-1]),
        sigma_p_R0=float(s31p_R0),
    )
