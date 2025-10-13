"""
velocity_scaling_plots.py  (support module)

Utilities to assemble first/second-order fields at speed V and to measure
how the PDE residuals scale with V, plus a plotting helper.

This module DOES NOT run anything on import. Call its functions from your main.

Expected F (FirstOrder-like) interface:
    F.R0, F.Z, F.P, F.Khat0, F.m0
    F.s11(r: array) -> array
    F.m11(r: array) -> array

Expected SO (SecondOrderAll-like) interface:
    SO.A.r, SO.A.s0, SO.A.m0, SO.A.s2   (arrays over r-grid for branch A)
    SO.B.r, SO.B.s0, SO.B.m0, SO.B.s2   (arrays over r-grid for branch B)

Public API:
    - D_factory_vdW(e_a=0.0, m_inf=10.0) -> callable D(m)
    - build_sigma_m(F, SO, V, order=1|2) -> (sigma(r,theta), m(r,theta))
    - sigma_residual_L2(F, sigma, m, r, th) -> float
    - m_residual_L2(F, sigma, m, r, th, D_of_m, K0, V) -> float
    - compute_velocity_scaling(F, SO, D_of_m, D0, Vs=None, Nr=180, Nth=128)
         -> dict with arrays {'V','sigma_F','sigma_SO','m_F','m_SO'}
    - plot_velocity_scaling(result_dict, *, save=False, prefix='velocity_scaling', show=True)

Notes
-----
* We choose σ0 ≡ P*m0 so that the σ-equation steady interior is satisfied:
      ZΔσ0 - σ0 + P m0 = 0   (since Δσ0=0).
* The m-equation residual uses K = K̂0 * D(m0) = F.Khat0 * D0.
* V is oriented along +x so V·∇ = V (cosθ ∂r - sinθ (1/r) ∂θ).
"""

from typing import Callable, Dict, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Diffusion helpers --------------------

def D_factory_vdW(e_a: float = 0.0, m_inf: float = 10.0) -> Callable:
    """Return a callable D(m) = m_inf^2/(m_inf - m)^2 - e_a*m."""
    def D(m):
        return (m_inf**2)/((m_inf - m)**2) - e_a*m
    return D

# -------------------- Field assembly up to order in V ------------------

def build_sigma_m(F, SO, V: float, order: int = 1):
    """
    Construct σ(r,θ), m(r,θ) up to given order in V using F and SO.
    order=1: σ = σ0 + V σ1 cosθ;   m = m0 + V m1 cosθ
    order=2: add V^2 (n=0,2) from SO.A+SO.B
    Returns two vectorized callables (r,θ)->array.
    """
    sigma0_const = F.P * F.m0  # makes ZΔσ0 - σ0 + P m0 = 0
    s1 = lambda r: F.s11(r)
    m1 = lambda r: F.m11(r)

    if order == 1:
        def sigma(r, th): return sigma0_const + V*s1(r)*np.cos(th)
        def m(r, th):     return F.m0 + V*m1(r)*np.cos(th)
        return sigma, m

    # second-order add-ons
    rA, s0A, m0A, s2A, m2A = SO.A.r, SO.A.s0, SO.A.m0, SO.A.s2, SO.A.m2
    rB, s0B, m0B, s2B, m2B = SO.B.r, SO.B.s0, SO.B.m0, SO.B.s2, SO.B.m2

    def interp(rr, rx, fy): return np.interp(rr, rx, fy)
    s0_tot = lambda rr: interp(rr, rA, s0A) + interp(rr, rB, s0B)
    m0_tot = lambda rr: interp(rr, rA, m0A) + interp(rr, rB, m0B)
    s2_tot = lambda rr: interp(rr, rA, s2A) + interp(rr, rB, s2B)
    m2_tot = lambda rr: interp(rr, rA, m2A) + interp(rr, rB, m2B)

    def sigma(r, th):
        return sigma0_const + V*s1(r)*np.cos(th) + (V**2)*( s0_tot(r) + s2_tot(r)*np.cos(2*th) )
    def m(r, th):
        return F.m0 + V*m1(r)*np.cos(th) + (V**2)*( m0_tot(r) +m2_tot(r)*np.cos(2*th))
    return sigma, m

# -------------------- Discrete operators & residuals -------------------

def _laplacian_polar(f: Callable, r: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Δf on a tensor grid using centered finite differences in polar coords."""
    R, TH = np.meshgrid(r, th, indexing='ij')
    Fv = f(R, TH)
    Fr  = np.gradient(Fv, r,  axis=0, edge_order=2)
    Frr = np.gradient(Fr,  r,  axis=0, edge_order=2)
    dth = th[1] - th[0]
    Fth   = np.gradient(Fv, dth, axis=1, edge_order=2)
    Fthth = np.gradient(Fth, dth, axis=1, edge_order=2)
    Rsafe = R.copy(); Rsafe[Rsafe == 0] = r[1]
    Lap = Frr + (1.0/Rsafe)*Fr + (1.0/(Rsafe**2))*Fthth
    # center-row regularization (limit r→0)
    Lap[0, :] = Frr[0, :] + Fthth[0, :]/(r[1]**2)
    return Lap

def sigma_residual_L2(F, sigma: Callable, m: Callable, r: np.ndarray, th: np.ndarray) -> float:
    """|| ZΔσ - σ + P m ||_{L2(B_R0)} with area weight r dr dθ."""
    Lap = _laplacian_polar(sigma, r, th)
    R = F.Z * Lap - sigma(*np.meshgrid(r, th, indexing='ij')) + F.P * m(*np.meshgrid(r, th, indexing='ij'))
    dth = th[1]-th[0]; dr = r[1]-r[0]
    return float(np.sqrt(np.sum((R**2)*r[:,None]) * dr * dth))

def m_residual_L2(F, sigma: Callable, m: Callable, r: np.ndarray, th: np.ndarray,
                  D_of_m: Callable, K0: float, V: float) -> float:
    """
    || -V·∇m - ∇·( D(m)∇m - K0 m ∇σ ) ||_{L2(B_R0)} with area weight.
    V is along +x: V·∇ = V(cosθ ∂r - sinθ (1/r) ∂θ).
    """
    R, TH = np.meshgrid(r, th, indexing='ij')
    M  = m(R, TH)
    Sg = sigma(R, TH)

    Mr  = np.gradient(M, r,  axis=0, edge_order=2)
    dth = th[1] - th[0]
    Mth = np.gradient(M, dth, axis=1, edge_order=2)
    Sr  = np.gradient(Sg, r,  axis=0, edge_order=2)
    Sth = np.gradient(Sg, dth, axis=1, edge_order=2)

    Rsafe = R.copy(); Rsafe[Rsafe == 0] = r[1]
    grad_m_r,  grad_m_th  = Mr,        Mth/Rsafe
    grad_s_r,  grad_s_th  = Sr,        Sth/Rsafe

    VdotGradm = V*(np.cos(TH)*grad_m_r - np.sin(TH)*grad_m_th)

    Dm = D_of_m(M)
    flux_r  = Dm*grad_m_r  - K0*M*grad_s_r
    flux_th = Dm*grad_m_th - K0*M*grad_s_th

    d_r_fr   = np.gradient(Rsafe*flux_r, r, axis=0, edge_order=2)
    d_th_fth = np.gradient(flux_th, dth, axis=1, edge_order=2)
    div_flux = (1.0/Rsafe)*d_r_fr + (1.0/Rsafe)*d_th_fth

    Resid = -VdotGradm - div_flux

    dth = th[1]-th[0]; dr = r[1]-r[0]
    return float(np.sqrt(np.sum((Resid**2)*r[:,None]) * dr * dth))

# -------------------- Orchestrators -----------------------------------

def compute_velocity_scaling(F, SO, D_of_m: Callable, D0: float,
                             Vs: Optional[np.ndarray] = None,
                             Nr: int = 180, Nth: int = 128):
    """
    Compute L2 residuals vs V for both equations.
    Returns dict with arrays: {'V','sigma_F','sigma_SO','m_F','m_SO'}.
    """
    if Vs is None:
        Vs = np.geomspace(1e-4, 1e-1, 12)
    Vs = np.asarray(Vs, dtype=float)

    r  = np.linspace(0.0, F.R0, Nr)
    th = np.linspace(0.0, 2*np.pi, Nth, endpoint=False)
    K0 = F.Khat0 * D0

    sig_F = []; sig_SO = []; m_F = []; m_SO = []
    for V in Vs:
        s1, m1 = build_sigma_m(F, SO, V, order=1)
        s2, m2 = build_sigma_m(F, SO, V, order=2)
        sig_F .append(sigma_residual_L2(F, s1, m1, r, th))
        sig_SO.append(sigma_residual_L2(F, s2, m2, r, th))
        m_F   .append(m_residual_L2(F, s1, m1, r, th, D_of_m, K0, V))
        m_SO  .append(m_residual_L2(F, s2, m2, r, th, D_of_m, K0, V))

    return {
        'V': Vs,
        'sigma_F':  np.array(sig_F),
        'sigma_SO': np.array(sig_SO),
        'm_F':      np.array(m_F),
        'm_SO':     np.array(m_SO),
    }

def compare_first_second_order_fields(F, SO, V: float,
                                      *, Nr: int = 180, Nth: int = 128) -> Dict[str, np.ndarray]:
    """
    Evaluate the first- and second-order reconstructions of σ and m on a polar grid
    and measure their differences.

    Returns a dict containing:
      - 'r', 'theta': 1D grids.
      - 'sigma_first', 'sigma_second', 'm_first', 'm_second': field samples (Nr×Nth).
      - 'sigma_diff_L2', 'm_diff_L2': area-weighted L2 norms of the differences.
      - 'sigma_diff_max', 'm_diff_max': max-norms of the differences.
    """
    r = np.linspace(0.0, F.R0, Nr)
    th = np.linspace(0.0, 2*np.pi, Nth, endpoint=False)
    R, TH = np.meshgrid(r, th, indexing='ij')

    sigma_F, m_F = build_sigma_m(F, SO, V, order=1)
    sigma_SO, m_SO = build_sigma_m(F, SO, V, order=2)

    sigma_first = sigma_F(R, TH)
    sigma_second = sigma_SO(R, TH)
    m_first = m_F(R, TH)
    m_second = m_SO(R, TH)

    dsigma = sigma_second - sigma_first
    dm = m_second - m_first

    dr = r[1] - r[0] if Nr > 1 else F.R0
    dth = th[1] - th[0] if Nth > 1 else 2*np.pi
    area_weight = r[:, None]

    sigma_diff_L2 = float(np.sqrt(np.sum((dsigma**2) * area_weight) * dr * dth))
    m_diff_L2 = float(np.sqrt(np.sum((dm**2) * area_weight) * dr * dth))

    return dict(
        r=r,
        theta=th,
        sigma_first=sigma_first,
        sigma_second=sigma_second,
        m_first=m_first,
        m_second=m_second,
        sigma_diff_L2=sigma_diff_L2,
        m_diff_L2=m_diff_L2,
        sigma_diff_max=float(np.max(np.abs(dsigma))),
        m_diff_max=float(np.max(np.abs(dm))),
    )

def plot_velocity_scaling(result,
                          *, save: bool = False, prefix: str = "velocity_scaling",
                          show: bool = True, title_suffix: str = "") -> None:
    """
    Plot the residual-vs-V curves produced by compute_velocity_scaling().
    """
    V = result['V']
    sig_F, sig_SO = result['sigma_F'], result['sigma_SO']
    m_F,   m_SO   = result['m_F'],     result['m_SO']

    if save:
        out_dir = Path(prefix).parent
        if out_dir != Path("."):
            out_dir.mkdir(parents=True, exist_ok=True)

    # σ plot
    plt.figure()
    plt.loglog(V, sig_F, 'o-', label='σ: First-order (F)')
    plt.loglog(V, sig_SO, 's-', label='σ: Second-order (F+SO)')
    plt.xlabel("V"); plt.ylabel(r"$\| Z\Delta\sigma - \sigma + P m \|_{L^2}$")
    plt.title("Residual scaling vs V (σ-equation) " + title_suffix)
    plt.legend()
    if save:
        plt.savefig(f"{prefix}_sigma.png", dpi=180, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    # m plot
    plt.figure()
    plt.loglog(V, m_F, 'o-', label='m: First-order (F)')
    plt.loglog(V, m_SO, 's-', label='m: Second-order (F+SO)')
    plt.xlabel("V"); plt.ylabel(r"$\| -V\cdot\nabla m - \nabla\cdot(D(m)\nabla m - K m \nabla\sigma) \|_{L^2}$")
    plt.title("Residual scaling vs V (m-equation) " + title_suffix)
    plt.legend()
    if save:
        plt.savefig(f"{prefix}_m.png", dpi=180, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
