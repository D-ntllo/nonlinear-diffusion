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
    - sigma_residual_L2(F, sigma, m, r, th, *, scale_theta=None) -> float
    - m_residual_L2(F, sigma, m, r, th, D_of_m, K0, V, *, scale_theta=None) -> float
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

from typing import Callable, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


from second_order import *
# -------------------- Diffusion helpers --------------------

def D_vdW(e_a: float = 0.0, m_inf: float = 10.0) -> Tuple[Callable, Callable, Callable]:
    """Return a callable D(m) = m_inf^2/(m_inf - m)^2 - e_a*m and its derivatives"""
    def D(m):
        return (m_inf**2)/((m_inf - m)**2) - e_a*m
    
    def Dp(m):
        return (2*m_inf**2)/((m_inf-m)**3) - e_a
    
    def Dpp(m):
        return (6*m_inf**2)/((m_inf-m)**4)
    return D, Dp, Dpp

# -------------------- Discrete operators & residuals -------------------

def _shape_scale(F, SO,D, Dp, V: float, th: np.ndarray) -> np.ndarray:
    """
    Return the θ-dependent scale factor 1 + ε(θ) that maps the base circular
    domain r∈[0,R0] to the V^2-perturbed boundary
        R(θ) = R0 + V^2 (ρ20 + ρ22 cos 2θ).
    """
    m0 =F.m0

    rho0 = (1/D(m0)**2*getattr(SO, "rho20A", 0.0) + Dp(m0)/D(m0)**3 * getattr(SO, "rho20B", 0.0))
    rho2 = (1/D(m0)**2*getattr(SO, "rho22A", 0.0) + Dp(m0)/D(m0)**3 * getattr(SO, "rho22B", 0.0))
    chi = rho0 + rho2 * np.cos(2.0 * th)
    return 1.0 + (V**2 / F.R0) * chi


def _laplacian_polar(f: Callable, r: np.ndarray, th: np.ndarray,
                     scale_theta: Optional[np.ndarray] = None) -> np.ndarray:
    """Δf on a tensor grid using centered finite differences in polar coords."""
    R, TH = np.meshgrid(r, th, indexing='ij')
    Fv = f(R, TH)

    if scale_theta is None:
        scale_array = np.ones((1, th.size))
    else:
        scale_array = scale_theta[np.newaxis, :]
    scale = scale_array

    Fr  = np.gradient(Fv, r,  axis=0, edge_order=2)
    Frr = np.gradient(Fr,  r,  axis=0, edge_order=2)
    dth = th[1] - th[0]
    Fth   = np.gradient(Fv, dth, axis=1, edge_order=2)
    Fthth = np.gradient(Fth, dth, axis=1, edge_order=2)

    Rphys = R * scale
    Fr_phys = Fr / scale
    Frr_phys = Frr / (scale**2)

    Rsafe = Rphys.copy()
    Rsafe[0, :] = r[1] * scale_array[0, :]

    Lap = Frr_phys + (1.0/Rsafe)*Fr_phys + (1.0/(Rsafe**2))*Fthth
    # center-row regularization (limit r→0)
    Lap[0, :] = Frr_phys[0, :] + Fthth[0, :]/((r[1]**2)*(scale_array[0, :]**2))
    return Lap

def sigma_residual_L2(F, sigma: Callable, m: Callable, r: np.ndarray, th: np.ndarray,
                      *, scale_theta: Optional[np.ndarray] = None) -> float:
    """|| ZΔσ - σ + P m ||_{L2(B_R0)} with area weight r dr dθ."""
    Lap = _laplacian_polar(sigma, r, th, scale_theta=scale_theta)
    sigma_vals = sigma(*np.meshgrid(r, th, indexing='ij'))
    m_vals = m(*np.meshgrid(r, th, indexing='ij'))
    Resid = F.Z * Lap - sigma_vals + F.P * m_vals
    dth = th[1]-th[0]; dr = r[1]-r[0]
    if scale_theta is None:
        scale_arr = np.ones((1, th.size))
    else:
        scale_arr = scale_theta[np.newaxis, :]
    weight = (r[:, None]) * (scale_arr**2)
    return float(np.sqrt(np.sum((Resid**2)*weight) * dr * dth))

def m_residual_L2(F, sigma: Callable, m: Callable, r: np.ndarray, th: np.ndarray,
                  D_of_m: Callable, K0: float, V: float,
                  *, scale_theta: Optional[np.ndarray] = None) -> float:
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

    if scale_theta is None:
        scale_arr = np.ones((1, th.size))
    else:
        scale_arr = scale_theta[np.newaxis, :]
    scale = scale_arr

    Rphys = R * scale
    Rsafe = Rphys.copy()
    Rsafe[0, :] = r[1] * scale_arr[0, :]

    grad_m_r = Mr / scale
    grad_s_r = Sr / scale
    grad_m_th  = Mth / Rsafe
    grad_s_th  = Sth / Rsafe

    VdotGradm = V*(np.cos(TH)*grad_m_r - np.sin(TH)*grad_m_th)

    Dm = D_of_m(M)
    flux_r  = Dm*grad_m_r  - K0*M*grad_s_r
    flux_th = Dm*grad_m_th - K0*M*grad_s_th

    d_r_fr   = np.gradient(Rphys*flux_r, r, axis=0, edge_order=2)
    d_th_fth = np.gradient(flux_th, dth, axis=1, edge_order=2)
    div_flux = (1.0/(Rsafe*scale))*d_r_fr + (1.0/Rsafe)*d_th_fth

    Resid = -VdotGradm - div_flux

    dth = th[1]-th[0]; dr = r[1]-r[0]
    weight = (r[:, None]) * (scale_arr**2)
    return float(np.sqrt(np.sum((Resid**2)*weight) * dr * dth))

# -------------------- Orchestrators -----------------------------------

def compute_velocity_scaling(F, SO, D: Callable, Dp: Callable,
                             Vs: Optional[np.ndarray] = None,
                             Nr: int = 180, Nth: int = 128):
    """
    Compute L2 residuals vs V for both equations.
    Returns dict with arrays: {'V','sigma_F','sigma_SO','m_F','m_SO'}.
    """
    if Vs is None:
        Vs = np.geomspace(1e-7, 1e-1, 20)
    Vs = np.asarray(Vs, dtype=float)

    r  = np.linspace(0.0, F.R0, Nr)
    th = np.linspace(0.0, 2*np.pi, Nth, endpoint=False)
    K0 = F.Khat0 * D(F.m0)

    sig_F = []; sig_SO = []; m_F = []; m_SO = []
    for V in Vs:
        s1, m1 = build_sigma_m(F, SO, D, Dp, V, order=1)
        s2, m2 = build_sigma_m(F, SO, D, Dp, V, order=2)
        sig_F .append(sigma_residual_L2(F, s1, m1, r, th))
        scale_theta = _shape_scale(F, SO, D, Dp, V, th)
        sig_SO.append(sigma_residual_L2(F, s2, m2, r, th, scale_theta=scale_theta))
        m_F   .append(m_residual_L2(F, s1, m1, r, th, D, K0, V))
        m_SO  .append(m_residual_L2(F, s2, m2, r, th, D, K0, V, scale_theta=scale_theta))

    return {
        'V': Vs,
        'sigma_F':  np.array(sig_F),
        'sigma_SO': np.array(sig_SO),
        'm_F':      np.array(m_F),
        'm_SO':     np.array(m_SO),
    }

def compare_first_second_order_fields(F, SO, D, Dp, V: float,
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

    sigma_F, m_F = build_sigma_m(F, SO, D, Dp, V, order=1)
    sigma_SO, m_SO = build_sigma_m(F, SO, D, Dp, V, order=2)

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
    # if style == "log":
    #     # σ plot
    #     plt.figure()
    #     plt.loglog(V, sig_F, 'o-', label='σ: First-order (F)')
    #     plt.loglog(V, sig_SO, 's-', label='σ: Second-order (F+SO)')
    #     plt.xlabel("V"); plt.ylabel(r"$\| Z\Delta\sigma - \sigma + P m \|_{L^2}$")
    #     plt.title("Residual scaling vs V (σ-equation) " + title_suffix)
    #     plt.legend()
    #     if save:
    #         plt.savefig(f"{prefix}_sigma.png", dpi=180, bbox_inches='tight')
    #     if show:
    #         plt.show()
    #     else:
    #         plt.close()

    #     # m plot
    #     plt.figure()
    #     plt.loglog(V, m_F, 'o-', label='m: First-order (F)')
    #     plt.loglog(V, m_SO, 's-', label='m: Second-order (F+SO)')
    #     plt.xlabel("V"); plt.ylabel(r"$\| -V\cdot\nabla m - \nabla\cdot(D(m)\nabla m - K m \nabla\sigma) \|_{L^2}$")
    #     plt.title("Residual scaling vs V (m-equation) " + title_suffix)
    #     plt.legend()
    #     if save:
    #         plt.savefig(f"{prefix}_m.png", dpi=180, bbox_inches='tight')
    #     if show:
    #         plt.show()
    #     else:
    #         plt.close()
    # else:
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
