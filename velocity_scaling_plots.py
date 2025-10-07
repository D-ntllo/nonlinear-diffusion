#!/usr/bin/env python3
"""
velocity_scaling_plots.py

Make log-log plots of the PDE residuals vs V for:
  (1) sigma-equation:  Z Δσ - σ + P m
  (2) m-equation:     -V·∇m - ∇·( D(m)∇m - K m ∇σ ),   with  K = K̂0 * D(m0)

It uses your existing code in k2_2d_v2.py to build the first-order fields (F)
and the second-order corrections (SO).

Usage (from the same folder as k2_2d_v2.py):
    python velocity_scaling_plots.py --P 0.015 --Z 1.25 --gamma 8.86 --save

If k2_2d_v2.py is elsewhere:
    python velocity_scaling_plots.py --module-path /path/to/k2_2d_v2.py --P ...

Arguments you might customize:
    --P, --Z, --gamma    : model parameters
    --vmin, --vmax, --nV : V range for the logspace grid
    --Nr, --Nth          : grid resolution for residual evaluation
    --second-order-N     : initial nodes for second-order solver
    --eA, --m-inf        : vdW diffusion parameters (D(m) = m_inf^2/(m_inf - m)^2 - eA*m)
    --save               : save figures as PNG instead of only showing them
"""

import argparse
import importlib.util
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# -------------------- import k2_2d_v2 ---------------------------------
def import_k2(module_path: str = None):
    if module_path is None:
        # try normal import first
        try:
            import k2_2d_v2 as k2
            return k2
        except Exception as e:
            print("[warn] Could not import k2_2d_v2 on sys.path; use --module-path to point to k2_2d_v2.py")
            raise
    else:
        spec = importlib.util.spec_from_file_location("k2mod", module_path)
        k2 = importlib.util.module_from_spec(spec)
        sys.modules["k2mod"] = k2
        spec.loader.exec_module(k2)
        return k2

# -------------------- math helpers ------------------------------------
def D_factory_vdW(e_a: float = 0.0, m_inf: float = 10.0) -> Callable:
    """Return a callable D(m) = m_inf^2/(m_inf - m)^2 - e_a*m."""
    def D(m):
        return (m_inf**2)/((m_inf - m)**2) - e_a*m
    return D

def laplacian_polar(f, r, th):
    """Δf on tensor grid (r,θ) via centered finite differences."""
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

def sigma_residual_L2(F, sigma, m, r, th):
    """|| ZΔσ - σ + P m ||_{L2(B_R0)} with area weight r dr dθ."""
    Lap = laplacian_polar(sigma, r, th)
    R = F.Z * Lap - sigma(*np.meshgrid(r, th, indexing='ij')) + F.P * m(*np.meshgrid(r, th, indexing='ij'))
    dth = th[1]-th[0]; dr = r[1]-r[0]
    return float(np.sqrt(np.sum((R**2)*r[:,None]) * dr * dth))

def m_residual_L2(F, sigma, m, r, th, D_of_m, K0, V):
    """
    || -V·∇m - ∇·( D(m)∇m - K0 m ∇σ ) ||_{L2(B_R0)} with area weight.
    V is along +x: V·∇ = V(cosθ ∂_r - sinθ (1/r) ∂_θ).
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

# -------------------- assemble fields up to order in V -----------------
def build_sigma_m(F, SO, V, order=1):
    """Construct σ(r,θ), m(r,θ) up to given order in V using F and SO."""
    sigma0_const = F.P * F.m0  # makes ZΔσ0 - σ0 + P m0 = 0
    s1 = lambda r: F.s11(r)
    m1 = lambda r: F.m11(r)
    if order == 1:
        def sigma(r, th): return sigma0_const + V*s1(r)*np.cos(th)
        def m(r, th):     return F.m0 + V*m1(r)*np.cos(th)
        return sigma, m
    # second-order add-ons
    rA, s0A, m0A, s2A = SO.A.r, SO.A.s0, SO.A.m0, SO.A.s2
    rB, s0B, m0B, s2B = SO.B.r, SO.B.s0, SO.B.m0, SO.B.s2
    def interp(rr, rx, fy): return np.interp(rr, rx, fy)
    s0_tot = lambda rr: interp(rr, rA, s0A) + interp(rr, rB, s0B)
    m0_tot = lambda rr: interp(rr, rA, m0A) + interp(rr, rB, m0B)
    s2_tot = lambda rr: interp(rr, rA, s2A) + interp(rr, rB, s2B)
    def sigma(r, th): return sigma0_const + V*s1(r)*np.cos(th) + (V**2)*( s0_tot(r) + s2_tot(r)*np.cos(2*th) )
    def m(r, th):     return F.m0 + V*m1(r)*np.cos(th) + (V**2)*( m0_tot(r) )
    return sigma, m

# -------------------- main plotting routine ----------------------------
def plot_velocity_scaling_sigma_and_m(k2, P, Z, gamma, eA, m_inf, vmin, vmax, nV, Nr, Nth, second_order_N, save):
    # Build first order and second order
    F  = k2.build_first_order(P, Z, gamma)
    SO = k2.compute_second_order_elimination(F, D=1.0, modes=(0,2), N_init=second_order_N)
    # diffusion and K
    D_of_m = D_factory_vdW(e_a=eA, m_inf=m_inf)
    D0, _, _ = k2.vdW_D_at_m0(e_a=eA, F=F)
    K0 = F.Khat0 * D0

    # grids and V list
    Vs = np.geomspace(vmin, vmax, nV)
    r  = np.linspace(0.0, F.R0, Nr)
    th = np.linspace(0.0, 2*np.pi, Nth, endpoint=False)

    Lsig1, Lsig2, Lm1, Lm2 = [], [], [], []

    for V in Vs:
        s1, m1 = build_sigma_m(F, SO, V, order=1)
        s2, m2 = build_sigma_m(F, SO, V, order=2)
        Lsig1.append(sigma_residual_L2(F, s1, m1, r, th))
        Lsig2.append(sigma_residual_L2(F, s2, m2, r, th))
        Lm1.append(m_residual_L2(F, s1, m1, r, th, D_of_m, K0, V))
        Lm2.append(m_residual_L2(F, s2, m2, r, th, D_of_m, K0, V))

    # --- σ plot ---
    plt.figure()
    plt.loglog(Vs, Lsig1, 'o-', label='σ: First-order (F)')
    plt.loglog(Vs, Lsig2, 's-', label='σ: Second-order (F+SO)')
    plt.xlabel("V"); plt.ylabel(r"$\| Z\Delta\sigma - \sigma + P m \|_{L^2}$")
    plt.title(f"Residual scaling vs V (σ-equation)  (P={P}, Z={Z}, γ={gamma})")
    plt.legend()
    if save:
        plt.savefig("velocity_scaling_sigma.png", dpi=180, bbox_inches='tight')
    else:
        plt.show()

    # --- m plot ---
    plt.figure()
    plt.loglog(Vs, Lm1, 'o-', label='m: First-order (F)')
    plt.loglog(Vs, Lm2, 's-', label='m: Second-order (F+SO)')
    plt.xlabel("V"); plt.ylabel(r"$\| -V\cdot\nabla m - \nabla\cdot(D(m)\nabla m - K m \nabla\sigma) \|_{L^2}$")
    plt.title(f"Residual scaling vs V (m-equation)  (P={P}, Z={Z}, γ={gamma})")
    plt.legend()
    if save:
        plt.savefig("velocity_scaling_m.png", dpi=180, bbox_inches='tight')
    else:
        plt.show()

    return np.array(Vs), np.array(Lsig1), np.array(Lsig2), np.array(Lm1), np.array(Lm2)

# -------------------- CLI ----------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot residual scaling vs V for sigma and m equations using k2_2d_v2.")
    ap.add_argument("--module-path", type=str, default=None, help="Path to k2_2d_v2.py (if not importable).")
    ap.add_argument("--P", type=float, required=True)
    ap.add_argument("--Z", type=float, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--eA", type=float, default=0.0, help="vdW diffusion parameter e_A (0 = linear)")
    ap.add_argument("--m-inf", type=float, default=10.0, help="vdW diffusion parameter m_inf")
    ap.add_argument("--vmin", type=float, default=1e-4)
    ap.add_argument("--vmax", type=float, default=1e-1)
    ap.add_argument("--nV",   type=int,   default=12)
    ap.add_argument("--Nr",   type=int,   default=180)
    ap.add_argument("--Nth",  type=int,   default=128)
    ap.add_argument("--second-order-N", type=int, default=300, help="Initial nodes for second-order solver")
    ap.add_argument("--save", action="store_true", help="Save PNGs instead of showing interactively")
    args = ap.parse_args()

    k2 = import_k2(args.module_path)
    _ = plot_velocity_scaling_sigma_and_m(
        k2,
        P=args.P, Z=args.Z, gamma=args.gamma,
        eA=args.eA, m_inf=args.m_inf,
        vmin=args.vmin, vmax=args.vmax, nV=args.nV,
        Nr=args.Nr, Nth=args.Nth,
        second_order_N=args.second_order_N,
        save=args.save
    )

if __name__ == "__main__":
    main()
