from typing import Any, Dict

from steady_state import *
from first_order import *
from second_order import *
from compute_Ai import *

def _num_deriv(f: np.ndarray, x: np.ndarray):
    """Return (f', f'') via second-order finite differences on x."""
    fp  = np.gradient(f, x, edge_order=2)
    fpp = np.gradient(fp, x, edge_order=2)
    return fp, fpp


def _Ln(n: int, f: np.ndarray, r: np.ndarray):
    fp, fpp = _num_deriv(f,r)
    """Modal radial operator L_n f = f'' + (1/r) f' - (n^2/r^2) f, using arrays."""
    return fpp + (1.0/r)*fp - (n**2)/(r**2)*f, fp

def _piece_mode_arrays(piece, mode: int):
    """Extract (r, s_mode, m_mode) from a SecondOrderPiece."""
    r = piece.r
    if mode == 0:
        return r, piece.s0, piece.m0
    elif mode == 2:
        return r, piece.s2, piece.m2
    else:
        raise ValueError("Mode must be 0 or 2 for this checker.")
    
def check_second_order_pde_discrete(
    F,
    SO,
    *,
    modes=(0, 2),
    weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Discrete check of the second-order PDEs:

      Z L_n σ_{2S,n}  - (σ_{2S,n} - P m_{2S,n}) = 0
      L_n m_{2S,n}    - (Khat0 m0) L_n σ_{2S,n} - q_{S,n}(r) = 0

    for S in {A,B}, n in {0,2}. Uses finite differences on the arrays in SO.

    BC checks:
      • center:   n=0 ⇒ σ'(r→0)=0, m'(r→0)=0;  n=2 ⇒ σ(0)=0, m(0)=0
      • outer:    σ'(R0)=0, m'(R0)=0
      • boundary/shape (149).4:
           s0(R0) + (2π + γ/R0^2) ρ20S = 0,
           s2(R0) − (3γ/R0^2) ρ22S = 0.

    Returns a dict with total 'loss', per-piece/mode details, and parameters.
    """
    if weights is None:
        weights = dict(
            pde=1.0, bc_center=1.0, bc_outer=1.0, bc_shape=1.0
        )

    R0 = float(F.R0); Z = float(F.Z); P = float(F.P); m0 = float(F.m0); gamma =float(F.gamma)
    Khat0 = F.Khat0

    # Forcings built from first-order fields
    Qf = make_forcings_A_B_func(F)  # provides qA0,qA2,qB0,qB2

    def area_l2(r, x):  # (∫ r |x|^2 dr)^{1/2}
        return np.sqrt(np.trapz(r * (x**2), r))

    out_details: Dict[str, Any] = dict(A={}, B={})

    total_loss = 0.0
    for S_name, piece in (("A", SO.A), ("B", SO.B)):
        for n in modes:
            # --- arrays for this piece/mode
            r, s, m = _piece_mode_arrays(piece, n)
            # derivatives & modal Laplacians
            Ln_s, sp = _Ln(n, s, r)
            Ln_m, mp = _Ln(n, m, r)

            # --- PDE residuals
            res_sigma = Z*Ln_s - (s - P*m)
            if S_name == "A":
                q = Qf["qA0"](r) if n == 0 else Qf["qA2"](r)
            else:
                q = Qf["qB0"](r) if n == 0 else Qf["qB2"](r)
            res_m = Ln_m - (Khat0*m0)*Ln_s - q

            # L2 norms
            res_sigma_L2 = area_l2(r, res_sigma)
            res_m_L2     = area_l2(r, res_m)
            pde_L2_sum   = res_sigma_L2 + res_m_L2

            # --- BC residuals (discrete)
            if n == 0:
                # regularity: derivatives vanish at r≈0
                bc_center_s = abs(sp[0])
                bc_center_m = abs(mp[0])
            else:
                # regularity: functions vanish at r≈0
                bc_center_s = abs(s[0])
                bc_center_m = abs(m[0])

            # outer Neumann
            bc_outer_s = abs(sp[-1])
            bc_outer_m = abs(mp[-1])

            # --- boundary/shape relation (149).4)
            if n == 0:
                rho20S = getattr(SO, f"rho20{S_name}")
                bc_shape = abs(s[-1] + (2.0*np.pi + gamma/R0**2) * rho20S)
            else:
                rho22S = getattr(SO, f"rho22{S_name}")
                bc_shape = abs(s[-1] - (3.0*gamma/R0**2) * rho22S)

            # weighted loss for this (S,n)
            loss_Sn = (weights["pde"] * pde_L2_sum
                       + weights["bc_center"] * (bc_center_s + bc_center_m)
                       + weights["bc_outer"]  * (bc_outer_s  + bc_outer_m)
                       + weights["bc_shape"]  * bc_shape)

            total_loss += loss_Sn

            out_details[S_name][n] = dict(
                r_min=float(r[0]), r_max=float(r[-1]),
                res_sigma_L2=res_sigma_L2,
                res_m_L2=res_m_L2,
                pde_L2_total=pde_L2_sum,
                bc_center_s=bc_center_s, bc_center_m=bc_center_m,
                bc_outer_s=bc_outer_s,   bc_outer_m=bc_outer_m,
                bc_shape=bc_shape,
                loss=loss_Sn
            )

    return dict(
        loss=total_loss,
        details=out_details,
        params=dict(R0=R0, Z=Z, P=P, m0=m0, Khat0=Khat0, modes=list(modes))
    )
