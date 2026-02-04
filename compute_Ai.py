import numpy as np
from typing import Callable

from first_order import *
from second_order import *

#---- nested quadratures for Ã_i in (38)–(41) ----------

def _nested_quads(R0: float,
                  U: Callable[[np.ndarray], np.ndarray],
                  Up: Callable[[np.ndarray], np.ndarray],  #
                  f: np.ndarray,                           # f ≡ f1(r) sampled on grid r
                  r: np.ndarray,
                  P: float) -> float:
    """
    Compute:
      P * ∫_0^{R0} [ (1/2) * S2(r) + (r^2/2) * Fr(r) + (r^2/(2 R0^2)) * S2_total ] * U(r) dr
    where
      S2(r)      = ∫_0^r s^2 f(s) ds,
      Fr(r)      = ∫_r^{R0} f(s) ds,
      S2_total   = ∫_0^{R0} s^2 f(s) ds,
    using trapezoidal rules on the grid r.

    Args:
        R0: upper limit of integration (should equal r[-1])
        U:  callable returning U(r) array
        Up: callable for U'(r) (unused here; kept to match the previous signature)
        f:  array of f1(r) values on the same grid as r
        r:  1D, strictly increasing grid from 0 to R0
        P:  scalar prefactor

    Returns:
        Scalar value of the integral.
    """
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)

    if r.ndim != 1 or f.shape != r.shape:
        raise ValueError("r must be 1D and f must have the same shape as r.")
    dr = np.diff(r)
    if np.any(dr <= 0):
        raise ValueError("r must be strictly increasing.")

    # S2(r) = ∫_0^r s^2 f(s) ds (cumulative trapezoid)
    S2 = np.concatenate((
        [0.0],
        np.cumsum(0.5 * ((r[1:]**2)*f[1:] + (r[:-1]**2)*f[:-1]) * dr)
    ))
    S2_total = S2[-1]

    # Fr(r) = ∫_r^{R0} f(s) ds (reverse cumulative trapezoid)
    Fr_rev_cum = np.cumsum(0.5 * (f[::-1][1:] + f[::-1][:-1]) * dr[::-1])
    Fr = np.concatenate((Fr_rev_cum[::-1], [0.0]))

    Ur = U(r)
    integrand = (0.5 * S2
                 + 0.5 * (r**2) * Fr
                 + 0.5 * (r**2) * (S2_total / (R0**2))) * Ur

    return P * np.trapz(integrand, r)

# ---------- f1..f4 from (161)–(164) ----------
def _compute_fi_arrays(F: FirstOrder, SO: SecondOrderAll) -> Dict[str, np.ndarray]:
    r = SO.A.r
    s = r  # naming to mirror the paper

    # First-order hats and derivatives
    m11, m11p, m11pp = F.m11(s), F.m11p(s), F.m11pp(s)
    s11, s11p, s11pp = F.s11(s), F.s11p(s), F.s11pp(s)

    # Second order pieces (A/B) and derivatives (central differences)
    def d1(y):  return np.gradient(y, s, edge_order=2)
    def d2(y):  return np.gradient(np.gradient(y, s, edge_order=2), s, edge_order=2)

    m20A, m22A = SO.A.m0, SO.A.m2
    m20B, m22B = SO.B.m0, SO.B.m2
    s20A, s22A = SO.A.s0, SO.A.s2
    s20B, s22B = SO.B.s0, SO.B.s2

    m20A_p, m22A_p = d1(m20A), d1(m22A)
    m20B_p, m22B_p = d1(m20B), d1(m22B)
    m20A_pp, m22A_pp = d2(m20A), d2(m22A)
    m20B_pp, m22B_pp = d2(m20B), d2(m22B)

    s20A_p, s22A_p = d1(s20A), d1(s22A)
    s20B_p, s22B_p = d1(s20B), d1(s22B)
    s20A_pp, s22A_pp = d2(s20A), d2(s22A)
    s20B_pp, s22B_pp = d2(s20B), d2(s22B)

    K0h = F.Khat0  # K̂0

    # --- f1 (unchanged)
    f1 = (1.0 / (8.0 * r**2)) * m11 * (
        -m11**2
        + 6.0 * r**2 * (m11p**2)
        + 3.0 * r * m11 * (m11p + r * m11pp)
    )

    # --- f2 (UPDATED)
    f2 = (1.0 / (8.0 * r**2)) * (
        -8.0 * m11 * m20B
        -4.0 * m11 * m22B
        +8.0 * r * m20B * m11p
        +4.0 * r * m22B * m11p
        +8.0 * r * m11 * m20B_p
        +16.0 * r**2 * m11p * m20B_p
        +4.0 * r * m11 * m22B_p
        +8.0 * r**2 * m11p * m22B_p
        +8.0 * r**2 * m20B * m11pp
        +4.0 * r**2 * m22B * m11pp
        +8.0 * r**2 * m11 * m20B_pp
        +4.0 * r**2 * m11 * m22B_pp
    )

    # --- f3 (UPDATED)
    f3 = (1.0 / (8.0 * r**2)) * (
        -8.0 * m11 * m20A
        -4.0 * m11 * m22A
        +8.0 * r * m20A * m11p
        +4.0 * r * m22A * m11p
        +8.0 * r * m11 * m20A_p
        +16.0 * r**2 * m11p * m20A_p
        +4.0 * r * m11 * m22A_p
        +8.0 * r**2 * m11p * m22A_p
        +8.0 * r**2 * m20A * m11pp
        +4.0 * r**2 * m22A * m11pp
        +8.0 * r**2 * m11 * m20A_pp
        +4.0 * r**2 * m11 * m22A_pp
        +8.0 * r**2 * m20B_p
        +4.0 * r**2 * m22B_p
        +8.0 * r * m22B
        +K0h * (
            8.0 * m20B * s11
            -4.0 * m22B * s11
            +8.0 * m11 * s22B
            -8.0 * r * m20B * s11p
            -4.0 * r * m22B * s11p
            -8.0 * r**2 * m20B_p * s11p
            -4.0 * r**2 * m22B_p * s11p
            -8.0 * r * m11 * s20B_p
            -4.0 * r * m11 * s22B_p
            -4.0 * r**2 * m11p * s22B_p
            -8.0 * r**2 * m20B * s11pp
            -4.0 * r**2 * m22B * s11pp
            -8.0 * r**2 * m11p * s20B_p
            -8.0 * r**2 * m11 * s20B_pp
            -4.0 * r**2 * m11 * s22B_pp
        )
    )

    # --- f4
    f4 = (1.0 / (8.0 * r**2)) * (
        8.0 * r**2 * m20A_p
        +4.0 * r**2 * m22A_p
        +8.0 * r * m22A
        +K0h * (
            8.0 * m20A * s11
            -4.0 * m22A * s11
            +8.0 * m11 * s22A
            -8.0 * r * m20A * s11p
            -4.0 * r * m22A * s11p
            -8.0 * r**2 * m20A_p * s11p
            -4.0 * r**2 * m22A_p * s11p
            -8.0 * r * m11 * s20A_p
            -4.0 * r * m11 * s22A_p
            -4.0 * r**2 * m11p * s22A_p
            -8.0 * r**2 * m20A * s11pp
            -4.0 * r**2 * m22A * s11pp
            -8.0 * r**2 * m11p * s20A_p
            -8.0 * r**2 * m11 * s20A_pp
            -4.0 * r**2 * m11 * s22A_pp
        )
    )


    return dict(f1=f1, f2=f2, f3=f3, f4=f4)

# ---------- A-tilde_i and A_i via (38)–(41) and (42) ----------
def compute_Ais(F: FirstOrder, SO: SecondOrderAll) -> Dict[str, float]:
    r = SO.A.r
    R0, P = F.R0, F.P
    U, Up  = F.U, F.Up

    # --- A0
    # A0 = -(Z R0)/Khat0^2 + P m0 * ∫ r ( m11(r) + r/Khat0 ) U(r) dr
    IU_kernel = np.trapz(r*(F.s11(r)) * U(r), r )
    A0 = -(F.Z * R0) / (F.Khat0**2) + P * F.m0 * IU_kernel

    # these are still needed for Ã3/Ã4 boundary pieces
    IU_r2 = np.trapz( (r**2) * U(r), r)

    fis = _compute_fi_arrays(F, SO)

    # --- A1
    A1t = -_nested_quads(R0, U, Up, fis["f1"], r, P)
    

    # --- A2
    A2t = -_nested_quads(R0, U, Up, fis["f2"], r, P)



    # --- A3
    A3t  = -_nested_quads(R0, U, Up, fis["f3"], r, P)

    A3t -= IU_r2*P*(
        (SO.rho20B+SO.rho22B/2)*(F.Khat0*F.m0*F.s11pp_R0-F.m11pp_R0)
        +SO.rho22B*F.m11_R0/R0**2
    )

    A3t += F.Z*R0*(SO.rho20B+SO.rho22B/2)*(F.s11pp_R0- F.alpha * J1p(F.alpha) / (R0*F.Khat0 * J1(F.alpha)))


    # --- A4
    A4t  = -_nested_quads(R0, U, Up, fis["f4"], r, P)

    A4t -= IU_r2*P*(
        (SO.rho20A+SO.rho22A/2)*(F.Khat0*F.m0*F.s11pp_R0-F.m11pp_R0)
        +SO.rho22A*F.m11_R0/R0**2
    )

    A4t += F.Z*R0*(SO.rho20A+SO.rho22A/2)*(F.s11pp_R0- F.alpha * J1p(F.alpha) / (R0*F.Khat0 * J1(F.alpha)))

    # normalize
    A1 = A1t / A0
    A2 = A2t / A0
    A3 = A3t / A0
    A4 = A4t / A0

    #print("A1",A1, "A2", A2, "A3", A3, "A4",A4)
    return dict(A0=A0, A1=A1, A2=A2, A3=A3, A4=A4,
                meta=dict(R0=F.R0, m0=F.m0, Khat0=F.Khat0, alpha=F.alpha))


# ---------- K2 from (37)/(19) ----------
def K2_from_Ai(A: Dict[str, float], D0: float, Dp: float, Dpp: float) -> float:
    return (A["A1"]*Dpp/(D0**2) + A["A2"]*(Dp*Dp)/(D0**3) + A["A3"]*Dp/(D0**2) + A["A4"]/(D0))
