import numpy as np

from first_order import *
from second_order import *

#---- nested quadratures for Ã_i in (38)–(41) ----------

def _nested_quads(R0: float,
                  U: Callable[[np.ndarray], np.ndarray],
                  Up: Callable[[np.ndarray], np.ndarray],
                  f: np.ndarray,
                  r: np.ndarray,
                  P: float) -> float:
    S2 = np.cumsum(0.5*( (r[1:]**2)*f[1:] + (r[:-1]**2)*f[:-1]) * np.diff(r))
    S2 = np.concatenate([[0.0], S2])          # ∫_0^r s^2 f(s) ds
    Fr = np.cumsum(0.5*(f[::-1][1:] + f[::-1][:-1])*np.diff(r)[::-1])[::-1]
    Fr = np.concatenate([Fr, [0.0]])          # ∫_r^{R0} f(s) ds

    I1 = (P/R0**2) * np.trapz((r**2)*U(r), r) * np.trapz((r**2)*f, r)
    I2 =  P * np.trapz( (r**2)*U(r) * Fr, r)
    I3 = -P * np.trapz( U(r) * S2, r)
    return I1 + I2 + I3

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

    # --- f1 (161)
    f1 = -(1.0/(4.0*s**2)) * (
            m11**3
            - 6.0*(s**2)*m11*(m11p**2)
            - 3.0*s*(m11**2)*(m11p + s*m11pp)
         )

    # --- f2 (162)  (uses B-piece only)
    f2 = -(1.0/(s**2)) * (
            2.0*m11*m20B + 4.0*m11*m22B
          - 2.0*s*m20B*m11p - 1.0*s*m22B*m11p
          - 2.0*s*m11*m20B_p - 4.0*(s**2)*m11p*m20B_p
          - 1.0*s*m11*m22B_p - 2.0*(s**2)*m11p*m22B_p
          - 2.0*(s**2)*m20B*m11pp - (s**2)*m22B*m11pp
          - 2.0*(s**2)*m11*m20B_pp - (s**2)*m11*m22B_pp
         )

    # --- f3 (163)  (A/B mix + σ_B; all hats/ρ-independent here)
    f3 = -(1.0/(4.0*s**2)) * (
          8.0*m11*m20A - 4.0*K0h*s11*m20B + 4.0*m11*m22A + 2.0*K0h*s11*m22B
        - 4.0*K0h*m11*s22B
        - 8.0*s*m20A*m11p - 4.0*s*m22A*m11p
        + 4.0*K0h*s*m20B*s11p + 2.0*K0h*s*m22B*s11p
        - 8.0*s*m11*m20A_p - 16.0*(s**2)*m11p*m20A_p - 4.0*(s**3)*m20B_p
        + 4.0*K0h*(s**2)*s11p*m20B_p
        - 4.0*s*m11*m22A_p - 8.0*(s**2)*m11p*m22A_p - 2.0*(s**3)*m22B_p
        + 2.0*K0h*(s**2)*s11p*m22B_p
        + 4.0*K0h*s*m11*s20B_p + 4.0*K0h*(s**2)*m11p*s20B_p
        + 2.0*K0h*s*m11*s22B_p + 2.0*K0h*(s**2)*m11p*s22B_p
        - 8.0*(s**2)*m20A*m11pp - 4.0*(s**2)*m22A*m11pp
        + 4.0*K0h*(s**2)*m20B*s11pp + 2.0*K0h*(s**2)*m22B*s11pp
        - 8.0*(s**2)*m11*m20A_pp - 4.0*(s**2)*m11*m22A_pp
        + 4.0*K0h*(s**2)*m11*s20B_pp + 2.0*K0h*(s**2)*m11*s22B_pp
        )

    # --- f4 (164)  (A-piece with σ_A)
    f4 = -(1.0/(4.0*s**2)) * (
         -4.0*K0h*s11*m20A + 2.0*K0h*s11*m22A - 4.0*K0h*m11*s22A
         + 4.0*K0h*s*m20A*s11p + 2.0*K0h*s*m22A*s11p
         - 4.0*(s**3)*m20A_p + 4.0*K0h*(s**2)*s11p*m20A_p
         - 2.0*(s**3)*m22A_p + 2.0*K0h*(s**2)*s11p*m22A_p
         + 4.0*K0h*s*m11*s20A_p + 4.0*K0h*(s**2)*m11p*s20A_p
         + 2.0*K0h*s*m11*s22A_p + 2.0*K0h*(s**2)*m11p*s22A_p
         + 4.0*K0h*(s**2)*m20A*s11pp + 2.0*K0h*(s**2)*m22A*s11pp
         + 4.0*K0h*(s**2)*m11*s20A_pp + 2.0*K0h*(s**2)*m11*s22A_pp
         )

    return dict(f1=f1, f2=f2, f3=f3, f4=f4)

# ---------- A-tilde_i and A_i via (38)–(41) and (42) ----------
def compute_Ais(F: FirstOrder, SO: SecondOrderAll) -> Dict[str, float]:
    r = SO.A.r
    R0, P = F.R0, F.P
    U, Up  = F.U, F.Up

    # --- A0 from your corrected eq. (36):
    # A0 = (Z R0)/Khat0^2 - P m0 * ∫ ( m11(r) + r/Khat0 ) U(r) dr
    IU_kernel = np.trapz( (F.m11(r) + r/F.Khat0) * U(r), r )
    A0 = (F.Z * R0) / (F.Khat0**2) - P * F.m0 * IU_kernel

    # these are still needed for Ã3/Ã4 boundary pieces
    IU_r2 = np.trapz( (r**2) * U(r), r )

    fis = _compute_fi_arrays(F, SO)

    # (38)–(39)
    A1t = _nested_quads(R0, U, Up, fis["f1"], r, P)
    A2t = _nested_quads(R0, U, Up, fis["f2"], r, P)

    # (40) + boundary terms
    A3t  = _nested_quads(R0, U, Up, fis["f3"], r, P)
    A3t += P * ( -2.0*SO.rho22B*F.m11_R0
                 + (F.Khat0*F.m0 - 1.0)*(R0**2)*(2.0*SO.rho20B + SO.rho22B)*F.m11pp_R0
               ) * ( IU_r2 / (2.0*R0**2) )
    A3t += ( F.alpha * J1p(F.alpha) / (F.Khat0 * J1(F.alpha)) ) * ( -2.0*SO.rho20B + SO.rho22B ) * F.Z
    A3t += (2.0*SO.rho20B + SO.rho22B) * F.m11pp_R0

    # (41) + boundary terms
    A4t  = _nested_quads(R0, U, Up, fis["f4"], r, P)
    A4t += P * ( -2.0*SO.rho22A*F.m11_R0
                 + (F.Khat0*F.m0 - 1.0)*(R0**2)*(2.0*SO.rho20A + SO.rho22A)*F.m11pp_R0
               ) * ( IU_r2 / (2.0*R0**2) )
    A4t += ( F.alpha * J1p(F.alpha) / (2.0*F.Khat0 * J1(F.alpha)) ) * ( -2.0*SO.rho20A + SO.rho22A ) * F.Z
    A4t += (2.0*SO.rho20A + SO.rho22A) * F.m11pp_R0

    # normalize
    A1 = A1t / A0
    A2 = A2t / A0
    A3 = A3t / A0
    A4 = A4t / A0
    return dict(A0=A0, A1=A1, A2=A2, A3=A3, A4=A4,
                meta=dict(R0=F.R0, m0=F.m0, Khat0=F.Khat0, alpha=F.alpha))


# ---------- K2 from (37)/(19) ----------
def K2_from_Ai(A: Dict[str, float], D0: float, Dp: float, Dpp: float) -> float:
    return (A["A1"]*Dpp/(D0**4) + A["A2"]*(Dp*Dp)/(D0**5) + A["A3"]*Dp/(D0**4) + A["A4"]/(D0**3))