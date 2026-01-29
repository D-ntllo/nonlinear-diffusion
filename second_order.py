import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

from steady_state import *
from first_order import *

from scipy.special import iv as I, kv as K, ivp as Ip, kvp as Kp, jv as J, yv as Y, jvp as Jp, yvp as Yp


@dataclass
class SecondOrderPiece:
    r: np.ndarray
    m0: np.ndarray
    m2: np.ndarray
    s0: np.ndarray
    s2: np.ndarray

@dataclass
class SecondOrderAll:
    A: SecondOrderPiece
    B: SecondOrderPiece
    rho20A: float
    rho22A: float
    rho20B: float
    rho22B: float


def make_forcings_A_B_func(F):
    """
    Return callables qA0(r), qA2(r), qB0(r), qB2(r) built from the analytic
    first-order fields in F: s11, s11p, s11pp, m11, m11p, m11pp.
    Conventions (paper eqs. (149)–(150)):
      A: Δ m2A − K̂0 m0 Δ σ2A = − e1·∇m̂1 + K̂0(∇m̂1·∇σ̂1 + m̂1 Δσ̂1)
      B: Δ m2B − K̂0 m0 Δ σ2B = − (1/2) Δ( m̂1^2 )
    with  m̂1 = m(r) cosθ,  σ̂1 = s(r) cosθ,  and  L_n f = f'' + (1/r) f' − n^2 f / r^2.
    """
    s, sp, spp = F.s11, F.s11p, F.s11pp
    m, mp, mpp = F.m11, F.m11p, F.m11pp
    Khat0 = F.Khat0

    '''
    # A-piece pieces (projected to n=0 and n=2)
    '''
    qA0 = lambda r: -(0.5*(mp(r)+m(r)/r) - Khat0/2*(mp(r)*sp(r)+m(r)*spp(r)+m(r)*sp(r)/r))
    qA2 = lambda r: -(0.5*(mp(r)-m(r)/r) - Khat0/2*(mp(r)*sp(r)+m(r)*spp(r)+m(r)*sp(r)/r-2*m(r)*s(r)/r**2))
    '''
    # B-piece: −(1/2) Δ(m̂1^2). With m̂1^2 = 1/2 m^2 (1 + cos 2θ):
    # projections give qB0 = −¼ L0(m^2), qB2 = −¼ L2(m^2)
    '''
    qB0 = lambda r: -(0.5*(mp(r)*mp(r)+m(r)*mpp(r)+1/r*m(r)*mp(r)))
    qB2 = lambda r: -(0.5*(mp(r)*mp(r)+m(r)*mpp(r)+1/r*m(r)*mp(r))-m(r)*m(r)/r**2)


    return dict(qA0=qA0, qA2=qA2, qB0=qB0, qB2=qB2)


# ---- small utilities ----
def _cumtrapz_from_zero(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid ∫_0^x y(s) ds on the given grid (x[0] is the lower limit)."""
    cs = np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(x))
    return np.concatenate(([0.0], cs))

# add a helper (suffix integral) next to _cumtrapz_from_zero
def _cumtrapz_to_R0(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Cumulative trapezoid ∫_x^R0 y(s) ds on the given grid, returned as an array B with:
        B[i] = ∫_{x[i]}^{x[-1]} y(s) ds,  B[-1] = 0.
    """
    n = len(x)
    if n == 1:
        return np.zeros_like(x, dtype=float)
    dx = np.diff(x)
    areas = 0.5 * (y[:-1] + y[1:]) * dx
    B = np.zeros(n, dtype=float)
    B[:-1] = np.cumsum(areas[::-1])[::-1]
    B[-1] = 0.0
    return B

def _build_mesh(R0: float, eps_factor: float, N_init: int, mesh_power: float) -> np.ndarray:
    eps = max(1e-12, eps_factor*R0)
    s = np.linspace(0.0, 1.0, int(N_init)+1)
    return eps + (R0 - eps) * (s**mesh_power)

# ---- L_n^{-1} q via Green's integral (regular at 0, Neumann at R0) ----
def _Ln_inverse_neumann(
    n: int,
    q: Callable[[np.ndarray], np.ndarray],
    r: np.ndarray,
    R0: float,
    neuman_bc: float = 0.0,   # enforce u'(R0) = neuman_bc (default 0.0)
) -> np.ndarray:
    """
    Solve L_n u = q on (0,R0), regular at 0, with Neumann condition u'(R0) = neuman_bc.

    For n > 0, the regular general solution is
        u(r) = a r^n + (r^n)/(2n) ∫_0^r s^{1-n} q(s) ds - (r^{-n})/(2n) ∫_0^r s^{n+1} q(s) ds,
    and choosing
        a = neuman_bc / (n R0^{n-1}) - (1/(2n)) I1(R0) - (1/(2n)) R0^{-2n} I2(R0)
    ensures u'(R0) = neuman_bc exactly.

    For n = 0, regularity eliminates the ln r homogeneous piece, so u'(R0) is fixed by q:
        u'(r) = (1/r) ∫_0^r s q(s) ds  ⇒  u'(R0) = (1/R0) ∫_0^{R0} s q(s) ds.
    A nonzero target neuman_bc therefore requires the compatibility ∫_0^{R0} s q(s) ds = neuman_bc * R0.
    """
    rr = r
    qv = q(rr)

    if n > 0:
        # I1(r) = ∫_0^r s^{1-n} q(s) ds,  I2(r) = ∫_0^r s^{n+1} q(s) ds
        I1 = _cumtrapz_from_zero((rr**(1-n)) * qv, rr)
        I2 = _cumtrapz_from_zero((rr**(n+1)) * qv, rr)

        # a from u'(R0) = neuman_bc:
        # u'(R0) = a n R0^{n-1} + 0.5 R0^{n-1} I1(R0) + 0.5 R0^{-n-1} I2(R0)
        a = (neuman_bc / (n * R0**(n-1))) - (I1[-1] / (2*n)) - (R0**(-2*n) * I2[-1] / (2*n))

        u = a * (rr**n) + (rr**n) * I1 / (2*n) - (rr**(-n)) * I2 / (2*n)
        return u

    else:
        # n = 0: u(r) = C - ∫_0^r s ln s q(s) ds + (ln r) ∫_0^r s q(s) ds
        F1 = _cumtrapz_from_zero(rr * qv, rr)            # ∫_0^r s q(s) ds
        ln_r = np.log(rr)
        F2 = _cumtrapz_from_zero(rr * ln_r * qv, rr)     # ∫_0^r s ln s q(s) ds

        # Compatibility for regular Neumann: u'(R0) = F1(R0)/R0 must equal neuman_bc
        comp = F1[-1] / R0
        # Keep behavior non-intrusive: warn if incompatible (you may raise instead)
        if abs(comp - neuman_bc) > 1e-10 * (abs(comp) + 1.0):
            # print or log a warning if you have a logger; we just pass silently here:
            # warnings.warn(f"n=0 Neumann target {neuman_bc} incompatible with forcing; using u'(R0)={comp}.")
            pass

        C = 0.0  # gauge (constant nullspace)
        u = C - F2 + ln_r * F1
        return u

# ---- Solve σ from (L_n + κ^2)σ = f with σ'(R0)=0 using modified-Bessel integral repr. ----
def _solve_sigma_bessel(n: int, kappa: float, f: np.ndarray, r: np.ndarray, R0: float) -> Tuple[np.ndarray, float]:
    """
    Solve (L_n + kappa^2) σ = f on (0,R0) with σ'(R0)=0, regular at r=0.

    Green's representation for '+' sign (ordinary Bessels):

    This constant A enforces σ'(R0)=0 by construction.
    """
    rr = np.asarray(r, dtype=float)
    fv = np.asarray(f, dtype=float)
    assert rr.ndim == 1 and fv.shape == rr.shape, "r and f must be 1D arrays of same length."
    assert np.all(np.diff(rr) > 0), "r must be strictly increasing."
    assert np.isclose(rr[-1], R0), "R0 must equal r[-1] for this routine."

    kr  = kappa * rr
    kr0 = kappa * R0

    # fundamental solutions and derivatives (w.r.t. argument)
    Jn  = lambda x: J(n, x)
    Yn  = lambda x: Y(n, x)
    Jnp = lambda x: Jp(n, x)   # d/dx J_n(x)
    Ynp = lambda x: Yp(n, x)   # d/dx Y_n(x)

    denom = Jnp(kr0)
    if np.isclose(denom, 0.0):
        raise ZeroDivisionError("Neumann resonance: J_n'(kappa*R0) ≈ 0; adjust kappa or R0.")
    

    fracnum = (Y(n-1,kappa*R0)+Y(n+1, kappa*R0))* _cumtrapz_from_zero(rr*Jn(kr)*fv,rr)[-1]
    fracden = J(n-1, kappa*R0) - J(n+1, kappa*R0)
    frac = fracnum/ fracden


    integral = _cumtrapz_from_zero(rr*Yn(kr)*fv,rr)[-1]
    A = 1/2*np.pi*(frac + integral)


    sigma = -np.pi/2*Jn(kr)*_cumtrapz_from_zero(rr*Yn(kr)*fv,rr) + np.pi/2*Yn(kr)*_cumtrapz_from_zero(rr*Jn(kr)*fv,rr)+A*Jn(kr)

    sR0 = 1/2*np.pi*Yn(kappa*R0) * _cumtrapz_from_zero(rr*Jn(kr)*fv,rr)[-1] + Jn(kappa*R0)*(A - 1/2*np.pi*_cumtrapz_from_zero(rr*Yn(kr)*fv,rr)[-1])

    # c = -Ynp(kr0) / denom

    # phi = Jn(kr)                  # left-regular
    # psi = Yn(kr) + c * Jn(kr)     # right-Neumann

    # # prefix and suffix weighted integrals
    # I2 = _cumtrapz_from_zero(rr * phi * fv, rr)   # ∫_0^r s J_n(κ s) f(s) ds
    # B  = _cumtrapz_to_R0(rr * psi * fv, rr)       # ∫_r^{R0} s ψ̂(κ s) f(s) ds

    # sigma = (np.pi/2.0) * (psi * I2 + phi * B)

    # # σ(R0): since B(R0)=0, I2(R0)=∫_0^{R0} s J_n(κ s) f(s) ds
    # sR0 = float((np.pi/2.0) * (Yn(kr0) + c * Jn(kr0)) * I2[-1])

    return sigma, sR0

# ---- Main: compute second order via integrals (no BVPs) ----
def compute_second_order(
    F,
    *,
    modes=(0, 2),
    eps_factor: float = 1e-3,
    N_init: int = 400,
    mesh_power: float = 3.0,
) -> SecondOrderAll:

    R0 = float(F.R0); Z = float(F.Z); P = float(F.P); m0 = float(F.m0); Khat0 = float(F.Khat0); gamma = float(F.gamma)
    # Forcings q_{S,n}(r)
    Qf = make_forcings_A_B_func(F)

    # mesh for each piece (same for all to keep life simple)
    r = _build_mesh(R0, eps_factor, N_init, mesh_power)

    # κ^2 = (P Khat0 m0 - 1)/Z  (positive on the physical branch)
    kappa2 = (P*Khat0*m0 - 1.0)/Z
    if kappa2 <= 0:
        raise RuntimeError(f"kappa^2 <= 0: (P*Khat0*m0-1)/Z = {kappa2}. Pick a physical Khat0 branch.")
    kappa = np.sqrt(kappa2)

    def solve_piece(q0_fn, q2_fn) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        # u_n = L_n^{-1} q_n  (regular at 0, Neumann at R0)
        out = {}
        sR0s = {}
        for n, qfn in ((0, q0_fn), (2, q2_fn)):
            if n not in modes:
                out[n] = (np.zeros_like(r), np.zeros_like(r)); sR0s[n] = 0.0
                continue
            u = _Ln_inverse_neumann(n, qfn, r, R0)
            f_sigma = -(P/Z)*u
            sigma, sR0 = _solve_sigma_bessel(n, kappa, f_sigma, r, R0)
            m = u + Khat0*m0*sigma
            out[n] = (sigma, m); sR0s[n] = sR0
        return out[0][0], out[2][0], out[0][1], out[2][1], sR0s[0], sR0s[2]

    # A piec.  e
    s0A, s2A, m0A, m2A, sR0_0A, sR0_2A = solve_piece(Qf["qA0"], Qf["qA2"])
    # B piece
    s0B, s2B, m0B, m2B, sR0_0B, sR0_2B = solve_piece(Qf["qB0"], Qf["qB2"])


    ## At this point n=0 solutions need to be adjusted  to respect the integral condition for m and rho.

    def _update_solns_by_A(sigma_sol: np.ndarray, m_sol: np.ndarray, sR0_sol: float) -> Tuple[np.ndarray, np.ndarray, float]:
        Intm = _cumtrapz_from_zero(m_sol, r)[-1]

        Anum = (P*Khat0*m0-1)*(m0*R0**2*sR0_sol+ gamma* Intm - 2*np.pi*R0**2*Intm)
        Aden = (-m0*P-R0+Khat0*m0*P*R0)*(-gamma + 2*np.pi*R0**2)

        A = Anum/Aden
        sA = -P*A/ (P*Khat0*m0-1)

        return sigma_sol+sA, m_sol+A, sR0_sol+sA
    
    def _update_solns_by_A_new(sigma_sol: np.ndarray, m_sol: np.ndarray, sR0_sol: float) -> Tuple[np.ndarray, np.ndarray, float]:
        
        int_m = _cumtrapz_from_zero(m_sol*r,r)[-1]

        rho20Atilde= - sR0_sol / (2.0*np.pi + gamma/R0**2)

        Cm_num = 4*np.pi*m0*R0*rho20Atilde + 2*np.pi*int_m

        Cm_den = 4*np.pi*m0*R0*(-P*1/(2.0*np.pi + gamma/R0**2)) + 2*np.pi*R0**2

        Cm = - Cm_num / Cm_den

        Cs = P*Cm

        return sigma_sol+Cs, m_sol+Cm, sR0_sol+Cs
    
    s0A, m0A, sR0_0A = _update_solns_by_A_new(s0A, m0A, sR0_0A)
    s0B, m0B, sR0_0B = _update_solns_by_A_new(s0B, m0B, sR0_0B)

    # Map σ(R0) -> ρ via (149).4
    rho20A = - sR0_0A / (2.0*np.pi + gamma/R0**2) if 0 in modes else 0.0
    rho22A =   (R0**2)*sR0_2A / (3.0*gamma)       if 2 in modes else 0.0
    rho20B = - sR0_0B / (2.0*np.pi + gamma/R0**2) if 0 in modes else 0.0
    rho22B =   (R0**2)*sR0_2B / (3.0*gamma)       if 2 in modes else 0.0

    A_piece = SecondOrderPiece(r=r, m0=m0A, m2=m2A, s0=s0A, s2=s2A)
    B_piece = SecondOrderPiece(r=r, m0=m0B, m2=m2B, s0=s0B, s2=s2B)

    return SecondOrderAll(A=A_piece, B=B_piece,
                          rho20A=rho20A, rho22A=rho22A,
                          rho20B=rho20B, rho22B=rho22B)



# -------------------- Field assembly up to order in V ------------------

def build_sigma_m(F, SO, D, Dp, V: float, order: int = 1):
    """
    Construct σ(r,θ), m(r,θ) up to given order in V using F and SO.
    order=1: σ = σ0 + V σ1 cosθ;   m = m0 + V m1 cosθ
    order=2: add V^2 (n=0,2) from SO.A+SO.B
    Returns two vectorized callables (r,θ)->array.
    """
    m0 = F.m0
    sigma0 = F.P * m0  # makes ZΔσ0 - σ0 + P m0 = 0
    s1 = lambda r: F.s11(r)
    m1 = lambda r: F.m11(r)

    if order == 1:
        def sigma(r, th): return sigma0 + V*s1(r)*np.cos(th)
        def m(r, th):     return F.m0 + V*m1(r)*np.cos(th)
        return sigma, m

    # second-order add-ons
    rA, s0A, m0A, s2A, m2A = SO.A.r, SO.A.s0, SO.A.m0, SO.A.s2, SO.A.m2
    rB, s0B, m0B, s2B, m2B = SO.B.r, SO.B.s0, SO.B.m0, SO.B.s2, SO.B.m2

    
    s0_tot = lambda rr: 1/D(m0)**2*np.interp(rr, rA, s0A) + Dp(m0)/D(m0)**3 * np.interp(rr, rB, s0B)
    m0_tot = lambda rr: 1/D(m0)**2*np.interp(rr, rA, m0A) + Dp(m0)/D(m0)**3 * np.interp(rr, rB, m0B)
    s2_tot = lambda rr: 1/D(m0)**2*np.interp(rr, rA, s2A) + Dp(m0)/D(m0)**3 * np.interp(rr, rB, s2B)
    m2_tot = lambda rr: 1/D(m0)**2*np.interp(rr, rA, m2A) + Dp(m0)/D(m0)**3 * np.interp(rr, rB, m2B)

    def sigma(r, th):
        return sigma0 + V*s1(r)*np.cos(th) + (V**2)*( s0_tot(r) + s2_tot(r)*np.cos(2*th) )
    def m(r, th):
        return F.m0 + V*m1(r)*np.cos(th) + (V**2)*( m0_tot(r) +m2_tot(r)*np.cos(2*th))
    return sigma, m
           