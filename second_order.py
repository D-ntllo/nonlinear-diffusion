import numpy as np
from typing import Callable, Optional
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

def _build_mesh(R0: float, eps_factor: float, N_init: int, mesh_power: float) -> np.ndarray:
    eps = max(1e-12, eps_factor*R0)
    s = np.linspace(0.0, 1.0, int(N_init)+1)
    return eps + (R0 - eps) * (s**mesh_power)



# ---- L_n^{-1} q via Green's integral (regular at 0, Neumann at R0) ----
def _Ln_inverse_neumann(n: int, q: Callable[[np.ndarray], np.ndarray], r: np.ndarray, R0: float) -> np.ndarray:
    rr = r
    qv = q(rr)

    if n > 0:
        # I1(r) = ∫_0^r s^{1-n} q(s) ds,  I2(r) = ∫_0^r s^{n+1} q(s) ds
        I1 = _cumtrapz_from_zero((rr**(1-n))*qv, rr)
        I2 = _cumtrapz_from_zero((rr**(n+1))*qv, rr)
        # A from u'(R0)=0
        #A = -0.5*( I1[-1] + (R0**(-2*n))*I2[-1] )/n
        #A = - 2*float(q(np.array(R0)))/ (n**2 * R0**(n-2)) + 1/(2*n)*I1[-1] -1/(2*n)*R0**(-2*n) * I2[-1] 
        A = -1/(2*n)*I1[-1] -R0**(-2*n)/(2*n)*I2[-1]
        u = A*(rr**n) + (rr**n)*I1/(2*n) - (rr**(-n))*I2/(2*n)
        return u
    else:
        # n = 0: u(r) = A - ∫_0^r s ln s q(s) ds + (ln r) ∫_0^r s q(s) ds
        F1 = _cumtrapz_from_zero(rr*qv, rr)
        # handle ln at first point safely
        ln_r = np.log(rr)
        F2 = _cumtrapz_from_zero(rr*ln_r*qv, rr)
        # compatibility for Neumann: u'(R0) = (1/R0) F1(R0) = 0
        comp = F1[-1]
        if abs(comp) > 1e-10*abs(F1[-1] + 1e-30):
            # You can raise if you prefer strictness:
            # raise RuntimeError(f"Neumann compatibility fails for n=0: ∫_0^{R0} s q(s) ds = {comp}")
            pass
        A = 0.0  # gauge (constant is nullspace)
        u = A - F2 + ln_r*F1
        return u

# ---- Solve σ from (L_n + κ^2)σ = f with σ'(R0)=0 using modified-Bessel integral repr. ----
def _solve_sigma_bessel(n: int, kappa: float,f: Callable[[np.ndarray], np.ndarray], r: np.ndarray, R0: float) -> Tuple[np.ndarray, float]:
    rr = r
    kr = kappa*rr
    fv = f(rr)

    # I1(r) = ∫_0^r s Y_n(κ s) f(s) ds,  I2(r) = ∫_0^r s J_n(κ s) f(s) ds
    I1 = _cumtrapz_from_zero(rr * Y(n, kappa*rr) * fv, rr)
    I2 = _cumtrapz_from_zero(rr * J(n, kappa*rr) * fv, rr)

    # A from σ'(R0)=0:
    kr0 = kappa*R0
    A= np.pi/2*(-(Yp(n, kr0)/Jp(n, kr0)) * I2[-1] + I1[-1])

    # σ(r) = J_n(κ r)[A - (π/2) * I1(r)] + (π/2) * Y_n(κ r) * I2(r)
    sigma = J(n, kr)*(A - (np.pi/2)*I1) + (np.pi/2)*Y(n, kr)*I2

    # also return σ(R0)
    sR0 = float(J(n, kr0)*(A - (np.pi/2)*I1[-1]) + (np.pi/2)*Y(n, kr0)*I2[-1])
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

    # A piece
    s0A, s2A, m0A, m2A, sR0_0A, sR0_2A = solve_piece(Qf["qA0"], Qf["qA2"])
    # B piece
    s0B, s2B, m0B, m2B, sR0_0B, sR0_2B = solve_piece(Qf["qB0"], Qf["qB2"])

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