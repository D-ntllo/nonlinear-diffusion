import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
from scipy import optimize, special
from scipy.integrate import solve_bvp
from typing import Optional,Any
from findiff import FinDiff, Diff, Laplacian

import matplotlib.pyplot as plt

# --- Bessel helpers
J1   = lambda x: special.jv(1, x)
J1p  = lambda x: special.jvp(1, x, 1)
J1pp = lambda x: special.jvp(1, x, 2)

# ---------- root utilities ----------
def solve_R0(P: float, gamma: float, rmax: float = 100.0) -> float:
    """Largest positive root of: 0 = -γ/R + 1 - πR^2 - P/(πR^2).  (eq. 13)"""
    def F(R):
        return -gamma/R + 1.0 - np.pi*R**2 - P/(np.pi*R**2)
    xs = np.geomspace(1e-4, rmax, 1200)
    vals = F(xs)
    roots = []
    for a,b,fa,fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa*fb < 0:
            roots.append(optimize.brentq(F, a, b))
    if not roots:
        raise RuntimeError("Failed to find R0; widen rmax or check (P, gamma).")
    return max(roots)

def solve_Khat0(
    P: float, Z: float, R0: float, m0: float,
    *,
    root_index: int = 0,
    Khat0_max: float = 5e3,
    n_samples: int = 20000,
) -> Tuple[float, float]:
    """
    Solve for Khat0 = K0 / D(m0) from (14)-(15) without D:
        G(Khat0) = P*m0 - (1/Khat0) * [ J1(α) / (α J1'(α)) ] = 0,
        α = (R0/√Z)*sqrt(P*m0*Khat0 - 1).

    Returns:
        (Khat0, alpha) for the chosen root_index (0 = principal).

    Notes:
    - Domain: Khat0 > 1/(P*m0) to make α real.
    - Skips singularities where J1'(α)=0 by ignoring NaN intervals.
    """
    if min(P, Z, R0, m0) <= 0:
        raise ValueError("P, Z, R0, m0 must be positive.")

    Kmin = (1.0 / (P * m0)) * (1 + 1e-12)  # tiny offset above threshold

    def alpha_of(Khat0: float) -> float:
        return (R0/np.sqrt(Z)) * np.sqrt(max(P*m0*Khat0 - 1.0, 0.0))

    def G(Khat0: float) -> float:
        if Khat0 <= Kmin:
            return np.nan
        a = alpha_of(Khat0)
        j1p = J1p(a)
        if not np.isfinite(a) or a == 0.0 or not np.isfinite(j1p) or j1p == 0.0:
            return np.nan  # singular; skip this point
        return P*m0 - (J1(a) / (a * j1p)) / Khat0

    # Scan to find sign-change brackets between singularities
    K_grid = np.linspace(Kmin*(1+1e-6), Khat0_max, n_samples)
    Gvals  = np.array([G(K) for K in K_grid])

    brackets = []
    i = 0
    while i < len(K_grid) - 1:
        if not np.isfinite(Gvals[i]):
            i += 1; continue
        j = i + 1
        while j < len(K_grid) and not np.isfinite(Gvals[j]):
            j += 1
        if j >= len(K_grid):
            break
        if Gvals[i] * Gvals[j] < 0:
            brackets.append((K_grid[i], K_grid[j]))
        i = j

    if not brackets:
        raise RuntimeError("No sign change found for G(Khat0). Increase Khat0_max or n_samples.")

    if root_index < 0 or root_index >= len(brackets):
        raise RuntimeError(f"root_index={root_index} out of range (found {len(brackets)} bracket(s)).")

    a, b = brackets[root_index]
    sol = optimize.root_scalar(G, bracket=(a, b), method="brentq", maxiter=500)
    if not sol.converged:
        raise RuntimeError(f"root_scalar failed: {sol.flag}")

    Khat0 = float(sol.root)
    alpha = alpha_of(Khat0)
    return Khat0, alpha

# ---------- first-order fields (D-independent hats) ----------
@dataclass
class FirstOrder:
    R0: float; m0: float; Z: float; P: float; gamma: float
    Khat0: float; alpha: float
    s11: Callable[[np.ndarray], np.ndarray]
    s11p: Callable[[np.ndarray], np.ndarray]
    s11pp: Callable[[np.ndarray], np.ndarray]
    m11: Callable[[np.ndarray], np.ndarray]
    m11p: Callable[[np.ndarray], np.ndarray]
    m11pp: Callable[[np.ndarray], np.ndarray]
    U: Callable[[np.ndarray], np.ndarray]
    Up: Callable[[np.ndarray], np.ndarray]
    m11_R0: float
    m11pp_R0: float

def build_first_order(P: float, Z: float, gamma: float,
                      root_index: int = 0) -> FirstOrder:
    R0 = solve_R0(P, gamma)
    m0 = 1.0/(np.pi*R0**2)
    Khat0, alpha = solve_Khat0(P, Z, R0, m0, root_index=root_index)
    den = (P*Khat0*m0 - 1.0)
    if den <= 0: raise RuntimeError("P*K̂0*m0 - 1 must be > 0.")
    j1p_a = J1p(alpha)

    def s11(r):
        return (P*m0*r - (R0/Khat0)*(J1(alpha/R0*r)/(alpha*j1p_a))) / den
    def s11p(r):
        return ((P*m0) - (1/Khat0)*(J1p(alpha/R0*r)/j1p_a)) / den
    def s11pp(r):
        return (-(1/Khat0)*(alpha/R0)*(J1pp(alpha/R0*r)/j1p_a)) / den

    def m11(r):   return Khat0*m0*s11(r) - m0*r
    def m11p(r):  return Khat0*m0*s11p(r) - m0
    def m11pp(r): return Khat0*m0*s11pp(r)

    denomU = J1(alpha)
    if abs(denomU) < 1e-14: raise RuntimeError("J1(α)≈0 causes ill-conditioned U.")
    def U(r):  return J1(alpha*r/R0)/denomU
    def Up(r): return (alpha/R0)*J1p(alpha*r/R0)/denomU

    m11_R0   = float(m11(np.array([R0])))
    m11pp_R0 = float(m11pp(np.array([R0])))

    return FirstOrder(R0, m0, Z, P,gamma, Khat0, alpha,
                      s11, s11p, s11pp, m11, m11p, m11pp, U, Up, m11_R0, m11pp_R0)

# ---------- second-order A/B BVPs (n=0,2) ----------


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

# ============================================================
#  Forcings (callables) that match (149)–(150) exactly
# ============================================================

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

    # L_n acting on analytic fields (vectorized)
    def L_n(n, f, fp, fpp):
        return lambda r: fpp(r) + (1.0/r)*fp(r) - (n*n)/(r*r)*f(r)

    L1s = L_n(1, s, sp, spp)     # Δ(σ̂1) = L1s(r) * cosθ
    '''
    # A-piece pieces (projected to n=0 and n=2)
    E0 = lambda r: 0.5*(mp(r) + m(r)/r)                # from e1·∇m̂1
    E2 = lambda r: 0.5*(mp(r) - m(r)/r)
    G0 = lambda r: 0.5*(mp(r)*sp(r) + (m(r)*s(r))/(r*r))  # ∇m̂1·∇σ̂1
    G2 = lambda r: 0.5*(mp(r)*sp(r) - (m(r)*s(r))/(r*r))
    H0 = lambda r: 0.5*m(r)*L1s(r)                     # m̂1 Δσ̂1 ⇒ same in n=0,2
    H2 = H0

    # Assemble A forcings (move RHS of (149).2 to RHS): q_A = -E + K̂0(G+H)
    qA0 = lambda r: -E0(r) + K0h*(G0(r) + H0(r))
    qA2 = lambda r: -E2(r) + K0h*(G2(r) + H2(r))
    '''
    qA0 = lambda r: -(0.5*(mp(r)+m(r)/r) - Khat0/2*(mp(r)*sp(r)+m(r)*spp(r)+m(r)*sp(r)/r))
    qA2 = lambda r: -(0.5*(mp(r)-m(r)/r) - Khat0/2*(mp(r)*sp(r)+m(r)*spp(r)+m(r)*sp(r)/r-2*m(r)*s(r)/r**2))
    '''
    # B-piece: −(1/2) Δ(m̂1^2). With m̂1^2 = 1/2 m^2 (1 + cos 2θ):
    # projections give qB0 = −¼ L0(m^2), qB2 = −¼ L2(m^2)
    def m2(r):    return m(r)*m(r)
    def m2p(r):   return 2.0*m(r)*mp(r)
    def m2pp(r):  return 2.0*(mp(r)*mp(r) + m(r)*mpp(r))
    L0_m2 = lambda r: m2pp(r) + (1.0/r)*m2p(r)
    L2_m2 = lambda r: m2pp(r) + (1.0/r)*m2p(r) - (4.0/(r*r))*m2(r)

    qB0 = lambda r: -0.25*L0_m2(r)
    qB2 = lambda r: -0.25*L2_m2(r)
    '''
    qB0 = lambda r: -(0.5*(mp(r)*mp(r)+m(r)*mpp(r)+1/r*m(r)*mp(r)))
    qB2 = lambda r: -(0.5*(mp(r)*mp(r)+m(r)*mpp(r)+1/r*m(r)*mp(r))-m(r)*m(r)/r**2)


    return dict(qA0=qA0, qA2=qA2, qB0=qB0, qB2=qB2)


# ============================================================
#  Invert L_n u = g(r) with original BCs (regular at 0, Neumann at R0)
# ============================================================

def _solve_Ln_inverse(
    n, g_func, R0, *,
    eps_factor=1e-3, N_init=200, mesh_power=4.0,
    tol=1e-5, max_nodes=300000,
    # start homotopy strictly > 0 to avoid the singular λ=0 case
    homotopy=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
):
    """
    Solve u'' + (1/r) u' - (n^2/r^2) u = g(r)
    BCs:
      - n = 0: u(eps) = 0  (pins the constant),  u'(R0) = 0
      - n > 0: u(eps) = 0  (regular),            u'(R0) = 0
    No renormalization; eps > 0 only to avoid division by zero.
    Returns (r, u(r), u'(R0), u_func)
    """
    import numpy as np
    from scipy.integrate import solve_bvp

    eps = max(1e-12, eps_factor * R0)

    # clustered initial mesh near eps
    s = np.linspace(0., 1., N_init+1)
    r_mesh = eps + (R0 - eps) * (s**mesh_power)

    def make_ode(lam):
        def ode(r, y):
            u, p = y
            gv = lam * np.asarray(g_func(r), dtype=float)
            up = p
            pp = -(1.0/r)*p + (n**2/(r*r))*u + gv
            return np.vstack([up, pp])
        return ode

    def bc(ya, yb):
        uL, pL = ya
        uR, pR = yb
        # left BC: pin value at eps; for n>0 this is also the regular condition u(0)=0
        left = uL                 # u(eps) = 0
        right = pR                # u'(R0) = 0
        return np.array([left, right], dtype=float)

    # initial guess: zero (satisfies both BCs)
    y0 = np.zeros((2, r_mesh.size))

    sol = None
    for lam in homotopy:
        ode = make_ode(lam)
        if sol is None:
            sol = solve_bvp(ode, bc, r_mesh, y0, tol=tol, max_nodes=max_nodes)
        else:
            y_guess = sol.sol(sol.x)
            sol = solve_bvp(ode, bc, sol.x, y_guess, tol=tol, max_nodes=max_nodes)
        if not sol.success:
            raise RuntimeError(f"L_n inverse (n={n}, λ={lam}) failed: {sol.message}")

    rr = sol.x
    u  = sol.y[0].copy()
    upr_R0 = float(sol.y[1, -1])

    # callable that evaluates u on arbitrary r in [eps, R0]
    def u_of_r(r):
        r = np.atleast_1d(r)
        vals = sol.sol(r)[0]
        return np.asarray(vals, dtype=float)

    return rr, u, upr_R0, u_of_r



# ============================================================
#  Eliminate m, then solve a single σ-BVP
# ============================================================

def _solve_sigma_via_elimination(n, f_func, F, D=1.0, *,
                                 eps_factor=1e-3, N_init=200, mesh_power=4.0,
                                 tol=1e-5, max_nodes=300000):
    """
    Given (2): D L_n m − K0 m0 L_n σ = f(r),
    write m = α σ + u + A r^n  (n≥1) or m = α σ + u + A (n=0),
      where α = K0 m0 / D, and u solves L_n u = f/D with regularity + Neumann BC.
    Regularity ⇒ drop the r^{-n} (or ln r) term.  Neumann at R0 fixes A for n≥1.
    Substitute into (1): Z L_n σ = σ − P m, i.e.
         Z L_n σ − c σ = h(r),   with c = 1 − P α,  h = −P (u + A r^n).
    Solve σ with regular at 0 and Neumann at R0; reconstruct m; return σ(R0).
    """
    R0, Z, P, K0, m0 = F.R0, F.Z, F.P, F.Khat0, F.m0

    # Step 1: u from L_n u = f/D
    g = (lambda r: np.asarray(f_func(r), float)/D)
    r_u, u, upr_R0, u_of_r = _solve_Ln_inverse(
        n, g, R0,
        eps_factor=eps_factor, N_init=N_init, mesh_power=mesh_power,
        tol=tol, max_nodes=max_nodes
    )

    # Step 2: A from Neumann m'(R0)=0 (σ'(R0)=0 will hold for σ-BVP)
    if n == 0:
        A = 0.0
    else:
        A = - upr_R0 / (n * (R0**(n-1)))

    alpha = (K0 * m0) / D
    c = 1.0 - P * alpha
    h_of_r = (lambda r: -P * (u_of_r(r) + (A * (np.asarray(r, float)**n) if n>0 else 0.0)))

    # Step 3: σ-BVP  :  Z(σ'' + σ'/r − n^2 σ / r^2) − c σ = h(r)
    eps = max(1e-12, eps_factor*R0)
    s = np.linspace(0., 1., N_init+1)
    r_mesh = eps + (R0 - eps) * (s**mesh_power)

    def ode_sigma(r, y):
        σ, v = y
        h = np.asarray(h_of_r(r), float)
        vp = (c*σ + h)/Z - (1.0/r)*v + (n**2/(r*r))*σ
        return np.vstack([v, vp])

    def bc_sigma(ya, yb):
        σL, vL = ya
        σR, vR = yb
        left  = vL if n==0 else σL   # regularity at 0
        right = vR                   # Neumann at R0
        return np.array([left, right], float)

    # initial guess that matches left BC
    if n == 0:
        σ0 = np.zeros_like(r_mesh)
    else:
        σ0 = (r_mesh/R0)**n * 0.0
    v0 = np.gradient(σ0, r_mesh, edge_order=2)
    y0 = np.vstack([σ0, v0])

    sol = solve_bvp(ode_sigma, bc_sigma, r_mesh, y0, tol=tol, max_nodes=max_nodes)
    if not sol.success:
        raise RuntimeError(f"σ-BVP (n={n}) failed: {sol.message}")

    r = sol.x
    σ = sol.y[0]

    # Step 4: reconstruct m
    m = alpha*σ + u_of_r(r) + (A*(r**n) if n>0 else 0.0)
    s_at_R0 = float(σ[-1])
    return r, σ, m, s_at_R0


# ============================================================
#  Top-level: compute second order via elimination for A & B
# ============================================================

def compute_second_order_elimination(
    F,
    gamma: float,
    D: Optional[float] = None,
    *,
    modes=(0, 2),
    eps_factor: float = 1e-3,
    N_init: int = 200,
    mesh_power: float = 4.0,
    tol: float = 1e-5,
    max_nodes: int = 300000,
) -> SecondOrderAll:
    """
    Build A/B forcings from F's analytic first-order solution, solve each
    mode n∈modes via the elimination route, and convert σ(R0) → ρ via (149).4.

    Returns:
        SecondOrderAll dataclass with:
          - A, B: SecondOrderPiece (r, m0, m2, s0, s2) on a per-piece common grid
          - rho20A, rho22A, rho20B, rho22B
    """
    if D is None:
        D = getattr(F, "D", 1.0)

    R0 = float(F.R0)
    Qf = make_forcings_A_B_func(F)

    # Solve each requested mode for both pieces
    results_A: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    results_B: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    if 0 in modes:
        r0A, s0A, m0A, sA0 = _solve_sigma_via_elimination(
            0, Qf["qA0"], F, D,
            eps_factor=eps_factor, N_init=N_init, mesh_power=mesh_power,
            tol=tol, max_nodes=max_nodes
        )
        results_A[0] = (r0A, s0A, m0A, sA0)

        r0B, s0B, m0B, sB0 = _solve_sigma_via_elimination(
            0, Qf["qB0"], F, D,
            eps_factor=eps_factor, N_init=N_init, mesh_power=mesh_power,
            tol=tol, max_nodes=max_nodes
        )
        results_B[0] = (r0B, s0B, m0B, sB0)

    if 2 in modes:
        r2A, s2A, m2A, sA2 = _solve_sigma_via_elimination(
            2, Qf["qA2"], F, D,
            eps_factor=eps_factor, N_init=N_init, mesh_power=mesh_power,
            tol=tol, max_nodes=max_nodes
        )
        results_A[2] = (r2A, s2A, m2A, sA2)

        r2B, s2B, m2B, sB2 = _solve_sigma_via_elimination(
            2, Qf["qB2"], F, D,
            eps_factor=eps_factor, N_init=N_init, mesh_power=mesh_power,
            tol=tol, max_nodes=max_nodes
        )
        results_B[2] = (r2B, s2B, m2B, sB2)

    # Map boundary values σ(R0) → ρ via (149).4:
    #   s0 = −(2π + γ/R0^2) ρ20,     s2 = (3γ/R0^2) ρ22
    def _rho_from_s(outdict: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, float]]):
        rho20 = 0.0
        rho22 = 0.0
        if 0 in outdict:
            rho20 = - outdict[0][3] / (2.0*np.pi + gamma / R0**2)
        if 2 in outdict:
            rho22 =   outdict[2][3] * (R0**2) / (3.0 * gamma)
        return rho20, rho22

    rho20A, rho22A = _rho_from_s(results_A)
    rho20B, rho22B = _rho_from_s(results_B)

    # Build per-piece common grids and interpolate both modes to those grids
    def _make_piece(res: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> SecondOrderPiece:
        # Determine a safe common [r_min, R0] and size
        rmins = []
        Ns = []
        for n, (r, sig, m, sR) in res.items():
            rmins.append(float(np.min(r)))
            Ns.append(int(len(r)))
        rmin = max(min(rmins) if rmins else 0.0, 1e-12)
        N_common = max(Ns) if Ns else 2  # fallback tiny grid if empty

        r_common = np.linspace(rmin, R0, N_common)

        def interp(r_src, v_src):
            if r_src is None or v_src is None:
                return np.zeros_like(r_common)
            return np.interp(r_common, np.asarray(r_src), np.asarray(v_src))

        # Mode 0
        if 0 in res:
            r0, s0, m0, _ = res[0]
            s0c = interp(r0, s0)
            m0c = interp(r0, m0)
        else:
            s0c = np.zeros_like(r_common)
            m0c = np.zeros_like(r_common)

        # Mode 2
        if 2 in res:
            r2, s2, m2, _ = res[2]
            s2c = interp(r2, s2)
            m2c = interp(r2, m2)
        else:
            s2c = np.zeros_like(r_common)
            m2c = np.zeros_like(r_common)

        return SecondOrderPiece(r=r_common, m0=m0c, m2=m2c, s0=s0c, s2=s2c)

    A_piece = _make_piece(results_A)
    B_piece = _make_piece(results_B)

    return SecondOrderAll(
        A=A_piece,
        B=B_piece,
        rho20A=rho20A, rho22A=rho22A,
        rho20B=rho20B, rho22B=rho22B)

# ---------- nested quadratures for Ã_i in (38)–(41) ----------

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

# ---------- one-call pipeline ----------
def compute_K2(P: float, Z: float, gamma: float,
               D0: float, Dp: float, Dpp: float,
               *,
               root_index: int = 0,
               N_second_order: int = 1200) -> Dict[str, float]:
    F  = build_first_order(P, Z, gamma, root_index=root_index)
    SO = compute_second_order(F, gamma, N=N_second_order)
    Ai = compute_Ais(F, SO)
    K2 = K2_from_Ai(Ai, D0, Dp, Dpp)
    out = dict(K2=K2, A1=Ai["A1"], A2=Ai["A2"], A3=Ai["A3"], A4=Ai["A4"],
               R0=Ai["meta"]["R0"], m0=Ai["meta"]["m0"], Khat0=Ai["meta"]["Khat0"], alpha=Ai["meta"]["alpha"])
    return out

#Example (linear diffusion): set D0=1, Dp=0, Dpp=0 ⇒ only A4 contributes.

def plot_full_second_order_fields(
    SO,
    F=None,                   # if provided, will use F.R0; else inferred from SO
    *,
    Nr=300,                   # radial samples for plotting
    Ntheta=241,               # angular samples for plotting
    rho_scale=1.0,            # scale factor for polar boundary visualization
    save=False,
    prefix="second_order_full",
    show=True
):
    """
    Plot full σ₂(r,θ), m₂(r,θ) and ρ₂(θ) using results from compute_second_order_elimination.

    SO structure (per compute_second_order_elimination):
      SO['A'][0 or 2] = {'r': r, 'sigma': σ_n(r), 'm': m_n(r), 's_R0': σ_n(R0)}
      SO['B'][0 or 2] = {...}
      SO['rho'] = {'20A','22A','20B','22B'}

    Parameters
    ----------
    F : object with attribute R0 (optional). If not provided, R0 is inferred.
    Nr, Ntheta : grid sizes for the plots
    rho_scale : visualize R(θ) = R0 + rho_scale * ρ₂(θ) in the polar plot
    save : save PNGs in the current directory
    prefix : filename prefix for saved figures
    show : call plt.show() at the end
    """

    # --- helpers
    def _piece_modes(piece):
        return SO.get(piece, {})

    def _get_mode(piece, n, key):
        """Return (r, arr) for piece 'A' or 'B', mode n, field key 'sigma'|'m' if present, else (None,None)."""
        dct = _piece_modes(piece)
        if n in dct and key in dct[n]:
            return np.asarray(dct[n]['r']), np.asarray(dct[n][key])
        return None, None

    # R0: prefer F.R0, else infer from the largest r endpoint we have
    if F is not None and hasattr(F, "R0"):
        R0 = float(F.R0)
    else:
        R0 = 0.0
        for S in ("A","B"):
            for n in _piece_modes(S):
                R0 = max(R0, float(np.max(SO[S][n]["r"])))
        if R0 <= 0:
            raise ValueError("Could not infer R0; pass F with F.R0 or ensure SO has r arrays.")

    # Inner radius for plotting: avoid literal r=0 if the data begins at eps>0
    rmins = []
    for S in ("A","B"):
        for n in _piece_modes(S):
            rmins.append(float(np.min(SO[S][n]["r"])))
    rmin = max(min(rmins) if rmins else 0.0, 1e-12)

    # Plot grids
    r_plot = np.linspace(rmin, R0, Nr)
    theta = np.linspace(0.0, 2.0*np.pi, Ntheta)
    R, TH = np.meshgrid(r_plot, theta, indexing="xy")     # TH shape (Nθ, Nr)
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    C2 = np.cos(2.0*TH)

    # Interpolate modal profiles to r_plot (missing modes → zeros)
    def interp_or_zero(r_src, f_src):
        if r_src is None or f_src is None:
            return np.zeros_like(r_plot)
        return np.interp(r_plot, r_src, f_src)

    # σ modes (sum A and B)
    sigma0 = (
        interp_or_zero(*_get_mode("A", 0, "sigma")) +
        interp_or_zero(*_get_mode("B", 0, "sigma"))
    )
    sigma2 = (
        interp_or_zero(*_get_mode("A", 2, "sigma")) +
        interp_or_zero(*_get_mode("B", 2, "sigma"))
    )

    # m modes (sum A and B)
    m0 = (
        interp_or_zero(*_get_mode("A", 0, "m")) +
        interp_or_zero(*_get_mode("B", 0, "m"))
    )
    m2 = (
        interp_or_zero(*_get_mode("A", 2, "m")) +
        interp_or_zero(*_get_mode("B", 2, "m"))
    )

    # Expand to full fields on (r,θ) grid: f(r,θ) = f0(r) + f2(r)*cos 2θ
    SIGMA = (sigma0[np.newaxis, :] + sigma2[np.newaxis, :] * C2)
    Mfull = (m0[np.newaxis, :]     + m2[np.newaxis, :]     * C2)

    # --- ρ₂(θ) from SO['rho']
    rho = SO.get("rho", {})
    rho20 = float(rho.get("20A", 0.0)) + float(rho.get("20B", 0.0))
    rho22 = float(rho.get("22A", 0.0)) + float(rho.get("22B", 0.0))
    rho2_theta = rho20 + rho22 * np.cos(2.0*theta)

    # ---- Plot σ₂(r,θ)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    pcm1 = ax1.pcolormesh(X, Y, SIGMA, shading="auto")
    cb1 = fig1.colorbar(pcm1, ax=ax1)
    cb1.set_label(r"$\sigma_2(r,\theta)$")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(r"Second-order $\sigma_2(r,\theta)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    if save:
        fig1.savefig(f"{prefix}_sigma2_field.png", dpi=150, bbox_inches="tight")

    # ---- Plot m₂(r,θ)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    pcm2 = ax2.pcolormesh(X, Y, Mfull, shading="auto")
    cb2 = fig2.colorbar(pcm2, ax=ax2)
    cb2.set_label(r"$m_2(r,\theta)$")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title(r"Second-order $m_2(r,\theta)$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    if save:
        fig2.savefig(f"{prefix}_m2_field.png", dpi=150, bbox_inches="tight")

    # ---- Plot ρ₂(θ): cartesian (θ vs ρ₂)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    ax3.plot(theta, rho2_theta)
    ax3.set_xlabel(r"$\theta$")
    ax3.set_ylabel(r"$\rho_2(\theta)$")
    ax3.set_title(r"Boundary mode $\rho_2(\theta)$")
    ax3.set_xlim([0, 2*np.pi])
    if save:
        fig3.savefig(f"{prefix}_rho2_theta.png", dpi=150, bbox_inches="tight")

    # ---- Polar contour of the perturbed boundary: R(θ) = R0 + rho_scale * ρ₂(θ)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1,1,1, projection="polar")
    Rb = R0 + rho_scale * rho2_theta
    ax4.plot(theta, Rb)
    ax4.set_title(r"Boundary contour: $R(\theta)=R_0 + \mathrm{scale}\cdot\rho_2(\theta)$")
    # Optional: show reference circle R0
    ax4.plot(theta, R0*np.ones_like(theta), linestyle="--")
    if save:
        fig4.savefig(f"{prefix}_rho2_polar.png", dpi=150, bbox_inches="tight")

    if show:
        plt.show()

import numpy as np
from typing import Optional, Tuple

def vdW_D_at_m0(e_a: float,
                *,
                m0: Optional[float] = None,
                m_inf: float = 10.0,
                F: Optional[object] = None) -> Tuple[float, float, float]:
    """
    Van der Waals diffusion (paper eq. (20)):
        D(m) = m_inf^2 / (m_inf - m)^2 - e_a * m

    Returns (D(m0), D'(m0), D''(m0)) given e_a and either m0 directly or F.m0.

    Args
    ----
    e_a   : cooperative binding parameter e_A
    m0    : steady-state myosin density (if None, will try F.m0)
    m_inf : saturation concentration m_∞ (default 10.0)
    F     : optional first-order object with attribute m0

    Raises
    ------
    ValueError if m0 is not provided and F.m0 is unavailable, or if m_inf == m0.

    Notes
    -----
    D'(m)  =  2 m_inf^2 / (m_inf - m)^3 - e_a
    D''(m) =  6 m_inf^2 / (m_inf - m)^4
    """
    if m0 is None:
        if F is not None and hasattr(F, "m0"):
            m0 = float(F.m0)
        else:
            raise ValueError("Please provide m0 or pass F with attribute m0.")
    M = float(m_inf)
    m0 = float(m0)

    denom = M - m0
    if denom == 0.0:
        raise ValueError("m_inf equals m0 → D and its derivatives blow up; choose m_inf != m0.")

    D0   = (M**2) / (denom**2) - e_a * m0
    Dp   = (2.0 * M**2) / (denom**3) - e_a
    Dpp  = (6.0 * M**2) / (denom**4)
    return D0, Dp, Dpp

def K2_vs_eA(Ai, F, eA_vals, m_inf=10.0, pole_tol=1e-10):
    """
    Vectorized K2(eA) for the van-der-Waals diffusion:
      D(m)   = m_inf^2 / (m_inf - m)^2 - eA * m
      D'(m)  = 2 m_inf^2 / (m_inf - m)^3 - eA
      D''(m) = 6 m_inf^2 / (m_inf - m)^4
    evaluated at m = m0 (from F.m0).

    Ai: dict with A1..A4 from compute_Ais(F, SO)
    F : first-order container with attribute m0
    eA_vals: scalar or 1D array of eA values
    m_inf: saturation parameter m_∞ (default 10.0, same as paper’s figures)
    pole_tol: near-zero threshold for D0 to avoid blowups

    Returns: numpy array K2(eA) (NaN where D0≈0).
    """
    m0 = float(F.m0)
    M  = float(m_inf)
    den = M - m0
    if den == 0.0:
        raise ValueError("m_inf equals m0 → diffusion singular; choose m_inf != m0.")

    # Pieces that don't depend on eA
    Dpp = (6.0 * M**2) / (den**4)
    Dp0 = (2.0 * M**2) / (den**3)      # D'(m0) at eA = 0
    D00 = (M**2) / (den**2)            # D(m0)  at eA = 0

    eA_vals = np.atleast_1d(eA_vals).astype(float)
    Dp = Dp0 - eA_vals                 # linear in eA
    D0 = D00 - eA_vals * m0            # linear in eA

    K2 = np.full_like(eA_vals, np.nan)
    good = np.abs(D0) > pole_tol       # avoid pole where D0 -> 0

    A1, A2, A3, A4 = Ai["A1"], Ai["A2"], Ai["A3"], Ai["A4"]
    D0g, Dpg = D0[good], Dp[good]
    K2[good] = (A1*Dpp/(D0g**4)
                + A2*(Dpg**2)/(D0g**5)
                + A3*Dpg/(D0g**4)
                + A4/(D0g**3))
    return K2


def functional(first_order_solution,second_order_solution):
    # input: have 201x1 tuples ~ discretized function over [0,R0]
    # output: 201x201 matrices ~ functions over [0,R0] x [0, 2pi]
    
    def mult_r(matrix, r_vector):
        """
        Performs row-wise multiplication of a matrix by a radius-dependent vector.
    
        Each row `i` of the matrix is multiplied by the scalar value `r_vector[i]`.
    
        Args:
            matrix (np.ndarray): A 2D NumPy array.
            r_vector (np.ndarray): A 1D NumPy array representing the scaling
                                   factors for each row.
    
        Returns:
            np.ndarray: The resulting 2D array after row-wise multiplication.
            
        Raises:
            ValueError: If the number of rows in the matrix does not match the
                        length of the r_vector.
            TypeError: If inputs are not NumPy arrays or have incorrect dimensions.
        """
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            raise TypeError("Input 'matrix' must be a 2D NumPy array.")
        if not isinstance(r_vector, np.ndarray) or r_vector.ndim != 1:
            raise TypeError("Input 'r_vector' must be a 1D NumPy array.")
            
        if matrix.shape[0] != len(r_vector):
            raise ValueError(f"Shape mismatch: Number of matrix rows ({matrix.shape[0]}) "
                             f"must equal the length of r_vector ({len(r_vector)}).")
        
        # Reshape the 1D r_vector to a column vector (n, 1) to enable
        # broadcasting across the matrix rows.
        return matrix * r_vector[:, np.newaxis]

    def mult_theta(matrix, theta_vector):
        """
        Performs column-wise multiplication of a matrix by a theta-dependent vector.
    
        Each column `j` of the matrix is multiplied by the scalar value `theta_vector[j]`.
    
        Args:
            matrix (np.ndarray): A 2D NumPy array.
            theta_vector (np.ndarray): A 1D NumPy array representing the scaling
                                       factors for each column.
    
        Returns:
            np.ndarray: The resulting 2D array after column-wise multiplication.
    
        Raises:
            ValueError: If the number of columns in the matrix does not match the
                        length of the theta_vector.
            TypeError: If inputs are not NumPy arrays or have incorrect dimensions.
        """
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            raise TypeError("Input 'matrix' must be a 2D NumPy array.")
        if not isinstance(theta_vector, np.ndarray) or theta_vector.ndim != 1:
            raise TypeError("Input 'theta_vector' must be a 1D NumPy array.")
    
        if matrix.shape[1] != len(theta_vector):
            raise ValueError(f"Shape mismatch: Number of matrix columns ({matrix.shape[1]}) "
                             f"must equal the length of theta_vector ({len(theta_vector)}).")
        
        # A 1D vector naturally broadcasts across the matrix columns.
        return matrix * theta_vector
    
    def gen_square_mat_with_tup_cols(tup):
        n = len(tup) # 201
        tuple_list = [tup] * n
        
        # 2. Stack them as columns
        matrix = np.column_stack(tuple_list)
            
        return matrix
    
    # now to generate a proper version with multiplication r_i = r_i * tup_i
    def scale_rows_by_v_element_wise(mat, cVec):
        return mat
    
    def scale_matrix_by_a(mat, a):
        a = np.array(a)
        return mat * a[:, np.newaxis]

    # generate domains (r = array(201,1), theta = array(1,201))
    r_domain = second_order_solution.A.r
    # r_domain = r_domain[:, np.newaxis] # assert: 201x1
    theta_domain = np.linspace(0, 2*np.pi, len(second_order_solution.A.r))
    
    # generate underlying functions from parts
    cos_2Theta = np.cos(2 * theta_domain)
    
    # For A
    a_sols = second_order_solution.A
    a_m0_mat, a_m2_mat = [gen_square_mat_with_tup_cols(tup) for tup in [a_sols.m0, a_sols.m2]]
    a_s0_mat, a_s2_mat = [gen_square_mat_with_tup_cols(tup) for tup in [a_sols.s0, a_sols.s2]]
    a_rho0, a_rho2 = second_order_solution.rho20A, second_order_solution.rho22A
    
    # final funcs
    m2A = a_m0_mat + mult_theta(a_m2_mat, cos_2Theta)
    s2A = a_s0_mat + mult_theta(a_s2_mat, cos_2Theta)
    rho2A = (np.ones(len(cos_2Theta)) *  a_rho0) + (cos_2Theta * a_rho2)
    
    # For B
    b_sols = second_order_solution.B
    b_m0_mat, b_m2_mat = [gen_square_mat_with_tup_cols(tup) for tup in [b_sols.m0, b_sols.m2]]
    b_s0_mat, b_s2_mat = [gen_square_mat_with_tup_cols(tup) for tup in [b_sols.s0, b_sols.s2]]
    b_rho0, b_rho2 = second_order_solution.rho20B, second_order_solution.rho22B
    
    # final funcs
    m2B = b_m0_mat + mult_theta(b_m2_mat, cos_2Theta)
    s2B = b_s0_mat + mult_theta(b_s2_mat, cos_2Theta)
    rho2B = (np.ones(len(cos_2Theta)) *  b_rho0) + (cos_2Theta * b_rho2)
    
    # finally: 
    # assert m2A,m0A,s2B,s0B are 201x201 and A[i,j] = f(m, sig) index-wise
    # assert rho2A, rho2B are 1x201 and A[j] = f(sig) index-wise
    
    # ### Obtain Residual for system(149), (150)
    # constants(verify internal consistency)
    P,Z,gamma = 0.0159, 0.016, 0.005
    R0 = first_order_solution.R0
    m0 = 1.0/(np.pi*R0**2)
    
    K0_hat = first_order_solution.Khat0
    
    # lets fetch the boundary condition residual for A and B
    
    # fetch r-partials at the boundary
    dr = r_domain[1] - r_domain[0] # Assuming r_domain is your radial grid
    dtheta = 2* np.pi / len(theta_domain)
    
    m2A_r_boundary = (m2A[-1, :] - m2A[-2, :]) / dr
    s2A_r_boundary = (s2A[-1, :] - s2A[-2, :]) / dr

    m2B_r_boundary = (m2B[-1, :] - m2B[-2, :]) / dr
    s2B_r_boundary = (s2B[-1, :] - s2B[-2, :]) / dr
    
    # --- generate Boundary Residuals ---
    
    # for A
    term1 = m2A_r_boundary**2
    term2 = (s2A[-1, :] + (2 * np.pi * a_rho0) + (gamma/(R0**2)) * (a_rho0 - (3 * a_rho2 * cos_2Theta)))**2
    term3 = s2A_r_boundary**2
    
    bcA_residual = np.sum(term1 + term2 + term3)
    
    # for B
    term1 = m2B_r_boundary**2
    term2 = (s2B[-1, :] + (2 * np.pi * b_rho0) + (gamma/(R0**2)) * (b_rho0 - (3 * b_rho2 * cos_2Theta)))**2
    term3 = s2B_r_boundary**2
    
    bcB_residual = np.sum(term1 + term2 + term3)
        
    
    # Generate neccissary derivitives for computations (2)
    
    # first order
    m2A_r, m2A_theta = np.gradient(m2A, dr, dtheta)
    s2A_r, s2A_theta = np.gradient(s2A, dr, dtheta)
    m2B_r, m2B_theta = np.gradient(m2B, dr, dtheta)
    s2B_r, s2B_theta = np.gradient(s2B, dr, dtheta)
    
    # second order
    m2A_rr, _ = np.gradient(m2A_r, dr, dtheta)
    _, m2A_thetatheta = np.gradient(m2A_theta, dr, dtheta)
    m2B_rr, _ = np.gradient(m2B_r, dr, dtheta)
    _, m2B_thetatheta = np.gradient(m2B_theta, dr, dtheta)
    
    s2A_rr, _ = np.gradient(s2A_r)
    _, s2A_thetatheta = np.gradient(s2A_theta)
    s2B_rr, _ = np.gradient(s2B_r)
    _, s2B_thetatheta = np.gradient(s2B_theta)
    
    # scaling terms for theta partials(kinda wrong, but is the second index of grad)
    m2A_theta = mult_r(m2A_theta, -1/r_domain)
    s2A_theta = mult_r(s2A_theta, -1/r_domain)
    m2B_theta = mult_r(m2A_theta, -1/r_domain)
    s2B_theta = mult_r(s2A_theta, -1/r_domain)
    
    
    # laplacians
    m2A_lap = m2A_rr + mult_r(m2A_r, 1/r_domain) + mult_r(m2A_thetatheta, 1/np.square(r_domain))
    s2A_lap = s2A_rr + mult_r(s2A_r, 1/r_domain) + mult_r(s2A_thetatheta, 1/np.square(r_domain))
    m2B_lap = m2B_rr + mult_r(m2B_r, 1/r_domain) + mult_r(m2B_thetatheta, 1/np.square(r_domain))
    s2B_lap = s2B_rr + mult_r(s2B_r, 1/r_domain) + mult_r(s2B_thetatheta, 1/np.square(r_domain))
    
    
    # --- System A Residual ---
    sysA_residual = 0
    
    # calculate term1
    term1A = -1 * Z * s2A_lap + s2A - P * m2A
    
    # calculate term2
    
    # 1. grad_m1 dot grad_m2 = some vector calc
    # unpack needed functions from (1) solutions
    s11, s11p, s11pp = first_order_solution.s11, first_order_solution.s11p, first_order_solution.s11pp
    m11, m11p, m11pp = first_order_solution.m11, first_order_solution.m11p, first_order_solution.m11pp
    # evaluate on r_domain
    s11, s11p, s11pp = s11(r_domain), s11p(r_domain), s11pp(r_domain)
    m11, m11p, m11pp = m11(r_domain), m11p(r_domain), m11pp(r_domain)
    # stack into matrix
    s11_mat, s11p_mat = gen_square_mat_with_tup_cols(s11), gen_square_mat_with_tup_cols(s11p)
    s11pp_mat = gen_square_mat_with_tup_cols(s11pp)
    m11_mat, m11p_mat = gen_square_mat_with_tup_cols(m11), gen_square_mat_with_tup_cols(m11p)
    m11pp_mat = gen_square_mat_with_tup_cols(m11pp)
    
    # apply trig functions
    cos_cVec, sin_cVec = np.cos(theta_domain), np.sin(theta_domain)
    # functions
    m1 = mult_theta(m11_mat, cos_cVec)
    s1 = mult_theta(s11_mat, cos_cVec)
    
    # first order
    m1_r = mult_theta(m11p_mat,cos_cVec)
    s1_r = mult_theta(s11p_mat, cos_cVec)
    m1_theta = mult_theta(m11_mat, -1*sin_cVec)
    s1_theta = mult_theta(s11_mat, -1*sin_cVec)
    
    # second order
    m1_rr = mult_theta(m11pp_mat, cos_cVec)
    m1_rtheta = mult_theta(m11p_mat, -1*sin_cVec)
    m1_thetatheta = mult_theta(m11_mat, -1*cos_cVec)
    
    s1_rr = mult_theta(s11pp_mat, cos_cVec)
    s1_rtheta = mult_theta(s11p_mat, -1*sin_cVec)
    s1_thetatheta = mult_theta(s11_mat, -1*cos_cVec)
    
    # laplacian
    m1_lap = m1_rr + mult_r(m1_r, 1/r_domain) + mult_r(m1_thetatheta, 1/np.square(r_domain))
    s1_lap = s1_rr + mult_r(s1_r, 1/r_domain) + mult_r(s1_thetatheta, 1/np.square(r_domain))
    
    
    # 1. computing dot product in polar 
    m1_s1_dot = (m1_r * s1_r) + mult_r(m1_theta * s1_theta, 1/np.square(r_domain))
    
    # 2. e_1 * grad(m1) = m1_r * cos + m1_theta * sin
    e_1_m1A = mult_theta(m1_r,cos_cVec) - mult_r(mult_theta(m1_theta, sin_cVec), 1/r_domain)
    
    
    
    term2A = m2A_lap - (K0_hat*m0*s2A_lap) + e_1_m1A - (K0_hat*m1_s1_dot) - (K0_hat*m1*s1_lap)
    
    # combining terms and ssd
    sysA_residual = np.sum(term1A**2 + term2A**2)
    
    
    # --- system B residual --- 
    
    term1 = (-1 * Z * s2B_lap) + s2B - (P * m2B)
    # Term 2 from your residual calculation
    grad_m1_sq = (m1_r**2) + mult_r(m1_theta**2, 1/np.square(r_domain))

    # Then, calculate the Laplacian of m1 squared
    lap_m1_sq = 2 * m1 * m1_lap + 2 * grad_m1_sq
    
    term2 = m2B_lap - (K0_hat * m0 * s2B_lap) + (0.5 * lap_m1_sq) 
    
    # Square each term element-wise and sum them all to get the final scalar residual
    sysB_residual = np.sum(term1**2 + term2**2)
    
        
    return {
        "total": sum([sysA_residual,sysB_residual,bcA_residual,bcB_residual]),
        "sysA":sysA_residual,
        "sysB":sysB_residual,
        "bcA":bcA_residual,
        "bcB":bcB_residual,
        "sysA term1": sum(term1A**2)[0],
        "sysA term2": sum(term2A**2)[0]
        }


def functional_1st_order(first_order_solution, eps = 1e-10, gran = 1000):
    # system constants
    P, Z, gamma = first_order_solution.P, first_order_solution.Z, first_order_solution.gamma
    R0 = first_order_solution.R0
    m0 = 1.0/(np.pi*R0**2)
    
    # First order in r,theta
    # FIX: Start from r=0 if solution is defined there, or handle r->0 limit
    r = np.linspace(eps, R0, gran)  # Changed from 0.1 to 0
    theta = np.linspace(0, 2*np.pi, gran)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    dr, dtheta = r[1] - r[0], theta[1] - theta[0]
    
    cos_cVec, sin_cVec = np.cos(Theta), np.sin(Theta)
    
    K0_hat = first_order_solution.Khat0
    
    # unpack needed functions from (1) solutions
    s11, s11p, s11pp = first_order_solution.s11, first_order_solution.s11p, first_order_solution.s11pp
    m11, m11p, m11pp = first_order_solution.m11, first_order_solution.m11p, first_order_solution.m11pp
    
    # Handle r=0 specially to avoid evaluation issues
    R_safe = R.copy()
    R_safe[0, :] = 1e-10  # Small value to avoid division by zero
    
    # functions
    m1 = m11(R) * np.cos(Theta)
    s1 = s11(R) * np.cos(Theta)
    
    # first order
    m1_r = m11p(R) * np.cos(Theta)
    s1_r = s11p(R) * np.cos(Theta)
    m1_theta = m11(R) * -1*np.sin(Theta)
    s1_theta = s11(R) * -1*np.sin(Theta)
    
    # FIX: Correct Laplacian calculation
    # Define derivative operators
    d2_dr2 = FinDiff(0, dr, 2)
    d_dr = FinDiff(0, dr, 1)
    d2_dtheta2 = FinDiff(1, dtheta, 2, periodic=True)
    
    # Apply derivatives separately, then combine with position-dependent coefficients
    m1_rr = d2_dr2(m1)
    m1_r_computed = d_dr(m1)
    m1_thetatheta = d2_dtheta2(m1)
    
    s1_rr = d2_dr2(s1)
    s1_r_computed = d_dr(s1)
    s1_thetatheta = d2_dtheta2(s1)
    
    # Laplacian in polar coordinates: ∂²f/∂r² + (1/r)∂f/∂r + (1/r²)∂²f/∂θ²
    m1_lap = m1_rr + (1/R) * m1_r_computed + (1/R**2) * m1_thetatheta
    s1_lap = s1_rr + (1/R) * s1_r_computed + (1/R**2) * s1_thetatheta
    
    # Fix at r=0: Use L'Hôpital's rule for smooth functions
    # lim_{r->0} Lap(f) = 2*f_rr(0) for axisymmetric part
    if r[0] == 0:
        m1_lap[0, :] = 2 * m1_rr[0, :]
        s1_lap[0, :] = 2 * s1_rr[0, :]
    
    # --- Boundary Residual ---
    # Compute derivatives at boundary using FinDiff
    m1_r_boundary = d_dr(m1)[-1, :]
    s1_r_boundary = d_dr(s1)[-1, :]
    
    # Check what the actual boundary condition should be for s1
    # From your system, it looks like s1_r should be related to K0
    K0_cos = 1/K0_hat * np.cos(theta)
    
    # FIX: Normalize by number of boundary points
    boundary_res = np.sum(m1_r_boundary**2 + s1[-1,:]**2 + (s1_r_boundary - K0_cos)**2) / len(theta)
    print(f"Boundary residual: {boundary_res}")
    
    # --- System Residual ---
    # PDEs from system (149) or similar:
    # -Z*Δσ₁ + σ₁ - P*m₁ = 0
    # Δm₁ - K₀*m₀*Δσ₁ = 0
    
    pde1_residual = -Z*s1_lap + s1 - P*m1
    pde2_residual = m1_lap - K0_hat*m0*s1_lap
    
    # FIX: Normalize by total number of interior points
    # Exclude boundary if needed
    interior_points = gran * gran  # or (gran-1) * gran if excluding boundary
    
    # Using L2 norm (sum of squares)
    sys_res_L2 = (np.sum(pde1_residual**2) + np.sum(pde2_residual**2)) / interior_points
    print(f"System residual (L2): {sys_res_L2}")
    
    # Using L1 norm (sum of absolute values) - what you had
    sys_res_L1 = (np.sum(np.abs(pde1_residual)) + np.sum(np.abs(pde2_residual))) / interior_points
    print(f"System residual (L1): {sys_res_L1}")
    
    # Debugging: check individual equation residuals
    print(f"PDE1 max residual: {np.max(np.abs(pde1_residual))}")
    print(f"PDE2 max residual: {np.max(np.abs(pde2_residual))}")
    
    # Check where residuals are largest
    print(f"PDE2 max location: {np.unravel_index(np.argmax(np.abs(pde2_residual)), pde2_residual.shape)}")
    print(f"R0: {R0}")
    print(max(r))
    
    return {
        'boundary_residual': boundary_res,
        'system_residual_L2': sys_res_L2,
        'system_residual_L1': sys_res_L1,
        'pde1_residual': pde1_residual,
        'pde2_residual': pde2_residual
    }


def _num_deriv(f: np.ndarray, x: np.ndarray):
    """1st and 2nd derivatives with respect to nonnegative x via np.gradient (2nd order)."""
    fp  = np.gradient(f, x, edge_order=2)
    fpp = np.gradient(fp, x, edge_order=2)
    return fp, fpp

def check_first_order_pde_discrete_from_F(
    F,
    *,
    n: int = 1,                 # harmonic (1 for first order)
    N: int = 2000,              # radial samples (uniform)
    eps_factor: float = 1e-6,   # start >0 to avoid literal 1/r at r=0
    weights: Dict[str, float] = None,
    D0: Optional[float] = None  # only needed if F has K0 but not Khat0
) -> Dict[str, Any]:
    """
    Check eq. (143) in modal form using *discrete* derivatives:
        Z L_n σ1  - (σ1 - P m1) = 0
        L_n m1    - (Khat0 m0) L_n σ1 = 0
    with  L_n f = f'' + (1/r) f' - (n^2/r^2) f.

    BCs checked (discretely):
        σ1(0)=0, m1(0)=0  (regularity),
        σ1'(R0)=0, m1'(R0)=0  (Neumann)   — derivatives at R0 from one-sided finite diff.

    Returns a dict with scalar 'loss', component norms, BC mismatches, and grid info.
    """
    if weights is None:
        weights = dict(bc_center=1.0, bc_outer=1.0)

    R0 = float(F.R0); Z = float(F.Z); P = float(F.P); m0 = float(F.m0)
    # resolve Khat0
    if hasattr(F, "Khat0"):
        Khat0 = float(F.Khat0)
    elif hasattr(F, "K0"):
        D0_eff = D0 if D0 is not None else (float(F.D0) if hasattr(F, "D0") else None)
        if D0_eff is None:
            raise ValueError("Need D0 to compute Khat0 = K0/D0 (F lacks Khat0 and D0).")
        Khat0 = float(F.K0) / D0_eff
    else:
        raise ValueError("F must have either Khat0 or K0 (with D0).")

    # --- uniform radial grid (avoid r=0 to keep 1/r terms finite) ---
    eps = max(1e-12, eps_factor*R0)
    r = np.linspace(eps, R0, int(N))

    # --- sample fields from F (arrays only) ---
    s = np.asarray(F.s11(r), float)
    m = np.asarray(F.m11(r), float)

    # --- numerical derivatives (arrays only) ---
    sp,  spp  = _num_deriv(s, r)
    mp,  mpp  = _num_deriv(m, r)

    # --- modal Laplacians (original PDE in polar for cosθ ⇒ n=1) ---
    Ln_s = spp + (1.0/r)*sp - (n**2)/(r**2)*s
    Ln_m = mpp + (1.0/r)*mp - (n**2)/(r**2)*m

    # --- PDE residuals of (143) ---
    res_sigma = Z*Ln_s - (s - P*m)              # should be 0
    res_m     = Ln_m   - (Khat0*m0)*Ln_s        # should be 0

    # area-weighted L2 norms: ( ∫ r |res|^2 dr )^{1/2}
    def l2r(x): return np.sqrt(np.trapz(r*(x**2), r))
    res_sigma_L2 = l2r(res_sigma)
    res_m_L2     = l2r(res_m)
    pde_L2_total = res_sigma_L2 + res_m_L2

    # --- BC mismatches (discrete) ---
    # Center (regularity): fields vanish ~ r^1 ⇒ measure |value at first point|
    bc_center_s = abs(s[0])
    bc_center_m = abs(m[0])

    # Outer Neumann: use one-sided derivative at R0 from our discrete gradient
    bc_outer_s = abs(Khat0*sp[-1]-1)
    bc_outer_m = abs(mp[-1])

    bc_center = bc_center_s + bc_center_m
    bc_outer  = bc_outer_s  + bc_outer_m

    # total scalar loss
    loss = pde_L2_total + weights.get("bc_center",1.0)*bc_center + weights.get("bc_outer",1.0)*bc_outer

    return dict(
        loss=loss,
        components=dict(
            res_sigma_L2=res_sigma_L2,
            res_m_L2=res_m_L2,
            pde_L2_total=pde_L2_total,
            bc_center_s=bc_center_s, bc_center_m=bc_center_m,
            bc_outer_s=bc_outer_s,   bc_outer_m=bc_outer_m,
            bc_center_total=bc_center, bc_outer_total=bc_outer
        ),
        params=dict(R0=R0, Z=Z, P=P, m0=m0, Khat0=Khat0, n=n),
        grid=dict(r_min=float(r[0]), r_max=float(r[-1]), N=int(N))
    )

def _num_deriv(f: np.ndarray, x: np.ndarray):
    """Return (f', f'') via second-order finite differences on x."""
    fp  = np.gradient(f, x, edge_order=2)
    fpp = np.gradient(fp, x, edge_order=2)
    return fp, fpp

def _Ln(n: int, f: np.ndarray, r: np.ndarray, fp: Optional[np.ndarray]=None, fpp: Optional[np.ndarray]=None):
    """Modal radial operator L_n f = f'' + (1/r) f' - (n^2/r^2) f, using arrays."""
    if fp is None or fpp is None:
        fp, fpp = _num_deriv(f, r)
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
    gamma: float,
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

    R0 = float(F.R0); Z = float(F.Z); P = float(F.P); m0 = float(F.m0)
    Khat0 = float(getattr(F, "Khat0", None)) if hasattr(F, "Khat0") else None
    if Khat0 is None:
        raise ValueError("F must provide Khat0 for the second-order PDE check.")

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

if __name__ == "__main__":
    #P, Z, gamma = 0.1, 0.2, 0.005
    #P,Z,gamma = 0.17, 0.27, 7.3
    #P,Z, gamma = 0.1, 1.25, 0.0886 #-- interesting
    P,Z,gamma = 0.015915, 0.159155, 0.0025
    #D0, Dp, Dpp = 1.0, 0.0, 0.0

    F = build_first_order(P=P, Z=Z, gamma=gamma)
    #report = check_first_order_compatibility_from_F(F, D0=None, n=1, N=2000)
    #print("loss =", report["loss"])
    #print(report["components"]) 
    #print(first_ord.s11p)
    #print(F.Khat0*F.s11p(F.R0)-1)
    #x_arr = np.linspace(-1,1,100)
    #y_arr = [F.s11(s) for s in x_arr]

    result = check_first_order_pde_discrete_from_F(F)
    #result=functional_1st_order(F)
    print(result)
    #plt.plot(x_arr, y_arr)
    #plt.show()
    SO = compute_second_order_elimination(F, gamma, D=1.0,   # set D if (2) has it; else leave 1.0
                                      modes=(0,2),
                                      eps_factor=1e-3, N_init=200, mesh_power=4.0,
                                      tol=1e-5, max_nodes=300000)

    # Access results
    #rho20A, rho22A = SO['rho']['20A'], SO['rho']['22A']
    #rho20B, rho22B = SO['rho']['20B'], SO['rho']['22B']
    #print(rho20A)
    #plot_full_second_order_fields(SO, F=F, Nr=400, Ntheta=361, rho_scale=1.0, save=False, show=True)

   #F  = build_first_order(P, Z, gamma)
    SO = compute_second_order_elimination(F, gamma)
    Ai = compute_Ais(F, SO)

    # F: first-order object (with Khat0), SO: your SecondOrderAll from compute_second_order_elimination
    report2 = check_second_order_pde_discrete(F, SO, gamma, modes=(0,2))
    print("second-order total loss =", report2["loss"])
    for S in ("A","B"):
        for n in (0,2):
            if n in report2["details"][S]:
                d = report2["details"][S][n]
                print(f"{S}, n={n}: pde_L2={d['pde_L2_total']:.3e}, "
                    f"bc_center={d['bc_center_s']+d['bc_center_m']:.3e}, "
                    f"bc_outer={d['bc_outer_s']+d['bc_outer_m']:.3e}, "
                    f"shape={d['bc_shape']:.3e}")

    #eport = functional(F, SO)

    #print(report)
    #print(Ai)
    #print(K2_from_Ai(Ai, 1,0,0))
    #D0, Dp, Dpp = vdW_D_at_m0(e_a=0, F=F, m_inf=10.0)
    #print(K2_from_Ai(Ai, D0,Dp,Dpp))

    #eA_grid = np.linspace(0.0, 0.7, 400)
    #K2_grid = K2_vs_eA(Ai, F, eA_grid, m_inf=10.0)

    #plt.figure()
    #plt.plot(eA_grid, K2_grid, lw=2)
    #plt.xlabel(r"$e_A$")
    #plt.ylabel(r"$K_2(e_A)$")
    #plt.title(r"$K_2$ vs $e_A$ (Ai fixed)")
   
    #print(res)
    pass
