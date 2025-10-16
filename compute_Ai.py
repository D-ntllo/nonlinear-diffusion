import numpy as np
from typing import Iterable, Tuple, List
import matplotlib.pyplot as plt
from itertools import product

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
    print("A0", A0) 
    # these are still needed for Ã3/Ã4 boundary pieces
    IU_r2 = np.trapz( (r**2) * U(r), r )

    fis = _compute_fi_arrays(F, SO)

    # (38)–(39)
    A1t = _nested_quads(R0, U, Up, fis["f1"], r, P)
    print("A1t", A1t)
    A2t = _nested_quads(R0, U, Up, fis["f2"], r, P)
    print("A2t", A2t)
    # (40) + boundary terms
    A3t  = _nested_quads(R0, U, Up, fis["f3"], r, P)
    print("A3t", A3t)
    A3t += P * ( -2.0*SO.rho22B*F.m11_R0
                 + (F.Khat0*F.m0 - 1.0)*(R0**2)*(2.0*SO.rho20B + SO.rho22B)*F.m11pp_R0
               ) * ( IU_r2 / (2.0*R0**2) )
    print("A3t1", A3t)
    A3t += ( F.alpha * J1p(F.alpha) / (F.Khat0 * J1(F.alpha)) ) * ( -2.0*SO.rho20B + SO.rho22B ) * F.Z
    print("A3t2", A3t)
    A3t += (2.0*SO.rho20B + SO.rho22B) * F.m11pp_R0
    print("A3t3", A3t)

    # (41) + boundary terms
    A4t  = _nested_quads(R0, U, Up, fis["f4"], r, P)
    print("A4t", A4t)
    A4t += P * ( -2.0*SO.rho22A*F.m11_R0
                 + (F.Khat0*F.m0 - 1.0)*(R0**2)*(2.0*SO.rho20A + SO.rho22A)*F.m11pp_R0
               ) * ( IU_r2 / (2.0*R0**2) )
    print("A4t1", A4t)
    A4t += ( F.alpha * J1p(F.alpha) / (2.0*F.Khat0 * J1(F.alpha)) ) * ( -2.0*SO.rho20A + SO.rho22A ) * F.Z
    print("A4t2", A4t)
    A4t += (2.0*SO.rho20A + SO.rho22A) * F.m11pp_R0
    print("A4t3", A4t)

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


# -- VdW model


def D_vdW(e_a: float = 0.0, m_inf: float = 10.0) -> Tuple[Callable, Callable, Callable]:
    """Return a callable D(m) = m_inf^2/(m_inf - m)^2 - e_a*m and its derivatives"""
    def D(m):
        return (m_inf**2)/((m_inf - m)**2) - e_a*m
    
    def Dp(m):
        return (2*m_inf**2)/((m_inf-m)**3) - e_a
    
    def Dpp(m):
        return (6*m_inf**2)/((m_inf-m)**4)
    return D, Dp, Dpp



def K2_of_eA(F, SO, eA_values: Iterable[float], m_inf: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute K2(e_A) by fixing Ai (from F, SO) and varying the vdW parameter e_A.
    Returns (eA_array, K2_array).
    """
    eA_values = np.asarray(list(eA_values), dtype=float)

    # Compute Ai once (independent of e_A)
    Ailst = compute_Ais(F, SO)

    K2_list = []
    for eA in eA_values:
        D, Dp, Dpp = D_vdW(e_a=eA, m_inf=m_inf)
        m0 = F.m0
        # guard against singular diffusion when m0 ~ m_inf
        if np.isclose(m_inf - m0, 0.0):
            K2_list.append(np.nan)
            continue
        K2 = K2_from_Ai(Ailst, D(m0), Dp(m0), Dpp(m0))
        K2_list.append(float(K2))

    return eA_values, np.array(K2_list)

def plot_K2_of_eA(F, SO, eA_values: Iterable[float], m_inf: float = 10.0,
                  *, save: bool = False, filename: str = "K2_of_eA.png"):
    """
    Plot K2 as a function of e_A for the vdW diffusion.
    """

    eA, K2 = K2_of_eA(F, SO, eA_values, m_inf=m_inf)
    plt.figure()
    plt.plot(eA, K2, marker='o')
    plt.xlabel(r"$e_A$")
    plt.ylabel(r"$K_2(e_A)$")
    plt.title(r"$K_2$ vs $e_A$ (vdW diffusion)")
    plt.grid(True, alpha=0.3)
    if save:
        plt.savefig(filename, dpi=180, bbox_inches='tight')
    else:
        plt.show()
    return eA, K2



def find_bifurcation_change_to_negative(
    F,
    SO,
    eA_min: float,
    eA_max: float,
    *,
    m_inf: float = 10.0,
    n_scan: int = 401,
    tol: float = 1e-10,
    max_iter: int = 60,
) -> Dict[str, object]:
    """
    Scan e_A in [eA_min, eA_max], detect K2 sign changes, refine with bisection.
    Uses your existing K2_of_eA(F, SO, eA_values, m_inf).
    Returns:
      {
        'changed_to_negative': bool,
        'crossings': [ {'ea_left','ea_right','K2_left','K2_right','root','left_sign','right_sign'} ... ],
        'samples':   { 'eA': array, 'K2': array }
      }
    """
    # coarse scan
    e_grid = np.linspace(eA_min, eA_max, int(n_scan))
    eA, K2 = K2_of_eA(F, SO, e_grid, m_inf=m_inf)

    def iszero(x):  # robust zero test
        return np.isfinite(x) and np.isclose(x, 0.0, rtol=1e-12, atol=1e-14)
    def sgn(x):
        return 0 if iszero(x) else (1 if x > 0 else -1)

    crossings: List[Dict[str, float]] = []

    # helper that evaluates K2 at a single eA (still via your K2_of_eA)
    def eval_K2(e):
        return float(K2_of_eA(F, SO, [e], m_inf=m_inf)[1][0])

    for i in range(len(eA) - 1):
        a, b = eA[i], eA[i+1]
        fa, fb = K2[i], K2[i+1]
        if not np.isfinite(fa) or not np.isfinite(fb):
            continue

        sa, sb = sgn(fa), sgn(fb)

        # If there's a sign change across the interval, refine by bisection
        if sa * sb < 0:
            lo, hi = a, b
            flo, fhi = fa, fb
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                fmid = eval_K2(mid)
                if not np.isfinite(fmid):
                    break
                smid = sgn(fmid)
                # keep the sign-changing half
                if sgn(flo) * smid <= 0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid
                if abs(hi - lo) <= tol:
                    break
            root = 0.5 * (lo + hi)
            crossings.append({
                'ea_left': a, 'ea_right': b,
                'K2_left': fa, 'K2_right': fb,
                'root': root,
                'left_sign': float(sa),
                'right_sign': float(sb),
            })
        else:
            # also record exact-zero endpoints if they exist
            if sa == 0 or sb == 0:
                root = a if sa == 0 else b
                crossings.append({
                    'ea_left': a, 'ea_right': b,
                    'K2_left': fa, 'K2_right': fb,
                    'root': root,
                    'left_sign': float(sa),
                    'right_sign': float(sb),
                })

    # flag if any crossing is +→−
    changed_to_negative = any(c['left_sign'] > 0 and c['right_sign'] < 0 for c in crossings)

    return {
        'changed_to_negative': changed_to_negative,
        'crossings': crossings,
        'samples': {'eA': eA, 'K2': K2},
    }

def has_change_to_negative(F, SO, eA_min: float, eA_max: float, **kwargs) -> bool:
    """Convenience wrapper that returns only the boolean flag."""
    res = find_bifurcation_change_to_negative(F, SO, eA_min, eA_max, **kwargs)
    return bool(res['changed_to_negative'])


def sweep_params_for_bif_change(
    P_vals: Iterable[float],
    Z_vals: Iterable[float],
    gamma_vals: Iterable[float],
    *,
    eA_min: float,
    eA_max: float,
    m_inf: float = 10.0,
    n_scan: int = 401,
    tol: float = 1e-10,
    max_iter: int = 60,
    second_order_N: int = 300,
    modes: Tuple[int, int] = (0, 2),
    D_for_SO: float = 1.0,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """
    Sweep a grid of (P, Z, gamma). For each triple:
      1) build F = build_first_order(P, Z, gamma)
      2) build SO = compute_second_order_elimination(F, D=D_for_SO, modes=modes, N_init=second_order_N)
      3) check if K2(e_A) changes to negative on [eA_min, eA_max] using find_bifurcation_change_to_negative
    Returns a list of records:
      {
        'P','Z','gamma','ok', 'changed_to_negative',
        'root_eA' (or None), 'crossings', 'error' (if any)
      }
    """
    results: List[Dict[str, object]] = []

    for P, Z, gamma in product(P_vals, Z_vals, gamma_vals):
        rec = {'P': P, 'Z': Z, 'gamma': gamma, 'ok': False,
               'changed_to_negative': False, 'root_eA': None,
               'crossings': [], 'error': None}
        try:
            if verbose:
                print(f"[sweep] P={P}, Z={Z}, gamma={gamma} → building F", flush=True)
            F = build_first_order(P, Z, gamma)

            if verbose:
                print(f"[sweep] ... building SO", flush=True)
            SO = compute_second_order(F, modes=modes, N_init=second_order_N)

            if verbose:
                print(f"[sweep] ... scanning e_A in [{eA_min}, {eA_max}]", flush=True)
            res = find_bifurcation_change_to_negative(
                F, SO, eA_min, eA_max,
                m_inf=m_inf, n_scan=n_scan, tol=tol, max_iter=max_iter
            )

            rec['ok'] = True
            rec['changed_to_negative'] = bool(res['changed_to_negative'])
            rec['crossings'] = res['crossings']

            # If multiple crossings exist, pick the first +→− crossing as root_eA
            roots = [c['root'] for c in res['crossings'] if c['left_sign'] > 0 and c['right_sign'] < 0]
            rec['root_eA'] = None if not roots else float(roots[0])

        except Exception as e:
            rec['error'] = f"{type(e).__name__}: {e}"

        results.append(rec)

    if verbose:
        print("\n=== Summary ===")
        for r in results:
            if not r['ok']:
                print(f"  (fail) P={r['P']} Z={r['Z']} γ={r['gamma']}  -> {r['error']}")
            else:
                flag = "YES" if r['changed_to_negative'] else "no"
                print(f"  P={r['P']} Z={r['Z']} γ={r['gamma']}  change_to_negative? {flag}"
                      + (f", root_eA≈{r['root_eA']:.10f}" if r['root_eA'] is not None else ""))

    return results