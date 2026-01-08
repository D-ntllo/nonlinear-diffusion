import numpy as np
from typing import Iterable, Tuple, List, Callable
from typing import Dict

import matplotlib.pyplot as plt
from itertools import product

from first_order import *
from second_order import *
from compute_Ai import *
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


def eA_max_vdW_strictly_positive(m_inf: float = 10.0, *, strict: bool = True, eps: float = 1e-12) -> float:
    """
    Return the sharp upper bound on e_A such that

        D(m) = m_inf^2/(m_inf - m)^2 - e_A*m

    is > 0 for all m in [0, m_inf).

    The threshold occurs at m = m_inf/3, where D hits 0 when e_A = 27/(4*m_inf).
    Therefore strict positivity requires e_A < 27/(4*m_inf).

    If strict=True, returns a slightly smaller value than the sharp threshold.
    If strict=False, returns the sharp threshold (note: D is not strictly positive at that value).
    """
    if m_inf <= 0:
        raise ValueError("m_inf must be positive.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    eA_star = 27.0 / (4.0 * m_inf)  # sharp threshold (supremum)

    if not strict:
        return eA_star

    # ensure a strict inequality margin
    return max(0.0, eA_star * (1.0 - eps))




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



def plot_K2_of_eA(F, SO, eA_values, m_inf: float = 10.0,
                  *, save: bool = False, filename: str = "K2_of_eA.png"):
    """
    Plot K2 as a function of e_A for the vdW diffusion.
    Solid line, dashed y=0 line, and a red point labeled e_A^* at K2=0 (or closest).
    No background grid.
    """
    eA, K2 = K2_of_eA(F, SO, eA_values, m_inf=m_inf)

    eA = np.asarray(eA, dtype=float)
    K2 = np.asarray(K2, dtype=float)

    # --- find e_A^* where K2(e_A) = 0 ---
    eA_star, K2_star = None, None
    zero_idx = np.where(np.isclose(K2, 0.0, atol=1e-12, rtol=0.0))[0]
    if zero_idx.size > 0:
        i = int(zero_idx[0])
        eA_star, K2_star = float(eA[i]), float(K2[i])
    else:
        s = np.sign(K2)
        change = np.where(s[:-1] * s[1:] < 0)[0]
        if change.size > 0:
            i = int(change[0])
            eA_star = float(np.interp(0.0, [K2[i], K2[i+1]], [eA[i], eA[i+1]]))
            K2_star = 0.0
        else:
            i = int(np.argmin(np.abs(K2)))
            eA_star, K2_star = float(eA[i]), float(K2[i])

    plt.figure()
    plt.plot(eA, K2, linestyle='-', linewidth=2)
    plt.axhline(0.0, linestyle='--', linewidth=1, alpha=0.7)

    if eA_star is not None:
        plt.scatter([eA_star], [K2_star], color='red', zorder=5)
        plt.annotate(r"$e_A^*$",
                     xy=(eA_star, K2_star),
                     xytext=(8, 10),
                     textcoords="offset points",
                     color='red')

    plt.xlabel(r"$e_A$")
    plt.ylabel(r"$K_2$")
    #plt.title(r"$K_2$ vs $e_A$ (vdW diffusion)")

    # remove background grid
    plt.grid(False)

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