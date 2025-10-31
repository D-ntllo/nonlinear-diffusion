import numpy as np
from typing import Callable
from dataclasses import dataclass


from steady_state import *

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
    s11_R0: float
    s11pp_R0: float

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
    s11_R0   = float(s11(np.array([R0])))
    s11pp_R0 = float(s11pp(np.array([R0])))

    return FirstOrder(R0, m0, Z, P,gamma, Khat0, alpha,
                      s11, s11p, s11pp, m11, m11p, m11pp, U, Up, m11_R0, m11pp_R0, s11_R0, s11pp_R0)