from steady_state import *
from first_order import *
from second_order import *
from compute_Ai import *
from verify_numerics import *


if __name__ == "__main__":
    P,Z,gamma = 0.015915, 0.159155, 0.0025
    F = build_first_order(P=P, Z=Z, gamma=gamma)
    SO = compute_second_order(F)

    print(check_second_order_pde_discrete(F,SO))

    Ai = compute_Ais(F, SO)

    print(K2_from_Ai(Ai, 1,0,0))
    