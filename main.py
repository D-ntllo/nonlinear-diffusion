from steady_state import *
from first_order import *
from second_order import *
from compute_Ai import *
from verify_numerics import *
from velocity_scaling_plots import *

if __name__ == "__main__":
    P,Z,gamma = 0.015915, 0.159155, 0.0025
    F = build_first_order(P=P, Z=Z, gamma=gamma)
    SO = compute_second_order(F)

    print(check_second_order_pde_discrete(F,SO)["loss"])


    # diffusion model and K
    D, Dp, Dpp = D_vdW(e_a=0.0, m_inf=10.0)  # or your own D(m)
    #D0 = D_of_m(F.m0)                            # D(m0)

    # compute arrays and plot
    result = compute_velocity_scaling(F, SO, D, Dp, Nr=180, Nth=128)

    comp = compare_first_second_order_fields(F, SO, D, Dp, V = 0.01)
    print(comp)
    plot_velocity_scaling(result, save=True, prefix="figs/velocity_scaling", show=False, title_suffix=f"(P={P}, Z={Z}, γ={gamma})")