from steady_state import *
from first_order import *
from second_order import *
from compute_Ai import *
from verify_numerics import *
from velocity_scaling_plots import *
from check_k2 import *
from vdW_diff import *


from k2_optimizer import optimize_k2_boundary_loss

if __name__ == "__main__":


    P_exp = 0.015915
    Z_exp = 0.159155
    gamma_exp = 0.0025

    P,Z,gamma = 0.016, 0.16, 0.00001
    F = build_first_order(P=P_exp, Z=Z_exp, gamma=gamma_exp)
    SO = compute_second_order(F)

    #print(check_second_order_pde_discrete(F,SO)["loss"])

    #print("K0 ", F.Khat0)

    # diffusion model and K
    D, Dp, Dpp = D_vdW(e_a=0.0, m_inf=10.0)  # or your own D(m)
    As = compute_Ais(F,SO)
    #print(As)

    k2_vdw = K2_from_Ai(As, D(F.m0), Dp(F.m0), Dpp(F.m0))
    k2_const = K2_from_Ai(As, 1, 0, 0)


    k2_opt = optimize_k2_boundary_loss(F, SO,  lambda x: 1,  lambda x: 0,  lambda x: 0, k2_guess=0.0)
    k2_formula = K2_from_Ai(compute_Ais(F, SO), 1, 0, 0)
    print("k2_opt", k2_opt, "k2_formula", k2_formula)

    # print("K2 (vdW diffusion)")
    # print(k2_vdw)
    # print("K2 (constant diffusion)")
    # print(k2_const)

    # print("error at k2=0")
    # print(check_k2_consistent(F, SO, D, Dp, Dpp, 0))

    # print("error at K2")
    # print(check_k2_consistent(F, SO, D, Dp, Dpp, k2_vdw))


    # plot_K2_of_eA(F, SO, np.linspace(0.5, 0.7, 100),
    #           save=True)

    #D0 = D_of_m(F.m0)                            # D(m0)

    # compute arrays and plot
   # result = compute_velocity_scaling(F, SO, D, Dp, Nr=180, Nth=128)

    #err = check_second_order_pde_discrete(F,SO)


    # print(err)

    # comp = compare_first_second_order_fields(F, SO, D, Dp, V = 0.01)
    # print(comp)
    # plot_velocity_scaling(result, save=True, prefix="figs/velocity_scaling", show=False, title_suffix=f"(P={P}, Z={Z}, γ={gamma})")

    #sigma, m = build_sigma_m(F,SO, D, Dp, V=0.01, order=2)

# sweep e_A
    res = find_bifurcation_change_to_negative(F, SO, 0.0, 1.0, m_inf=10.0, n_scan=801)
    print("change to negative?", res['changed_to_negative'])
    for c in res['crossings']:
        print(f"ea in [{c['ea_left']:.4f}, {c['ea_right']:.4f}] "
          f"K2_left={c['K2_left']:.3e}, K2_right={c['K2_right']:.3e}, root≈{c['root']:.8f}, "
          f"dir {int(c['left_sign'])}→{int(c['right_sign'])}")
        

#     P_vals     = np.linspace(0.01, 0.025, 5)
#     Z_vals     = np.linspace(0.01, 0.25, 5)          # or a small linspace
#     gamma_vals = np.linspace(0.001, 0.01, 4)

#     res = sweep_params_for_bif_change(
#         P_vals, Z_vals, gamma_vals,
#         eA_min=0.0, eA_max=0.7,
#         m_inf=10.0,
#         n_scan=801,             # dense e_A sampling improves bracketing
#         second_order_N=300,
#         verbose=True
#     )
    
    # Extract all parameter triples with a +→− crossing
    #hits = [r for r in res if r.get('ok') and r.get('changed_to_negative')]
    #print(f"\nFound {len(hits)} triples with change-to-negative:")
    #for h in hits:
        #print(f"  P={h['P']}, Z={h['Z']}, γ={h['gamma']}, root_eA≈{h['root_eA']}")  
