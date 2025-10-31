from steady_state import *
from first_order import *
from second_order import *
from compute_Ai import *
from verify_numerics import *
from velocity_scaling_plots import *
from check_k2 import *

if __name__ == "__main__":
    P,Z,gamma = 0.015915, 0.159155, 0.0025
    F = build_first_order(P=P, Z=Z, gamma=gamma)
    SO = compute_second_order(F)

    print(check_second_order_pde_discrete(F,SO)["loss"])


    # diffusion model and K
    D, Dp, Dpp = D_vdW(e_a=0.0, m_inf=10.0)  # or your own D(m)
    As = compute_Ais(F,SO)

    k2 = K2_from_Ai(As, D(F.m0),Dp(F.m0), Dpp(F.m0))
    k2 = K2_from_Ai(As, 1,0,0)
    #print(check_k2_consistent(F, SO, D, Dp, Dpp, k2))
    #D0 = D_of_m(F.m0)                            # D(m0)

    # compute arrays and plot
    #result = compute_velocity_scaling(F, SO, D, Dp, Nr=180, Nth=128)

    #err = check_second_order_pde_discrete(F,SO)

    #print(err)

    #comp = compare_first_second_order_fields(F, SO, D, Dp, V = 0.01)
    #print(comp)
    #plot_velocity_scaling(result, save=True, prefix="figs/velocity_scaling", show=False, title_suffix=f"(P={P}, Z={Z}, γ={gamma})")

    #sigma, m = build_sigma_m(F,SO, D, Dp, V=0.01, order=2)

# sweep e_A
    res = find_bifurcation_change_to_negative(F, SO, 0.0, 1.0, m_inf=10.0, n_scan=801)
    print("change to negative?", res['changed_to_negative'])
    for c in res['crossings']:
        print(f"ea in [{c['ea_left']:.4f}, {c['ea_right']:.4f}] "
          f"K2_left={c['K2_left']:.3e}, K2_right={c['K2_right']:.3e}, root≈{c['root']:.8f}, "
          f"dir {int(c['left_sign'])}→{int(c['right_sign'])}")
        

    # P_vals     = np.linspace(0.01, 0.025, 5)
    # Z_vals     = [1.25]          # or a small linspace
    # gamma_vals = np.linspace(0.001, 0.01, 4)

    # res = sweep_params_for_bif_change(
    #     P_vals, Z_vals, gamma_vals,
    #     eA_min=0.0, eA_max=1.0,
    #     m_inf=10.0,
    #     n_scan=801,             # dense e_A sampling improves bracketing
    #     second_order_N=300,
    #     verbose=True
    # )
    
    # # Extract all parameter triples with a +→− crossing
    # hits = [r for r in res if r.get('ok') and r.get('changed_to_negative')]
    # print(f"\nFound {len(hits)} triples with change-to-negative:")
    # for h in hits:
    #     print(f"  P={h['P']}, Z={h['Z']}, γ={h['gamma']}, root_eA≈{h['root_eA']}")  