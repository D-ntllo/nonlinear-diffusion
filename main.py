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

    print(check_second_order_pde_discrete(F,SO))


    As = compute_Ais(F,SO)
    print(As)


    #print("K0 ", F.Khat0)

    #plt.plot(SO.A.r, SO.A.m2)
    #plt.show()

    DD, Dp, Dpp = D_vdW(e_a=0.58, m_inf=10.0)  
   
  
    #res = compute_velocity_scaling(F, SO, DD, Dp, Vs=np.logspace(1, 1e5, 10),)
    #plot_velocity_scaling(res, save=True, prefix="figs/velocity_scaling")



    k2_vdw = K2_from_Ai(As, DD(F.m0), Dp(F.m0), Dpp(F.m0))
    k2_const = K2_from_Ai(As, 1, 0, 0)
    print("k2_const:", k2_const)


    k2_opt = optimize_k2_boundary_loss(F, SO,  DD,  Dp,  Dpp, k2_guess=0.0)
    k2_formula = K2_from_Ai(compute_Ais(F, SO), DD(F.m0), Dp(F.m0), Dpp(F.m0))
    print("k2_formula", k2_formula)

    
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
