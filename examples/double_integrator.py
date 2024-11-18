# This example shows how to use the PHLAME to generate a trajectory 
# to move the double integrator system from some initial to final position


import ipdb
import numpy as np
import matplotlib.pyplot as plt
import os

from phlame.aghf import PostAghf
from phlame.parameter_set import ParameterSetBase
from phlame.experiment import Experiment
from phlame.utils import cleanup_folder


Initial_guesses = [
    # this is the line
    np.array([[ 6.25      ],
        [ 7.5       ],
        [ 8.75      ],
        [-3.23223305],
        [-1.03553391],
        [-3.23223305]]),
    # line with some gaussian noise
    np.array([[ 5.44914078],
       [ 9.54364417],
       [ 8.56363249],
       [-2.17518399],
       [ 0.18093025],
       [-2.12729329]]),
    # Random points set 1
    np.array([[-10.57657819],
        [  8.86109353],
        [ -2.87770441],
        [ -1.48095107],
        [ -8.38785171],
        [  2.36880279]]),
    #  Random points set 2
    np.array([[-15.52153202],
        [ 15.1250421 ],
        [ -3.84099933],
        [ -7.52405408],
        [-14.01876775],
        [ 14.04282182]])
]

for ii in range(len(Initial_guesses)):
    
    # Parameter for ParameterSetBase
    p = 4 # Number of Chebyshev nodes
    k = 1e3 # Penalty parameter
    s_max = 100 # Maximum value of s
    
    # ode solver settings
    abs_tol = 1e-4
    rel_tol = 1e-4
    method_name = "cvode"
    max_steps = 1e8
    ns_points = int(1e2)
    run_name = "test"
    
    dt = 1e-4 # time step size for integrating the u^2
    
    # Get base directory path
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    
    # Set up number of bodies and initial and final states
    N = 1 # Number of links (joint angles dimension)
    j_type = (2 * np.ones(shape=(N, 1))).astype(dtype=np.double, order='F')
    fp_urdf = os.path.abspath(os.path.join(base_dir, 
                            f"urdfs/pendulums/pen-{int(N)}.urdf"))
    X0 = np.array([10, -0.3]).reshape(-1, 1).astype(dtype=np.double, order='F')
    Xf = np.array([5, 0]).reshape(-1, 1).astype(dtype=np.double, order='F')
    
    # Parameters for Experiment
    timeout = 60
    print_debug = False

    single_pset = ParameterSetBase(p, N, X0, Xf, run_name, s_max, k, abs_tol, rel_tol, 
                                method_name, max_steps, ns_points, j_type, fp_urdf)
    # single_pset.init_ps_values = single_pset.generate_random_ps_traj()
    single_pset.init_ps_values = Initial_guesses[ii]

    pset = [ single_pset ]
       
    # solve the AGHF. Setting mode to doubleint to use AGHF RHS that corresponds to double integrator dynamics
    result = Experiment.run_single_static(single_pset, timeout, mode="general", use_jacobian=False, print_debug=False)
    
    # Do post processing of AGHF solution to see evolution of things along s
    post_aghf = PostAghf(result, mode="doubleint")
    ps_values_ini = post_aghf.first_result.sol[0, :].reshape(-1, 1)
    ps_values_fin = post_aghf.first_result.sol[-1, :].reshape(-1, 1)

    # Compute the u sq by extracting the acceleration
    q_ini, qd_ini, u_ini, t_ini = post_aghf.compute_qdd(ps_values_ini, dt)
    q_fin, qd_fin, u_fin, t_fin = post_aghf.compute_qdd(ps_values_fin, dt)
    get_u_sq = lambda u, dt: dt * np.sum( np.sum(u * u, axis=1) )
    u_sq_ini = get_u_sq(u_ini, dt)
    u_sq_fin = get_u_sq(u_fin, dt)
    print("Initial u squared: ", u_sq_ini)
    print("Final u squared: ", u_sq_fin)

    # Compute how the action functional, norm u^2 and other values evolved as the AGHF solves
    sol = post_aghf.first_result.sol
    vel_dif_vec, accel_dif_vec, norm_ps_values_ds_vec, u_sq_vec, _ = \
        post_aghf.compute_along_s_doubleint(dt, sol)

    print("#" * 80)
    print("Result number: ", ii)
    print_ini_fin = lambda x, run_name: print(f"{run_name} <> Ini: {x[0]}, Fin: {x[-1]}")
    print_ini_fin(vel_dif_vec, 'vel_dif_norm')
    print_ini_fin(accel_dif_vec, 'acc_dif_norm')
    print_ini_fin(norm_ps_values_ds_vec, 'norm_ps_values_ds')
    print_ini_fin(u_sq_vec, 'u_sq')

    idx_min_u_sq = np.argmin(u_sq_vec)
    idx_min_norm_ps_values_ds = np.argmin(norm_ps_values_ds_vec)
    print(f"Minimum u squared: {u_sq_vec[idx_min_u_sq]}, s_value: {post_aghf.s_span[idx_min_u_sq]}")
    print(f"Minimum norm_ps_values_ds: {norm_ps_values_ds_vec[idx_min_norm_ps_values_ds]}, s_value: {post_aghf.s_span[idx_min_norm_ps_values_ds]}")

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_fin, q_fin, label='x_fin')
    plt.plot(t_fin, q_ini, label='x_ini')

    plt.ylabel('x_fin')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t_fin, qd_fin, label='xd_fin')
    plt.plot(t_fin, qd_ini, label='xd_ini')
    plt.ylabel('xd_fin')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t_fin, u_fin, label='u_fin')
    plt.plot(t_fin, u_ini, label='u_ini')
    plt.ylabel('u_fin')
    plt.xlabel('Time')
    plt.legend()
    plt.title(f"Result number: {ii}")
    plt.tight_layout()
plt.show()
