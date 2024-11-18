# This example shows how to use PHLAME to generate a trajectory 
# to move the double pendulum from some initial to final position

import ipdb
import numpy as np
import matplotlib.pyplot as plt
import os
import unittest

from phlame.aghf import PostAghf
from phlame.parameter_set import generate_parameter_sets
from phlame.parameter_set import ParameterSetBase
from phlame.experiment import Experiment
from phlame.utils import cleanup_folder
from phlame.plotting import plot_along_s_span_from_vecs


# Parameter for ParameterSetBase
p = 7 # Number of Chebyshev nodes
k = 1e3 # Penalty parameter
s_max = 10 # Maximum value of s

# ode solver settings
abs_tol = 1e-10
rel_tol = 1e-10
method_name = "cvode"
max_steps = 1e2
ns_points = int(1e2)
run_name = "double-pendulum-swingup" # Name of the run

dt = 1e-4 # time step size for integrating the u^2

# Get base directory path
base_dir = os.path.join(os.path.dirname(__file__), "..")

# Set up number of bodies and initial and final states
N = 2 # Number of links (joint angles dimension)
j_type = (2 * np.ones(shape=(N, 1))).astype(dtype=np.double, order='F')
fp_urdf = os.path.join(base_dir, f"urdfs/pendulums/pen-{int(N)}.urdf")
X0 = np.zeros(shape=(2*N, 1), dtype=np.double, order='F')
X0[0] = -np.pi/2
Xf = np.zeros(shape=(2*N, 1), dtype=np.double, order='F')
Xf[0] = np.pi/2

# Parameters for Experiment
timeout = 60
print_debug = False
folder_store = "temp_store"
fp_folder_store = os.path.join(base_dir, "examples/", folder_store)

# create folder if it does not exist, clear contents if it does
cleanup_folder(fp_folder_store)

pset = ParameterSetBase(p, N, X0, Xf, run_name, s_max, k, abs_tol, rel_tol, 
                                        method_name, max_steps, ns_points, j_type, fp_urdf)
# solve the AGHF
result = Experiment.run_single_static(pset, timeout, mode="general", use_jacobian=False, print_debug=False)
print(f"Finished solving in: {result.t_solve} seconds.")

# Do post processing of AGHF solution to see evolution of things along s
post_aghf = PostAghf(result)

# Get the pseudospectral node values of the initial guess and final solution of the AGHF
ps_values_ini = post_aghf.first_result.sol[0, :].reshape(-1, 1)
ps_values_fin = post_aghf.first_result.sol[-1, :].reshape(-1, 1)
t_interp = np.linspace(-1, 1, int((2 / dt) + 1) )

q_ini, qd_ini, qdd_ini = post_aghf.get_q_qd_qdd(ps_values_ini, t_interp)
q_fin, qd_fin, qdd_fin = post_aghf.get_q_qd_qdd(ps_values_fin, t_interp)
u_ini, _ = post_aghf.compute_u_sq(q_ini, qd_ini, qdd_ini, dt)
u_fin, _ = post_aghf.compute_u_sq(q_fin, qd_fin, qdd_fin, dt)
print("norm u^2 of the AGHF solution: ", u_fin)


sol = post_aghf.first_result.sol

# Compute how the action functional, norm u^2 and other values evolved as the AGHF solves 
action_functional_vec, vel_dif_vec, accel_dif_vec, \
    norm_dps_values_ds_vec, u_sq_vec, real_s_span, _, _, _, _ = \
    post_aghf.compute_along_s(dt, sol)
    
# Plot the evolution of the action functional,
# the dynamics errors (vel_dif = (xdp1-xp2), accel_dif = (xddp1 - xdp2)), 
# the norm of the ps_values (which is a proxy for dx/ds) and norm u^2
plot_along_s_span_from_vecs(action_functional_vec, vel_dif_vec, 
                                accel_dif_vec, norm_dps_values_ds_vec, 
                                u_sq_vec, real_s_span)
plt.show()