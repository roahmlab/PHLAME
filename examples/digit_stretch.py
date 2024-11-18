# This example shows how to use the PHLAME to generate trajectory to 
# get Digit to do a yoga stretch

from phlame.plotting import plot_along_s_span
from phlame.plotting import print_results
from phlame.utils import load_from_pickle
from phlame.aghf import PostAghf
from phlame.fwd_simulation import ode_fwd_integrate
from phlame.fwd_simulation import get_u_curr
from phlame.utils import interpolate_matrix
from scipy.integrate import solve_ivp
from phlame.aghf import PostAghf
from phlame.experiment import Experiment
from phlame.parameter_set import ParameterSetBase
from phlame.plotting import plot_along_s_span_from_vecs
import math

import ipdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import os

# Parameter for ParameterSetBase
p = 6 # Number of Chebyshev nodes
k = 100000000 # Penalty parameter
s_max = 20 # Maximum value of s

# ode solver settings
abs_tol = 1e-4
rel_tol = 1e-4
method_name = "cvode"
max_steps = 1e8
ns_points = int(1e2)
run_name = "Stretch-Digit" # Name of the run


dt = 1e-2 # # time step size for integrating the u^2

# Get base directory path
base_dir = os.path.join(os.path.dirname(__file__), "..")

# Set up number of bodies and initial and final states
N = 22 # Number of links (joint angles dimension)
j_type = (2 * np.ones(shape=(N, 1))).astype(dtype=np.double, order='F')
fp_urdf = os.path.abspath( os.path.join( base_dir, "urdfs/digit/digit-v3-modified-pinned.urdf" ) )

# Standing configuration
X0 = np.array([0.10, 0, 0, 0, -0.20, 0, -0.30, 0, 0, 0, 0, -0.20, 0, -0.10, 0.080, 0.020, -0.10, -0.10, 0, 0, 0, 0, #position states
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1, order='F') #velocity states

# Stretching configuration
Xf = np.array([0, -0.20, 0.80, -0.40, 1.1, 0.30, -0.40, 0, 0, 0, 0, 0, 0, -0.60, -0.40, 1, 0, 0, -0.30, 3.14100, 0, -0.70, #position states
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1, order='F')        #velocity states

# Parameters for Experiment
timeout = 200
print_debug = False

pset = ParameterSetBase(p=p, N=N, X0=X0, Xf=Xf, name=run_name, s_max=s_max, k=k, abs_tol=abs_tol, 
                          rel_tol=rel_tol, method_name=method_name, max_steps=max_steps, 
                          ns_points=ns_points, j_type=j_type, fp_urdf=fp_urdf)

# Solving the AGHF without jacobian
# result = Experiment.run_single_static(pset=pset, timeout=timeout)

# Solving the AGHF with the jacobian
result = Experiment.run_single_static(pset=pset, timeout=timeout, use_jacobian=True, print_debug=False)

print(f"Finished solving in: {result.t_solve} seconds.")
sol = result.sol

# Do post processing of AGHF solution to see evolution of things along s
post_aghf = PostAghf(result) # This class has the functionality to compute things along the s-span


# Uncomment this block to plot how the action functional, norm u^2 and other values evolved as the AGHF solves
#### -------------- /#####
# print(f"Computing intermediate quantities along the s-span")
# action_functional_vec, vel_dif_vec, accel_dif_vec, norm_dps_values_ds_vec, \
#                u_sq_vec, real_s_span, Q_mat, Qd_mat, Qdd_mat, U_mat = post_aghf.compute_along_s(dt, sol)

# plot_along_s_span_from_vecs(action_functional_vec, vel_dif_vec, accel_dif_vec, 
#                                 norm_dps_values_ds_vec, u_sq_vec, real_s_span)
#### -------------- /#####

# Obtain the final solution
ps_values = sol[-1, :]
t_interp = np.linspace(-1, 1, int((2 / dt) + 1) )


# Compute u_sq of the aghf solution
q, qd, qdd = post_aghf.get_q_qd_qdd(ps_values, t_interp)
U_sol = post_aghf.compute_u_matrix(q, qd, qdd)
X_aghf_t, _ = post_aghf.get_X_Xd_spec_t(ps_values, t_interp)
u_sq_aghf, _ = post_aghf.compute_u_sq(q, qd, qdd, dt)
print("u^2 of the AGHF solution: ", u_sq_aghf)

# Forward simulate the AGHF solution
k_pos = 10
k_vel = 10
lam_ode = lambda tau, x: ode_fwd_integrate(tau, x, U_sol, t_interp, post_aghf.model, 
                                           X_aghf_t, t_interp, k_pos,
                                           k_vel)
x0 = post_aghf.X0
t_simul = [ t_interp[0], t_interp[-1] ]
solution = solve_ivp(lam_ode, t_simul, x0.squeeze(), method='RK45', t_eval=t_interp)

# Calculate number of rows needed
num_cols = 4
num_rows = math.ceil(post_aghf.N / num_cols)

# Plot all trajectories (q)
fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6*num_rows))
axs = axs.flatten()  # Flatten the array of axes for easy iteration
for ii in range(post_aghf.N):
    ax = axs[ii]
    ax.plot(solution.t, solution.y[ii, :], label=f'Integrated Trajectory, q_{ii+1}')
    ax.plot(t_interp, X_aghf_t[:, ii], label=f'AGHF solution q_{ii+1}')
    ax.legend()
    
# Remove empty subplots
for ax in axs[post_aghf.N:]:
    fig.delaxes(ax)
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)  # Increase space between subplots vertically 


# Plot all velocities (qd)
fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6*num_rows))
axs = axs.flatten()  # Flatten the array of axes for easy iteration
for ii in range(post_aghf.N):
    ax = axs[ii]
    idx = ii + post_aghf.N  # Adjusted index for the second set of trajectories
    ax.plot(solution.t, solution.y[idx, :], label=f'Integrated Trajectory qd_{ii+1}')
    ax.plot(t_interp, X_aghf_t[:, idx], label=f'AGHF solution qd_{ii+1}')
    ax.legend()
    
# Remove empty subplots
for ax in axs[post_aghf.N:]:
    fig.delaxes(ax)
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)  # Increase space between subplots vertically 

# get actual u
U_actual = np.zeros(shape=(t_interp.shape[0], post_aghf.N))

for ii in range(solution.y.shape[1]):
    U_actual[ii, :] = get_u_curr(t_interp[ii], solution.y[:, ii], U_sol, t_interp, 
                                 post_aghf.model, X_aghf_t, t_interp, k_pos, k_vel)
    

# plot u at each joint
fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6*num_rows))
axs = axs.flatten()
for ii in range(post_aghf.N):
    ax = axs[ii]
    ax.plot(solution.t, U_actual[:, ii], label=f'u{ii}: Actual/real')
    ax.plot(solution.t, U_sol[:, ii], label=f'u{ii}: extracted from AGHF solution')
    ax.legend()
for ax in axs[post_aghf.N:]:
    fig.delaxes(ax)
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)  # Increase space between subplots vertically 


norm_u_real = np.sum(U_actual * U_actual, axis=1)
u_sq_real = dt * np.sum(norm_u_real)
print("u^2 real when using a controller: ", u_sq_real)

# obtain the inf norm
x_right_sol = solution.y[:, -1].reshape(-1, 1)
dif_bc = Xf - x_right_sol
dif_pos = Xf[:N, 0] - x_right_sol[:N, 0]
dif_vel = Xf[N : , 0] - x_right_sol[N : , 0]

inf_norm_all = np.linalg.norm(dif_bc, ord=np.inf)
inf_norm_pos = np.linalg.norm(dif_pos, ord=np.inf)
inf_norm_vel = np.linalg.norm(dif_vel, ord=np.inf)

print("Inf norm all: ", inf_norm_all)
print("Inf norm pos: ", inf_norm_pos)
print("Inf norm vel: ", inf_norm_vel)

plt.show()