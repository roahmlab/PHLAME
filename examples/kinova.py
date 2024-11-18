# This example shows how to use PHLAME to generate a trajectory 
# to move the Kinova from some initial to final position

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
from phlame.parameter_set import ParameterSetActivatedSimpleSphere
from phlame.plotting import plot_along_s_span_from_vecs
import math

import ipdb
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat, loadmat

# Parameter for ParameterSetBase
p = 7  # Number of Chebyshev nodes
k = 10000.0  # Penalty parameter
s_max = 1  # Maximum value of s

# ode solver settings
abs_tol = 1e-4
rel_tol = 1e-4
method_name = "cvode"
max_steps = 1e8
ns_points = int(1e2)
run_name = "kinova"  # Name of the run

dt = 1e-2  # Time step size for integrating the u^2

# Get base directory path
base_dir = os.path.join(os.path.dirname(__file__), "..")

# Set up number of bodies and urdf file
N = 7  # Number of links (joint angles dimension)
j_type = (2 * np.ones(shape=(N, 1))).astype(dtype=np.double, order='F')
fp_urdf = os.path.abspath(os.path.join(base_dir, "urdfs/kinova-gen3/kinova.urdf"))

# Set up constraint hyper-parameters
k_cons = 1e6
c_cons = 50

# Parameters for Experiment
do_plot = False  # choose whether to plot the results
timeout = 200
print_debug = False

print("Parameters: ")
print("p: ", p)
print("s_max: ", s_max)
print("k: ", k)

# Setup scenarios to load obstacle data and initial and final positions from
FOLDER_SCENARIOS = os.path.abspath(os.path.join(base_dir, "scenarios/kinova_no_obstacles"))
SCENARIO_NAME = f"kinova_example"
FP_SCENARIO = os.path.join(FOLDER_SCENARIOS, f"{SCENARIO_NAME}.mat")

# Load the scenario data
data_scenario = loadmat(FP_SCENARIO)
q_traj = data_scenario['q_traj'].astype(dtype=np.double, order='F')
q_start = q_traj[0, :].reshape(N, 1)
q_end = q_traj[-1, :].reshape(N, 1)
qd_start = data_scenario['qd_start'].astype(dtype=np.double, order='F').reshape(N, 1)
qd_end = data_scenario['qd_end'].astype(dtype=np.double, order='F').reshape(N, 1)
X0 = np.concatenate((q_start, qd_start)).astype(dtype=np.double, order='F')
Xf = np.concatenate((q_end, qd_end)).astype(dtype=np.double, order='F')

pset = ParameterSetBase(
    p=p, N=N, X0=X0, Xf=Xf, name=run_name, s_max=s_max, k=k, abs_tol=abs_tol, 
    rel_tol=rel_tol, method_name=method_name, max_steps=max_steps, ns_points=ns_points, 
    j_type=j_type, fp_urdf=fp_urdf, scenario_name=SCENARIO_NAME, q_traj=q_traj
)

result = Experiment.run_single_static(pset=pset, timeout=timeout, use_jacobian=True, print_debug=False)

print(f"Finished solving in: {result.t_solve} seconds.")

# Do post processing of AGHF solution to see evolution of things along s
sol = result.sol
post_aghf = PostAghf(result)

# Uncomment this block to plot the quantities along the s-span
# action_functional_vec, vel_dif_vec, accel_dif_vec, norm_dps_values_ds_vec, \
#     u_sq_vec, real_s_span, Q_mat, Qd_mat, Qdd_mat, U_mat = post_aghf.compute_along_s(dt, sol)
# plot_along_s_span_from_vecs(action_functional_vec, vel_dif_vec, accel_dif_vec, 
#     norm_dps_values_ds_vec, u_sq_vec, real_s_span)

# Obtain the final solution
ps_values = sol[-1, :]
t_interp = np.linspace(-1, 1, int((2 / dt) + 1))

# Compute u_sq of the aghf solution
q, qd, qdd = post_aghf.get_q_qd_qdd(ps_values, t_interp)
U_sol = post_aghf.compute_u_matrix(q, qd, qdd)
X_aghf_t, _ = post_aghf.get_X_Xd_spec_t(ps_values, t_interp)
u_sq_aghf, _ = post_aghf.compute_u_sq(q, qd, qdd, dt)
print("u^2 of the AGHF solution: ", u_sq_aghf)

# Forward simulate the AGHF solution
k_pos = 100
k_vel = 100
method_name_int = 'BDF'  # RK45
lam_ode = lambda tau, x: ode_fwd_integrate(tau, x, U_sol, t_interp, post_aghf.model, 
                                           X_aghf_t, t_interp, k_pos, k_vel)
X0 = post_aghf.X0
t_simul = [t_interp[0], t_interp[-1]]
solution = solve_ivp(lam_ode, t_simul, X0.squeeze(), method=method_name_int, t_eval=t_interp)

# Get actual u
U_actual = np.zeros(shape=(t_interp.shape[0], post_aghf.N))

for ii in range(solution.y.shape[1]):
    U_actual[ii, :] = get_u_curr(t_interp[ii], solution.y[:, ii], U_sol, t_interp, 
                                 post_aghf.model, X_aghf_t, t_interp, k_pos, k_vel)

# Calculate number of rows needed
num_cols = 4
num_rows = math.ceil(post_aghf.N / num_cols)

if do_plot:
    # Plot all trajectories (q)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows))
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

    # Plot all velocities (qd)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows))
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

    # Plot u at each joint
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows))
    axs = axs.flatten()
    for ii in range(post_aghf.N):
        ax = axs[ii]
        ax.plot(solution.t, U_actual[:, ii], label=f'u{ii}: Actual/real')
        ax.plot(solution.t, U_sol[:, ii], label=f'u{ii}: extracted from AGHF solution')
        ax.legend()
    for ax in axs[post_aghf.N:]:
        fig.delaxes(ax)
    plt.tight_layout()

norm_u_real = np.sum(U_actual * U_actual, axis=1)
u_sq_real = dt * np.sum(norm_u_real)
print("u^2 real when using a controller: ", u_sq_real)

# Obtain the inf norm
x_right_sol = solution.y[:, -1].reshape(-1, 1)
dif_bc = Xf - x_right_sol
dif_pos = Xf[:N, 0] - x_right_sol[:N, 0]
dif_vel = Xf[N:, 0] - x_right_sol[N:, 0]

# Get the norms of the dynamics violation
inf_norm_all = np.linalg.norm(dif_bc, ord=np.inf)
inf_norm_pos = np.linalg.norm(dif_pos, ord=np.inf)
inf_norm_vel = np.linalg.norm(dif_vel, ord=np.inf)

print("Inf norm all: ", inf_norm_all)
print("Inf norm pos: ", inf_norm_pos)
print("Inf norm vel: ", inf_norm_vel)

plt.show()