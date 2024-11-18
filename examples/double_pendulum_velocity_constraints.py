# This example shows how to use PHLAME to generate a trajectory 
# to move the double pendulum from some initial to final position while enforcing state constraints

import ipdb
import numpy as np
import matplotlib.pyplot as plt
import os

from phlame.cheb import Cheb
from phlame.aghf import Aghf
from phlame.aghf import PostAghf
from phlame.parameter_set import generate_parameter_sets
from phlame.parameter_set import ParameterSetActivatedState
from phlame.experiment import Experiment
from phlame.utils import cleanup_folder


# Parameter for ParameterSetBase
p = 11 # Number of Chebyshev nodes
k = 1e5 # Penalty parameter
s_max = 100 # Maximum value of s

# ode solver settings
abs_tol = 1e-10
rel_tol = 1e-10
method_name = "cvode"
max_steps = 1e5
ns_points = int(1e2)
run_name = "double-pendulum-swingup_statecons" # Name of the run

dt = 1e-2 # time step size for integrating the u^2

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

# Set up constraint hyper-parameters
k_cons = 1e5
c_cons = 50

# Setup state limits
x_max = np.array([50,50,3,1], dtype=np.double, order='F')

# Parameters for Experiment
timeout = 60
print_debug = False

pset = ParameterSetActivatedState(p, N, X0, Xf, run_name, s_max, 
                            k, abs_tol, rel_tol, method_name, 
                            max_steps, ns_points, j_type, fp_urdf,
                            c_cons, k_cons, -x_max, x_max)

result = Experiment.run_single_static(pset, timeout, mode="general")
print(f"Finished solving in: {result.t_solve} seconds.")

# Do post processing of AGHF solution to see evolution of things along s
post_aghf = PostAghf(result)

ps_values_ini = post_aghf.first_result.sol[0, :].reshape(-1, 1)
ps_values_fin = post_aghf.first_result.sol[-1, :].reshape(-1, 1)
t_interp = np.linspace(-1, 1, int((2 / dt) + 1) )

# q_ini, qd_ini, qdd_ini = post_aghf.get_q_qd_qdd(ps_values_ini, t_interp)
q_fin, qd_fin, qdd_fin = post_aghf.get_q_qd_qdd(ps_values_fin, t_interp)

## Plot q_fin, qd_fin in a 2x2 plot
fig, axs = plt.subplots(2, 2, sharex=True)  # 2x2 subplots sharing the x-axis
plt.xlabel("Time")
axs[0, 0].plot(t_interp, q_fin[:, 0])
axs[0, 0].grid()
axs[0, 0].set_ylabel("q1")

axs[0, 1].plot(t_interp, q_fin[:, 1])
axs[0, 1].grid()
axs[0, 1].set_ylabel("q2")

axs[1, 0].plot(t_interp, qd_fin[:, 0])
axs[1, 0].grid()
axs[1, 0].set_ylabel("q1 dot")

axs[1, 1].plot(t_interp, qd_fin[:, 1])
axs[1, 1].grid()
axs[1, 1].set_ylabel("q2 dot")

plt.tight_layout()

plt.show()