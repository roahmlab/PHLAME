
import ipdb
import numpy as np
import os
from scipy.io import loadmat
import unittest
import ipdb

import aghf_pybind
from phlame.cheb import Cheb
from phlame.aghf import Aghf

import ipdb
import numpy as np
import matplotlib.pyplot as plt
import os
import unittest

from phlame.aghf import PostAghf
from phlame.parameter_set import generate_parameter_sets
from phlame.parameter_set import ParameterSetActivatedState
from phlame.experiment import Experiment
from phlame.utils import cleanup_folder


class TestAghfSolnDoublePendulumActivatedStateConstraints(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.join(os.path.dirname(__file__), "..")
        fp_data = os.path.join(base_dir, "tests/matlab_files/data_double_pendulum_aghf_soln_state_constraints.mat")
        self.data = loadmat(fp_data)
        self.fp_urdf = os.path.abspath( os.path.join(base_dir,"urdfs/pendulums/pen-2.urdf"))
        self.N = int(self.data['NB'][0][0])
        self.j_type = (2 * np.ones(shape=(self.N, 1))).astype(dtype=np.double, order='F')        
        self.p = int( self.data["N"][0][0] )

        self.x0 = self.data["x0"].astype(dtype=np.double, order='F')
        self.xf = self.data["xf"].astype(dtype=np.double, order='F')
        self.s_span = self.data["s_span"].reshape(-1).astype(dtype=np.double, order='F')
        self.k = self.data["K"][0][0]
        self.c_cons = self.data["c_cons"][0][0]
        self.chebnodes = self.data["chebnodes"].reshape(-1).astype(dtype=np.double, order='F')
        self.D = self.data["D"].astype(dtype=np.double, order='F')
        self.D2 = self.data["D2"].astype(dtype=np.double, order='F')
        self.k_cons = self.data["k_cons"][0][0]
        self.x_max = self.data["x_max"].reshape(-1).astype(dtype=np.double, order='F')
        self.aps = self.data["aps"].reshape(-1).astype(dtype=np.double, order='F')
        self.q1_t = self.data["q1_t"].reshape(-1).astype(dtype=np.double, order='F')
        self.q2_t = self.data["q2_t"].reshape(-1).astype(dtype=np.double, order='F')
        self.q1d_t = self.data["q1d_t"].reshape(-1).astype(dtype=np.double, order='F')
        self.q2d_t = self.data["q2d_t"].reshape(-1).astype(dtype=np.double, order='F')
        self.dt = self.data["dt"][0][0]
        self.t = self.data["t"].reshape(-1).astype(dtype=np.double, order='F')
        self.abs_tol = self.data["abs_tol"][0][0]
        self.rel_tol = self.data["rel_tol"][0][0]
        self.init_ps_values = self.data["init_ps_values"].astype(dtype=np.double, order='F').reshape(-1)

        self.method_name = "cvode"
        self.max_steps = 1e5
        self.ns_points = int(1e2)
        self.X0 = np.reshape(self.x0, newshape=(2*self.N, 1), order='F')
        self.Xf = np.reshape(self.xf, newshape=(2*self.N, 1), order='F')

        # Parameters for Experiment
        self.timeout = 60
        print_debug = False

    def test_run(self):
        print("####### Running test_double_pendulum_state_constraints_fullsolve_jacobian")
        pset = ParameterSetActivatedState(self.p, self.N, self.X0, self.Xf, "test", self.s_span[-1], 
                                    self.k, self.abs_tol, self.rel_tol, self.method_name, 
                                    self.max_steps, self.ns_points, self.j_type, self.fp_urdf,
                                    self.c_cons, self.k_cons, -self.x_max, self.x_max)
        # Override init_ps_values with the values used in matlab. This functionality should be moved
        # to ParameterSetBase soon.
        pset.init_ps_values = self.init_ps_values
        
        res = Experiment.run_single_static(pset, self.timeout, mode="general", use_jacobian=False)
        print("Time solve without jacobian: ", res.t_solve)
        res_jac = Experiment.run_single_static(pset, self.timeout, mode="general", use_jacobian=True)
        print("Time solve jacobian: ", res_jac.t_solve)
        
        # Check that the solutions with the jacobian and without the jacobian are the same
        np.testing.assert_allclose(res.sol[-1, :], res_jac.sol[-1, :])
        
        post_aghf = PostAghf(res)

        ps_values_ini = post_aghf.first_result.sol[0, :].reshape(-1, 1)
        ps_values_fin = post_aghf.first_result.sol[-1, :].reshape(-1, 1)
        dt = self.dt
        num_points = int((2 / dt) + 1)
        t = np.linspace(-1, 1, num_points)

        # q_ini, qd_ini, qdd_ini = post_aghf.get_q_qd_qdd(ps_values_ini, t)
        q_fin, qd_fin, qdd_fin = post_aghf.get_q_qd_qdd(ps_values_fin, t)

        np.testing.assert_allclose(q_fin[:, 0], self.q1_t, atol=1e-6)
        np.testing.assert_allclose(q_fin[:, 1], self.q2_t, atol=1e-6)
        np.testing.assert_allclose(qd_fin[:, 0], self.q1d_t, atol=1e-6)
        np.testing.assert_allclose(qd_fin[:, 1], self.q2d_t, atol=1e-6)

        ## Plot q_fin, qd_fin in a 2x2 plot
        fig, axs = plt.subplots(2, 2, sharex=True)  # 2x2 subplots sharing the x-axis
        plt.xlabel("Time")
        axs[0, 0].plot(t, q_fin[:, 0], label="python/c++")
        axs[0, 0].plot(self.t, self.q1_t, label="matlab")
        axs[0, 0].legend()
        axs[0, 0].grid()
        axs[0, 0].set_ylabel("q1")

        axs[0, 1].plot(t, q_fin[:, 1], label="python/c++")
        axs[0, 1].plot(self.t, self.q2_t, label="matlab")
        axs[0, 1].legend()
        axs[0, 1].grid()
        axs[0, 1].set_ylabel("q2")

        axs[1, 0].plot(t, qd_fin[:, 0], label="python/c++")
        axs[1, 0].plot(self.t, self.q1d_t, label="matlab")
        axs[1, 0].legend()
        axs[1, 0].grid()
        axs[1, 0].set_ylabel("q1 dot")

        axs[1, 1].plot(t, qd_fin[:, 1])
        axs[1, 1].plot(self.t, self.q2d_t, label="matlab")
        axs[1, 1].legend()
        axs[1, 1].grid()
        axs[1, 1].set_ylabel("q2 dot")


        plt.legend()
        # plt.show()
        
if __name__ == "__main__":
    unittest.main()