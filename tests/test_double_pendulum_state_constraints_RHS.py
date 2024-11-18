import ipdb
import numpy as np
import os
from scipy.io import loadmat
import unittest
import ipdb

import aghf_pybind
from phlame.cheb import Cheb
from phlame.aghf import Aghf

class TestAghfActivatedStateConstraintsDoublePendulum(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.join(os.path.dirname(__file__), "..")
        fp_data_aghf_dp = os.path.join(base_dir, "tests/matlab_files/data_aghf_state_constraint_activated.mat")
        self.data = loadmat(fp_data_aghf_dp)
        self.fp_urdf = os.path.abspath( os.path.join(base_dir,"urdfs/pendulums/pen-2.urdf"))
        self.N = 2
        self.j_type = (2 * np.ones(shape=(self.N, 1))).astype(dtype=np.double, order='F')

        self.num_tests = int(self.data['num_tests'][0][0])
        self.test_X = self.data['test_X'].astype(dtype=np.double, order='F')
        self.test_Xd = self.data['test_Xd'].astype(dtype=np.double, order='F')
        self.test_Xdd = self.data['test_Xdd'].astype(dtype=np.double, order='F')
        self.test_k = self.data['test_k'].reshape(-1).astype(dtype=np.double, order='F')
        self.rhs_res = self.data['rhs_res'].astype(dtype=np.double, order='F')
        self.c_cons = self.data['c_cons']        
        self.test_x_max = self.data["test_x_max"].astype(dtype=np.double, order='F')

    def test_update_AGHF_with_state_constraints(self):
        print("####### Running test_double_pendulum_state_constraints_RHS")
        curr_dXi_ds = np.zeros(shape=(1, 4))
        mat_dXi_ds = np.zeros(shape=(4, self.num_tests))
        for ii in range(self.num_tests):
            k = self.test_k[ii]
            k_cons = k
            c_cons = self.c_cons[0][0]
            X = self.test_X[ii, :].reshape(1, -1).astype(dtype=np.double, order='F')
            Xd = self.test_Xd[ii, :].reshape(1, -1).astype(dtype=np.double, order='F')
            Xdd = self.test_Xdd[ii, :].reshape(1, -1).astype(dtype=np.double, order='F')
            x_max = self.test_x_max[:, ii].reshape(-1).astype(dtype=np.double, order='F')
            aghf_wrapper = aghf_pybind.AghfPybindWrapper(self.fp_urdf, k, self.j_type)
            aghf_wrapper.set_activated_state_limits(-x_max, x_max, c_cons, k_cons)
            aghf_wrapper.compute_AGHF_RHS(X, Xd, Xdd, 1, curr_dXi_ds)
            mat_dXi_ds[:, ii] = curr_dXi_ds
        
        np.testing.assert_allclose(mat_dXi_ds, self.rhs_res)

if __name__ == "__main__":
    unittest.main()