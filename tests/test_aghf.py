import ipdb
import numpy as np
import os
from scipy.io import loadmat
import unittest

import aghf_pybind
from phlame.cheb import Cheb
from phlame.aghf import Aghf

class TestAghfDoublePendulum(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.join(os.path.dirname(__file__), "..")
        fp_data_aghf_dp = os.path.join(base_dir, "tests/matlab_files/data_aghf_double_pendulum.mat")
        self.data = loadmat(fp_data_aghf_dp)
        self.fp_urdf = os.path.abspath( os.path.join(base_dir,"urdfs/pendulums/pen-2.urdf"))
        self.N = 2
        self.j_type = (2 * np.ones(shape=(self.N, 1))).astype(dtype=np.double, order='F')
        self.num_tests = int(self.data['num_tests'][0][0])
        self.test_X = self.data['test_X']
        self.test_Xd = self.data['test_Xd']
        self.test_Xdd = self.data['test_Xdd']
        self.test_k = self.data['test_k'].reshape(-1)
        self.rhs_res = self.data['rhs_res']
        self.num_tests_barr = int(self.data['num_tests_barr'][0][0])
        self.test_X_barr = self.data['test_X_barr']
        self.test_Xd_barr = self.data['test_Xd_barr']
        self.test_Xdd_barr = self.data['test_Xdd_barr']
        self.test_k_barr = self.data['test_k_barr'].reshape(-1)
        self.single_obs_mat = self.data['single_obs_mat']
        self.tau = self.data['tau'][0][0]
        self.rhs_res_barr = self.data['rhs_res_barr']

    
    def test_update_AGHF_no_obstacles(self):
        print("####### Running test_aghf")
        curr_dXi_ds = np.zeros(shape=(1, 4))
        mat_dXi_ds = np.zeros(shape=(4, self.num_tests))
        for ii in range(self.num_tests):
            k = self.test_k[ii]
            X = self.test_X[:, ii].reshape(1, -1)
            Xd = self.test_Xd[:, ii].reshape(1, -1)
            Xdd = self.test_Xdd[:, ii].reshape(1, -1)
            aghf_wrapper = aghf_pybind.AghfPybindWrapper(self.fp_urdf, k, self.j_type)
            
            aghf_wrapper.compute_AGHF_RHS(X, Xd, Xdd, 1, curr_dXi_ds)
            mat_dXi_ds[:, ii] = curr_dXi_ds
        np.testing.assert_allclose(mat_dXi_ds, self.rhs_res)

if __name__ == "__main__":
    unittest.main()