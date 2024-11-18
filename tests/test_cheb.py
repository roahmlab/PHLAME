import numpy as np
import os
from scipy.io import loadmat
import unittest
import ipdb

from phlame.cheb import Cheb
from phlame.utils import interpolate_matrix

class TestChebFunctions(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.join(os.path.dirname(__file__), "..")
        fp_data_cheb = os.path.join(base_dir, "tests/matlab_files/data_cheb.mat")
        self.data = loadmat(fp_data_cheb)
        self.p = int( self.data['p'][0][0] )
        self.N = int( self.data['N'][0][0] )
        self.X0 = np.reshape(self.data['X0'] , (2*self.N, 1)).astype(np.double, order='F')
        self.Xf = np.reshape(self.data['Xf'] , (2*self.N, 1)).astype(np.double, order='F')
        self.cheb_ins = Cheb(self.p, self.N, self.X0, self.Xf)

    def test_compute_cheb_t(self):
        """
        _compute_cheb_t: computes chebnodes(now called cheb_t)
            TEST: Compute cheb_t in matlab using cheb_helper_function and store it. Then compare with the 
                result in python.
        """
        print("####### Running test_cheb tests")
        cheb_t_py = self.cheb_ins._compute_cheb_t()
        cheb_t_ma = self.data['cheb_t'].reshape(-1)
        
        np.testing.assert_array_almost_equal(cheb_t_py, cheb_t_ma)
    
    def test_compute_D(self):
        """
        _compute_D: computes the differentiation matrix
            TEST: idem as _compute_cheb_t
        """
        D_ma = self.data['D']
        D_py = self.cheb_ins._compute_D()
        
        np.testing.assert_array_almost_equal(D_py, D_ma)
    
    def test_get_q_mat(self):
        """
        get_q_mat:
            Each row represent time. Each column represent a different degree of the polynomial
            TEST: idem as _compute_D
        """
        t = self.data['t'].reshape(-1)
        q_mat_ma = self.data['q_mat']
        q_mat_py = self.cheb_ins.get_q_mat(t)
        np.testing.assert_array_almost_equal(q_mat_py, q_mat_ma)

    def test_get_cheb_coeffs(self):
        """
        _get_cheb_coeffs: Given pseudo-spectral values computes the corresponding chebyshev coefficients.
            TEST: idem as -compute_D, comparing with chebinterp(from matlab).
        """

        Xi_col = self.data['Xi_col'].reshape(-1)
        coeffs_ma = self.data['coeffs'].reshape(-1)
        coeffs_py = self.cheb_ins.get_cheb_coeffs(Xi_col)
        np.testing.assert_array_almost_equal(coeffs_ma, coeffs_py)

    def test_encode_decode(self):
        """
        encode, decode: try encoding and decoding and making sure we obtain the same input.
        """
        n_ps = (self.p-1) * 2 * self.N
        ps_values = np.random.randn(n_ps, 1)
        Xi = self.cheb_ins.encode(ps_values)
        ps_values_recon = self.cheb_ins.decode(Xi)
        np.testing.assert_array_almost_equal(ps_values, ps_values_recon)

    def test_get_joints_trajectory(self):
        """
        get_joints_trajectory: Generates a trajectory from ps_values and t
            TEST: make sure that the ps_values are part of the trajectory at times
        NOTE: This test might not be extensive enough as we don't check that the boundary conditions are satisfied.
        """
        n_ps = (self.p-1) * 2 * self.N
        ps_values = np.random.randn(n_ps, 1)
        Xi = self.cheb_ins.encode(ps_values)
        num_times = int(1e4)
        t = np.linspace(-1, 1, num_times)
        result_traj, q_states, q_dot_states = self.cheb_ins.get_joints_trajectory(ps_values, t)
        # interpolate the values of the trajectory at the times of the Xi `cheb_t`
        Xi_recon = interpolate_matrix(result_traj, t, self.cheb_ins.cheb_t)
        np.testing.assert_array_almost_equal(Xi, Xi_recon)
        
    def test_generate_ps_traj_line(self):
        """
        generate_ps_traj_line: Generates a initial trajectory that is a line connecting Xf and X0.
            TEST: make sure that the ps_values are part of the trajectory at times        
        """
        # generates ps_values associated connecting Xf with X0(time is flipped)
        ps_values = self.cheb_ins.generate_ps_traj_line()
        Xi = self.cheb_ins.encode(ps_values)
        num_times = int(1e4)
        t = np.linspace(-1, 1, num_times)
        result_traj, q_states, q_dot_states = self.cheb_ins.get_joints_trajectory(ps_values, t)
        Xi_recon = interpolate_matrix(result_traj, t, self.cheb_ins.cheb_t)
        np.testing.assert_array_almost_equal(Xi, Xi_recon)
    
    def generate_ps_traj_from_q_traj(self):
        """
        generate_ps_traj_from_q_traj: 
            TEST: make sure that the ps_values are part of the trajectory at times        
        """
        num_times = int(1e4)
        t = np.linspace(-1, 1, num_times)
        q_traj = np.random.randn(num_times, self.N)
        ps_values = self.cheb_ins.generate_ps_traj_from_q_traj(q_traj)
        result_traj, q_states, q_dot_states = self.cheb_ins.get_joints_trajectory(ps_values, t)
        Xi = self.cheb_ins.encode(ps_values)
        Xi_recon = interpolate_matrix(result_traj, t, self.cheb_ins.cheb_t)
        np.testing.assert_array_almost_equal(Xi, Xi_recon)

    def test_generate_ps_traj_min_accel_v2(self):
        # Make sure that the initial condition line satifies all the constraints of the LP
        p = 5
        N = 3
        X0 = np.zeros(shape=(2*N, 1), dtype=np.double, order='F')
        X0[0][0] = -np.pi/2
        Xf = np.zeros(shape=(2*N, 1), dtype=np.double, order='F')
        Xf[0][0] = np.pi/2
        cheb = Cheb(p, N, X0, Xf)
        fake_Xi = cheb.encode(cheb.generate_ps_traj_line())
        Xi_p1 = fake_Xi[:, 0 : cheb.N]
        Xi_p2 = cheb.D @ Xi_p1
        # Update velocities with the correct one after applying D
        cheb.Xf[cheb.N : ] = Xi_p2[0, :].reshape(-1, 1)
        cheb.X0[cheb.N : ] = Xi_p2[-1, :].reshape(-1, 1)

        Z = np.concatenate( ( Xi_p1.T.flatten(), np.array([1_000_000]) ) )

        _, A_ub, b_ub, A_eq, b_eq = cheb.generate_ps_traj_min_accel_v2(test=True)

        np.testing.assert_array_almost_equal(A_eq @ Z, b_eq)

        res_ub = A_ub@Z <= b_ub
        np.testing.assert_array_almost_equal(res_ub, np.ones_like(res_ub))


if __name__ == "__main__":
    unittest.main()