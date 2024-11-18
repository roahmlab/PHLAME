from scipy.optimize import linprog
from sympy import Matrix
from sympy import symbols
from sympy import chebyshevt
from sympy import diff
from sympy import lambdify
from scipy.special import chebyt
from scipy.linalg import block_diag
import numpy as np
import ipdb

from .utils import apply_func_to_vector
from .utils import interpolate_matrix

class Cheb:
    def __init__(self, p, N, X0, Xf):
        """
        Args:
            - p (int): degree of the polynomial
            - N (int): number of bodies
            - X0 (np.ndarray): (2*N, 1): [q; qd] boundary condition left
            - Xf (np.ndarray): (2*N, 1): [q; qd] boundary condition right
        """
        self.p = p
        self.N = N
        self.X0 = X0
        self.Xf = Xf
        self.cheb_t = self._compute_cheb_t() # NOTE: the time vector is [1, ... , -1]
        self.D = self._compute_D()
        self.D2 = np.linalg.matrix_power(self.D, 2)
        self.D_ps = self.D[1 : -1, 1 : -1].astype(dtype=np.double, order='F')
        self.D2_ps = self.D2[1 : -1, 1 : -1].astype(dtype=np.double, order='F')

    def _compute_cheb_t(self):
        """
        NOTE: The time vector is [1, ... , -1]
        Computes chebnodes
        Returns:
            - chebnodes (np.ndarray): (l,)
        """
        return np.cos(np.pi * np.arange(self.p + 1) / self.p).reshape(-1)
    
    def _compute_D(self):
        """
        Computes the differentiation matrix
        Args:
            - cheb_t (np.ndarray): Times at which we will evaluate the value of the function.
            - p (int): Degree of the chebyshev polynomial
        Returns:
            - D (np.ndarray): (p+1, p+1)
        """

        D = np.zeros((self.p+1, self.p+1))
        c = np.ones(self.cheb_t.shape)
        c[0] = 2
        c[-1] = 2

        for i in range(self.p+1):
            for j in range(self.p+1):
                if (i == 0 and j == 0):
                    D[i,j] = (2 * self.p**2 + 1)/6
                elif( i == self.p and j == self.p):
                    D[i,j] = -(2 * self.p**2 + 1)/6
                elif(i==j):
                    D[i,j] = -self.cheb_t[i]/(2*(1 - self.cheb_t[i]**2)) 
                else:
                    D[i,j] = c[i]/c[j] * (-1)**(i+j)/(self.cheb_t[i] - self.cheb_t[j])
        return D

    def get_q_mat(self, t):
        """
        Each row represent time. Each column represent a different degree of the polynomial
        Args:
            - t (np.ndarray): (l,)
        Returns:
            - q_mat (np.ndarray): (len(t), p+1)
        """
        q_mat = np.zeros(shape=(t.shape[0], self.p+1))
        for ii in range(self.p+1):
            q_mat[:, ii] = chebyt(ii)(t)
        return q_mat
        
    def get_cheb_coeffs(self, Xi_col):
        """
        Given pseudo-spectral values computes the corresponding chebyshev coefficients.
        Args:
            - Xi_col (np.ndarray): (p+1,)
        Returns:
            - coeffs (np.ndarray): (p+1,)
        """
        p = len(Xi_col) - 1

        barc = np.ones(p+1)
        barc[0] = 2
        barc[-1] = 2

        coeffs = np.zeros(Xi_col.shape)
        for k in range(p+1):
            coeffs[k] = 2/(barc[k] * p) * np.dot(np.cos(k * np.pi * np.arange(p+1) / p), Xi_col/barc)
        return coeffs

    def encode(self, ps_values):
        """
        Goes from ps_values(the only ones we are evolving) to Xi, a matrix (p+1, 2*N)
        Args:
            - ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
        Returns:
            - Xi (np.ndarray): ( (p+1) , 2 * N)
        """
        aps_helper = np.reshape(ps_values, (self.p - 1, self.N * 2), 'F')
        Xi = np.vstack([self.Xf.T, aps_helper, self.X0.T])
        return Xi
    
    def decode(self, Xi):
        """
        Goes from Xi to ps_values
        Args:
            - Xi (np.ndarray): ( (p+1) , 2 * N)
        Returns:
            - ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
        """
        removed_bcs = Xi[ 1 : -1, :]
        ps_values = removed_bcs.T.flatten().reshape(-1, 1)
        return ps_values
    
    def decode_evol(self, Xi_e):
        """
        Goes from Xi_e to ps_values, Xi_e corresponds to the values that we evolve
        Args:
            - Xi_e (np.ndarray): ( (p-1), 2*N): states
        Returns:
            - ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
        """
        return Xi_e.T.flatten().reshape(-1, 1)
    
    def get_Xi_dXi_ddXi_evol(self, ps_values):
        """
        Given ps_values compute Xi_e, dXi_e, ddXi_e that correspond to 
        Xi, dXi, ddXi but without the boundaries
        Args:
            - ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
        Returns:
            - Xi_e (np.ndarray): ( (p-1), 2*N): states
            - dXi_e (np.ndarray): ( (p-1), 2*N): first derivative of the states
            - ddXi_e (np.ndarray): ( (p-1), 2*N): second derivative of the states
        """
        Xi = self.encode(ps_values)
        dXi = np.dot(self.D, Xi).astype(dtype=np.double, order='F')
        ddXi = np.dot(self.D2, Xi).astype(dtype=np.double, order='F')

        Xi_e = np.asfortranarray(Xi[ 1 : -1, :])
        dXi_e = np.asfortranarray(dXi[ 1 : -1, :])
        ddXi_e = np.asfortranarray(ddXi[ 1 : -1, :])

        return Xi_e, dXi_e, ddXi_e

    def get_joints_trajectory(self, ps_values, t):
        """
        Generates the trajectory of q(t), qd(t) given the ps_values.
        Args:
            - ps_values (np.ndarray): pseudo spectral values that we are evolving: ( (p-1) * 2*N , 1 )
            - t (np.ndarray): time at which we want values of the trajectory, has to be [-1, 1] : (l,)
        Returns:
            - result_traj (np.ndarray): (t.shape[0], 2*N): result_traj[i, j] represents the position or velocity
                of joint `j` at time t[i]. Position if i < N, else speed.
            - q_states (np.ndarray): (t.shape[0], N): q_states[i, j] represents the position of joint `j` at time
                t[i]
            - qd_states (np.ndarray): (t.shape[0], N): qd_states[i, j] represents the speed of joint `j` at time
                t[i]
        """

        # make sure that t first value is -1 and last value is 1, else raise an exception
        if t[0] != -1 or t[-1] != 1:
            raise Exception("t has to be from -1 to 1")

        Xi = self.encode(ps_values)
        result_traj = np.zeros(shape=(t.shape[0], 2*self.N))
        q_mat = self.get_q_mat(t)

        for ii in range(2 * self.N):
            Xi_col = Xi[:, ii]
            result_traj[:, ii] = q_mat @ self.get_cheb_coeffs(Xi_col)

        q_states = result_traj[:, :self.N]
        q_dot_states = result_traj[:, self.N:]
        return result_traj, q_states, q_dot_states
    
    def get_X_Xd_spec_t(self, ps_values, t):
        """
        NOTE: This follows the control extraction the way it is done in Matlab.
        Args:
            - ps_values (np.ndarray): pseudo spectral values that we are evolving: ( (p-1) * 2*N , 1 )
            - t (np.ndarray): time at which we want values of the trajectory, has to be [-1, 1] : (l,)
        Returns:
            - X_spec_t (np.ndarray): (len(t), 2*N)
            - Xd_spec_t (np.ndarray): (len(t), 2*N)
        """

        # Containers where data will be stored
        X_spec_t = np.zeros(shape = (t.shape[0], self.N * 2) ) # q
        Xd_spec_t = np.zeros(shape = (t.shape[0], self.N * 2) ) # qd
        
        # Get the pseudo-spectral values at the chebyshev nodes
        Xi = self.encode(ps_values)
        dXi = np.dot(self.D, Xi)
        q_mat = self.get_q_mat(t)

        for ii in range(self.N * 2):
            X_spec_t[:, ii] = q_mat @ self.get_cheb_coeffs( Xi[:, ii] )
            Xd_spec_t[:, ii] = q_mat @ self.get_cheb_coeffs(dXi[:, ii] )
        return X_spec_t, Xd_spec_t

    def get_q_qd_qdd(self, ps_values, t):
        """
        Args:
            - ps_values (np.ndarray): pseudo spectral values that we are evolving: ( (p-1) * 2*N , 1 )
            - t (np.ndarray): time at which we want values of the trajectory, has to be [-1, 1] : (l,)
        Returns:
            - q (np.ndarray): (len(t), N)
            - qd (np.ndarray): (len(t), N)
            - qdd (np.ndarray): (len(t), N)
        """
        X_spec_t, Xd_spec_t = self.get_X_Xd_spec_t(ps_values, t)
        q = X_spec_t[:, : self.N]
        qd = X_spec_t[:, self.N : ]
        qdd = Xd_spec_t[:, self.N : ]
        return q, qd, qdd

    def generate_ps_traj_line(self):
        """
        Generates an initial trajectory that is a line connecting Xf and X0.
        Returns:
            - init_ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
        """
        Nt = self.p + 1
        
        # only for positions, Xi original has N more columns corresponding to velocity
        init_Xi_pos = np.zeros(shape=(Nt, self.N))

        for ii in range(self.N):
            init_Xi_pos[:, ii] = np.linspace(self.Xf[ii][0], self.X0[ii][0], Nt)
        
        init_Xi_vel = self.D @ init_Xi_pos
        
        init_Xi = np.hstack(( init_Xi_pos, init_Xi_vel))
        init_ps_values = self.decode(init_Xi)
        return init_ps_values
    
    def generate_Xi_traj_line_bc_unsatisfied(self):
        """
        Generates a initial trajectory that is a line connecting Xf and X0.
        Returns:
            - init_ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
        """
        Nt = self.p + 1
        
        # only for positions, Xi original has N more columns corresponding to velocity
        init_Xi_pos = np.zeros(shape=(Nt, self.N))

        for ii in range(self.N):
            init_Xi_pos[:, ii] = np.linspace(self.Xf[ii][0], self.X0[ii][0], Nt)
        
        init_Xi_vel = self.D @ init_Xi_pos
        
        init_Xi = np.hstack(( init_Xi_pos, init_Xi_vel))
        return init_Xi
    
    def generate_ps_traj_min_accel_v2(self, test=False):
        """
        Generate a initial trajectory that has the minimum acceleration.
        This is achieved by solving a linear program where the decision variables 
        are the the Pseudo-Spectral values for position only. `_old_generate_ps_traj_min_accel`
        has as decision variables the whole Xi.
        
        Returns:
            - init_ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
        """
        Nt = self.p + 1
        n_dvs = 1 + Nt * self.N

        # Cost function
        c = np.zeros(n_dvs)
        c[-1] = 1
        
        # Equality constraints
        A_eq = np.zeros( shape = (4 * self.N, n_dvs ) )
        b_eq = np.zeros( shape = ( 4 * self.N ) )
        
        # q0 and qf
        b_eq[ 0 : self.N ] = self.Xf[0 : self.N, 0].reshape(-1)
        b_eq[ self.N : 2 * self.N ] = self.X0[0 : self.N, 0].reshape(-1)
        
        # qd_0 and qd_f
        b_eq[ 2 * self.N : 3 * self.N ] = self.Xf[self.N : 2 * self.N, 0].reshape(-1)
        b_eq[ 3 * self.N : 4 * self.N ] = self.X0[self.N : 2 * self.N, 0].reshape(-1)
        # Filling A_eq matrix : positions
        for ii in range(self.N):
            
            # for qf
            A_eq[ii, ii * Nt] = 1
            
            # for q0
            A_eq[ii + self.N, ii*Nt + self.p] = 1
        
        # Filling velocities
        for ii in range(self.N):
            idx_row = 2 * self.N + ii
            
            # for qd0
            A_eq[idx_row, ii * Nt : (ii+1)*Nt] = self.D[0, :]
            
            # for qdf
            A_eq[idx_row + self.N, ii * Nt : (ii+1)*Nt] = self.D[-1, :]

        # Inequality constraints
        A_ub = np.zeros(shape = ( 2 * Nt * self.N, n_dvs ) )
        b_ub = np.zeros( 2 * Nt * self.N)
        A_ub[:, -1] = -1
        B_block = block_diag( *( [self.D2]*self.N ) )
        A_ub[ 0 : Nt * self.N, 0 : Nt * self.N ] = B_block
        A_ub[ Nt * self.N : 2 * Nt * self.N, 0 : Nt * self.N ] = -B_block

        # solve linear program
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[None, None])
        x_vec = res.x[:-1]
        Xi_p1 = x_vec.reshape( (self.N, self.p+1) ).T
        Xi_p2 = self.D @ Xi_p1
        Xi = np.hstack((Xi_p1, Xi_p2))
        init_ps_values = self.decode(Xi)

        if test:
            return init_ps_values, A_ub, b_ub, A_eq, b_eq
        else:
            return init_ps_values

    def generate_ps_traj_min_accel(self):
        """
        Generate a initial trajectory that has the minimum acceleration.
        This is achieved by solving a linear program where the decision variables 
        are the pseudo-spectral values.
        """

        Nt = self.p + 1
        
        # the last decision variable is `t` from the epigraph trick
        n_dvs = 1 + Nt * self.N * 2
        

        # Cost function
        c = np.zeros(n_dvs)
        c[-1] = 1
        
        # Equality constraints
        A_eq = np.zeros(shape = (6 * self.N, n_dvs) )
        b_eq = np.zeros(6 * self.N)
        
        # q0 and qf
        b_eq[ 0 : self.N ] = self.Xf[0 : self.N, 0].reshape(-1)
        b_eq[ self.N : 2 * self.N ] = self.X0[0 : self.N, 0].reshape(-1)
        
        # qd_0 and qd_f, that reside in Xi
        b_eq[ 2 * self.N : 3 * self.N ] = self.Xf[self.N : 2 * self.N, 0].reshape(-1)
        b_eq[ 3 * self.N : 4 * self.N ] = self.X0[self.N : 2 * self.N, 0].reshape(-1)
        
        # qd_0 and qd_f, that reside in dXi
        b_eq[ 4 * self.N : 5 * self.N ] = self.Xf[self.N : 2 * self.N, 0].reshape(-1)
        b_eq[ 5 * self.N : 6 * self.N ] = self.X0[self.N : 2 * self.N, 0].reshape(-1)

        # Filling A_eq matrix : positions and velocities
        for ii in range(self.N):
            
            # for qf
            A_eq[ii, ii * Nt] = 1
            
            # for q0
            A_eq[ii + self.N, ii*Nt + self.p] = 1
            
            # for qd_f
            A_eq[ii + 2 * self.N, ii*Nt + Nt * self.N] = 1
            
            # for qd_0
            A_eq[ii + 3 * self.N, ii*Nt + Nt * self.N + self.p] = 1


        # Filling velocities
        for ii in range(self.N):
            idx_row = 4 * self.N + ii
            # for qd0
            A_eq[idx_row, ii * Nt : (ii+1)*Nt] = self.D[0, :]
            # for qdf
            A_eq[idx_row + self.N, ii * Nt : (ii+1)*Nt] = self.D[-1, :]

        # Inequality constraints
        A_ub = np.zeros(shape = ( Nt * 2 * self.N, n_dvs ) )
        b_ub = np.zeros(Nt * 2 * self.N)
        A_ub[:, -1] = -1
        B_block = block_diag( *( [self.D2]*self.N ) )
        A_ub[ 0 : Nt * self.N, 0 : Nt * self.N ] = B_block
        A_ub[ Nt * self.N : 2 * Nt * self.N, 0 : Nt * self.N ] = -B_block
        # This is only bounding the accelerations in D^2 @ Xi

        
        Xi = self.encode(self.generate_ps_traj_line())
        Z = np.concatenate( (Xi.T.flatten(), np.array([100]) ) )
        np.set_printoptions(linewidth=200)

        # solve linear program
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[None, None])
        x_vec = res.x[:-1]
        sol = x_vec.reshape( (2*self.N, self.p+1) ).T
        
    def generate_random_ps_traj(self):
        """
        Generate a random initial trajectory that satisfies the boundary conditions.
        Returns:
            - init_ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
        """
        Nt = self.p + 1
        
        # Form Xi but only for positions
        init_Xi_pos = np.random.randn(Nt, self.N) * 10
        
        # Enforce BC
        init_Xi_pos[0, :] = self.Xf[0][0]
        init_Xi_pos[-1, :] = self.X0[0][0]
        
        # Get the velocity
        init_Xi_vel = self.D @ init_Xi_pos

        init_Xi = np.hstack(( init_Xi_pos, init_Xi_vel))
        init_ps_values = self.decode(init_Xi)
        return init_ps_values
    
    def generate_ps_traj_from_q_traj(self, q_traj):
        """
        Given a trajectory between t0=0 and tf=2, we interpolate to obtain
        the ps values. Then, use the D matrix to obtain its derivative.
        Args:
            - q_traj: (num_times, N)
        Returns:
            - init_ps_values: ( (p-1) * 2*N , 1 )
        """
        n_samples_traj = q_traj.shape[0]
        t_traj = np.linspace(0, 2, n_samples_traj)
        query_times = self.cheb_t + 1 # This goes from 2 -> 0
        
        # NOTE: In PS world our trajectories(Xi) go from t=1 to t=-1
        # in the trajectory q_traj, it goes from 0 to a positive number.
        # However, when we form Xi; the first row corresponds to the final state Xf
        # and the last row corresponds to the first state. We have to reflect that here

        # The fact that query_times goes from 2 -> 0 implies that interp_q first row
        # correspond to the final time and that last row to the first time
        interp_q = interpolate_matrix(q_traj, t_traj, query_times)
        interp_dq = self.D @ interp_q

        init_Xi = np.hstack( ( interp_q, interp_dq))
        init_ps_values = self.decode(init_Xi)
        return init_ps_values
