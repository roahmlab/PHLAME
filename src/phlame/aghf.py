from .cheb import Cheb
from .parameter_set import ParameterSetBase
from .parameter_set import ParameterSetActivatedState
from .parameter_set import ParameterSetActivatedSimpleSphere
from .utils import load_from_pickle
import aghf_pybind
from .result import ResultBase
from .control_extraction import compute_u_matrix_static
from .control_extraction import compute_u_sq_doubleint

import ipdb
import numpy as np
import pinocchio as pin
import scipy.integrate as integrate


class Aghf(Cheb):
    def __init__(self, pset, print_debug=False, log_fn="log.txt", abs_tol_lagr=1e-10, 
                 rel_tol_lagr=1e-10, mode="general", use_jacobian=True):
        """
        Args:
            - pset (ParameterSetBase): instance of ParameterSetBase
            - print_debug (bool): True means to print debugging information and compute the lagrangian
            - log_fn (str): which log file to print the output of the print statement
            - abs_tol_lagr (float): absolute tolerance in the integration of the lagrangian
            - rel_tol_lagr (float): relative tolerance in the integration of the lagrangian
            - mode (str): "general" or in ["doubleint", "doubleint_vel_cons"]
            - use_jacobian (bool): whether to use the jacobian in the computation of the rhs_pde
        """

        super().__init__(p=pset.p, N=pset.N, X0=pset.X0, Xf=pset.Xf)
        self.model = pin.buildModelFromUrdf(pset.fp_urdf)

        self.dXi_e_ds = np.zeros(shape=(self.p-1, 2*self.N), dtype=np.double, order='F')
        self.J_rhs = np.zeros(shape=( (self.p-1) * 2 * self.N, (self.p-1) * 2 * self.N ), order='F')

        self.print_debug = print_debug
        self.k = pset.k
        self.fp_urdf = pset.fp_urdf
        self.j_type = pset.j_type
        self.mode = mode

        if hasattr(pset, 'k_cons'):
            # This is the code that will be used with the activated penalty function
            # b_i(x) = k_cons * g_i(x)^2 * S( g_i( x ) )
            self.k_cons = pset.k_cons

        if mode == "general":
            # This mode is used when solving the OCP for a .urdf
            if use_jacobian:
                self.aghf_wrapper = aghf_pybind.AghfPybindWrapper(self.fp_urdf, self.k, self.j_type, 
                                                                  self.D_ps, self.D2_ps)
            else:
                self.aghf_wrapper = aghf_pybind.AghfPybindWrapper(self.fp_urdf, self.k, self.j_type)
        elif mode in ["doubleint", "doubleint_vel_cons"]:
            self.aghf_wrapper = aghf_pybind.AghfPybindWrapper(self.k, self.N)
        else:
            raise ValueError('''mode has to be "general" or in ["doubleint", 
                             "doubleint_vel_cons"].''')

        self.log_fn = log_fn
        self.abs_tol_lagr = abs_tol_lagr
        self.rel_tol_lagr = rel_tol_lagr

        # TODO: this logic has to be modified for the case where we have both state and sphere 
        # obstacle constraints.
        # set up state constraints
        if isinstance(pset, ParameterSetActivatedState):
            print("Setting activated state constraints")
            self.aghf_wrapper.set_activated_state_limits(pset.x_lower, pset.x_upper, pset.c_cons, 
                                                         self.k_cons)
        elif isinstance(pset, ParameterSetActivatedSimpleSphere):
            print("Setting activated obstacle constraints")
            self.aghf_wrapper.set_activated_obstacles(pset.obs_mat, pset.c_cons, pset.k_cons)

        
    def rhs_pde_scikits(self, s, ps_values, d_ps_values_ds_sk):
        """
        This function is called by the scikits ODE solver to compute the right hand side of the 
        PDE.
        Args:
            - s (float): value of how far along we are in the deformation
            - ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
            - d_ps_values_ds_sk (np.ndarray): (2*N*(p-1)) : flattened dXi/ds not considering the boundary conditions
        Modified in-place:
            - d_ps_values_ds_sk (np.ndarray): (2*N*(p-1)) : flattened dXi/ds not considering the boundary conditions
        """

        d_ps_values_ds = self.rhs_pde(s, ps_values)
        d_ps_values_ds_sk[:] = d_ps_values_ds.flatten()
        return 0

    def rhs_pde(self, s, ps_values):
        """
        Args:
            - s (float): value of how far along we are in the deformation
            - ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
        Returns:
            - d_ps_values_ds (np.ndarray): (2*N*(p-1)) : flattened dXi/ds not considering the boundary
                conditions
        """
        Xi_e, dXi_e, ddXi_e = self.get_Xi_dXi_ddXi_evol(ps_values)

        if self.mode == "general":
            self.aghf_wrapper.compute_AGHF_RHS(Xi_e, dXi_e, ddXi_e, self.p - 1 , self.dXi_e_ds)
        elif self.mode == "doubleint":
            self.aghf_wrapper.compute_AGHF_RHS_doubleint(Xi_e, dXi_e, ddXi_e, self.p - 1, self.dXi_e_ds)
        elif self.mode == "doubleint_vel_cons":
            self.aghf_wrapper.compute_AGHF_RHS_doubleint_vel_cons(Xi_e, dXi_e, ddXi_e, self.p - 1, 
                                                            self.dXi_e_ds)

        d_ps_values_ds = self.decode_evol(self.dXi_e_ds)

        if self.print_debug:
            # means we compute the lagrangian and print stuff for all attempted steps
            Xi_e, dXi_e, ddXi_e = self.get_Xi_dXi_ddXi_evol(ps_values)
            action_functional = self.get_action_functional(Xi_e, dXi_e)
            log_string = "s_value value: %f norm(dxds): %f Actional: %f" % \
                (s, np.linalg.norm(d_ps_values_ds), action_functional)
            print(log_string)
            with open(self.log_fn, 'a') as log_file:
                print(log_string, file=log_file)

        return d_ps_values_ds
    
    def compute_numerical_jacobian(self, s, ps_values, h=1e-8):
        """
        Computes the Jacobian of the rhs_pde function using central difference scheme.
        NOTE: THIS IS ONLY USED FOR DEBUGGING PURPOSES!
        
        Args:
            - s (float): value of how far along we are in the deformation
            - ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
            - h (float): perturbation size for finite difference
        
        Returns:
            - jacobian (np.ndarray): Jacobian matrix of shape ((p-1) * 2*N, (p-1) * 2*N)
        """
        n = ps_values.shape[0]
        jacobian = np.zeros((n, n))

        for i in range(n):
            ps_values_pos = np.copy(ps_values)
            ps_values_neg = np.copy(ps_values)
            
            ps_values_pos[i] += h
            ps_values_neg[i] -= h
            f_pos = self.rhs_pde(s, ps_values_pos).reshape(-1)
            f_neg = self.rhs_pde(s, ps_values_neg).reshape(-1)
            
            jacobian[:, i] = (f_pos - f_neg) / (2 * h)
        
        return jacobian
        
    def jac_rhs_pde_scikits(self, s, ps_values, d_ps_values_ds, J):
        """
        Args:
            - s (float): value of how far along we are in the deformation
            - ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
            - d_ps_values_ds (np.ndarray): (2*N*(p-1)) : flattened dXi/ds not considering the boundary conditions
        Modified in-place:
            - J (np.ndarray): (2 * N * (p-1), 2 * N * (p-1)): jacobian matrix that will be modified in place
        """

        Xi_e, dXi_e, ddXi_e = self.get_Xi_dXi_ddXi_evol(ps_values)
        self.aghf_wrapper.compute_PSAGHF_jac(Xi_e, dXi_e, ddXi_e, self.p - 1, self.J_rhs)
        J[:, :] = self.J_rhs[:, :]
        return 0

    def get_action_functional(self, Xi, dXi):
        """
        Calculate the action functional for given Xi and dXi.

        This method computes the action functional by integrating the provided 
        integrand function over a specified range, ignoring the boundaries.

        Parameters:
            - Xi (np.ndarray): The state variable for which the action functional is to be computed.
            - dXi (np.ndarray): The derivative of the state variable Xi.

        Returns:
            - The computed value of the action functional (float)
        """
        integrand_lam = lambda t: self.integrand_action_functional(t, Xi, dXi)
        # When integrating for the lagrangian we ignore the boundaries.
        action_functional = integrate.quad(integrand_lam, self.cheb_t[-2], self.cheb_t[1],
                                           epsabs=self.abs_tol_lagr, 
                                           epsrel=self.rel_tol_lagr)[0]
        return action_functional

    def integrand_action_functional(self, t, Xi, dXi):
        # NOTE: In the future this will be computed in c++.
        """
        Computes the lagrangian for any t
        Args:
            - t (float): time
            - Xi (np.ndarray): ( (p+1) , 2 * N)
            - dXi (np.ndarray): ( (p+1) , 2 * N): time derivative of Xi
        """
        base_time = self.get_q_mat(np.array([t])).squeeze()

        X = np.zeros(shape=(1, 2 * self.N))
        dX = np.zeros(shape=(1, 2 * self.N))

        for ii in range(2 * self.N):
            X[0, ii] = base_time @ self.get_cheb_coeffs(Xi[:, ii])
            dX[0, ii] = base_time @ self.get_cheb_coeffs(dXi[:, ii])
        L_store = np.zeros(1)
        self.update_L(X, dX, L_store)
        return L_store[0]

    def update_L(self, X, dX, L_store):
        # NOTE: In the future this will be computed in c++.
        # NOTE: This does not take into account any kind of penalty function.
        """
        Updates L_store with the value of the lagrangian.
        Args:
            - X: (num_samples, 2*N) : each row corresponds to a state
            - dX: (num_samples, 2*N) : each row corresponds to a time derivative state
        Modified in-place:
            - L_store: (num_samples,): each value corresponds to the lagrangian for that sample
        """
        u = np.zeros(self.N)
        dX_minus_Fd = np.zeros(2 * self.N)
        K_diag = np.eye(self.N) * self.k
        G = np.eye(2 * self.N)
        D = np.eye(self.N)
        data = self.model.createData()
        G[ : self.N, : self.N ] = K_diag
        num_samples = X.shape[0]

        for ii in range(num_samples):
            # Extract
            X_p1 = X[ ii, : self.N ]
            X_p2 = X[ ii, self.N : ]
            dX_p1 = dX[ ii, : self.N ]
            dX_p2 = dX[ ii, self.N : ]
            
            # Compute acceleration
            ddQ = pin.aba(self.model, data, X_p1, X_p2, u)
            
            # Compute parts of the lagrangian
            dX_minus_Fd[ : self.N ] = dX_p1 - X_p2
            dX_minus_Fd[ self.N : ] = dX_p2 - ddQ
            
            # Compute the mass matrix and update G
            pin.crba(self.model, data, X_p1)
            D[:, :] = data.M
            G[self.N : self.N + self.N, self.N : self.N + self.N] = np.dot(D, D)
            L_store[ii] = dX_minus_Fd.T @ G @ dX_minus_Fd

class PostAghf(Aghf):
    def __init__(self, fp_result, fp_urdf=None, mode="general"):
        """
        Initializes the AGHF class with the given result file path or ResultBase instance.
        Args:
            - fp_result (str or ResultBase): The file path to the result pickle file or a ResultBase instance.
            - fp_urdf (str, optional): The file path to the URDF file. Defaults to None.
            - mode (str, optional): The mode of operation. Defaults to "general".
        """
        # Have not considered the case where the stored pickle is a list of dictionaries
        # versus when each pickle is a dictionary.

        # `temp_result` can be a List[Result] or Result
        # List[Result] means that we used Experiment.run_sequential with num_trials > 1
        # Result means that we used Experiment.run_sequential with num_trials = 1 OR Experiment.run_parallel

        if isinstance(fp_result, str):
            temp_result = load_from_pickle(fp_result)
        elif isinstance(fp_result, ResultBase):
            temp_result = fp_result
        else:
            raise ValueError("fp_result has to be a string or a ResultBase instance")
        
        self.s_span = None
        # This is a temporary thing
        self.first_result = None

        if isinstance(temp_result, list):
            self.pset = ParameterSetBase.create_instance_from_result(temp_result[0])
            self.first_result = temp_result[0]
            self.s_span = temp_result[0].s_span
            if hasattr(temp_result[0], "k_cons"):
                self.pset.c_cons = temp_result[0].c_cons
                self.pset.k_cons = temp_result[0].k_cons            
        else:
            self.pset = ParameterSetBase.create_instance_from_result(temp_result)
            self.first_result = temp_result
            self.s_span = temp_result.s_span
            if hasattr(temp_result, "k_cons"):
                self.pset.c_cons = temp_result.c_cons
                self.pset.k_cons = temp_result.k_cons
        
        if fp_urdf:
            self.pset.fp_urdf = fp_urdf

        super().__init__(self.pset)
        self.mode = mode
    
    def compute_d_ps_values_ds(self, Xi_e, dXi_e, ddXi_e):
        """
        Computes the derivative of the ps values with respect to s.
        This method computes the derivative of the ps values with respect to s
        based on the mode specified. It uses different AGHF (Adaptive Generalized
        Harmonic Function) RHS (Right-Hand Side) computation methods depending on
        the mode.
        Args:
            - Xi_e (np.ndarray): ( (p-1), 2*N)
            - dXi_e (np.ndarray): ( (p-1), 2*N) : The first derivative of Xi_e values.
            - ddXi_e (np.ndarray): ( (p-1), 2*N): The second derivative of Xi_e values.
        Returns:
            - d_ps_values_ds (np.ndarray): ( (p-1) * 2*N , 1 ) : The derivative of the ps values with respect to s.
        """
        if self.mode == "general":
            self.aghf_wrapper.compute_AGHF_RHS(Xi_e, dXi_e, ddXi_e, self.p - 1 , self.dXi_e_ds)
        elif self.mode == "doubleint":
            self.aghf_wrapper.compute_AGHF_RHS_doubleint(Xi_e, dXi_e, ddXi_e, self.p - 1 , self.dXi_e_ds)
        elif self.mode == "doubleint_vel_cons":
            self.aghf_wrapper.compute_AGHF_RHS_doubleint_vel_cons(Xi_e, dXi_e, ddXi_e, self.p - 1 , 
                                                            self.dXi_e_ds)
        else:
            raise ValueError('''mode has to be "general" or in ["doubleint", 
                             "doubleint_vel_cons"].''')
 
        d_ps_values_ds = self.decode_evol(self.dXi_e_ds)
        return d_ps_values_ds
    
    def compute_joint_vel_dif(self, Xi_e, dXi_e):
        """
        Computes the maximum average absolute difference between the joint velocities.
        This function calculates the absolute differences between the elements of 
        `Xi_e` and `dXi_e` for the specified range of columns, then computes the 
        average of these absolute differences along the specified axis, and finally 
        returns the maximum value of these averages.
        Args:
            - Xi_e (np.ndarray): ( (p-1), 2*N)
            - dXi_e (np.ndarray): ( (p-1), 2*N) : The first derivative of Xi_e values.
        Returns:
            - res (float): The maximum average absolute difference between the joint velocities.
        """
        abs_dif = np.abs(Xi_e[:, self.N : ] - dXi_e[:,  : self.N ])
        avg_abs_dif = np.sum(abs_dif, axis=0) / (self.p + 1)
        res = np.max(avg_abs_dif)

        return res
        
    def compute_joint_accel_dif(self, dXi_e, ddXi_e):
        """
        Computes the maximum average absolute difference between the joint accelerations.
        Args:
            - dXi_e (np.ndarray): ( (p-1) , 2 * N)
            - ddXi_e (np.ndarray): ( (p-1) , 2 * N)
        Returns:
            - res (float): The maximum average absolute difference between the joint accelerations.
        """
        abs_dif = np.abs(dXi_e[:, self.N : ] - ddXi_e[:,  : self.N ])
        avg_abs_dif = np.sum(abs_dif, axis=0) / (self.p + 1)
        res = np.max(avg_abs_dif)        
        return res    
    
    def compute_qdd(self, ps_values, dt):
        """
        Compute the position (q), velocity (qd), and acceleration (qdd) profiles over time.

        Parameters:
            - ps_values (np.ndarray): ( (p-1) * 2*N , 1 )
            - dt (float): The time step interval.

        Returns:
            - q (np.ndarray): (len(t), N)
            - qd (np.ndarray): (len(t), N)
            - qdd (np.ndarray): (len(t), N)
            - t (np.ndarray): len(t)
            
        """
        num_points = int((2 / dt) + 1) 
        t = np.linspace(-1, 1, num_points)        

        X_spec_t, Xd_spec_t = self.get_X_Xd_spec_t(ps_values, t)
        q = X_spec_t[:, : self.N]
        qd = X_spec_t[:, self.N : ]
        qdd = Xd_spec_t[:, self.N : ]
        return q, qd, qdd, t
    
    def compute_u_matrix(self, q, qd, qdd, num_threads=1):
        """
        Computes the torques for the given position, velocity, and acceleration profiles.
        Args:
            - q (np.ndarray): (len(t), N)
            - qd (np.ndarray): (len(t), N)
            - qdd (np.ndarray): (len(t), N)
            - num_threads (int): number of threads to use in the computation
        Returns:
            - u (np.ndarray): (len(t), N)
        """
        u = compute_u_matrix_static(q, qd, qdd, self.model, num_threads)
        return u

    def compute_u_sq(self, q, qd, qdd, dt):
        """
        Args:
            - q (np.ndarray): (len(t), N)
            - qd (np.ndarray): (len(t), N)
            - qdd (np.ndarray): (len(t), N)
            - dt (float): time step to use for the riemman sum
        Returns:
            - u_sq (float): value of the riemmann sum of the norm of the torques squared
        """
        u = self.compute_u_matrix(q, qd, qdd)

        # compute u_sq
        if len(u.shape) == 1:
            u_sq = np.sum(u * u, axis=0) * dt
        else:
            norm_u_real = np.sum(u * u, axis=1)
            # riemman sum to integrate
            u_sq = dt * np.sum(norm_u_real)
        return u_sq, u

    def compute_along_s(self, dt, sol):
        """
        Computes various metrics along the s_span for given trajectories.
        Args:
            - dt (float): Time step to use for the Riemann sum.
            - sol (np.ndarray): Solution array containing the trajectories.
        Returns:
            - tuple: A tuple containing the following elements:
                - action_functional_vec (np.ndarray): Array of action functional values for each trajectory.
                - vel_dif_vec (np.ndarray): Array of velocity differences for each trajectory.
                - accel_dif_vec (np.ndarray): Array of acceleration differences for each trajectory.
                - norm_dps_values_ds_vec (np.ndarray): Array of norms of d_ps_values_ds for each trajectory.
                - u_sq_vec (np.ndarray): Array of squared control inputs for each trajectory.
                - real_s_span (np.ndarray): The real s_span corresponding to the number of trajectories.
                - Q_mat (np.ndarray): Matrix of q values for each trajectory.
                - Qd_mat (np.ndarray): Matrix of qd values for each trajectory.
                - Qdd_mat (np.ndarray): Matrix of qdd values for each trajectory.
                - U_mat (np.ndarray): Matrix of control inputs for each trajectory.
        """
        
        num_trajectories = sol.shape[0]
        s_span = self.s_span
        real_s_span = s_span[ : num_trajectories]

        action_functional_vec = np.zeros(num_trajectories)
        vel_dif_vec = np.zeros(num_trajectories)
        accel_dif_vec = np.zeros(num_trajectories)
        norm_dps_values_ds_vec = np.zeros(num_trajectories)
        u_sq_vec = np.zeros(num_trajectories)
        

        num_points = int((2 / dt) + 1)
        t = np.linspace(-1, 1, num_points)

        Q_mat = np.zeros(shape=(num_trajectories, num_points, self.N), dtype=np.double)
        Qd_mat = np.zeros(shape=(num_trajectories, num_points, self.N), dtype=np.double)
        Qdd_mat = np.zeros(shape=(num_trajectories, num_points, self.N), dtype=np.double)
        U_mat = np.zeros(shape=(num_trajectories, num_points, self.N))

        for ii in range(num_trajectories):
            ps_values = sol[ii, :]
            Xi = self.encode(ps_values)
            dXi = np.dot(self.D, Xi).astype(dtype=np.double, order='F')
            ddXi = np.dot(self.D2, Xi).astype(dtype=np.double, order='F')
            Xi_e, dXi_e, ddXi_e = self.get_Xi_dXi_ddXi_evol(ps_values)
            
            action_functional_vec[ii] = self.get_action_functional(Xi, dXi)

            vel_dif_vec[ii] = self.compute_joint_vel_dif(Xi_e, dXi_e)
            accel_dif_vec[ii] = self.compute_joint_accel_dif(dXi_e, ddXi_e)
            d_ps_values_ds = self.compute_d_ps_values_ds(Xi_e, dXi_e, ddXi_e)

            norm_dps_values_ds_vec[ii] = np.linalg.norm(d_ps_values_ds, ord=np.inf)

            q, qd, qdd = self.get_q_qd_qdd(ps_values, t)
            Q_mat[ii, :, :] = q
            Qd_mat[ii, :, :] = qd
            Qdd_mat[ii, :, :] = qdd
            u_sq_U_mat = self.compute_u_sq(q, qd, qdd, dt)
            u_sq_vec[ii] = u_sq_U_mat[0]
            
            U_mat[ii, :, :] = u_sq_U_mat[1].reshape(-1, self.N)

        return action_functional_vec, vel_dif_vec, accel_dif_vec, norm_dps_values_ds_vec, \
               u_sq_vec, real_s_span, Q_mat, Qd_mat, Qdd_mat, U_mat

    def compute_along_s_doubleint(self, dt, sol):
        """
        Computes various metrics along the s parameter for a double integrator system.

        Args:
            - dt (float): Time step size.
            - sol (ndarray): Solution array containing the trajectories.

        Returns:
            - tuple: A tuple containing the following elements:
                - vel_dif_vec (ndarray): Vector of velocity differences for each trajectory.
                - accel_dif_vec (ndarray): Vector of acceleration differences for each trajectory.
                - norm_dps_values_ds_vec (ndarray): Vector of the infinity norm of the derivative of ps values with respect to s.
                - u_sq_vec (ndarray): Vector of squared control inputs for each trajectory.
                - real_s_span (ndarray): The real span of the s parameter for the given trajectories.
        """
        num_trajectories = sol.shape[0]
        vel_dif_vec = np.zeros(num_trajectories)
        accel_dif_vec = np.zeros(num_trajectories)
        norm_dps_values_ds_vec = np.zeros(num_trajectories)
        u_sq_vec = np.zeros(num_trajectories)
        num_points = int((2 / dt) + 1)
        t = np.linspace(-1, 1, num_points)
        s_span = self.s_span
        real_s_span = s_span[ : num_trajectories]

        for ii in range(num_trajectories):
            ps_values = sol[ii, :]
            Xi_e, dXi_e, ddXi_e = self.get_Xi_dXi_ddXi_evol(ps_values)
            vel_dif_vec[ii] = self.compute_joint_vel_dif(Xi_e, dXi_e)
            accel_dif_vec[ii] = self.compute_joint_accel_dif(dXi_e, ddXi_e)
            d_ps_values_ds = self.compute_d_ps_values_ds(Xi_e, dXi_e, ddXi_e)
            norm_dps_values_ds_vec[ii] = np.linalg.norm(d_ps_values_ds, ord=np.inf)
            q, qd, qdd = self.get_q_qd_qdd(ps_values, t)
            u_sq_vec[ii] = compute_u_sq_doubleint(qdd, dt)

        return vel_dif_vec, accel_dif_vec, norm_dps_values_ds_vec, u_sq_vec, real_s_span

    def compute_ini_fin_quantities(self, dt, sol):
        """
        Compute initial and final quantities for the given solution.
        This function calculates various quantities at the initial and final points of the solution `sol`.
        It computes the action functional, velocity difference, acceleration difference, norm of the 
        derivative of the phase space values, and the squared control input. Additionally, it computes 
        the joint positions, velocities, and accelerations over a time grid.
        Args:
            - dt (float): Time step size.
            - sol (numpy.ndarray): Solution array with shape (num_points, N).
        Returns:
            - tuple: A tuple containing the following elements:
                - action_functional_vec (numpy.ndarray): Array of action functional values at initial and final points.
                - vel_dif_vec (numpy.ndarray): Array of velocity differences at initial and final points.
                - accel_dif_vec (numpy.ndarray): Array of acceleration differences at initial and final points.
                - norm_ps_values_ds_vec (numpy.ndarray): Array of norms of the derivative of phase space values at initial and final points.
                - u_sq_vec (numpy.ndarray): Array of squared control input values at initial and final points.
                - Q_mat (numpy.ndarray): Matrix of joint positions over the time grid.
                - Qd_mat (numpy.ndarray): Matrix of joint velocities over the time grid.
                - Qdd_mat (numpy.ndarray): Matrix of joint accelerations over the time grid.
                - t (numpy.ndarray): Time grid array.
        """
        num_points = int((2 / dt) + 1)
        t = np.linspace(-1, 1, num_points)

        action_functional_vec = np.zeros(shape=(2))
        vel_dif_vec = np.zeros(shape=(2))
        accel_dif_vec = np.zeros(shape=(2))
        norm_ps_values_ds_vec = np.zeros(shape=(2))
        u_sq_vec = np.zeros(shape=(2))

        Q_mat = np.zeros(shape=(2, num_points, self.N))
        Qd_mat = np.zeros(shape=(2, num_points, self.N))
        Qdd_mat = np.zeros(shape=(2, num_points, self.N))
        

        for ii in range(2):
            idx = 0 if ii == 0 else -1

            ps_values = sol[idx, :].reshape(-1, 1)
            Xi = self.encode(ps_values)
            dXi = np.dot(self.D, Xi).astype(dtype=np.double, order='F')
            ddXi = np.dot(self.D2, Xi).astype(dtype=np.double, order='F')            
            Xi_e, dXi_e, ddXi_e = self.get_Xi_dXi_ddXi_evol(ps_values)

            action_functional_vec[ii] = self.get_action_functional(Xi, dXi)
            vel_dif_vec[ii] = self.compute_joint_vel_dif(Xi_e, dXi_e)
            accel_dif_vec[ii] = self.compute_joint_accel_dif(dXi_e, ddXi_e)
            norm_ps_values_ds_vec[ii] = np.linalg.norm(self.compute_d_ps_values_ds(Xi_e, dXi_e, ddXi_e))

            q, qd, qdd = self.get_q_qd_qdd(ps_values, t)
            Q_mat[ii, :, :] = q
            Qd_mat[ii, :, :] = qd
            Qdd_mat[ii, :, :] = qdd
            u_sq_vec[ii], _= self.compute_u_sq(q, qd, qdd, dt)

        res = action_functional_vec, vel_dif_vec, accel_dif_vec, \
            norm_ps_values_ds_vec, u_sq_vec, Q_mat, Qd_mat, Qdd_mat, t
        return res

    def get_q_traj_ini_fin(self, sol):
        """
        Computes the initial and final trajectory matrices from the solution matrix.

        Args:
            - sol (numpy.ndarray): A 2D array where each row represents a solution vector at a specific time.

        Returns:
            - tuple: A tuple containing two 2D numpy arrays:
                - q_traj_ini (numpy.ndarray): The initial trajectory matrix of shape (p + 1, N).
                - q_traj_end (numpy.ndarray): The final trajectory matrix of shape (p + 1, N).
        """
        q_traj_ini = np.zeros(shape=(self.p + 1, self.N))
        q_traj_end = np.zeros(shape=(self.p + 1, self.N))
        Q_mat = np.zeros(shape=(2, self.p + 1, self.N))
        for ii in range(2):
            idx = 0 if ii == 0 else -1
            ps_values = sol[idx, :].reshape(-1, 1)
            Xi = self.encode(ps_values)
            
            # np.flip so that the first row corresponds to the initial time
            Q_mat[ii, :, :] = np.flip(Xi[:, : self.N], axis=0)

        q_traj_ini = np.squeeze(Q_mat[0, :, :])
        q_traj_end = np.squeeze(Q_mat[1, :, :])

        return q_traj_ini, q_traj_end