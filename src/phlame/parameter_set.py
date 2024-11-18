from .constants import MINIMUM_S_SPAN
from .cheb import Cheb

from dataclasses import fields
from itertools import product
import numpy as np


class ParameterSetBase(Cheb):
    def __init__(self, p, N, X0, Xf, name, s_max, k, abs_tol, rel_tol, method_name, max_steps, 
                 ns_points, j_type, fp_urdf, IC_type = None, q_traj=None, scenario_name=None):   
        """
        Initializes the ParameterSet class.

        Args:
            - p (int): Number of Chebyshev nodes.
            - N (int): Number of links (joint angles dimension).
            - X0 (np.ndarray): Initial state vector (2*N, 1).
            - Xf (np.ndarray): Final state vector (2*N, 1).
            - name (str): Name of the run.
            - s_max (float): Maximum value of s.
            - k (int): Penalty parameter.
            - abs_tol (float): Absolute tolerance for the ODE solver.
            - rel_tol (float): Relative tolerance for the ODE solver.
            - method_name (str): Name of the ODE solver method.
            - max_steps (int): Maximum number of steps for the ODE solver.
            - ns_points (int): Number of sample points.
            - j_type (np.ndarray): Joint type array (2*N, 1).
            - fp_urdf (str): File path to the URDF file.
            - IC_type (str, optional): Initial condition type. Default is None.
            - q_traj (np.ndarray, optional): Trajectory matrix. Default is None.
            - scenario_name (str, optional): Name of the scenario. Default is None.

        Returns:
            - None
        """
        super().__init__(p, N, X0, Xf)
        self.name = name
        self.s_max = s_max
        self.k = k
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.method_name = method_name
        self.max_steps = max_steps
        self.ns_points = ns_points
        self.j_type = j_type
        self.fp_urdf = fp_urdf
        self.IC_type = IC_type
        self.scenario_name = scenario_name
        
        # Generate initial trajectory(line)
        if (self.IC_type is None):
            self.init_ps_values = self.generate_ps_traj_line()
        elif self.IC_type == 'low_acc':
            self.init_ps_values = self.generate_ps_traj_min_accel_v2()
            print('LP solved for initial acceleration initial condition')
        else:
            raise ValueError(f"IC_type: {self.IC_type} not recognized")
        
        #compute s_span    
        self.s_span = self._compute_s_span()

        # update init_ps_values to match q_traj
        if q_traj is not None:
            self.q_traj = q_traj
            self.init_ps_values = self.generate_ps_traj_from_q_traj(q_traj)
            self.validate_initial_trajectory()

    
    def __str__(self):
        return self.generate_info_string()
    
    def validate_initial_trajectory(self):
        """
        Validates that the initial trajectory `q_traj` satisfies the boundary conditions given in `X0` and `Xf`.

        Args:
            - self (ParameterSet): The instance of the ParameterSet class.

        Returns:
            - None

        Raises:
            - AssertionError: If the start of `q_traj` does not match `X0` or the end of `q_traj` does not match `Xf`.
        """
        # validates that the initial trajectory q_traj satisfies the BCs given in X0, Xf
        q_start_bc = self.X0[ 0 : self.N, 0].reshape(-1)
        q_end_bc = self.Xf[ 0 : self.N, 0].reshape(-1)

        q_end_sto = self.q_traj[-1, :].reshape(-1)
        q_start_sto = self.q_traj[0, :].reshape(-1)

        np.testing.assert_almost_equal(q_start_bc, q_start_sto, 
                                       err_msg="`q_traj` start doesn't match X0")
        np.testing.assert_almost_equal(q_end_bc, q_end_sto,
                                       err_msg="`q_traj` end doesn't match Xf")


    @classmethod
    def create_instance_from_result(cls, result):
        """
        Creates an instance of the ParameterSet class from a ResultBase instance.

        Args:
            - result (ResultBase): An instance of ResultBase containing the parameters for the ParameterSet.

        Returns:
            - ParameterSet: An instance of the ParameterSet class initialized with the parameters from the result.
        """
        if hasattr(result, 'scenario_name'):
            return cls(p=result.p, N=result.N, X0=result.X0, Xf=result.Xf, name=result.name, 
                    s_max=result.s_max, k=result.k,  abs_tol=result.abs_tol, 
                    rel_tol=result.rel_tol, method_name=result.method_name, 
                    max_steps=result.max_steps, ns_points=result.ns_points, j_type=result.j_type, 
                    fp_urdf=result.fp_urdf, scenario_name=result.scenario_name)
        else:
            return cls(p=result.p, N=result.N, X0=result.X0, Xf=result.Xf, name=result.name, 
                    s_max=result.s_max, k=result.k,  abs_tol=result.abs_tol, 
                    rel_tol=result.rel_tol, method_name=result.method_name, 
                    max_steps=result.max_steps, ns_points=result.ns_points, j_type=result.j_type, 
                    fp_urdf=result.fp_urdf)            

    def _compute_s_span(self):
        """
        Computes the span of the parameter `s` using a logarithmic scale.

        This method generates a span of `s` values starting from `MINIMUM_S_SPAN` to `self.s_max` 
        using a logarithmic scale with `self.ns_points` number of points. The span starts with 0 
        and then includes the logarithmically spaced values.

        Args:
            - self (ParameterSet): The instance of the ParameterSet class.

        Returns:
            - np.ndarray: An array containing the span of `s` values.
        """
        s_span = np.logspace(np.log10(MINIMUM_S_SPAN), np.log10(self.s_max), num=self.ns_points)
        return np.concatenate( ([0], s_span ) )
    
    def generate_info_string(self):
        """
        Generates a string containing the names and values of all fields in the ParameterSet instance.

        Args:
            - self (ParameterSet): The instance of the ParameterSet class.

        Returns:
            - str: A string containing the names and values of all fields, formatted as "field_name: field_value".
        """
        info_list = []
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            info_list.append(f"{field_name}: {field_value}")
        return ", ".join(info_list)

class ParameterSetActivatedState(ParameterSetBase):
    """
    Initializes the ParameterSet class for trajectory generation problems with state constraints.

    Args:
        - p (int): Degree of the polynomial.
        - N (int): Number of bodies.
        - X0 (np.ndarray): Initial state vector (2*N, 1).
        - Xf (np.ndarray): Final state vector (2*N, 1).
        - name (str): Name of the parameter set.
        - s_max (float): Maximum arc length.
        - k (int): Penalty parameter.
        - abs_tol (float): Absolute tolerance for the ODE solver.
        - rel_tol (float): Relative tolerance for the ODE solver.
        - method_name (str): Name of the ODE solver method.
        - max_steps (int): Maximum number of steps for the ODE solver.
        - ns_points (int): Number of sample points.
        - j_type (np.ndarray): Joint type array (2*N, 1).
        - fp_urdf (str): File path to the URDF file.
        - c_cons (float): Constraint activation function coefficient controlling how sharply the penalty activates upon constraint violation.
        - k_cons (float): Constraint penalty parameter.
        - x_lower (np.ndarray): Lower bounds for the state constraints.
        - x_upper (np.ndarray): Upper bounds for the state constraints.
        - q_traj (np.ndarray, optional): Trajectory matrix. Default is None.

    Returns:
        - None
    """
    def __init__(self, p, N, X0, Xf, name, s_max, k, abs_tol, rel_tol, method_name, max_steps, 
                 ns_points, j_type, fp_urdf, c_cons, k_cons, x_lower, x_upper, q_traj=None):
        super().__init__(p, N, X0, Xf, name, s_max, k, abs_tol, rel_tol, method_name, max_steps, 
                         ns_points, j_type, fp_urdf, q_traj=q_traj)
        self.c_cons = c_cons
        self.k_cons = k_cons
        self.x_lower = x_lower
        self.x_upper = x_upper
    
    def __str__(self):
        return "ParameterSetActivatedState"

    
class ParameterSetActivatedSimpleSphere(ParameterSetBase):
    def __init__(self, p, N, X0, Xf, name, s_max, k, abs_tol, rel_tol, method_name, max_steps, 
                 ns_points, j_type, fp_urdf, c_cons, k_cons, obs_mat, scenario_name, q_traj=None):
        """
        Initializes the ParameterSetActivatedSimpleSphere class for trajectory generation problems with sphere obstacle avoidance constraints.

        Args:
            - p (int): Degree of the polynomial.
            - N (int): Number of bodies.
            - X0 (np.ndarray): Initial state vector (2*N, 1).
            - Xf (np.ndarray): Final state vector (2*N, 1).
            - name (str): Name of the parameter set.
            - s_max (float): Maximum arc length.
            - k (int): Penalty parameter.
            - abs_tol (float): Absolute tolerance for the ODE solver.
            - rel_tol (float): Relative tolerance for the ODE solver.
            - method_name (str): Name of the ODE solver method.
            - max_steps (int): Maximum number of steps for the ODE solver.
            - ns_points (int): Number of sample points.
            - j_type (np.ndarray): Joint type array (2*N, 1).
            - fp_urdf (str): File path to the URDF file.
            - c_cons (float): Constraint activation function coefficient controlling how sharply the penalty activates upon constraint violation.
            - k_cons (float): Constraint penalty parameter.
            - obs_mat (np.ndarray): Obstacle matrix (num_obstacles, 4) where each row is [x, y, z, r] (center and radius of the sphere).
            - scenario_name (str): Name of the scenario.
            - q_traj (np.ndarray, optional): Trajectory matrix. Default is None.

        Returns:
            - None
        """
        super().__init__(p, N, X0, Xf, name, s_max, k, abs_tol, rel_tol, method_name, 
                         max_steps, ns_points, j_type, fp_urdf, q_traj=q_traj)
        self.c_cons = c_cons
        self.k_cons = k_cons
        self.obs_mat = obs_mat
        self.scenario_name = scenario_name
    
    def __str__(self):
        return "ParameterSetActivatedSimpleSphere"

def generate_parameter_sets(p_v, s_max_v, k_v, abs_tol_v, rel_tol_v, run_name, method_name, 
                            max_steps, ns_points, N, j_type, fp_urdf, X0, Xf, 
                            class_cons=ParameterSetBase, IC_type = None, x_max_lower=None, 
                            x_max_upper=None, c_cons_v=None, k_cons_v=None, obs_mat=None, 
                            q_traj=None, scenario_name=None):
    
    """
    Generates a list of parameter sets for trajectory generation problems, including those with state constraints and sphere obstacle avoidance constraints.

    Args:
        - p_v (list of int): List of degrees of the polynomial.
        - s_max_v (list of float): List of maximum arc lengths.
        - k_v (list of int): List of penalty parameters.
        - abs_tol_v (list of float): List of absolute tolerances for the ODE solver.
        - rel_tol_v (list of float): List of relative tolerances for the ODE solver.
        - run_name (str): Name of the run.
        - method_name (str): Name of the ODE solver method.
        - max_steps (int): Maximum number of steps for the ODE solver.
        - ns_points (int): Number of sample points.
        - N (int): Number of bodies.
        - j_type (np.ndarray): Joint type array (2*N, 1).
        - fp_urdf (str): File path to the URDF file.
        - X0 (np.ndarray): Initial state vector (2*N, 1).
        - Xf (np.ndarray): Final state vector (2*N, 1).
        - class_cons (type): Class constructor for the parameter set. Default is ParameterSetBase.
        - IC_type (str, optional): Initial condition type. Default is None.
        - x_max_lower (np.ndarray, optional): Lower bounds for the state constraints. Default is None.
        - x_max_upper (np.ndarray, optional): Upper bounds for the state constraints. Default is None.
        - c_cons_v (list of float, optional): List of constraint activation function coefficients. Default is None.
        - k_cons_v (list of float, optional): List of constraint penalty parameters. Default is None.
        - obs_mat (np.ndarray, optional): Obstacle matrix (4, num_obstacles) where each column is [x, y, z, r] (center and radius of the sphere). Default is None.
        - q_traj (np.ndarray, optional): Trajectory matrix. Default is None.
        - scenario_name (str, optional): Name of the scenario. Default is None.

    Returns:
        - list: A list of parameter set instances.
    """

    parameter_sets = []

    if class_cons  == ParameterSetBase:
        for p, s_max, k, abs_tol, rel_tol in product(p_v, s_max_v, k_v, abs_tol_v, rel_tol_v):
            curr_param_set = class_cons(name=run_name, p=p, s_max=s_max, k=k, abs_tol=abs_tol, 
                                        rel_tol=rel_tol, method_name=method_name, 
                                        max_steps=max_steps, ns_points=ns_points, N=N, j_type=j_type, 
                                        fp_urdf=fp_urdf, X0=X0, Xf=Xf, IC_type = IC_type, 
                                        q_traj=q_traj, scenario_name=scenario_name)
            parameter_sets.append(curr_param_set)
    elif class_cons in [ParameterSetActivatedState, ParameterSetActivatedSimpleSphere]:
        if (type(k_cons_v) == list) and ( type(c_cons_v) == list):
            for p, s_max, k, abs_tol, rel_tol, k_cons, c_cons in product(p_v, s_max_v, k_v, 
                                                                         abs_tol_v, rel_tol_v, 
                                                                         k_cons_v, c_cons_v):
                if k_cons is None:
                    k_cons = k

                if class_cons == ParameterSetActivatedState:
                    curr_param_set = class_cons(p=p, N=N, X0=X0, Xf=Xf, name=run_name, s_max=s_max, 
                                                k=k, abs_tol=abs_tol, rel_tol=rel_tol, 
                                                method_name=method_name, max_steps=max_steps,
                                                ns_points=ns_points, j_type=j_type, fp_urdf=fp_urdf, 
                                                c_cons=c_cons, k_cons=k_cons, x_lower=x_max_lower, 
                                                x_upper=x_max_upper, q_traj=q_traj)
                elif class_cons == ParameterSetActivatedSimpleSphere:
                    curr_param_set = class_cons(p=p, N=N, X0=X0, Xf=Xf, name=run_name, s_max=s_max, 
                                                k=k, abs_tol=abs_tol, rel_tol=rel_tol, 
                                                method_name=method_name, max_steps=max_steps,
                                                ns_points=ns_points, j_type=j_type, fp_urdf=fp_urdf, 
                                                c_cons=c_cons, k_cons=k_cons, obs_mat=obs_mat, 
                                                q_traj=q_traj, scenario_name=scenario_name)
                parameter_sets.append(curr_param_set)
    else:
        raise("Class: ", class_cons, " is not supported")
    
    return parameter_sets