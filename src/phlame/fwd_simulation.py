# provisionary functions to forward simulate

import numpy as np
import pinocchio as pin
from scipy.interpolate import interp1d
from .aghf import PostAghf
from .utils import load_from_pickle
import ipdb

def ode_fwd_integrate(t, x, U_sol, time_u, model, X, time_x, k_pos, k_vel):
    """
    Args:
        - t: float: time
        - x: np.array: (2*N,) : state vector
        - U_sol: np.array: ( len(t), N) : control signal matrix
        - time_u: np.array: ( len(t),) : time vector for U_sol
        - model: pinocchio.Model: model of the system
        - X: np.array: ( len(t), 2*N) : desired trajectory first n columns correspond to q, 
            next n cols to qd
        - time_x: np.array: ( len(t),) : time vector for X
        - k_pos: float: proportional gain for position error
        - k_vel: float: proportional gain for velocity error

    """
    N = model.nv
    q = x[ : N]
    qd = x[N : ]
    u_curr = get_u_curr(t, x, U_sol, time_u, model, X, time_x, k_pos, k_vel)
    data = model.createData()
    qdd = pin.aba(model, data,  q, qd, u_curr)

    dxdt = np.concatenate( (qd, qdd ) ) # this might be super slow
    return dxdt

def get_u_curr(t, x, U_sol, time_u, model, X, time_x, k_pos, k_vel):
    """
    Args:
        - t (float): time
        - x (np.ndarray): (2*N,) : state vector
        - U_sol (np.ndarray): (len(t), N) : control signal matrix
        - time_u (np.ndarray): (len(t),) : time vector for U_sol
        - model (pinocchio.Model): model of the system
        - X (np.ndarray): (len(t), 2*N) : desired trajectory first n columns correspond to q, 
            next n cols to qd
        - time_x (np.ndarray): (len(t),) : time vector for X
        - k_pos (float): proportional gain for position error
        - k_vel (float): proportional gain for velocity error

    Returns:
        - None
    """  
    N = model.nv
    q = x[ : N]
    qd = x[N : ]
    u_extracted = interp1d(time_u, U_sol.T, kind='linear', fill_value='extrapolate')(t)
    # u_extracted = U_interp.reshape(-1)

    x_des = interp1d(time_x, X.T, kind='linear', fill_value='extrapolate')(t)
    
    err_pos = q - x_des[ : N]
    err_vel = qd - x_des[N : ]
    u_curr = u_extracted - k_pos * err_pos - k_vel * err_vel
    return u_curr

def run_fwd_simulation(fp_pickle, s_query, dt, t):
    """
    Runs a forward simulation using the provided pickle file and query parameters.

    Args:
        - fp_pickle (str): File path to the pickle file containing the simulation results.
        - s_query (float): The specific query value for 's' to find the closest match in the simulation results.
        - dt (float): Time step for the simulation.
        - t (float): Current time for which the simulation is being run.
    """
    result = load_from_pickle(fp_pickle)
    post_aghf = PostAghf(fp_pickle)
    sol = result.sol
    num_trajectories = sol.shape[0]
    s_span = result.s_span
    real_s_span = s_span[ : num_trajectories]
    idx = np.argmin( np.abs( real_s_span - s_query ) )
    print("Actual s: ", real_s_span[idx])

    ps_values = sol[idx, :]
    q, qd, qdd = post_aghf.get_q_qd_qdd(ps_values, t)
    u = post_aghf.compute_u_matrix(q, qd, qdd)

    return