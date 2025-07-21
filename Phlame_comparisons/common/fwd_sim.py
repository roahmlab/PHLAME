import numpy as np
import math
import pinocchio as pin
import os
import ipdb
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def run_fwd_simulation_controller(U_sol, X, k_pos, k_vel, fp_urdf, dt_sim, dt_croco, t_total, num_times, X0, method_ivp='RK45'):
    """
    Input:
    ------
    U_sol:
    X:
    k_pos:
    k_vel:
    fp_urdf:
    t_final:
    dt: dt that was used in croco


    Returns:
    --------
    solution: solution dict produced by solve_ivp
    u_sq_real: u^2 produced by the controller

    """
    N = int(X0.shape[0] / 2)

    t_u = np.linspace(0, t_total-dt_croco, num_times)
    t_x = np.linspace(0, t_total, num_times + 1)

    model = pin.buildModelFromUrdf(fp_urdf)
    lam_ode = lambda tau, x: ode_fwd_integrate(tau, x, U_sol, t_u, model, 
                                            X, t_x, k_pos,
                                            k_vel)
    t_eval_fwd = np.arange(0.0, t_total + dt_sim, dt_sim)
    t_simul = [ t_eval_fwd[0], t_eval_fwd[-1] ]

    solution = solve_ivp(lam_ode, t_simul, X0.squeeze(), method=method_ivp, t_eval=t_eval_fwd)

    # get actual u
    U_actual = np.zeros(shape=(t_eval_fwd.shape[0], N))
    for ii in range(solution.y.shape[1]-1):
        U_actual[ii, :] = get_u_curr(t_eval_fwd[ii], solution.y[:, ii], U_sol, t_u, 
                                    model, X, t_x, k_pos, k_vel)
    u_sq_real = np.sum(U_actual**2) * dt_sim
    return solution, u_sq_real


def ode_fwd_integrate(t, x, U_sol, time_u, model, X, time_x, k_pos, k_vel):
    """
    Right hand side of xdot = f(x, u); where u = U_sol + k_pos * err_pos + k_vel * err_vel
    Args:
        t: float: time
        x: np.array: (2*N,) : state vector
        U_sol: np.array: ( len(t), N) : control signal matrix
        time_u: np.array: ( len(t),) : time vector for U_sol
        model: pinocchio.Model: model of the system
        X: np.array: ( len(t), 2*N) : desired trajectory first n columns correspond to q, next n cols to qd
        time_x: np.array: ( len(t),) : time vector for X
        k_pos: float: proportional gain for position error
        k_vel: float: proportional gain for velocity error

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
        t: float: time
        x: np.array: (2*N,) : state vector
        U_sol: np.array: ( len(t), N) : control signal matrix
        time_u: np.array: ( len(t),) : time vector for U_sol
        model: pinocchio.Model: model of the system
        X: np.array: ( len(t), 2*N) : desired trajectory first n columns correspond to q, next n cols to qd
        time_x: np.array: ( len(t),) : time vector for X
        k_pos: float: proportional gain for position error
        k_vel: float: proportional gain for velocity error

    """    
    N = model.nv
    q = x[ : N]
    qd = x[N : ]
    u_extracted = interp1d(time_u, U_sol.T, kind='linear', fill_value='extrapolate')(t)
    # u_extracted = U_interp.reshape(-1)

    x_des = interp1d(time_x, X.T, kind='linear', fill_value='extrapolate')(t)
    # ipdb.set_trace()
    err_pos = q - x_des[ : N]
    err_vel = qd - x_des[N : ]
    u_curr = u_extracted - k_pos * err_pos - k_vel * err_vel
    return u_curr