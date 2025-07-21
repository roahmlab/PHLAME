import numpy as np
import math
import os
from scipy.io import loadmat
from scipy.io import savemat
from itertools import product

from aligator_wrapper.run_urdf import gen_init_guess_from_q_traj
from aligator_wrapper.run_urdf import SphereObstacle

import sys
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), "../../src") ) )
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), ".." ) ) )

from phlame.utils import load_from_pickle
from phlame.utils import save_to_pickle
from phlame.utils import cleanup_folder


"""
NOTE: The name of this file is inaccurate. This code works for any robot.
"""

TIMEOUT_T_SOLVE = 99_999
TIMEOUT_UNHANDLED_EXCEPTION = 88_888

def get_x0_xf_obstacles(scenario_name, folder_scenarios):
    fp_mat = os.path.join(folder_scenarios, f"{scenario_name}.mat")
    data_scenario = loadmat(fp_mat)

    if data_scenario.get('obs_mat_values') is not None:
        obstacles = data_scenario['obs_mat_values'].astype(dtype=np.double, order='F')
        n_obstacles = obstacles.shape[1]
    else:
        n_obstacles = 0
        obstacles = None

    q_traj = data_scenario['q_traj'].astype(dtype=np.double, order='F')
    N = q_traj.shape[1]
    q_start = q_traj[0, :].reshape(N, 1)
    q_end = q_traj[-1, :].reshape(N, 1)
    qd_start = data_scenario['qd_start'].astype(dtype=np.double, order='F').reshape(N, 1)
    qd_end = data_scenario['qd_end'].astype(dtype=np.double, order='F').reshape(N, 1)

    x0 = np.concatenate( ( q_start, qd_start ) ).astype(dtype=np.double, order='F')
    xf = np.concatenate( ( q_end, qd_end ) ).astype(dtype=np.double, order='F')
    return x0, xf, obstacles

def generate_scenario_name_dt_to_IT(scenario_names, folder_scenarios, dt_vec, t_total, N):
    # Literal copy from croco_wrapper
    scenario_name_dt_to_IT = {scenario_idx: {} for scenario_idx in scenario_names}
    for scenario_name in scenario_names:
        for dt in dt_vec:
            fp_mat = os.path.join(folder_scenarios, f"{scenario_name}.mat")
            data_scenario = loadmat(fp_mat)
            q_traj = data_scenario['q_traj'].astype(dtype=np.double, order='F')
            N = q_traj.shape[1]
            q_start = q_traj[0, :].reshape(N, 1)
            q_end = q_traj[-1, :].reshape(N, 1)
            qd_start = data_scenario['qd_start'].astype(dtype=np.double, order='F').reshape(N, 1)
            qd_end = data_scenario['qd_end'].astype(dtype=np.double, order='F').reshape(N, 1)

            x0 = np.concatenate( ( q_start, qd_start ) ).astype(dtype=np.double, order='F')
            xf = np.concatenate( ( q_end, qd_end ) ).astype(dtype=np.double, order='F')
            num_times = math.ceil(t_total / dt)
            init_xs, init_us = gen_init_guess_from_q_traj(q_traj, N, num_times, x0, xf)
            scenario_name_dt_to_IT[scenario_name][dt] = (init_xs, init_us)
    return scenario_name_dt_to_IT
    
# very similar to croco_wrapper
def get_fn_base(idx, scenario_name, weight_x, weight_u, weight_xf, dt, robot_name, tol, mu_init, max_iters):
        return f"kinova_constrained_idx={idx}-sc_name={scenario_name}-weight_x={weight_x}" + \
                f"-weight_u={weight_u}-weight_xf={weight_xf}-dt={dt}-robot_name={robot_name}" + \
                f"-tol={tol}-mu_init={mu_init}-max_iters={max_iters}"

def create_callback_store_opt_result(fp_folder_storage, idx, scenario_name, weight_x, weight_u, weight_xf,
                                     dt_alig, folder_scenarios, robot_name, tol, mu_init, max_iters):
    
    """
    Similar to `create_callback_mp` only that here we only solve with aligator, thus, no forward simulation is done.
    """
    def task_done(future):
        fn_base = get_fn_base(idx, scenario_name, weight_x, weight_u, weight_xf,
                                dt_alig, robot_name, tol, mu_init, max_iters)
        fp_soln = "aligator_res_" + fn_base + "data_soln.pkl"
        fp_soln_store = os.path.join(fp_folder_storage, fp_soln)

        fp_mat = os.path.join(folder_scenarios, f"{scenario_name}.mat")
        data_scenario = loadmat(fp_mat)

        #### Literal copy of croco_wrapper.kinova_obstacles #########
        obstacles = data_scenario.get('obs_mat_values')
        if obstacles is None:
            n_obstacles = 0
        else:
            obstacles = data_scenario['obs_mat_values'].astype(dtype=np.double, order='F')
            n_obstacles = obstacles.shape[1]

        # Transpose required because of the way the c++ wrapper expects the matrix
        # obs_mat should be (n_obstacles, 4)
        q_traj = data_scenario['q_traj'].astype(dtype=np.double, order='F')
        N = q_traj.shape[1]
        q_start = q_traj[0, :].reshape(N, 1)
        q_end = q_traj[-1, :].reshape(N, 1)
        qd_start = data_scenario['qd_start'].astype(dtype=np.double, order='F').reshape(N, 1)
        qd_end = data_scenario['qd_end'].astype(dtype=np.double, order='F').reshape(N, 1)
        X0 = np.concatenate( ( q_start, qd_start ) ).astype(dtype=np.double, order='F')
        Xf = np.concatenate( ( q_end, qd_end ) ).astype(dtype=np.double, order='F')
        #### Literal copy of croco_wrapper.kinova_obstacles #########

        try:
            X, U, t_solve = future.result()
        except TimeoutError as error:
            print("Timeout was triggered.")
            X = None
            t_solve = TIMEOUT_T_SOLVE
            data_soln = None
            data_mat = None            
        except Exception as error:
            print("Uncaught exception when trying to access `future.result()` raised %s" % error)
            print(error)  # traceback of the function
            print("scenario_idx: ", scenario_name)
            print("weight_x: ", weight_x)
            print("weight_u: ", weight_u)
            print("weight_xf: ", weight_xf)
            print("dt: ", dt_alig)
            X = None
            t_solve = TIMEOUT_UNHANDLED_EXCEPTION
            data_soln = None
            data_mat = None       
        
        if X is None:
            # Add the other parameters that are only in aligator:
            # mu_init, max_iters, tol, maybe another one?
            data_soln = {
                "success": 0,
                "idx": idx,
                "scenario_name": scenario_name,
                "weight_x": weight_x,
                "weight_u": weight_u,
                "weight_xf": weight_xf,
                "dt_alig": dt_alig,
                "t_solve": t_solve,
                "n_obstacles": n_obstacles,
                "obstacles": obstacles,
                "tol": tol,
                "mu_init": mu_init,
                "max_iters": max_iters,
                "fn_base": fn_base,
            }
        else:
            # Means we didn't timeout and aligator solved successfully
            u_sq = np.sum(U**2) * dt_alig
            data_soln = {
                "success": 1,
                "X": X,
                "X0": X0,
                "Xf": Xf,
                "U": U,
                "t_solve": t_solve,
                "u_sq_soln": u_sq,
                "dt_alig": dt_alig,
                "idx": idx,
                "scenario_name": scenario_name,
                "weight_x": weight_x,
                "weight_u": weight_u,
                "weight_xf": weight_xf,
                "tol": tol,
                "mu_init": mu_init,
                "max_iters": max_iters,         
                "fp_soln_store": fp_soln_store,
                "n_obstacles": n_obstacles,
                "fn_base": fn_base,
                "obstacles": obstacles,                    
            }
        
        save_to_pickle(data_soln, fp_soln_store)

    return task_done