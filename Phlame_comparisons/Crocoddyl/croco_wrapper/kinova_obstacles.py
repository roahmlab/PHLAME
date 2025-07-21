
import numpy as np
import math
import os
import ipdb
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from scipy.io import loadmat
from scipy.io import savemat
import multiprocessing as mp
from pebble import ProcessPool
from itertools import product
from pprint import pprint

from croco_wrapper.run_urdf import run_scenario_obstacles

import sys
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), "../../src") ) )
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), ".." ) ) )

from phlame.utils import load_from_pickle
from phlame.utils import save_to_pickle
from phlame.utils import cleanup_folder

"""
NOTE: The name of the script kinova_obstacles.py is misleading. This code is written for a general articulated robotic system.
"""

TIMEOUT_T_SOLVE = 99_999
TIMEOUT_UNHANDLED_EXCEPTION = 88_888

def run_single(folder_scenarios, scenario_name, weight_x, weight_u, weight_xf, weight_obs, weight_obs_fin, alpha_log, dt, robot_name, frame_names, t_total,
               init_xs, init_us):

    fp_mat = os.path.join(folder_scenarios, f"{scenario_name}.mat")
    
    # get initial and final condition
    data_scenario = loadmat(fp_mat)

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
    x0 = np.concatenate( ( q_start, qd_start ) ).astype(dtype=np.double, order='F')
    xf = np.concatenate( ( q_end, qd_end ) ).astype(dtype=np.double, order='F')

    X, U, t_solve = run_scenario_obstacles(weight_x, weight_u, weight_xf, weight_obs, weight_obs_fin, robot_name, dt, t_total, alpha_log, frame_names, init_xs, init_us,
                                        obstacles, x0, xf, n_obstacles)
    # print("vars: after ", vars)

    return X, U, t_solve

def get_fn_base(idx, scenario_name, weight_x, weight_u, weight_xf, weight_obs, weight_obs_fin, alpha_log, dt, robot_name):
        return f"kinova_constrained_id={idx}-sc_idx={scenario_name}-weight_x={weight_x}" + \
                f"-weight_u={weight_u}-weight_xf={weight_xf}-w_obs={weight_obs}-w_obs_fin={weight_obs_fin}" + \
                f"alpha_log-={alpha_log}-dt={dt}-robot_name={robot_name}"    

def store_opt_result(X, U, t_solve, fp_folder_storage, idx, scenario_name, weight_x, weight_u, weight_xf, 
                                     weight_obs, weight_obs_fin, alpha_log, dt_croco, robot_name, folder_scenarios):
    fn_base = get_fn_base(idx, scenario_name, weight_x, weight_u, weight_xf, weight_obs, weight_obs_fin, alpha_log, dt_croco, robot_name)
    fn_soln = "croco_res_" +  fn_base + "data_soln.pkl"
    fp_soln_store = os.path.join(fp_folder_storage, fn_soln)

    fp_mat = os.path.join(folder_scenarios, f"{scenario_name}.mat")
    # get initial and final condition
    data_scenario = loadmat(fp_mat)


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

    if X is None:
        data_soln = {
            "success": 0,
            "idx": idx,
            "scenario_name": scenario_name,
            "weight_x": weight_x,
            "weight_u": weight_u,
            "weight_xf": weight_xf,
            "weight_obs": weight_obs,
            "weight_obs_fin": weight_obs_fin,
            "alpha_log": alpha_log,
            "dt_croco": dt_croco,
            "t_solve": t_solve,
            "n_obstacles": n_obstacles,
            "obstacles": obstacles,      
            "fn_base": fn_base,
        }

    else:
        # Means we didn't timeout and coroco solved successfully
        u_sq = np.sum(U**2) * dt_croco

        data_soln = {
            "success": 1,
            "X": X,
            "X0": X0,
            "Xf": Xf,
            "U": U,
            "t_solve": t_solve,
            "u_sq_soln": u_sq,
            "dt_croco": dt_croco,
            "idx": idx,
            "scenario_name": scenario_name,
            "weight_x": weight_x,
            "weight_u": weight_u,
            "weight_xf": weight_xf,
            "weight_obs": weight_obs,
            "weight_obs_fin": weight_obs_fin,
            "alpha_log": alpha_log,          
            "fp_soln_store": fp_soln_store,
            "n_obstacles": n_obstacles,
            "fn_base": fn_base,
            "obstacles": obstacles,
        }            
            
    if data_soln is not None:
        save_to_pickle(data_soln, fp_soln_store)    
    

def create_callback_store_opt_result(fp_folder_storage, idx, scenario_name, weight_x, weight_u, weight_xf, 
                                     weight_obs, weight_obs_fin, alpha_log, dt_croco, robot_name, folder_scenarios):
    """
    Similar to `create_callback_mp` only that we do not do the forward simulation.
    """
    def task_done(future):
        
        try:
            X, U, t_solve = future.result()
        except TimeoutError as error:
            print("Timeout was triggered.")
            X = None
            t_solve = TIMEOUT_T_SOLVE
        except Exception as error:
            print("Uncaught exception when trying to access `future.result()` raised %s" % error)
            print(error)  # traceback of the function
            print("scenario_name: ", scenario_name)
            print("weight_x: ", weight_x)
            print("weight_u: ", weight_u)
            print("weight_xf: ", weight_xf)
            print("weight_obs: ", weight_obs)
            print("weight_obs_fin: ", weight_obs_fin)
            print("alpha_log: ", alpha_log)
            print("dt_croco: ", dt_croco)
            
            X = None
            U = None
            t_solve = TIMEOUT_UNHANDLED_EXCEPTION
            
        store_opt_result(X, U, t_solve, fp_folder_storage, idx, scenario_name, weight_x, weight_u, weight_xf, 
                         weight_obs, weight_obs_fin, alpha_log, dt_croco, robot_name, folder_scenarios)            

    return task_done