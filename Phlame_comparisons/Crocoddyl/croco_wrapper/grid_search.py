import os
import math
from itertools import product

from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
from pebble import ProcessPool
import pandas as pd
import ipdb
import pinocchio as pin
import example_robot_data

from croco_wrapper.run_urdf import gen_init_guess_from_q_traj
from croco_wrapper.kinova_obstacles import create_callback_store_opt_result
from croco_wrapper.kinova_obstacles import store_opt_result
from croco_wrapper.kinova_obstacles import run_single
from croco_wrapper.kinova_obstacles import get_fn_base


import sys
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), "../../src") ) )
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), ".." ) ) )
from common.fwd_sim import run_fwd_simulation_controller

from phlame.utils import cleanup_folder


class GridSearch:
    def __init__(self, folder_scenarios, fp_urdf, timeout, max_workers_solve, scenario_names, k_p_vec, k_v_vec, weight_u_vec, 
                 weight_x_vec, weight_xf_vec, weight_obs_vec, alpha_log_vec, dt_vec, t_total, frame_names, fp_folder_store,
                 robot_name, dt_integration, ths, max_workers_fwd, method_ivp, N, mode_init_guess):
        self.folder_scenarios = folder_scenarios
        self.fp_urdf = fp_urdf
        self.timeout = timeout
        self.max_workers_solve = max_workers_solve
        self.scenario_names = scenario_names
        self.k_p_vec = k_p_vec
        self.k_v_vec = k_v_vec
        self.weight_u_vec = weight_u_vec
        self.weight_x_vec = weight_x_vec
        self.weight_xf_vec = weight_xf_vec
        self.weight_obs_vec = weight_obs_vec
        self.alpha_log_vec = alpha_log_vec
        self.dt_vec = dt_vec
        self.t_total = t_total
        self.fp_folder_store = fp_folder_store
        self.dt_integration = dt_integration
        self.t_eval_fwd = np.arange(0.0, t_total + self.dt_integration, self.dt_integration)
        self.ths = ths
        self.fp_cases = os.path.join(fp_folder_store, "cases_names.txt")
        self.max_workers_fwd = max_workers_fwd
        self.method_ivp = method_ivp
        self.robot_model = example_robot_data.load(robot_name).model

        self.robot_name = robot_name
        self.frame_names = frame_names
        self.N = N
        self.mode_init_guess = mode_init_guess
    
    def generate_scenario_name_dt_to_IT(self, mode_init_guess, model: pin.Model):
        scenario_name_dt_to_IT = {scenario_idx: {} for scenario_idx in self.scenario_names}
        for scenario_name in self.scenario_names:
            for dt in self.dt_vec:
                fp_mat = os.path.join(self.folder_scenarios, f"{scenario_name}.mat")
                data_scenario = loadmat(fp_mat)
                q_traj = data_scenario['q_traj'].astype(dtype=np.double, order='F')
                N = q_traj.shape[1]
                q_start = q_traj[0, :].reshape(N, 1)
                q_end = q_traj[-1, :].reshape(N, 1)
                qd_start = data_scenario['qd_start'].astype(dtype=np.double, order='F').reshape(N, 1)
                qd_end = data_scenario['qd_end'].astype(dtype=np.double, order='F').reshape(N, 1)

                x0 = np.concatenate( ( q_start, qd_start ) ).astype(dtype=np.double, order='F')
                xf = np.concatenate( ( q_end, qd_end ) ).astype(dtype=np.double, order='F')

                num_times = math.ceil(self.t_total / dt)


                init_xs, init_us = gen_init_guess_from_q_traj(q_traj, N, num_times, x0, xf, mode_init_guess, model)

                scenario_name_dt_to_IT[scenario_name][dt] = (init_xs, init_us)
        return scenario_name_dt_to_IT

    def run_gs(self):
        """
        Runs the grid search solving only with croco.

        """

        # wipes without asking
        cleanup_folder(self.fp_folder_store)

        cases = list( product(self.weight_u_vec, self.weight_x_vec, self.weight_xf_vec, self.weight_obs_vec, 
                              self.alpha_log_vec, self.dt_vec, self.scenario_names) )

        print("Total number of cases: ", len(list(cases)) )
              
        file_names = []
        idx = 0
        for weight_u, weight_x, weight_xf, weight_obs, alpha_log, dt, scenario_name in cases:
            # Create and store filenames in a txt so we can use it to resume later
            fn_base = get_fn_base(idx, scenario_name, weight_x, weight_u, weight_xf, weight_obs, weight_obs, alpha_log, dt, self.robot_name)
            file_names.append(fn_base)
            idx += 1
            
        with open(self.fp_cases, 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in file_names))

        scenario_name_dt_to_IT = self.generate_scenario_name_dt_to_IT(self.mode_init_guess, self.robot_model)

        print("Finished creation of initial guesses")

        with ProcessPool(max_workers=self.max_workers_solve) as pool:
            idx = 0
            for weight_u, weight_x, weight_xf, weight_obs, alpha_log, dt_croco, scenario_name in cases:
            
                weight_obs_fin = weight_obs
                init_xs, init_us = scenario_name_dt_to_IT[scenario_name][dt_croco]
                future = pool.schedule(
                    run_single,
                    args=(self.folder_scenarios, scenario_name, weight_x, weight_u, weight_xf, weight_obs, weight_obs_fin,
                            alpha_log, dt_croco, self.robot_name, self.frame_names, self.t_total, init_xs, init_us),
                    timeout=self.timeout,
                )

                callback = create_callback_store_opt_result(
                    fp_folder_storage=self.fp_folder_store, 
                    idx=idx, 
                    scenario_name=scenario_name,
                    weight_x=weight_x,
                    weight_u=weight_u,
                    weight_xf=weight_xf, 
                    weight_obs=weight_obs,
                    weight_obs_fin=weight_obs_fin,
                    alpha_log=alpha_log,
                    dt_croco=dt_croco,
                    robot_name=self.robot_name,
                    folder_scenarios=self.folder_scenarios
                )
                
                future.add_done_callback(callback)
            
                idx += 1
            
        pool.join()
        pool.close()
        
    def run_gs_seq(self):
        """
        Runs the grid search solving only with croco.

        """

        # wipes without asking
        cleanup_folder(self.fp_folder_store)

        cases = list( product(self.weight_u_vec, self.weight_x_vec, self.weight_xf_vec, self.weight_obs_vec, 
                              self.alpha_log_vec, self.dt_vec, self.scenario_names) )

        print("Total number of cases: ", len(list(cases)) )
              
        file_names = []
        idx = 0
        for weight_u, weight_x, weight_xf, weight_obs, alpha_log, dt, scenario_name in cases:
            # Create and store filenames in a txt so we can use it to resume later
            fn_base = get_fn_base(idx, scenario_name, weight_x, weight_u, weight_xf, weight_obs, weight_obs, alpha_log, dt, self.robot_name)
            file_names.append(fn_base)
            idx += 1
            
        with open(self.fp_cases, 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in file_names))

        scenario_name_dt_to_IT = self.generate_scenario_name_dt_to_IT(self.mode_init_guess, self.robot_model)

        print("Finished creation of initial guesses")
        
        idx = 0
        for weight_u, weight_x, weight_xf, weight_obs, alpha_log, dt_croco, scenario_name in cases:
        
            weight_obs_fin = weight_obs
            init_xs, init_us = scenario_name_dt_to_IT[scenario_name][dt_croco]
            X, U, t_solve = run_single(self.folder_scenarios, scenario_name, weight_x, weight_u, weight_xf, weight_obs, weight_obs_fin,
                            alpha_log, dt_croco, self.robot_name, self.frame_names, self.t_total, init_xs, init_us)
            
            store_opt_result(X, U, t_solve, fp_folder_storage=self.fp_folder_store, idx=idx, 
                             scenario_name=scenario_name, weight_x=weight_x, weight_u=weight_u,
                             weight_xf=weight_xf, weight_obs=weight_obs, weight_obs_fin=weight_obs_fin,
                             alpha_log=alpha_log, dt_croco=dt_croco, robot_name=self.robot_name,
                             folder_scenarios=self.folder_scenarios)
            
            idx += 1             

    def run_fwd_sim_single(self, fp_pkl, k_p, k_v):
        data_pkl = pd.read_pickle(fp_pkl)
        fn_base = data_pkl['fn_base']
        obstacles = data_pkl.get('obstacles')
        if obstacles is None:
            obstacles = np.nan

        if data_pkl.get('u_sq_soln') is not None:
                success_croco = 1
        else:
            success_croco = 0

        if success_croco == 0:
            res = {
                'success_croco': success_croco,
                "k_p": k_p,
                "k_v": k_v,
                "fp_pkl": fp_pkl,
                "fn_base": fn_base,
                "obstacles": obstacles,
                "fp_urdf": self.fp_urdf,
            }
            res['success_fwd_sim'] = 1
        else:
            U = data_pkl['U']
            X = data_pkl['X']
            X0 = data_pkl['X0']

            if data_pkl.get('dt_croco'):
                dt_opt = data_pkl['dt_croco']
            elif data_pkl.get('dt'):
                dt_opt = data_pkl['dt']
            else:
                raise("dt or dt_croco not in pickle: ", fp_pkl)

            num_times = math.ceil(self.t_total / dt_opt)

            try:
                solution, u_sq_real = run_fwd_simulation_controller(U, X, k_p, k_v, self.fp_urdf, self.dt_integration, 
                                                                    dt_opt, self.t_total, num_times, X0, method_ivp=self.method_ivp)
                res = {
                    'success_croco': 1,
                    'u_sq_real': u_sq_real,
                    'fn_base': fn_base,
                    "q_traj_soln": X[:, 0:self.N],
                    "X_soln": X,
                    "U_soln": U,
                    "q_traj_fwd": solution.y[0:self.N, :],
                    "qd_traj_fwd": solution.y[self.N:, :],
                    "obstacles": obstacles,
                    "fp_urdf": self.fp_urdf,
                    "solution_fwd": solution.y,
                    "k_p": k_p,
                    "k_v": k_v,
                    "fp_pkl": fp_pkl,
                }
                res['success_fwd_sim'] = 1
            except Exception as error:
                print("Unacaught exception in forward simulation: ", error)
                print("fp_pkl: ", fp_pkl)

                res = {
                    'success_croco': success_croco,
                    "k_p": k_p,
                    "k_v": k_v,
                    "fp_pkl": fp_pkl,
                    "fn_base": fn_base,
                    "obstacles": obstacles,
                    "fp_urdf": self.fp_urdf,
                }
                res['success_fwd_sim'] = 0

        return res

    def process_fwd_res(self, res):
        # if res['success_croco'] == 1:
        fn_base = "res_fwd_sim_" + res['fn_base'] +f"-k_p={res['k_p']}-k_v={res['k_v']}"+ "_data_m.mat"
        fp_mat = os.path.join(self.fp_folder_store, fn_base)
        res['fp_mat_store'] = fp_mat
        if res.get('obstacles') is None:
            res['obstacles'] = np.nan
        savemat(fp_mat, res)

    def create_callback_store_fwd_result(self):
        def task_done(future):
            try:
                res = future.result()
            except Exception as error:
                print("future.result() : uncaught exception in callback for fwd result: ", error)
                res = {'success_fwd_sim': 0}
            
            self.process_fwd_res(res)

        return task_done


    def run_fwd_sim_all(self, k_p_list, k_v_list, parallel=True):
        # get all pickles that start with "croco_res"

        pkl_files = filter(lambda x: x.startswith("croco_res"), os.listdir(self.fp_folder_store) )
        if parallel:
            with ProcessPool(max_workers=self.max_workers_fwd) as pool:
                for (pkl_fn, k_p, k_v) in product(pkl_files, k_p_list, k_v_list):
                    fp_pkl = os.path.join(self.fp_folder_store, pkl_fn)
                    future = pool.schedule(
                        self.run_fwd_sim_single,
                        args=(fp_pkl, k_p, k_v)
                    )
                    callback = self.create_callback_store_fwd_result()
                    future.add_done_callback(callback)
        else:
            for (pkl_fn, k_p, k_v) in product(pkl_files, k_p_list, k_v_list):
                fp_pkl = os.path.join(self.fp_folder_store, pkl_fn)
                res_fwd_sim = self.run_fwd_sim_single(fp_pkl, k_p, k_v)
                self.process_fwd_res(res_fwd_sim)


    def get_row_df(self, mat_fn):
        fp_mat = os.path.join(self.fp_folder_store, mat_fn)
        data_mat = loadmat(fp_mat)
        pkl_fn = data_mat['fp_pkl'][0]

        fp_pkl = os.path.join(self.fp_folder_store, pkl_fn)
        data_pkl = pd.read_pickle(fp_pkl)


        row_df = {
            "k_p": data_mat['k_p'][0][0],
            "k_v": data_mat['k_v'][0][0],
            "scenario_name": data_pkl['scenario_name'],
            "weight_x": data_pkl['weight_x'],
            "weight_u": data_pkl['weight_u'],
            "weight_xf": data_pkl['weight_xf'],
            "weight_obs": data_pkl['weight_obs'],
            "weight_obs_fin": data_pkl['weight_obs_fin'],
            "alpha_log": data_pkl['alpha_log'],
            "dt_croco": data_pkl['dt_croco'],
            "t_solve": data_pkl['t_solve'],
            "n_obstacles": data_pkl['n_obstacles'],
            "fp_pkl": data_pkl.get('fp_soln_store'),
            "fp_mat": fp_mat,
        }

        if data_pkl.get('u_sq_soln') is not None:
            row_df['u_sq_soln'] = data_pkl.get('u_sq_soln')
            row_df['t_solve'] = data_pkl.get('t_solve')
            X0 = data_pkl['X0']
            Xf = data_pkl['Xf']

            # load the corresponding .mat file
            # if it doesn't exist it means that either croco failed or the fwd simulation failed
            # mat_fp = os.path.join(self.fp_folder_store, "res_fwd_sim_" + data_pkl['fn_base'] + "_data_m.mat")
            # if os.path.isfile(mat_fp):
                # data_mat = loadmat(mat_fp)
            row_df['u_sq_real'] = data_mat['u_sq_real']
            solution_fwd = data_mat['solution_fwd']

            # Get the absolute difference in the final state between the croco integrated solution and the BC
            dif_fin_state = np.abs( solution_fwd[:, -1] - Xf.squeeze() )
            curr_inf_norm = np.linalg.norm(solution_fwd[:, -1] - Xf.squeeze(), ord=np.inf)

            # compute inf norm of position, first N states are positions
            inf_norm_pos = np.linalg.norm(solution_fwd[:self.N, -1] - Xf.squeeze()[:self.N], ord=np.inf)
            inf_norm_vel = np.linalg.norm(solution_fwd[self.N:, -1] - Xf.squeeze()[self.N:], ord=np.inf)

            row_df['inf_norm'] = curr_inf_norm
            row_df['inf_norm_pos'] = inf_norm_pos
            row_df['inf_norm_vel'] = inf_norm_vel

            for ii in range(self.N):
                row_df[f"abs_diff_q_{ii+1}"] = dif_fin_state[ii]
                row_df[f"abs_diff_qd_{ii+1}"] = dif_fin_state[self.N + ii]

            for ii, th in enumerate(self.ths):
                row_df[f"th_{ii+1}"] = th
                row_df[f"success_th_{ii+1}"] = (row_df["inf_norm"] <= th).astype(int)
                row_df[f"success_th_pos{ii+1}"] = (row_df["inf_norm_pos"] <= th).astype(int)
                row_df[f"success_th_vel{ii+1}"] = (row_df["inf_norm_vel"] <= th).astype(int)

        return row_df            

    def create_result_df(self):
        mat_files = filter(lambda x: x.startswith("res_fwd_sim_"), os.listdir(self.fp_folder_store) )
        rows = []

        for mat_file in mat_files:
            rows.append( self.get_row_df(mat_file) )
        
        fp_df = os.path.join(self.fp_folder_store, "data_table_croco.csv")
        df = pd.DataFrame(rows)
        df.to_csv(fp_df)