import os
import math
import re
from itertools import product

from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
from pebble import ProcessPool
import pandas as pd
import ipdb
import pinocchio as pin
import example_robot_data


import sys
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), "../../src") ) )
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), ".." ) ) )


from aligator_wrapper.kinova_obstacles import get_fn_base
from aligator_wrapper.kinova_obstacles import get_x0_xf_obstacles
from aligator_wrapper.kinova_obstacles import create_callback_store_opt_result
from aligator_wrapper.run_urdf import gen_init_guess_from_q_traj
from aligator_wrapper.run_urdf import run_single

from common.fwd_sim import run_fwd_simulation_controller
from common.utils import cleanup_folder


class GridSearch:
    def __init__(self, folder_scenarios, fp_urdf, timeout, max_workers_solve, scenario_names, k_p_vec, k_v_vec, weight_u_vec, weight_x_vec, weight_xf_vec,
                dt_vec, t_total, frame_names, fp_folder_store, robot_name, dt_integration, ths, max_workers_fwd, method_ivp, N, tol_vec, mu_init_vec,
                max_iters_vec, max_threads_aligator, verbose_level_aligator, mode_init_guess):
        self.folder_scenarios = folder_scenarios
        self.fp_urdf = fp_urdf
        self.timeout = timeout
        self.max_workers_solve = max_workers_solve
        self.scenario_names = scenario_names
        self.k_p_vec = k_p_vec
        self.k_v_vec =k_v_vec
        self.weight_u_vec = weight_u_vec
        self.weight_x_vec = weight_x_vec
        self.weight_xf_vec = weight_xf_vec
        self.dt_vec = dt_vec
        self.t_total = t_total
        self.fp_folder_store = fp_folder_store
        self.dt_integration = dt_integration
        self.t_eval_fwd = np.arange(0.0, t_total + self.dt_integration, self.dt_integration)
        self.ths = ths
        self.fp_cases = os.path.join(fp_folder_store, "cases_names.txt")
        self.max_workers_fwd = max_workers_fwd
        self.method_ivp = method_ivp

        self.tol_vec = tol_vec
        self.mu_init_vec = mu_init_vec
        self.max_iters_vec = max_iters_vec
        self.max_threads_aligator = max_threads_aligator
        self.verbose_level_aligator = verbose_level_aligator

        self.robot_name = robot_name
        self.frame_names = frame_names
        self.N = N
        self.u0 = np.zeros(N) # used to compute the cost function \int f( u(t) - u0 ) dt
        self.mode_init_guess = mode_init_guess
        self.robot_model = example_robot_data.load(robot_name).model

    
    def generate_scenario_name_dt_to_IT(self, mode_init_guess, model: pin.Model):
        # Literal copy from croco_wrapper
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
                init_xs, init_us = gen_init_guess_from_q_traj(q_traj, N, num_times, x0, xf, mode_init_guess, model, dt)
                scenario_name_dt_to_IT[scenario_name][dt] = (init_xs, init_us)
        return scenario_name_dt_to_IT

    def get_diff_cases(self, list_all_files, pkl_files):
        # list_all_files: does not have the pkl extension
        # pkl_files: have the .pkl extension

        if not list_all_files[0].startswith('aligator_res_'):
            list_all_files = [f"aligator_res_{fn}data_soln" for fn in list_all_files]

        fns_all = [f"{fn}.pkl" for fn in list_all_files]

        left_cases = list( set(fns_all) - set(pkl_files) )
        return left_cases
  
    def get_cases_from_fns(self, left_files):
        # left_files: contains the .pkl extension
        # example: kinova_constrained_idx=0-sc_name=scenario_12-weight_x=1e-06-weight_u=1e-05-weight_xf=1e-06-dt=0.01-robot_name=kinova_gen3-tol=0.001-mu_init=0.01-max_iters=1000
        # we want to extract (weight_u, weight_x, weight_xf, dt, sc_name, tol, mu_init, max_iters)

        # Define regex patterns for each target key
        patterns = {
            'idx': r'idx=(\d+)',
            'weight_u': r'weight_u=([\d.eE+-]+)(?=-|$)',
            'weight_x': r'weight_x=([\d.eE+-]+)(?=-|$)',
            'weight_xf': r'weight_xf=([\d.eE+-]+)(?=-|$)',
            'dt': r'dt=([\d.eE+-]+)(?=-|$)',
            'sc_name': r'sc_name=([\w\d_]+)(?=-|$)',
            'tol': r'tol=([\d.eE+-]+)(?=-|$)',
            'mu_init': r'mu_init=([\d.eE+-]+)(?=-|$)',
            'max_iters': r'max_iters=([0-9.eE+-]+)'
        }

        results = []  # Initialize a list to hold all the results

        # Iterate through each file string in left_files
        for file_str in left_files:
            case_data = {}  # Initialize dictionary to store values for the current string
            
            # Extract each target value using the corresponding regex pattern
            for key, pattern in patterns.items():

                # if key == "max_iters":
                    # ipdb.set_trace()

                match = re.search(pattern, file_str)  # Search for the pattern in the string
                if match:
                    value = match.group(1)  # Extract the value (first capturing group)

                    # Convert to float if not 'sc_name'
                    if (key == 'max_iters') or (key == 'idx'):
                        case_data[key] = int(value)
                    elif key == 'sc_name':
                        case_data[key] = value
                    else:
                        case_data[key] = float(value)
            
            # create tuple of values in the correct order
            vals = (
                case_data['idx'],
                case_data['weight_u'],
                case_data['weight_x'],
                case_data['weight_xf'],
                case_data['dt'],
                case_data['sc_name'],
                case_data['tol'],
                case_data['mu_init'],
                case_data['max_iters'],
            )
            
            results.append(vals)  # Append the dictionary to the results list

        return results

    
    def get_left_cases(self):
        # Read txt file
        with open(self.fp_cases, 'r') as file:
            list_all_files = [line.strip() for line in file]

        # Get list of all pkls created
        pkl_files = os.listdir(self.fp_folder_store)

        # Get the set of cases that is not in pkls
        left_files = self.get_diff_cases(list_all_files, pkl_files)

        # Extract weight_u, weight_x, weight_xf, dt_alig, scenario_name, tol, mu_init, max_iters and create cases variable
        left_cases = self.get_cases_from_fns(left_files)
        return left_cases        

    def continue_gs(self, using_pool=True):
        
        left_cases = self.get_left_cases()
        print("Left number of cases: ", len(list(left_cases)) )

        while len(left_cases) > 0:

            # Create scenario_name_dt_to_IT
            scenario_name_dt_to_IT = self.generate_scenario_name_dt_to_IT(self.mode_init_guess, self.robot_model)

            # Run parallel run
            print("Finished creation of initial guesses")
            if using_pool:
                with ProcessPool(max_workers=self.max_workers_solve) as pool:
                    for idx, weight_u, weight_x, weight_xf, dt_alig, scenario_name, tol, mu_init, max_iters in left_cases:
                        init_xs, init_us = scenario_name_dt_to_IT[scenario_name][dt_alig]
                        x0, xf, obstacles = get_x0_xf_obstacles(scenario_name, self.folder_scenarios)
                        future = pool.schedule(
                            run_single,
                            args=(init_xs, init_us, self.u0, x0, xf, obstacles, self.frame_names, self.robot_name, dt_alig, self.t_total, weight_x, weight_u, weight_xf,
                                tol, self.max_threads_aligator, mu_init, max_iters, self.verbose_level_aligator),
                            timeout=self.timeout,
                        )

                        callback = create_callback_store_opt_result(self.fp_folder_store, idx, scenario_name, 
                                                                    weight_x, weight_u, weight_xf, dt_alig, self.folder_scenarios, 
                                                                    self.robot_name, tol, mu_init, max_iters)
                        future.add_done_callback(callback)
            else:
                for idx, weight_u, weight_x, weight_xf, dt_alig, scenario_name, tol, mu_init, max_iters in cases:
                    init_xs, init_us = scenario_name_dt_to_IT[scenario_name][dt_alig]
                    x0, xf, obstacles = get_x0_xf_obstacles(scenario_name, self.folder_scenarios)
                    try:
                        X, U, t_solve = run_single(init_xs, init_us, self.u0, x0, xf, obstacles, self.frame_names, self.robot_name, dt_alig, self.t_total, weight_x, weight_u, weight_xf,
                                    tol, self.max_threads_aligator, mu_init, max_iters, self.verbose_level_aligator)
                    except Exception as error:
                        print("Uncaught exception when trying to access `future.result()` raised %s" % error)
                        print(error)  # traceback of the function                        

            left_cases = self.get_left_cases()
            print("Left number of cases: ", len(list(left_cases)) )

    
    def run_gs(self, using_pool=True):
        cleanup_folder(self.fp_folder_store)

        cases = list( product( self.weight_u_vec, self.weight_x_vec, self.weight_xf_vec, self.dt_vec, self.scenario_names, \
                              self.tol_vec, self.mu_init_vec, self.max_iters_vec ))

        print("Total number of cases: ", len(list(cases)) )
        file_names = []
        idx = 0

        for weight_u, weight_x, weight_xf, dt_alig, scenario_name, tol, mu_init, max_iters in cases:
            fn_base = get_fn_base(idx, scenario_name, weight_x, weight_u, weight_xf, dt_alig, self.robot_name, tol, mu_init, max_iters)
            file_names.append( 'aligator_res_' + fn_base + "data_soln")
            idx += 1
        
        with open(self.fp_cases, 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in file_names))

        scenario_name_dt_to_IT = self.generate_scenario_name_dt_to_IT(self.mode_init_guess, self.robot_model)

        print("Finished creation of initial guesses")
        if using_pool:
            with ProcessPool(max_workers=self.max_workers_solve) as pool:
                idx = 0
                for weight_u, weight_x, weight_xf, dt_alig, scenario_name, tol, mu_init, max_iters in cases:
                    init_xs, init_us = scenario_name_dt_to_IT[scenario_name][dt_alig]
                    x0, xf, obstacles = get_x0_xf_obstacles(scenario_name, self.folder_scenarios)
                    future = pool.schedule(
                        run_single,
                        args=(init_xs, init_us, self.u0, x0, xf, obstacles, self.frame_names, self.robot_name, dt_alig, self.t_total, weight_x, weight_u, weight_xf,
                            tol, self.max_threads_aligator, mu_init, max_iters, self.verbose_level_aligator),
                        timeout=self.timeout,
                    )

                    callback = create_callback_store_opt_result(self.fp_folder_store, idx, scenario_name, 
                                                                weight_x, weight_u, weight_xf, dt_alig, self.folder_scenarios, 
                                                                self.robot_name, tol, mu_init, max_iters)
                    future.add_done_callback(callback)
                    idx += 1
        else:
            # INcomplete implementation
            raise("Not implemented!")
            idx = 0
            for weight_u, weight_x, weight_xf, dt_alig, scenario_name, tol, mu_init, max_iters in cases:
                init_xs, init_us = scenario_name_dt_to_IT[scenario_name][dt_alig]
                x0, xf, obstacles = get_x0_xf_obstacles(scenario_name, self.folder_scenarios)
                try:
                    X, U, t_solve = run_single(init_xs, init_us, self.u0, x0, xf, obstacles, self.frame_names, self.robot_name, dt_alig, self.t_total, weight_x, weight_u, weight_xf,
                                tol, self.max_threads_aligator, mu_init, max_iters, self.verbose_level_aligator)
                except Exception as error:
                    print("Uncaught exception when trying to access `future.result()` raised %s" % error)
                    ipdb.set_trace()
                    print(error)  # traceback of the function


    def run_fwd_sim_single(self, fp_pkl, k_p, k_v):
        # An almost literal copy from croco_wrapper.grid_search
        data_pkl = pd.read_pickle(fp_pkl)
        fn_base = data_pkl['fn_base']
        obstacles = data_pkl.get('obstacles')
        if obstacles is None:
            obstacles = np.nan        
        
        if data_pkl.get('u_sq_soln') is not None:
                success_alig = 1
        else:
            success_alig = 0

        if success_alig == 0:
            res = {
                'success_alig': success_alig,
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

            if data_pkl.get('dt_alig'):
                dt_opt = data_pkl['dt_alig']
            else:
                raise("dt_alig not present in pickle: ", fp_pkl)

            num_times = math.ceil(self.t_total / dt_opt)


            try:
                solution, u_sq_real = run_fwd_simulation_controller(U, X, k_p, k_v, self.fp_urdf, self.dt_integration, 
                                                                    dt_opt, self.t_total, num_times, X0, method_ivp=self.method_ivp)
                res = {
                    'success_alig': 1,
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
                    'success_alig': success_alig,
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
        # Literal copy from croco_wrapper.grid_search
        # if res['success_alig'] == 1:
        fn_base = "res_fwd_sim_" + res['fn_base'] +f"-k_p={res['k_p']}-k_v={res['k_v']}"+ "_data_m.mat"
        fp_mat = os.path.join(self.fp_folder_store, fn_base)
        res['fp_mat_store'] = fp_mat
        if res.get('obstacles') is None:
            res['obstacles'] = np.nan

        savemat(fp_mat, res)
    
    def create_callback_store_fwd_result(self):
        # Literal copy from croco_wrapper.grid_search
        def task_done(future):
            try:
                res = future.result()
            except Exception as error:
                print("Unacaught exception in forward simulation: ", error)
                res = {'success_fwd_sim': 0}
            
            self.process_fwd_res(res)

        return task_done
    
    def run_fwd_sim_all(self, k_p_list, k_v_list, parallel=True):
        # Literal copy from croco_wrapper.grid_search

        # get all pickles that start with "croco_res"
        pkl_files = filter(lambda x: x.startswith("aligator_res"), os.listdir(self.fp_folder_store) )
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
        # Modified from croco_wrapper.grid_search, but mostly similar.
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
            "dt_alig": data_pkl['dt_alig'],
            "t_solve": data_pkl['t_solve'],
            "n_obstacles": data_pkl['n_obstacles'], 
            "tol": data_pkl['tol'],
            "mu_init": data_pkl['mu_init'],
            "max_iters": data_pkl['max_iters'],
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

        for mat_fn in mat_files:
            rows.append( self.get_row_df(mat_fn) )
        
        fp_df = os.path.join(self.fp_folder_store, "data_table_alig.csv")
        df = pd.DataFrame(rows)
        df['mode_init_guess'] = self.mode_init_guess
        df.to_csv(fp_df)
    