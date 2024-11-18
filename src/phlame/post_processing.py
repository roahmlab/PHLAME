"""
    The main goal of this class/functions/code is to generate a csv that summarizes the result
    of a set of experiments.
    It will have a secondary functionality of generating the tables that will be used in the paper.
"""

from concurrent.futures import ProcessPoolExecutor
from .utils import load_from_pickle
from .utils import save_to_pickle
from .utils import extract_substring_after
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from .aghf import PostAghf
from .fwd_simulation import ode_fwd_integrate
from .fwd_simulation import get_u_curr
import ipdb
from .utils import cleanup_folder
from scipy.integrate import solve_ivp
from typing import List, Dict, Any
import re
from scipy.io import loadmat
from scipy.io import savemat
import math
from .constants import ResultType

VALID_STOPPING_CRITERIA = ["u_sq", "acc_diff"]

class PostProcessing:
    def __init__(self, fp_folder_input, fp_folder_output, dt, stopping_criteria="u_sq"):
        """
        Args:
            - dt (float): Time step used in the integration to obtain u squared.
            - stopping_criteria (str): Criteria used to choose the best_s, can be ["u_sq", "acc_diff"].

        Returns:
            - None
        """
        self.fp_folder_input = fp_folder_input
        self.fp_folder_output = fp_folder_output
        cleanup_folder(self.fp_folder_output)
        self.dt = dt
        
        if stopping_criteria in VALID_STOPPING_CRITERIA:
            self.stopping_criteria = stopping_criteria
        else:
            raise(f"stopping criteria not valid because it is not one of {VALID_STOPPING_CRITERIA}(variable VALID_STOPPING_CRITERIA)")

    
    def process_pickle(self, fp):
        """
        Get list of all files in folder. (they should be pickles)
        
        In parallel, for each file:
            - Load pickle
            - Create dictionary with the row data
            - Return dictionary
        
        Create dataframe from list of dictionaries.
        Store dataframe as csv in `fp_folder_output`.

        Args:
            - folder_path (str): Path to the folder containing pickle files.
            - fp_folder_output (str): Path to the folder where the output CSV will be stored.

        Returns:
            - None
        """
        post_aghf = PostAghf(fp)
        
        # p, s_max, k, abs_tol, rel_tol, N
        p = post_aghf.pset.p
        s_max = post_aghf.pset.s_max
        k = post_aghf.pset.k
        abs_tol = post_aghf.pset.abs_tol
        rel_tol = post_aghf.pset.rel_tol
        N = post_aghf.N
        t_solve = post_aghf.first_result.t_solve
        run_name = post_aghf.pset.name
        ic_type = post_aghf.pset.IC_type
        scenario_name = post_aghf.pset.scenario_name

        k_cons = getattr(post_aghf.pset, "k_cons", None)
        c_cons = getattr(post_aghf.pset, "c_cons", None)
        
        # cout_ini, cout_fin, action_function_ini, action_function_fin
        # dif_velocities_ini, dif_velocities_end
        # dif_accelerations_ini, dif_accelerations_end
        # norm_u2_ini, norm_u2_end
        sol = post_aghf.first_result.sol
        
        if len(sol) == 0:
            data_dict = {
                'p': p,
                's_max': s_max,
                'k': k,
                'abs_tol': abs_tol,
                'rel_tol': rel_tol,
                'N': N,
                't_solve': 9999,
                'action_functional_ini': 0,
                'action_functional_fin': 0,
                'vel_dif_ini': 0,
                'vel_dif_fin': 0,
                'accel_dif_ini': 0,
                'accel_dif_fin': 0,
                'norm_ps_values_ds_ini': 0,
                'norm_ps_values_ds_fin': 0,
                'norm_u2_ini': 0,
                'norm_u2_fin': 0,
                'fp': fp,
                'name': run_name,
                'ic_type': ic_type,
                'x0': str(post_aghf.X0),
                'xf': str(post_aghf.Xf),
                'scenario_name': scenario_name,
            }
        else:
            action_functional_vec, vel_dif_vec, accel_dif_vec, norm_dps_values_ds_vec,\
                u_sq_vec, real_s_span, Q_mat, Qd_mat, Qdd_mat, U_mat = \
                    post_aghf.compute_along_s(self.dt, sol)
            
            name_to_data = {'action_functional': action_functional_vec, 'vel_dif': vel_dif_vec,
                            'accel_dif': accel_dif_vec, 'norm_dps_values_ds': norm_dps_values_ds_vec,
                            'u_sq': u_sq_vec}
            data_dict = {
                'p': p,
                's_max': s_max,
                'k': k,
                'abs_tol': abs_tol,
                'rel_tol': rel_tol,
                'N': N,
                't_solve': t_solve,
                'action_functional_ini': action_functional_vec[0],
                'action_functional_fin': action_functional_vec[-1],
                'vel_dif_ini': vel_dif_vec[0],
                'vel_dif_fin': vel_dif_vec[-1],
                'accel_dif_ini': accel_dif_vec[0],
                'accel_dif_fin': accel_dif_vec[-1],
                'norm_ps_values_ds_ini': norm_dps_values_ds_vec[0],
                'norm_ps_values_ds_fin': norm_dps_values_ds_vec[-1],
                'norm_u2_ini': u_sq_vec[0],
                'norm_u2_fin': u_sq_vec[-1],
                'fp': fp,
                'real_s_span': real_s_span,
                'name': run_name,
                'ic_type': ic_type,
                'x0': str(post_aghf.X0),
                'xf': str(post_aghf.Xf),       
                'scenario_name': scenario_name,         
            }

            # add vectors of values along the way for plotting
            for name, vec in name_to_data.items():
                data_dict[name] = vec

            # adding best values and the s value corresponding to the index of the best
            # for name, vec in name_to_data.items():
            #     best_idx = np.argmin(vec)
            #     data_dict[f'best_{name}'] = vec[best_idx]
            #     data_dict[f'best_idx_{name}'] = best_idx
            #     data_dict[f'best_s_{name}'] = real_s_span[best_idx]

            # we obtain the best idx and s with respect to u_sq
            # then we report the corresponding values for vel_dif, accel_dif, norm_dps_values_ds,
            # action functional
                
            if self.stopping_criteria == "u_sq":
                best_idx = np.argmin(u_sq_vec)
            elif self.stopping_criteria == "acc_diff":
                best_idx = np.argmin(accel_dif_vec)
            else:
                raise(f"stopping_criteria = {self.stopping_criteria} not supported.")


            data_dict['best_idx'] = best_idx
            data_dict['best_s'] = real_s_span[best_idx]
            data_dict['best_u_sq'] = u_sq_vec[best_idx]
            data_dict['best_vel_dif'] = vel_dif_vec[best_idx]
            data_dict['best_accel_dif'] = accel_dif_vec[best_idx]
            data_dict['best_norm_dps_values_ds'] = norm_dps_values_ds_vec[best_idx]

            # add trajectories
            data_dict['Q_mat'] = Q_mat
            data_dict['Qd_mat'] = Qd_mat
            data_dict['Qdd_mat'] = Qdd_mat
            data_dict['U_mat'] = U_mat

        if c_cons is not None:
            data_dict['c_cons'] = c_cons
        if k_cons is not None:
            data_dict['k_cons'] = k_cons

        return data_dict
    
    def process_pickle_simple(self, fp):
        post_aghf = PostAghf(fp)

    def run(self, fn_table, fn_all, max_workers=24, light=False):
        """
        Processes pickle files in parallel, extracts relevant data, and saves the results to CSV and pickle files.

        Args:
            - fn_table (str): Filename for the CSV table that stores the results without data along the way.
            - fn_all (str): Filename for the pickle file that stores the results with data along the way.
            - max_workers (int, optional): Maximum number of worker processes to use. Default is 24.
            - light (bool, optional): If True, processes the data in a lighter mode. Default is False.

        Returns:
            - tuple: A tuple containing two DataFrames:
                - df (pd.DataFrame): DataFrame with results without data along the way.
                - df_all (pd.DataFrame): DataFrame with results including data along the way.
        """
        files = os.listdir(self.fp_folder_input)
        fps = [os.path.join(self.fp_folder_input, f) for f in files if f.endswith(".pkl")]


        with ProcessPoolExecutor(max_workers) as executor:
            
            results = list(tqdm(executor.map(self.process_pickle, fps), total=len(fps)))
            # results = list(tqdm(executor.map(self.load_pickle, fps), total=len(fps)))

        rows_np_arrays = ['action_functional', 'vel_dif', 'accel_dif', 'norm_dps_values_ds', 'u_sq',
                          'real_s_span', 'Q_mat', 'Qd_mat', 'Qdd_mat', 'U_mat']
        results_csv = []
        for r in results:
            # create dictionary with the keys that are in r but not in rows_np_arrays
            r_csv = {k: r[k] for k in r if k not in rows_np_arrays}

            if 'c_cons' in r:
                r_csv['c_cons'] = r['c_cons']
            
            if 'k_cons' in r:
                r_csv['k_cons'] = r['k_cons']

            results_csv.append(r_csv)

        # create a dataframe that has no information along the way, this is easier to look at
        df = pd.DataFrame(results_csv)
        df.to_csv(os.path.join(self.fp_folder_output, fn_table) )

        # create a dataframe that has the information along the way, this is useful for plotting
        df_all = pd.DataFrame(results)
        save_to_pickle( df_all, os.path.join(self.fp_folder_output, fn_all) )
        return df, df_all

    def run_simple_csv(self, fn_table, max_workers=24):
        """
        Computes the values for the solution that depend only on the initial and final trajectory and saves the results to a CSV file.

        Args:
            - fn_table (str): Filename for the CSV table that stores the results.
            - max_workers (int, optional): Maximum number of worker processes to use. Default is 24.

        Returns:
            - None
        """
        files = os.listdir(self.fp_folder_input)
        fps = [os.path.join(self.fp_folder_input, f) for f in files]

        with ProcessPoolExecutor(max_workers) as executor:
            results = list(tqdm(executor.map(self.process_pickle_simple, fps), total=len(fps)))

class PostProcessingFwdSuccess():
    """
    was designed with the 1 to 5 link pendulum case
    """
    def __init__(self, fp_table_csv, fp_table_output, dt, ths, fp_folder_pickles, t_total, 
                 k_controller_list, N_to_fp_urdf, fp_ics=None, update_scenario_idx=True,
                 fp_folder_fwd_store=None, fp_folder_scenarios=None, method_int='RK45', 
                 force_s_max=False):
        """
        Initializes the PostProcessingFwdSuccess class.

        Args:
            - fp_table_csv (str): File path to the CSV table containing the results.
            - fp_table_output (str): File path to the output table.
            - dt (float): Time step to determine the grid over which the ODE integrator will return results.
            - ths (List[float]): List of thresholds to measure the success of a trial based on the max inf norm.
            - fp_folder_pickles (str): Folder path to store pickles.
            - t_total (float): Total simulation time.
            - k_controller_list (List[Tuple[float, float]]): List of tuples (kp, kv) representing gains for the controller used in forward simulation.
            - N_to_fp_urdf (dict): Dictionary mapping the number of links (N) to the file path of the URDF file.
            - fp_ics (str, optional): File path to the initial conditions. Default is None.
            - update_scenario_idx (bool, optional): Flag to update the scenario index. Default is True.
            - fp_folder_fwd_store (str, optional): Folder path to store forward simulation results. Default is None.
            - fp_folder_scenarios (str, optional): Folder path to the scenarios. Default is None.
            - method_int (str, optional): Integration method for the ODE solver. Default is 'RK45'.
            - force_s_max (bool, optional): Flag to force the maximum s value. Default is False.

        Returns:
            - None
        """
        self.fp_table_csv = fp_table_csv
        self.fp_table_output = fp_table_output
        self.dt = dt
        self.fp_ics = fp_ics
        self.ths = ths
        self.fp_folder_pickles = fp_folder_pickles
        self.t_total = t_total
        self.k_controller_list = k_controller_list
        self.update_scenario_idx = update_scenario_idx
        self.fp_folder_fwd_store = fp_folder_fwd_store
        self.fp_folder_scenarios = fp_folder_scenarios
        self.method_int = method_int
        self.force_s_max = force_s_max

        # NOTE: This is making a big assumption: that there will be ONLY one robot with N=1, N=2, ... links
        self.N_to_fp_urdf = N_to_fp_urdf

        if fp_ics is not None:
            self._load_ic() 

        self._generate_t_sim()
        self.input_df = pd.read_csv(fp_table_csv)
        self.max_N = self.input_df["N"].max()

    def _load_ic(self):
        """
        Loads initial conditions from a pickle file.

        This function loads the initial and final state matrices from a specified pickle file and sets the number of cases.

        Args:
            - self (PostProcessingFwdSuccess): The instance of the PostProcessingFwdSuccess class.

        Returns:
            - None
        """
        self.data_pkl = load_from_pickle(self.fp_ics)
        self.X0_mat = self.data_pkl['X0_mat']
        self.Xf_mat = self.data_pkl['Xf_mat']
        self.num_cases = self.X0_mat.shape[1]
    
    def _generate_t_sim(self):
        """
        Generates the simulation time vector.

        This function calculates the number of points based on the total simulation time and the time step,
        and generates a linearly spaced time vector from -1 to 1.

        Args:
            - self (PostProcessingFwdSuccess): The instance of the PostProcessingFwdSuccess class.

        Returns:
            - None
        """
        num_points = int( ( self.t_total / self.dt ) + 1)
        self.t_sim = np.linspace(-1, 1, num_points)

    def _get_pkl_fn(self, fp_pkl):
        """
        Extracts the filename from a given file path.

        Args:
            - fp_pkl (str): The file path to extract the filename from.

        Returns:
            - str: The extracted filename.
        """
        return os.path.basename(fp_pkl)

    def run_fwd_sim(self, fp_urdf, fp_pickle, s_query, k_p, k_v):
        """
        Runs a forward simulation using the AGHF solution and a specified controller.

        Args:
            - fp_urdf (str): File path to the URDF file.
            - fp_pickle (str): File path to the pickle file containing the AGHF solution.
            - s_query (float): The specific s value at which to evaluate the solution.
            - k_p (float): Proportional gain for the controller.
            - k_v (float): Derivative gain for the controller.

        Returns:
            - solution (OdeSolution): The solution object returned by solve_ivp, where solution.y is the solution trajectory.
            - u_sq_aghf (float): The u squared value obtained from the AGHF solution.
            - u_sq_real (float): The u squared value obtained when using the controller.
            - U_sol (np.ndarray): The control signal matrix from the AGHF solution.
            - U_actual (np.ndarray): The actual control signal matrix obtained from the forward simulation.
            - X_spec_t (np.ndarray): The desired state trajectory matrix.
        """
        result = load_from_pickle(fp_pickle)
        post_aghf = PostAghf(fp_pickle, fp_urdf)
        sol = result.sol

        if isinstance(sol, list):
            return None, None, None, None, None, None
        else:
            num_trajectories = sol.shape[0]
            s_span = result.s_span
            real_s_span = s_span[ : num_trajectories]


            idx = np.argmin( np.abs( real_s_span - s_query ) )

            ps_values = sol[idx, :]
            q, qd, qdd = post_aghf.get_q_qd_qdd(ps_values, self.t_sim)
            U_sol = post_aghf.compute_u_matrix(q, qd, qdd)
            X_spec_t, _ = post_aghf.get_X_Xd_spec_t(ps_values, self.t_sim)
            u_sq_aghf, _ = post_aghf.compute_u_sq(q, qd, qdd, self.dt)
            
            lam_ode = lambda tau, x: ode_fwd_integrate(tau, x, U_sol, self.t_sim, post_aghf.model,
                                                       X_spec_t, self.t_sim, k_p, k_v)
            x0 = post_aghf.X0
            solution = solve_ivp(lam_ode, [self.t_sim[0], self.t_sim[-1]], x0.squeeze(), 
                                 method=self.method_int, t_eval=self.t_sim)

            # get actual u
            U_actual = np.zeros(shape=(self.t_sim.shape[0], post_aghf.N))

            for ii in range(solution.y.shape[1]):
                U_actual[ii, :] = get_u_curr(self.t_sim[ii], solution.y[:, ii], U_sol, self.t_sim, 
                                            post_aghf.model, X_spec_t, self.t_sim, k_p, k_v)
            norm_u_real = np.sum(U_actual * U_actual, axis=1)
            u_sq_real = self.dt * np.sum(norm_u_real)

            return solution, u_sq_aghf, u_sq_real, U_sol, U_actual, X_spec_t 
    
    def _store_q_qd_diff(self, q_diff, qd_diff, res_dict, idx):
        """
        Stores the absolute differences in position and velocity for each joint in the result dictionary.

        This function modifies the result dictionary in place by storing the absolute differences in position (q_diff)
        and velocity (qd_diff) for each joint at the specified index.

        Args:
            - q_diff (np.ndarray): Array of absolute differences in position for each joint.
            - qd_diff (np.ndarray): Array of absolute differences in velocity for each joint.
            - res_dict (dict): Dictionary to store the results.
            - idx (int): Index at which to store the differences in the result dictionary.

        Returns:
            - None
        """
        # in place modifies res_dict
        n = q_diff.shape[0]
        for ii in range(n):
            res_dict[f"abs_diff_q_{ii+1}"][idx] = q_diff[ii]
            res_dict[f"abs_diff_qd_{ii+1}"][idx] = qd_diff[ii]

    def _get_scenario_idx(self, Xf_str):
        """
        Finds the index of the scenario that matches the given final state string.

        This function searches for the index of the scenario in `self.Xf_mat` that matches the given final state string `Xf_str`.
        It compares the first few characters of the string representation of the first value in each column of `self.Xf_mat` with `Xf_str`.

        Args:
            - Xf_str (str): The string representation of the final state to match.

        Returns:
            - int: The index of the matching scenario.

        Raises:
            - ValueError: If more than one match is found.
        """
        N_CHARS = 8
        for ii in range(self.num_cases):
            first_value = str( float(self.Xf_mat[0, ii]) )
            search_str = first_value[0 : N_CHARS]
            pattern = re.compile(search_str)
            matches = re.findall(pattern, Xf_str[0:int(2*N_CHARS)])
            if len(matches) == 1:
                return ii
            elif len(matches) > 1:
                raise

    def process_row(self, row_dict):
        """
        Processes a single row of the DataFrame by performing the following steps:
        1. Update scenario index.
        2. Forward simulate.
        3. Compute joint position absolute difference.
        4. Compute joint velocity absolute difference.
        5. Compute infinity norm.

        Args:
            - row_dict (dict): A dictionary that corresponds to a single row of the DataFrame.

        Returns:
            - dict: A dictionary containing the results with keys from row_dict plus additional keys:
                - inf_norm: Infinity norm of the difference between desired and actual final states.
                - inf_norm_pos: Infinity norm of the position difference.
                - inf_norm_vel: Infinity norm of the velocity difference.
                - abs_diff_q_{i}: Absolute difference in position for joint i.
                - abs_diff_qd_{i}: Absolute difference in velocity for joint i.
                - k_p: Proportional gain used in the controller.
                - k_v: Derivative gain used in the controller.
                - th_{i}: Threshold value i.
                - success_th_{i}: Success flag for threshold i.
                - best_u_sq_real: Best u squared value obtained from the forward simulation.
                - best_u_sq_aghf: Best u squared value obtained from the AGHF solution.
                - fp_fwd_mat: File path to the forward simulation results.
                - pct_decrease_act_func: Percentage decrease in the action functional.
        """

        res_dict = {}

        Xf_str = row_dict["xf"]

        # Get the scenario idx
        if self.update_scenario_idx:
            scenario_name = self._get_scenario_idx(Xf_str)
        else:
            scenario_name = row_dict.get('scenario_name')

        num_cases = len(self.k_controller_list)

        # update res_dict with all the data from row_dict plus updating the scenario_name
        for k, v in row_dict.items():
            if k == "scenario_name":
                res_dict[k] = [scenario_name] * num_cases
            else:
                res_dict[k] = [v] * num_cases

        # Add missings keys to res_dict
        res_dict["inf_norm"] = [9999] * num_cases
        res_dict["inf_norm_pos"] = [9999] * num_cases
        res_dict["inf_norm_vel"] = [9999] * num_cases

        for ii in range(self.max_N):
            res_dict[f"abs_diff_q_{ii+1}"] = [9999] * num_cases
            res_dict[f"abs_diff_qd_{ii+1}"] = [9999] * num_cases
        res_dict["k_p"] = [9999] * num_cases
        res_dict["k_v"] = [9999] * num_cases

        for ii, th in enumerate(self.ths):
            res_dict[f"th_{ii+1}"] = [th] * num_cases
            res_dict[f"success_th_{ii+1}"] = [0] * num_cases
        
        # add missing keys u_sq_real, u_sq_aghf
        res_dict["best_u_sq_real"] = [9999] * num_cases
        res_dict["best_u_sq_aghf"] = [9999] * num_cases
        res_dict["fp_fwd_mat"] = [9999] * num_cases

        # update percentage decrease in action functional
        act_ini = res_dict['action_functional_ini'][0]
        act_end = res_dict['action_functional_fin'][0]
        if act_ini != 0:
            pct_decrease_act_func = (act_ini - act_end) / act_ini
        else:
            pct_decrease_act_func = 9999
        res_dict["pct_decrease_act_func"] = [pct_decrease_act_func] * num_cases

        # run fwd simulation for each k_controller
        fn_pickle = self._get_pkl_fn(row_dict["fp"])
        fp_pickle = os.path.join(self.fp_folder_pickles, fn_pickle)
        result = load_from_pickle(fp_pickle)

        if result.type_result == ResultType.SUCCESS:
            if self.force_s_max:
                s_query = row_dict['s_max']
            else:
                s_query = res_dict["best_s"][0]
            N = row_dict["N"]
            fp_urdf = self.N_to_fp_urdf[N]
            xf_des = result.Xf
            # update k_controller in this file
            for ii, (k_p, k_v) in enumerate(self.k_controller_list):
                solution, u_sq_aghf, u_sq_real, U_sol, U_actual, X_spec_t = \
                    self.run_fwd_sim(fp_urdf, fp_pickle, s_query, k_p, k_v)

                if solution is not None:
                    if self.fp_folder_fwd_store is not None:
                        
                        # We always require scenario_name in the data
                        scenario_name = row_dict["scenario_name"] 

                        # load obstacles
                        fp_scenario_mat = os.path.join(self.fp_folder_scenarios, f"{scenario_name}.mat")
                        pre_data_mat = loadmat(fp_scenario_mat)

                        # Store fwd simulation solution
                        fn_old = extract_substring_after(row_dict["fp"], "name=")[:-3]
                        fn_fwd_mat = fn_old + f"-k_p={k_p}-k_v={k_v}.mat"
                        fp_fwd_mat = os.path.join(self.fp_folder_fwd_store, fn_fwd_mat)

                        if isinstance(pre_data_mat.get('obs_mat_values'), np.ndarray):
                            data_mat = {
                                "q_traj_ini_guess": pre_data_mat["q_traj"],
                                "q_traj_fwd": solution.y[0:N, :],
                                "obs_mat": pre_data_mat["obs_mat_values"],
                                "U_sol": U_sol,
                                "U_actual": U_actual,
                                "solution_y": solution.y,
                                "solution_t": solution.t,
                                "X_spec_t": X_spec_t,         
                            }
                        else:
                            data_mat = {
                                "q_traj_ini_guess": pre_data_mat["q_traj"],
                                "q_traj_fwd": solution.y[0:N, :],
                                "U_sol": U_sol,
                                "U_actual": U_actual,
                                "solution_y": solution.y,
                                "solution_t": solution.t,
                                "X_spec_t": X_spec_t,
                            }

                        res_dict["fp_fwd_mat"][ii] = fp_fwd_mat
                        savemat(fp_fwd_mat, data_mat)
        


                    q_diff = np.abs(xf_des[:N].squeeze() - solution.y[:N, -1])
                    qd_diff = np.abs(xf_des[N:].squeeze() - solution.y[N:, -1])
                    inf_norm = np.linalg.norm(xf_des.squeeze() - solution.y[:, -1], ord=np.inf)
                    inf_norm_pos = np.linalg.norm(q_diff, ord=np.inf)
                    inf_norm_vel = np.linalg.norm(qd_diff, ord=np.inf)

                    # update q diff, qd diff, k_controller
                    self._store_q_qd_diff(q_diff, qd_diff, res_dict, ii)
                    res_dict["k_p"][ii] = k_p
                    res_dict["k_v"][ii] = k_v
                    # update inf norm
                    res_dict["inf_norm"][ii] = inf_norm
                    res_dict["inf_norm_pos"][ii] = inf_norm_pos
                    res_dict["inf_norm_vel"][ii] = inf_norm_vel

                    # update successes
                    for jj, th in enumerate(self.ths):
                        res_dict[f"success_th_{jj+1}"][ii] = 1 if inf_norm < th else 0
                    
                    # update u^2
                    res_dict["best_u_sq_real"][ii] = u_sq_real
                    res_dict["best_u_sq_aghf"][ii] = u_sq_aghf
        
        return res_dict

    def create_df(self, list_res_dict: List[Dict[str, List[Any]]]) -> pd.DataFrame:
        """
        Creates a DataFrame from a list of dictionaries, where each dictionary's values are lists of the same length.

        Args:
            - list_res_dict (List[Dict[str, List[Any]]]): List of dictionaries to convert to DataFrame.

        Returns:
            - pd.DataFrame: Resulting DataFrame.
        """       
        df = pd.DataFrame()
        
        for d in list_res_dict:
            temp_df = pd.DataFrame(d)
            df = pd.concat([df, temp_df], ignore_index=True)
        
        return df
    
    def run(self, max_workers, debug=False):
        """
        Processes the input DataFrame in parallel or sequentially, depending on the debug flag, and saves the results to a CSV file.

        Args:
            - max_workers (int): Maximum number of worker processes to use for parallel processing.
            - debug (bool, optional): If True, processes the data sequentially for debugging purposes. Default is False.

        Returns:
            - None
        """
        if debug:
            list_of_dicts = self.input_df.to_dict(orient="records")
            results = []
            for the_dict in list_of_dicts:
                results.append(self.process_row(the_dict))
        else:
            # self.input_df
            list_of_dicts = self.input_df.to_dict(orient="records")

            
            with ProcessPoolExecutor(max_workers) as executor:
                results = list( tqdm( executor.map( self.process_row, list_of_dicts ), total=len( list_of_dicts ) ) )
            
            df_res = self.create_df(results)
            df_res.to_csv( self.fp_table_output )
