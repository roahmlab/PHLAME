"""
Functions to compare the results of crocoddyl and AGHF
"""

from scipy.io import loadmat
import numpy as np
import pandas as pd
import ipdb
import matplotlib.pyplot as plt
import os
from typing import Optional, Dict, Any
import gc
import os
import multiprocessing as mp

from phlame.aghf import PostAghf
from phlame.parameter_set import generate_parameter_sets
from phlame.parameter_set import ParameterSetBase
from phlame.parameter_set import ParameterSetActivatedSimpleSphere
from phlame.experiment import Experiment
from phlame.utils import cleanup_folder
from phlame.post_processing import PostProcessing
from phlame.post_processing import PostProcessingFwdSuccess


def get_best_pset(gr_df, cols_sort, ascending, top_k, pset_keys):
    """
    Selects the top k rows from a dataframe after sorting it based on specified columns.

    Args:
        gr_df (pd.DataFrame): The dataframe to be sorted and filtered.
        cols_sort (list of str): The columns to sort the dataframe by.
        ascending (bool or list of bool): Sort order for each column. True for ascending, False for descending.
        top_k (int): The number of top rows to select after sorting.
        pset_keys (list of str): The keys to include in the resulting dictionaries.

    Returns:
        list of dict: A list of dictionaries containing the top k rows with specified keys.
    """
    res = []
    gr_sorted = gr_df.sort_values(by=cols_sort, ascending=ascending)
    for ii in range(top_k):
        res.append({k: gr_sorted[k].iloc[ii] for k in pset_keys})
    return res

def make_boxplot(base_df, best_pset_list, cols_plot, pset_keys, sup_title_str, u_max, t_max):
    """
    Creates boxplots for specified columns of a filtered dataframe, with custom y-axis limits.

    Args:
        base_df (pd.DataFrame): The base dataframe to filter and plot.
        best_pset_list (list of dict): The list of best parameter sets to filter the dataframe.
        cols_plot (list of str): The columns to create boxplots for.
        pset_keys (str): The keys to group the boxplots by.
        sup_title_str (str): The overall title for the figure.
        u_max (float): The maximum value for the first y-axis.
        t_max (float): The maximum value for the second y-axis.

    Returns:
        None
    """
    filt_df = filter_df(base_df, best_pset_list)
    fig, axes = plt.subplots(nrows=1, ncols=len(cols_plot), figsize=(20, 5))
    for ii, (ax, col) in enumerate(zip(axes, cols_plot)):
        filt_df.boxplot(column=col, by=pset_keys, ax=ax)
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
        if ii == 0:
            ax.set_ylim(0, u_max + 2.5 * u_max)
        elif ii == 1:
            ax.set_ylim(0, t_max + 2.5 * t_max)

    plt.tight_layout()
    plt.suptitle(sup_title_str)

def filter_df(in_df, list_dict):
    """
    Filters in_df based on the conditions in list_dict. Any of the conditions has to be satisfied.

    Parameters:
    - in_df (pd.DataFrame): The input DataFrame to be filtered.
    - list_dict (list of dict): A list of dictionaries, where each dictionary contains conditions.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    # Start with a False condition to combine others with OR logic
    combined_condition = pd.Series([False] * len(in_df))
    
    for conditions in list_dict:
        # Start with a True condition for AND logic within this dictionary
        condition = pd.Series([True] * len(in_df))
        
        for col, value in conditions.items():
            condition &= (in_df[col] == value)
        
        # Combine with the existing combined_condition using OR logic
        combined_condition |= condition
    
    # Filter the DataFrame based on the combined condition
    res_df = in_df[combined_condition]
    
    return res_df


# For the new code for scatter plots

def pre_process_dicts(fps):
    """
    This assumes that the dfs have the exact same column
    """
    df_list = []
    for ic_name, fp_csv in fps.items():
        # ipdb.set_trace()

        # change the filepath up to storage to point to the backup location
        if not os.path.isfile(fp_csv):
            # use the backup location
            old_str = "/home/cesarch/repos/AGHF/new_pseudo_spectral/psaghf/storage/"
            new_str = "/mnt/ws-frb/users/cesarch/storage/"
            # ipdb.set_trace()
            fp_csv = fp_csv.replace(old_str, new_str)
            

        curr_df = pd.read_csv(fp_csv)
        curr_df['ic_name'] = ic_name
        df_list.append( curr_df.copy() )
    return pd.concat(df_list)

def pre_process_list(fps):
    df_list = []
    for fp, ic_name, N in fps:
        curr_df = pd.read_csv(fp)
        curr_df['ic_name'] = ic_name
        curr_df['N'] = N
        df_list.append( curr_df.copy() )
    return pd.concat( df_list )

def pre_process_fps(fps):
    if isinstance(fps, dict):
        df = pre_process_dicts(fps)
    elif isinstance(fps, list):
        df = pre_process_list(fps)
    elif fps is None:
        df = None
    else:
        raise("Unhandled type for fps")
    return df

def filter_df_th_top_sort(df, th_val, top_k, sort_col, N, max_t_solve, max_u2, kp_list, kv_list, 
                          filter_collision, scenario_name=None):
    # Does not modify the input

    if df is not None:
        curr_df = df.copy()
        
        # NOt the best way to handle this
        if 'inf_norm' not in curr_df:
            return None

        if scenario_name is not None:
            curr_df = curr_df[curr_df['scenario_name'] == scenario_name]
        

        curr_df = curr_df[curr_df['inf_norm'] != '']
        curr_df['t_solve'] = curr_df['t_solve'].astype(float)

        if filter_collision:
            curr_df = curr_df[curr_df['obs_check'] == '1']

        if 'k_p' in curr_df:
            curr_df['k_p'] = curr_df['k_p'].astype(float)
            curr_df['k_v'] = curr_df['k_v'].astype(float)

            curr_df = curr_df[curr_df['k_p'].isin(kp_list)]
            curr_df = curr_df[curr_df['k_v'].isin(kv_list)]

            # ipdb.set_trace()


        # fix u2 for aligator and croco
        if "u_sq_real" in curr_df:
            # ipdb.set_trace()
            curr_df["u_sq_real"] = curr_df["u_sq_real"].map(lambda x: str(x)[2:-2] )
            curr_df = curr_df[curr_df['u_sq_real'] != '']
            curr_df['u_sq_real'] = curr_df['u_sq_real'].astype(float)
        else:
            curr_df['best_u_sq_real'] = curr_df['best_u_sq_real'].astype(float)

        curr_df = curr_df[curr_df["N"] == N]
        curr_df = curr_df[curr_df['inf_norm'] < th_val]

        u2_col_name = get_u2_col(curr_df)

        curr_df = curr_df[curr_df['t_solve'] < max_t_solve]
        curr_df = curr_df[curr_df[u2_col_name].astype(float) < max_u2]


        if sort_col == "t_solve":
            curr_df = curr_df.sort_values(by=sort_col, ascending=True)
        elif sort_col == "u2":
            curr_df = curr_df.sort_values(by=u2_col_name, ascending=True)

        if top_k is not None:
            curr_df = curr_df.head(top_k)

        # ipdb.set_trace()

        return curr_df
    else:
        return None

def count_not_none(the_iterable):
    count = 0
    for v in the_iterable:
        if v is not None:
            count += 1
    return count

def get_u2_col(df):
    possible_vals = ["best_u_sq_real", "u_sq_real"]
    for val in possible_vals:
        if val in df:
            return val

def get_best_param(df: Optional['pd.DataFrame'], check_col_obs: str, 
                   cols_obs: list[str], cols_no_obs: list[str]) -> Optional[Dict[str, Any]]:
    # Check if the dataframe is None
    if df is None:
        return None
    
    # Check if the dataframe is empty
    if df.empty:
        return None

    # Return the appropriate parameter based on the presence of check_col_obs in the dataframe
    if check_col_obs in df:
        return df[cols_obs].iloc[0].to_dict()
    else:
        return df[cols_no_obs].iloc[0].to_dict()

def run_all(fps_aghf, fps_aligator, fps_croco, th_val, top_k, sort_col, N_list, exp_title, fp_folder_store,
            max_t_solve, max_u2, kp_list, kv_list, filter_collision=False, ylog=False, xlog=False, scenario_name=None,
            do_plot=True):
    """"
    Generates a scatter plot and 3 tables with the best results for the
    three methods. 

    Inputs:
    ------
    fps_aghf: List[str]: points to a csv
    fps_aligator: List[str]
    fps_croco: List[str]
    th_val: float : inf norm threshold of the BC
    top_k: number of successes to consider
    sort_col: value that will be used to rank results from best to worst
    """

    df_aghf = pre_process_fps(fps_aghf)
    # remove trivial results
    df_aghf = df_aghf[df_aghf['best_s'] != 0]

    df_croco = pre_process_fps(fps_croco)
    df_alig = pre_process_fps(fps_aligator)

    if 'obs_check' in df_aghf:
        df_aghf['obs_check'] = df_aghf['obs_check'].astype(str)
        df_croco['obs_check'] = df_croco['obs_check'].astype(str)
        df_alig['obs_check'] = df_alig['obs_check'].astype(str)

    best_parameters = {"aghf": [], "croco": [], "alig": []}

    for N in N_list:
        # apply filter, sorting and top
        top_aghf = filter_df_th_top_sort(df_aghf, th_val, top_k, sort_col, N, max_t_solve, max_u2, 
                                         kp_list, kv_list, filter_collision, scenario_name)
        top_croco = filter_df_th_top_sort(df_croco, th_val, top_k, sort_col, N, max_t_solve, max_u2, 
                                          kp_list, kv_list, filter_collision, scenario_name)
        top_alig = filter_df_th_top_sort(df_alig, th_val, top_k, sort_col, N, max_t_solve, max_u2, 
                                         kp_list, kv_list, filter_collision, scenario_name)
        
        # Create dictionary with the best parameters
        best_param_aghf = get_best_param(top_aghf, 'k_cons', 
                                         ['s_max', 'k_p', 'k_v', 'p', 'N', 'best_s', 'k',
                                          'scenario_name', 'k_cons', 'c_cons'],
            ['s_max', 'k_p', 'k_v', 'p', 'N', 'best_s', 'k', 'scenario_name'])
        best_param_croco = get_best_param(top_croco, 'weight_obs', 
            ['k_p', 'k_v', 'scenario_name', 'weight_x', 'weight_u', 'weight_xf', 'dt_croco', 'ic_name', 'weight_obs', 'weight_obs_fin', 'alpha_log'],
            ['k_p', 'k_v', 'scenario_name', 'weight_x', 'weight_u', 'weight_xf', 'dt_croco', 'ic_name'])
        best_param_alig = get_best_param(top_alig, 'weight_x',
            ['k_p', 'k_v', 'scenario_name', 'weight_x', 'weight_u', 'weight_xf', 'dt_alig', 'ic_name', 'tol', 'mu_init', 'max_iters'],
            ['k_p', 'k_v', 'scenario_name', 'weight_x', 'weight_u', 'weight_xf', 'dt_alig', 'ic_name', 'tol', 'mu_init', 'max_iters'],
        )
        
        # ipdb.set_trace()

        best_parameters['aghf'].append( (N, best_param_aghf ))
        best_parameters['croco'].append( (N, best_param_croco ))
        best_parameters['alig'].append( (N, best_param_alig ))

        # ipdb.set_trace()

        if do_plot:
        
            data = {
                "aghf": (top_aghf, "best_u_sq_real"),
                "croco": (top_croco, "u_sq_real"),
                "alig": (top_alig, "u_sq_real"),
            }
            
            method_to_color = {
                "aghf": "#1C7709",
                "croco": "#FFA719",
                "alig": "#D3168A",
            }
            # fig, axs = plt.subplots( nrows=1, ncols=3, sharex=True, sharey=True)
            # fig, axs = plt.subplots( nrows=1, ncols=3)
            fig, ax = plt.subplots( nrows=1, ncols=1)

            fig.set_size_inches(9.5, 5.5)
            fig.supxlabel('u^2 fwd sim')
            fig.supylabel('t_solve[s]')
            fig.suptitle(f"{exp_title}, N={N}, th_val={th_val}, sort_col={sort_col}")

            # for ii, (method_name, vals) in enumerate(data.items()):
            #     curr_df = vals[0]
            #     u2_name = vals[1]
            #     if curr_df is not None:
            #         axs[ii].scatter( curr_df[u2_name], curr_df["t_solve"], c=curr_df['inf_norm'], cmap='copper', label=method_name )
                
            #     axs[ii].legend()
            #     axs[ii].set_xticks(axs[ii].get_xticks(), axs[ii].get_xticklabels(), rotation=45, ha='right')
            #     axs[ii].grid()
            
            max_u = 0
            min_u = np.inf

            max_t = 0
            min_t = np.inf

            for ii, (method_name, vals) in enumerate(data.items()):
                curr_df = vals[0]
                u2_name = vals[1]
                # if method_name == "alig":
                    # ipdb.set_trace()
                if curr_df is not None:
                    print("N: ", N)
                    print("#"*100)
                    ax.scatter( curr_df[u2_name], curr_df["t_solve"], c=method_to_color[method_name], label=method_name )

                        
                    # ipdb.set_trace()
                    if curr_df.shape[0] > 0:
                        max_u = max( curr_df[u2_name].max(), max_u)
                        min_u = min( curr_df[u2_name].min(), min_u)

                        max_t = max( curr_df['t_solve'].max(), max_t)
                        min_t = min( curr_df['t_solve'].min(), min_t)

                    print("method name: ", method_name)
                    if method_name == "aghf":
                        if filter_collision:
                            print(curr_df[["best_s", "k", "p", "t_solve", "best_u_sq", "N", "ic_name", "k_cons", "c_cons", "scenario_name", "k_p", "k_v"]])
                        else:
                            print(curr_df[["best_s", "k", "p", "t_solve", "best_u_sq", "N", "ic_name", "scenario_name", "k_p", "k_v"]])
                            # ipdb.set_trace()
                    elif method_name == "croco":
                        print(curr_df[["dt_croco", "t_solve", "u_sq_real", "N", "ic_name", "scenario_name", "k_p", "k_v"]])
                    elif method_name == "alig":
                        print(curr_df[["dt_alig", "t_solve", "u_sq_real", "N", "ic_name", "scenario_name", "k_p", "k_v"]])
                    print("#"*100)

            ax.legend()
            ax.grid()
            # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
            if (N == 5) or xlog:
                ax.set_xscale('log')
                # ticks = np.logspace(np.floor(np.log10(min_u)), np.ceil(np.log10(max_u)), num=15)

            # Force specific ticks on the x-axis (choose appropriate values for your data)
            # 
            ticks = np.linspace(np.floor(min_u), np.ceil(max_u), num=15)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{tick:.2f}' for tick in ticks], rotation=45, ha='right')

            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Use scalar formatter for the ticks

            if ylog:
                ax.set_yscale('log')
                y_ticks = np.logspace(np.floor(np.log10(min_t)), np.ceil(np.log10(max_t)), num=15)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks])
                # ipdb.set_trace()

            fn = f"{exp_title}-N={N}-th_val={th_val}.pdf"
            fp = os.path.join(fp_folder_store, fn)
            plt.tight_layout()
            # plt.savefig(fp, format="pdf")

    return best_parameters

def generate_fp_folders_and_fns(fn_folder, fp_folder_base):
    fp_folder_store = os.path.join(fp_folder_base, fn_folder)

    fn_folder_output = fn_folder + "_csv"
    fp_folder_output = os.path.join(fp_folder_base, fn_folder_output)
    fn_table = "result_" + fn_folder + ".csv"
    fn_all = "result_" + fn_folder + "_all.csv"
    fn_fwd = "result_" + fn_folder + "_fwd_success.csv"

    # for post processing
    fp_folder_input = fp_folder_store
    fp_table_csv = os.path.join(fp_folder_output, fn_table)
    fp_table_output = os.path.join(fp_folder_output, fn_fwd)
    fp_folder_fwd_store = fp_folder_store
    fp_folder_pickles = fp_folder_input

    return fp_folder_store, fp_folder_output, fn_table, fn_all, fp_table_csv, fp_table_output, fp_folder_input, fp_folder_pickles, fp_folder_fwd_store

def get_x0_xf_q_traj(fp_folder_scenarios, scenario_name, N):
    fp_scenario = os.path.join(fp_folder_scenarios, scenario_name + ".mat")
    data_scenario = loadmat(fp_scenario)    
    q_traj = data_scenario['q_traj'].astype(dtype=np.double, order='F')
    q_start = q_traj[0, :].reshape(N, 1)
    q_end = q_traj[-1, :].reshape(N, 1)
    qd_start = data_scenario['qd_start'].astype(dtype=np.double, order='F').reshape(N, 1)
    qd_end = data_scenario['qd_end'].astype(dtype=np.double, order='F').reshape(N, 1)

    x0 = np.concatenate( ( q_start, qd_start ) ).astype(dtype=np.double, order='F')
    xf = np.concatenate( ( q_end, qd_end ) ).astype(dtype=np.double, order='F')

    # ipdb.set_trace()

    return x0, xf, q_traj


def custom_run_aghf(param_dict, abs_tol, rel_tol, dt_fwd, th_fwd, t_total, 
                    ns_points, method_name, max_steps, timeout, method_int, fp_folder_base, 
                    fp_folder_scenarios, fn_folder, fp_urdf, j_type, N_to_fp_urdf, 
                    force_s_max):

    # Inside fp_folder_base there will be a folder per experiment.
    

    fp_folder_store, fp_folder_output, fn_table, fn_all, fp_table_csv,\
        fp_table_output, fp_folder_input, fp_folder_pickles, fp_folder_fwd_store = \
        generate_fp_folders_and_fns(fn_folder, fp_folder_base)
    
    cleanup_folder(fp_folder_store)

    k_p = param_dict['k_p']
    k_v = param_dict['k_v']
    p_vec = [ int(param_dict['p']) ]
    s_max_vec = [ param_dict['s_max'] ]
    k_vec = [ param_dict['k'] ]
    scenario_name = param_dict['scenario_name']
    N = int(param_dict['N'])

    k_cons_v = [param_dict.get('k_cons') or None]
    c_cons_v = [param_dict.get('c_cons') or None]

    
    X0, Xf, q_traj = get_x0_xf_q_traj(fp_folder_scenarios, scenario_name, N)

    if k_cons_v[0] is not None:
        # get obs_mat
        fp_scenario = os.path.join(fp_folder_scenarios, scenario_name + ".mat")
        data_scenario = loadmat(fp_scenario)
        obs_mat = data_scenario['obs_mat_values'].astype(dtype=np.double, order='F').T

        parameter_sets = generate_parameter_sets(p_vec, s_max_vec, k_vec, [abs_tol], [rel_tol], "run_name", method_name, max_steps,
            ns_points, N, j_type, fp_urdf, X0, Xf, scenario_name=scenario_name, q_traj=q_traj, class_cons=ParameterSetActivatedSimpleSphere,
            k_cons_v=k_cons_v, c_cons_v=c_cons_v, obs_mat=obs_mat)
    else:
        parameter_sets = generate_parameter_sets(p_vec, s_max_vec, k_vec, [abs_tol], [rel_tol], "run_name", method_name, max_steps,
            ns_points, N, j_type, fp_urdf, X0, Xf, scenario_name=scenario_name, q_traj=q_traj, class_cons=ParameterSetBase)
    

    print("number of cases: ", len(parameter_sets))
    
    # ipdb.set_trace() 
    experiment = Experiment(parameter_sets, timeout, print_debug=False, folder_store=fp_folder_store)
    
    # ipdb.set_trace()
    experiment.run_seq_and_store()

    # experiment.run_parallel(max_workers=1)

    print("finished solving in parallel.")

    # TO POST PROCESS part 1
    pp = PostProcessing(fp_folder_input, fp_folder_output, dt=dt_fwd)
    _ = pp.run(fn_table, fn_all, 1)
    print("finished postprocessing")

    k_controller_list = [(k_p, k_v)]

    
    pp_fwd = PostProcessingFwdSuccess(fp_table_csv, fp_table_output, dt_fwd, [th_fwd], 
                                      fp_folder_pickles, t_total, k_controller_list, N_to_fp_urdf, 
                                      update_scenario_idx=False, 
                                      fp_folder_fwd_store=fp_folder_fwd_store,
                                      fp_folder_scenarios=fp_folder_scenarios,
                                      method_int=method_int, force_s_max=force_s_max)
    pp_fwd.run(1)
    print("finished forward simulating")

    del experiment
    del pp
    del pp_fwd
    gc.collect()
    

def run_sequential_with_repeats(fp_folder_params, list_fn_params, N_to_j_type, 
                                N_to_fp_urdf, fp_folder_store, N_to_fp_folder_scenarios,
                                num_repeats, rel_tol, abs_tol, stopping_criteria, dt_fwd, 
                                th_fwd, t_total, ns_points, method_name, max_steps, timeout,
                                method_int, lam_fn, force_s_max=False):

    for fn_params in list_fn_params:
        fp_param = os.path.join(fp_folder_params, fn_params)
        data_aghf = pd.read_pickle(fp_param)['aghf']

        for N, param_dict in data_aghf:
            scenario_name = param_dict['scenario_name']
            fp_urdf = N_to_fp_urdf[N]
            j_type = N_to_j_type[N]
            fp_folder_scenarios = N_to_fp_folder_scenarios[N]

            for n_repeat in range(num_repeats):
                print(f"scenario_name: {scenario_name}, N={N}")

                fn_folder = lam_fn(scenario_name, n_repeat, N)
                args_tuple = (param_dict, abs_tol, rel_tol, dt_fwd, th_fwd,
                              t_total, ns_points, method_name, max_steps, timeout, method_int,
                              fp_folder_store, fp_folder_scenarios, fn_folder, fp_urdf, j_type,
                              N_to_fp_urdf, force_s_max)
                # custom_run_aghf(*args_tuple)
                p = mp.Process(target=custom_run_aghf, args=args_tuple)
                p.start()
                p.join()

def get_param_across_scenarios(df, param_names, col_sort, u2_col):
    if df is None:
        return None

    # u2_col_name = get_u2_col(df)
    
    res = df.groupby(param_names).agg({
        "t_solve": 'mean',
        u2_col: 'mean',
        'th_1': 'count',
    })
    res = res.reset_index(drop=False)
    n_max = res['th_1'].max()
    res = res[ res['th_1'] == n_max ]
    # ipdb.set_trace()
    
    if col_sort == "t_solve":
        res = res.sort_values(by=col_sort)
    elif col_sort == "u2":
        res = res.sort_values(by=u2_col)
    else:
        raise("Unhandled `col_sort`.")
    
    return res

def get_avg_best_param(fps_aghf, fps_aligator, fps_croco, th_val, sort_col, N,
                       max_t_solve, max_u2, kp_list, kv_list, scenario_names, 
                       filter_collision=False):
    """ 
    
    """
    df_aghf = pre_process_fps(fps_aghf)
    # remove trivial results
    df_aghf = df_aghf[df_aghf['best_s'] != 0]

    df_croco = pre_process_fps(fps_croco)
    df_alig = pre_process_fps(fps_aligator)
    
    if 'obs_check' in df_aghf:
        if df_aghf is not None:
            df_aghf['obs_check'] = df_aghf['obs_check'].astype(str)
            df_aghf = df_aghf[ df_aghf['scenario_name'].isin(scenario_names) ]
        if df_croco is not None:
            df_croco['obs_check'] = df_croco['obs_check'].astype(str)
        if df_alig is not None:
            df_alig['obs_check'] = df_alig['obs_check'].astype(str)

    top_aghf = filter_df_th_top_sort(df_aghf, th_val, None, sort_col, N, max_t_solve, max_u2,
                                     kp_list, kv_list, filter_collision)
    top_croco = filter_df_th_top_sort(df_croco, th_val, None, sort_col, N, max_t_solve, max_u2,
                                      kp_list, kv_list, filter_collision)
    top_alig = filter_df_th_top_sort(df_alig, th_val, None, sort_col, N, max_t_solve, max_u2,
                                     kp_list, kv_list, filter_collision)
    
    ipdb.set_trace()
    
    # MAYBE MOVE THIS ELSEWHERE
    param_aghf_unc = ['p', 'N', 's_max', 'k'] + ['k_p', 'k_v']
    param_aghf_con = param_aghf_unc[:] + ['k_cons', 'c_cons']
    
    param_croco_unc = ['weight_x', 'weight_u', 'dt_croco', 'weight_xf', 'ic_name']  + ['k_p', 'k_v']
    param_alig_con = ['weight_x', 'weight_u', 'weight_xf', 'dt_alig', 'ic_name', 'tol', 
                      'mu_init', 'max_iters']  + ['k_p', 'k_v']

    if 'k_cons' in top_aghf.columns:
        filt_aghf = get_param_across_scenarios(top_aghf, param_aghf_con, sort_col, "best_u_sq_real")
        best_param_aghf = filt_aghf.iloc[0][param_aghf_con].to_dict()
    else:
        filt_aghf = get_param_across_scenarios(top_aghf, param_aghf_unc, sort_col, "best_u_sq_real")
        best_param_aghf = filt_aghf.iloc[0][param_aghf_unc].to_dict()
    
    filt_croco = get_param_across_scenarios(top_croco, param_croco_unc, sort_col, "u_sq_real")
    filt_alig = get_param_across_scenarios(top_alig, param_alig_con, sort_col, "u_sq_real")
    
    ipdb.set_trace()
    
    if filt_croco is not None:
        if filt_croco.shape[0] > 0:
            best_param_croco = filt_croco.iloc[0][param_croco_unc].to_dict()
        else:
            best_param_croco = None
    else:
        best_param_croco = None
    
    if (filt_alig is not None):
        if (filt_alig.shape[0] > 0):
            best_param_alig = filt_alig.iloc[0][param_alig_con].to_dict()
        else:
            best_param_alig = None
    else:
        best_param_alig = None

    return best_param_aghf, best_param_croco, best_param_alig

def run_batch_custom_aghf(fp_param, fp_urdf, N, j_type, fp_folder_scenarios, num_repeats,
                          scenario_names, fp_folder_base, force_s_max, lam_fn, abs_tol=1e-4, rel_tol=1e-4, dt_fwd=1e-2, 
                          th_fwd=0.05, t_total=2.0, ns_points=int(50), method_name="cvode", 
                          max_steps=int(1e8), timeout=60*7, method_int="BDF"):
    """
    Given a best parameter, a robot and scenarios, it runs each scenario `n_repeats`
    and stores the results for further processing.
    """

    if isinstance(fp_param, str):
        best_param = pd.read_pickle(fp_param)['aghf']
    elif isinstance(fp_param, dict):
        best_param = fp_param
    else:
        raise("fp_param valid types: str  or dict")
    N_to_fp_urdf = {N: fp_urdf}
    for scenario_name in scenario_names:
        best_param['scenario_name'] = scenario_name
        best_param['N'] = N
        best_param['best_s'] = best_param['s_max']
        
        for n_repeat in range(num_repeats):
            fn_folder = lam_fn(scenario_name, n_repeat)
            args_tuple = (best_param, abs_tol, rel_tol, dt_fwd, th_fwd, t_total,
                        ns_points, method_name, max_steps, timeout, method_int, fp_folder_base, 
                        fp_folder_scenarios, fn_folder, fp_urdf, j_type, N_to_fp_urdf, force_s_max)
            p = mp.Process(target=custom_run_aghf, args=args_tuple)
            p.start()
            p.join()
