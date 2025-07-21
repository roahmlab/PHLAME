# Reproduce the Crocoddyl comparison table for the Kinova arm without obstacles

import os
import pandas as pd
import ipdb
import numpy as np

from croco_wrapper.grid_search import GridSearch

def custom_run_croco(param_dict, N, N_to_fp_urdf, N_to_j_type, N_to_frame_names, N_to_robot_name, 
                        base_fp_folder, fn_folder, fp_folder_scenarios, timeout, max_workers_solve,
                        t_total, dt_int, th_fwd, max_workers_fwd, method_ivp):

    k_p = param_dict.get('k_p')
    k_v = param_dict.get('k_v')
    scenario_name = param_dict.get('scenario_name')
    weight_x = param_dict.get('weight_x')
    weight_u = param_dict.get('weight_u')
    weight_xf = param_dict.get('weight_xf')
    dt_croco = param_dict.get('dt_croco')
    ic_name = param_dict.get('ic_name')
    
    weight_obs = None if np.isnan(param_dict.get('weight_obs')) else param_dict.get('weight_obs')
    weight_obs_fin = None if np.isnan( param_dict.get('weight_obs_fin') ) else param_dict.get('weight_obs_fin')
    alpha_log = None if np.isnan( param_dict.get('alpha_log') ) else param_dict.get('alpha_log')

    fp_urdf = N_to_fp_urdf[N]
    j_type = N_to_j_type[N]
    frame_names = N_to_frame_names[N]
    robot_name = N_to_robot_name[N]
    
    fp_folder_store = os.path.join(base_fp_folder, fn_folder)
    
    gs = GridSearch(folder_scenarios=fp_folder_scenarios, fp_urdf=fp_urdf, timeout=timeout, 
                    max_workers_solve=max_workers_solve, scenario_names=[scenario_name], 
                    k_p_vec=[k_p], k_v_vec=[k_v], weight_u_vec=[weight_u], weight_x_vec=[weight_x],
                    weight_xf_vec=[weight_xf], weight_obs_vec=[weight_obs], 
                    alpha_log_vec=[alpha_log], dt_vec=[dt_croco],
                    t_total=t_total, frame_names=frame_names, fp_folder_store=fp_folder_store, 
                    robot_name=robot_name, dt_integration=dt_int, ths=[th_fwd], 
                    max_workers_fwd=max_workers_fwd, method_ivp=method_ivp, N=N, 
                    mode_init_guess=ic_name)

    gs.run_gs_seq()
    gs.run_fwd_sim_all([k_p], [k_v], parallel=True)
    gs.create_result_df()


METHOD_IVP = "BDF"
TH = 0.05
MAX_WORKERS_SOLVE = 1
MAX_WORKERS_FWD = 1
TIMEOUT = 60 * 5
T_TOTAL = 2.0
DT_INT = 1e-2

num_repeats = 1

N_to_fp_urdf = {ii: os.path.abspath(os.path.join(os.getcwd(), f"../urdfs/pendulums/pen-{int(ii)}.urdf")) \
                for ii in range(1, 6)}
N_to_fp_urdf[7] = os.path.abspath( os.path.join( os.getcwd(), "..", "urdfs/kinova.urdf" ) )
N_to_fp_urdf[22] = os.path.abspath( os.path.join( os.getcwd(), "..", "urdfs/digit/digit-v3-modified-pinned.urdf" ) )

N_to_j_type = {ii: (2 * np.ones(shape=(ii, 1))).astype(dtype=np.double, order='F') for ii in range(1, 6)}
N_to_j_type[7] = (3 * np.ones(shape=(7, 1))).astype(dtype=np.double, order='F')
N_to_j_type[22] = (2 * np.ones(shape=(22, 1))).astype(dtype=np.double, order='F')

N_to_frame_names = {ii: [f"joint_{jj}" for jj in range(1, ii + 1)] for ii in range(1, 6)}
N_to_frame_names[7] = [f"joint_{ii}" for ii in range(1, 7 + 1)]
frame_names_digit = ['left_toe_roll', 'left_toe_pitch', 'left_tarsus', 'left_knee', 'left_hip_pitch', 
               'left_hip_yaw', 'left_hip_roll', 'left_shoulder_roll', 'left_shoulder_pitch', 
               'left_shoulder_yaw', 'left_elbow', 'right_hip_roll', 'right_hip_yaw', 'right_hip_pitch', 
               'right_knee', 'right_tarsus', 'right_toe_pitch', 'right_toe_roll', 'right_shoulder_roll',
               'right_shoulder_pitch', 'right_shoulder_yaw', 'right_elbow']
N_to_frame_names[22] = frame_names_digit
N_to_robot_name = {
    7: "kinova_gen3",
    22: "digit_pinned",
}

N_to_folder_scenarios = {}
N_to_folder_scenarios[7] = os.path.abspath( os.path.join( os.getcwd(), "..", "kinova_saved_random_worlds" ) )
N_to_folder_scenarios[22] = os.path.abspath( os.path.join( os.getcwd(), "..", "digit_saved_pre_scenarios_and_scenarios" ) )

method_int = "BDF"
lam_fn = lambda scenario_name, n_repeat, N: \
    f"paper_croco_single_param_seq_N={N}_scenario_name={scenario_name}_n_repeat={n_repeat}".replace("name=", "n=")
    
N_to_scenario_names = {
    22: [
        "scenario_stretch",
        "scenario_step",
    ],
}

base_fp_folder = os.path.abspath( os.path.join( os.getcwd(), "..", "..", "storage_croco") )
N = 22

param_croco = {}
fp_urdf = N_to_fp_urdf[N]
j_type = N_to_j_type[N]
fp_folder_scenarios = N_to_folder_scenarios[N]


param_croco = {
    'weight_obs': np.nan,
    'weight_obs_fin': np.nan,
    'alpha_log': np.nan,
    'k_p': 100,
    'k_v': 100,
    'weight_x': 0.001,
    'weight_u': 0.0001,
    'dt_croco': 0.001,
    'weight_xf': 1000,
    'ic_name': 'x0_rnea',
}


for scenario_name in N_to_scenario_names[N]:
    param_croco['scenario_name'] = scenario_name

    for n_repeat in range(num_repeats):
        print(f"scenario_name: {scenario_name}, N={N}")
        fn_folder = lam_fn(scenario_name, n_repeat, N)

        custom_run_croco(param_croco, N, N_to_fp_urdf, N_to_j_type, N_to_frame_names, 
                            N_to_robot_name, base_fp_folder, fn_folder, fp_folder_scenarios, 
                            TIMEOUT, MAX_WORKERS_SOLVE, T_TOTAL, DT_INT, TH, MAX_WORKERS_FWD,
                            METHOD_IVP)
