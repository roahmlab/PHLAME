import os

import numpy as np

from phlame.comparison import run_batch_custom_aghf


fp_urdf = os.path.abspath( os.path.join( os.getcwd(), "Phlame_comparisons/urdfs/kinova.urdf" ) )

N = 7
j_type = (3 * np.ones(shape=(N, 1))).astype(dtype=np.double, order='F')
fp_folder_base = os.path.abspath( os.path.join( os.getcwd(), "storage_phlame/" ))


fp_folder_scenarios = os.path.abspath( os.path.join( os.getcwd(),"Phlame_comparisons/kinova_saved_random_worlds" ) )

num_repeats = 1
scenario_names = [
    'scenario_1_no_obs', 
    'scenario_2_no_obs', 
    'scenario_3_no_obs',
    'scenario_4_no_obs', 
    'scenario_5_no_obs',
    'scenario_6_no_obs', 
    'scenario_7_no_obs', 
    'scenario_8_no_obs',
    'scenario_9_no_obs', 
    'scenario_10_no_obs',
]
force_s_max = True
lam_fn = lambda scenario_name, n_repeat: \
    f"paper_seq_kinova_single_param_no_obstacles_scenario_={scenario_name}_n_repeat={n_repeat}"
fp_param = {
    's_max': 0.1,
    'k_p': 10,
    'k_v': 10,
    'p': 9,
    'k': 1e9,
}

run_batch_custom_aghf(fp_param, fp_urdf, N, j_type, fp_folder_scenarios, num_repeats,
                          scenario_names, fp_folder_base, force_s_max, lam_fn)
