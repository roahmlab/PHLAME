import os

import numpy as np

from phlame.comparison import run_batch_custom_aghf



fp_urdf = os.path.abspath( os.path.join( os.getcwd(), "Phlame_comparisons/urdfs/kinova.urdf" ) )
N = 7
j_type = (3 * np.ones(shape=(N, 1))).astype(dtype=np.double, order='F')
fp_folder_base = os.path.abspath( os.path.join( os.getcwd(), "storage_phlame/" ))

fp_folder_scenarios = os.path.abspath( os.path.join( os.getcwd(), "Phlame_comparisons/kinova_hard_obstacles" ) )
num_repeats = 1
scenario_names = [f"test_easy_{num}" for num in range(1, 11)]
    
force_s_max = True
lam_fn = lambda scenario_name, n_repeat: \
    f"paper_seq_kinova_single_param_WITH_obstacles_scenario_={scenario_name}_n_repeat={n_repeat}"
    
fp_param = {
    's_max': 0.1,
    'k': 1e9,
    'p': 8,
    'k_cons': 1e9,
    'c_cons': 1,
    'k_p': 10,
    'k_v': 10,
}

run_batch_custom_aghf(fp_param, fp_urdf, N, j_type, fp_folder_scenarios,
                      num_repeats, scenario_names, fp_folder_base, 
                      force_s_max, lam_fn)