# Compare scenario step, stretch

import os
import numpy as np

from phlame.comparison import run_batch_custom_aghf


fp_urdf = os.path.abspath( os.path.join( os.getcwd(), "Phlame_comparisons/urdfs/digit/digit-v3-modified-pinned-fixed-meshes.urdf" ) )
fp_folder_scenarios = os.path.abspath( os.path.join( os.getcwd(), "Phlame_comparisons/digit_saved_pre_scenarios_and_scenarios" ) )
fp_folder_base = os.path.abspath( os.path.join( os.getcwd(), "storage_phlame/" ))

N = 22
j_type = (2 * np.ones(shape=(N, 1))).astype(dtype=np.double, order='F')                        
num_repeats = 1
scenario_names = [
    "scenario_stretch",
    "scenario_step",
]
force_s_max = True
lam_fn = lambda scenario_name, n_repeat: \
    f"paper_seq_digit_single_param_no_obstacles_scenario_={scenario_name}_n_repeat={n_repeat}"
    
fp_param = {
    's_max': 1,
    'k_p': 100,
    'k_v': 100,
    'p': 6,
    'k': 1e7,
    }

run_batch_custom_aghf(fp_param, fp_urdf, N, j_type, fp_folder_scenarios, num_repeats,
                          scenario_names, fp_folder_base, force_s_max, lam_fn)


