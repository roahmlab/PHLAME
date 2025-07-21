# This script generates tables II, III, IV, V from the paper

from common_comparison import gen_unified_comparison_table
import ipdb
import os


name_to_scenarios = {
    'kinova_no_obs': [
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
    ],
    'digit_no_obs': [
        'scenario_step',
        'scenario_stretch',
    ],
    'digit_obs': [
        'scenario_2',
        'scenario_4',
        'scenario_6',
        'scenario_8',
    ],
    'kinova_obs': [
        'test_easy_1', 
        'test_easy_2', 
        'test_easy_3',
        'test_easy_4',
        'test_easy_5',
        'test_easy_6', 
        'test_easy_7', 
        'test_easy_8',
        'test_easy_9', 
        'test_easy_10',
    ]
}


FP_FOLDER_AGHF = os.path.abspath( os.path.join( os.getcwd(), "storage_phlame"))
FP_FOLDER_CROCO = os.path.abspath( os.path.join( os.getcwd(), "storage_croco"))
FP_FOLDER_ALIG = os.path.abspath( os.path.join( os.getcwd(), "storage_aligator"))


FP_FOLDER_STORE = os.path.abspath( os.path.join( os.getcwd(), "storage_tables_paper"))

fn_kino_no_obs = "kinova_no_obs"
fn_digit_no_obs = "digit_no_obs"
fn_digit_obs = "digit_obs"
fn_kinova_obs = "kinova_obs"
method_display_names = {
    "aghf": "PS-AGHF",
    "croco": "Crocoddyl",
    "alig": "Aligator",
}

# Generate comparison between Crocoddyl and Phlame for Kinova without obstacles
gen_unified_comparison_table(FP_FOLDER_AGHF, FP_FOLDER_CROCO, 
                                 FP_FOLDER_ALIG, FP_FOLDER_STORE, fn_kino_no_obs,
                                 None, None, 
                                 method_display_names, [],
                                 scenario_names=name_to_scenarios[fn_kino_no_obs])

# Generate comparison between Crocoddyl and Phlame for Pinned Digit without obstacles
gen_unified_comparison_table(FP_FOLDER_AGHF, FP_FOLDER_CROCO, 
                                 FP_FOLDER_ALIG, FP_FOLDER_STORE, fn_digit_no_obs,
                                 None, None, 
                                 method_display_names, [],
                                 scenario_names=name_to_scenarios[fn_digit_no_obs])

# Generate comparison between Aligator and Phlame for kinova with obstacles
gen_unified_comparison_table(FP_FOLDER_AGHF, FP_FOLDER_CROCO, 
                             FP_FOLDER_ALIG, FP_FOLDER_STORE, fn_kinova_obs,
                             None, None, 
                             method_display_names, [],
                             scenario_names=name_to_scenarios[fn_kinova_obs])

# Generate comparison between Aligator and Phlame for digit with obstacles
gen_unified_comparison_table(FP_FOLDER_AGHF, FP_FOLDER_CROCO, 
                             FP_FOLDER_ALIG, FP_FOLDER_STORE, fn_digit_obs,
                             None, None, 
                             method_display_names, ['alig'],
                             scenario_names=name_to_scenarios[fn_digit_obs])
