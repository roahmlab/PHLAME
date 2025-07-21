import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb


def get_all_scenario_names(fp_params, list_fn_params):
    all_scenario_names = []
    for fn_params in list_fn_params:
        fp_param = os.path.join(fp_params, fn_params)
        data_aghf = pd.read_pickle(fp_param)['aghf']
        for N, param_dict in data_aghf:
            scenario_name = param_dict['scenario_name']
            all_scenario_names.append(scenario_name)
    
    return all_scenario_names
            
def group_results(scenario_name, fp_storage, folders_list):

    folders_exp = list( 
                       filter( 
                              lambda folder_name: f"{scenario_name}_n_repeat" in folder_name, folders_list)
                       )
    t_solve_vec = np.zeros(len(folders_exp))
    u2_vec = np.zeros(len(folders_exp))
    solver_name = None
        
    if len(folders_exp) > 0:
        for ii, folder in enumerate(folders_exp):
            fp_folder = os.path.join( fp_storage, folder )
            if "csv" in folder:
                # aghf folder
                fn_csv = [ f for f in os.listdir(fp_folder) if "fwd_success" in f][0]
                fp_csv = os.path.join(fp_folder, fn_csv)
                df = pd.read_csv(fp_csv)
                # ADD CHECK IF PASSED COLLISION CHECK
                if 'n_collisions' in df:
                    if df['n_collisions'].iloc[0] > 0:
                        return None
                t_solve_vec[ii] = float(df['t_solve'].iloc[0])
                u2_vec[ii] = float(df['best_u_sq_real'].iloc[0])
                solver_name = "aghf"
                

            else:
                # aligator or croco
                fn_csv = [f for f in os.listdir(fp_folder) if "csv" in f][0]
                if "croco" in fn_csv:
                    solver_name = "croco"
                elif "alig" in fn_csv:
                    solver_name = "alig"
                else:
                    raise("Only expected aligator or crocoddyl")
                
                fp_csv = os.path.join(fp_folder, fn_csv)
                df = pd.read_csv(fp_csv)
                # ADD CHECK IF PASSED COLLISION CHECK
                if 'n_collisions' in df:
                    if df['n_collisions'].iloc[0] == 'solver_failed':
                        return None
                    if df['n_collisions'].iloc[0] > 0:
                        return None
                if df['t_solve'].iloc[0] == 99999:
                    return None
                t_solve_vec[ii] = float(df['t_solve'].iloc[0])
                u2_vec[ii] = float( df['u_sq_real'].iloc[0][2:-2] )


        res_dict = {"avg_t_solve": np.mean(t_solve_vec),
                    "avg_u2": np.mean(u2_vec),
                    "std_t_solve": np.std(t_solve_vec),
                    "std_u2": np.std(u2_vec),
                    "scenario_name": scenario_name,
                    "method_name": solver_name,
                    "u2_vec": u2_vec,
                    "t_solve_vec": t_solve_vec,
                    }
        return res_dict
    else:
        return None
    

    
def gen_comparison_by_scenario_name(all_dict_data, fp_figure, value_key, 
                                    scenario_order, method_colors, 
                                    method_display_names, y_log, 
                                    force_methods=None, fontsize=14):
    """
    Generates a pdf corresponding to a barplot with a y-axis and specified colors for each method.
    On the x-axis, we have scenario_name in a specified order.
    For each x-tick in the x-axis, we have barplots corresponding to a different method_name.
    On the y-axis, we either have avg_t_solve or avg_u2, depending on value_key.
    
    Input:
    ------
    all_dict_data: List[Dict], each dict has keys
        'avg_t_solve', 'avg_u2', 'scenario_name', 'method_name'
    fp_figure: str: filepath of where to store the vectorized pdf
    value_key: str: either 'avg_u2' or 'avg_t_solve'
    scenario_order: List[Tuple[str, str]]: Each tuple contains 
        (actual_scenario_name, display_scenario_name)
    method_colors: Dict[str, str]: A dictionary specifying the color for each method_name
    method_display_names: Dict[str, str]: A dictionary specifying how to display each method_name in the plot
    y_log: bool: If True, use a logarithmic scale for the y-axis
    force_methods: List[str] (optional): If provided, add these methods even if they don't appear in the data
    fontsize: int (optional): Font size for all text elements in the plot (default is 12)
    """
    # Extract the actual scenario names and display names
    scenarios = [item[0] for item in scenario_order]
    display_names = [item[1] for item in scenario_order]
    
    # Extract unique method names
    methods = sorted(set(item['method_name'] for item in all_dict_data))
    
    # Create a nested dictionary to hold values for each (scenario, method)
    data = {scenario: {method: None for method in methods} for scenario in scenarios}
    data_std = {scenario: {method: None for method in methods} for scenario in scenarios}
    
    if value_key == "avg_t_solve":
        std_key = "std_t_solve"
    elif value_key == "avg_u2":
        std_key = "std_u2"
    else:
        raise(f"value_key: {value_key} is not supported.")
    
    # Populate the data dictionary
    for entry in all_dict_data:
        scenario = entry['scenario_name']
        method = entry['method_name']
        value = entry[value_key]
        value_std = entry[std_key]
        if scenario in data:
            data[scenario][method] = value
            data_std[scenario][method] = value_std
    
    # Convert data dictionary into a DataFrame
    df = pd.DataFrame(data).T  # Transpose to have scenarios as rows and methods as columns
    df_std = pd.DataFrame(data_std).T
            
    if force_methods is not None:
        for force_method in force_methods:
            if force_method not in df:
                methods += [force_method]
                df[force_method] = 0

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine positions for the x-ticks
    x = np.arange(len(scenarios))  # the label locations
    width = 0.15  # width of the bars

    # Plot each method as a separate bar set with specified colors and display names
    for i, method in enumerate(methods):
        y_values = df[method].values
        y_std = df_std[method].values
        color = method_colors.get(method, 'gray')  # Default to 'gray' if color not specified
        display_name = method_display_names.get(method, method)  # Default to the original name if not specified
        # y_values_lower = np.log10(y_values) - np.log10(y_std + y_values)
        # y_values_upper = np.log10(y_values + y_std) - np.log10(y_values)
        # ax.bar(x + i * width, y_values, width, label=display_name, color=color, yerr=y_std, capsize=8)
        ax.bar(x + i * width, y_values, width, label=display_name, color=color)
        
        # Add standard deviation as text above each bar
        # ipdb.set_trace()
        # for j, (y_val, y_std_val) in enumerate(zip(y_values, y_std)):
        #     if not np.isnan(y_std_val):
        #         ax.text(
        #             x[j] + i * width,       # X position of the text (aligned with each bar)
        #             y_val + 0.05 * y_val,   # Y position (just above the bar, 5% higher than the bar height)
        #             f"±{y_std_val:.4f} s",     # Text showing the std value formatted to 2 decimal places
        #             ha='center',            # Center horizontally
        #             va='bottom',            # Align text to the bottom
        #             fontsize=fontsize * 0.8, # Slightly smaller than the default fontsize
        #             rotation=90,
        #         )
        
        # ax.errorbar(x, y_values, yerr=y_std, ecolor=color, 
        #             color=color, label=display_name)

    # ax.set_ylim(10**5)
    # Set logarithmic scale for y-axis
    if y_log:
        ax.set_yscale('log')
    
    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set y-axis label based on the value_key
    if value_key == 'avg_t_solve':
        ax.set_ylabel('Solve time [s]', fontsize=fontsize)
    elif value_key == 'avg_u2':
        ax.set_ylabel(r'$\int ||u(t)||^2 dt$', fontsize=fontsize)
    
    # Set tick positions and labels for the x-axis
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(display_names, rotation=0, ha="right", fontsize=fontsize)
    ax.legend(title="Methods", fontsize=fontsize, title_fontsize=fontsize)
    ax.set_xlabel("N", fontsize=fontsize)
    
    
    
    # Set x-axis label font size
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    
    # ax.margins(y=0.2)

    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig(fp_figure, format='pdf')
    plt.close()

def gen_table_by_scenario_name(all_dict_data, fp_figure, value_key, scenario_order, 
                               method_display_names, force_methods=None, fontsize=14):
    """
    Generates a pdf corresponding to a table with scenario names as rows and methods as columns.
    The cells of the table contain either avg_t_solve or avg_u2 values depending on value_key.

    Input:
    ------
    all_dict_data: List[Dict], each dict has keys
        'avg_t_solve', 'avg_u2', 'scenario_name', 'method_name'
    fp_figure: str: filepath of where to store the vectorized pdf
    value_key: str: either 'avg_u2' or 'avg_t_solve'
    scenario_order: List[Tuple[str, str]]: Each tuple contains 
        (actual_scenario_name, display_scenario_name)
    method_display_names: Dict[str, str]: A dictionary specifying how to display each method_name in the table
    force_methods: List[str] (optional): If provided, add these methods even if they don't appear in the data
    fontsize: int (optional): Font size for all text elements in the table (default is 14)
    """
    # Extract the actual scenario names and display names
    scenarios = [item[0] for item in scenario_order]
    display_names = [item[1] for item in scenario_order]
    
    # Extract unique method names
    methods = sorted(set(item['method_name'] for item in all_dict_data))
    
    # Create a nested dictionary to hold values for each (scenario, method)
    data = {scenario: {method: None for method in methods} for scenario in scenarios}
    
    # Populate the data dictionary
    for entry in all_dict_data:
        scenario = entry['scenario_name']
        method = entry['method_name']
        value = entry[value_key]
        if scenario in data:
            data[scenario][method] = value
    
    # Convert data dictionary into a DataFrame
    df = pd.DataFrame(data).T  # Transpose to have scenarios as rows and methods as columns
    
    
    # Force additional methods if specified
    if force_methods is not None:
        for force_method in force_methods:
            if force_method not in df:
                methods += [force_method]
                df[force_method] = None  # or 0 if you prefer to fill with zeros
    
                    
    
    # Apply display names to methods
    df.columns = [method_display_names.get(col, col) for col in df.columns]
    df.index = display_names

    # Plotting the table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # Hide axes

    # Create a table from the DataFrame
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#f2f2f2'] * len(df.columns),  # Light gray for column headers
                     rowColours=['#f2f2f2'] * len(df.index))   # Light gray for row headers

    # Set font sizes
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.5)  # Adjust scale as necessary
    
    # Save the table as a PDF
    plt.tight_layout()
    plt.savefig(fp_figure, format='pdf')
    plt.close()
    
def gen_comparison_by_scenario_name_all(fp_folder_aghf, fp_folder_croco, fp_folder_alig, fp_folder_store, 
                                        fn_store, fp_folder_params, list_fn_params, scenario_order, method_colors,
                                        method_display_names, y_log_u2, y_log_t_solve, force_methods=[]):

    folders_aghf = [] if fp_folder_aghf is None else [ f for f in os.listdir(fp_folder_aghf) if f.endswith("csv") ]
    folders_croco = [] if fp_folder_croco is None else os.listdir(fp_folder_croco)
    folders_alig = [] if fp_folder_alig is None else os.listdir(fp_folder_alig)
    
    all_scenario_names = get_all_scenario_names(fp_folder_params, list_fn_params)
    all_dict_data = [ ]
    
    for scenario_name in all_scenario_names:
        res_aghf = group_results(scenario_name, fp_folder_aghf, folders_aghf)
        res_croco = group_results(scenario_name, fp_folder_croco, folders_croco)
        res_alig = group_results(scenario_name, fp_folder_alig, folders_alig)
        if res_aghf is not None:
            all_dict_data.append(res_aghf)
        if res_croco is not None:
            all_dict_data.append(res_croco)
        if res_alig is not None:
            all_dict_data.append(res_alig)
    
        
    fp_u2 = os.path.join(fp_folder_store, f"u2_{fn_store}.pdf")
    fp_t_solve = os.path.join(fp_folder_store, f"t_solve_{fn_store}.pdf")
        
    gen_comparison_by_scenario_name(all_dict_data, fp_u2, "avg_u2", scenario_order, 
                                    method_colors, method_display_names, y_log_u2, force_methods)
    gen_comparison_by_scenario_name(all_dict_data, fp_t_solve, "avg_t_solve", scenario_order,
                                    method_colors, method_display_names, y_log_t_solve, force_methods)
    
def compute_avg_std(df, col_data, method_name, scenarios_intersection):
    # col_data: : ['u2_vec', 't_solve_vec']
    data_col = df[df.method_name == method_name][col_data]
    all_values = np.concatenate( data_col.tolist() )
    avg_val = np.mean(all_values)
    std_val = np.std(all_values)
    n = data_col.shape[0]
    # ipdb.set_trace()
    
    # get the values in the intersection
    data_intersection = df[ (df.method_name == method_name) & (df.scenario_name.isin(scenarios_intersection))][col_data]
    values_intersection = np.concatenate(data_intersection.tolist())
    avg_int = np.mean(values_intersection)
    std_int = np.std(values_intersection)
    
    return avg_val, std_val, n, avg_int, std_int

def dataframe_to_custom_latex(df: pd.DataFrame, filepath: str, value_key: str):
    """
    Converts a DataFrame to a LaTeX table, formatting specific columns to include ± 3 times the standard deviation,
    and saves it to the specified filepath.

    Parameters:
        df (pd.DataFrame): The input DataFrame with expected columns.
        filepath (str): The path where the .tex file will be stored.
    """
    # Identify columns for average and standard deviation
    avg_cols = [f'average: {value_key}', f'average intersection: {value_key}']
    std_cols = [f'std: {value_key}', f'std intersection: {value_key}']
    
    # Create a new DataFrame to store modified values
    df_latex = df.copy()
    
    for avg_col, std_col in zip(avg_cols, std_cols):
        if avg_col in df.columns and std_col in df.columns:
            # Modify the average columns to include ± 3 * standard deviation
            def apply_format(row):
                if isinstance(row[avg_col], (int, float)):
                    return f"{round(row[avg_col], 2)} ± {round(3 * row[std_col], 2)}"
                return row[avg_col]  # Return the original string value if it's already a string
            
            # Apply the formatting row-wise
            df_latex[avg_col] = df_latex.apply(apply_format, axis=1)
            
            # Remove the standard deviation columns
            df_latex.drop(columns=[std_col], inplace=True)
    
    columns_to_include = ['method_name', f'average: {value_key}', 'Number of successes', f'average intersection: {value_key}']
    save_df_to_latex(filepath, df_latex, columns_to_include)

def save_df_to_latex(fp, df, cols_include):
    
    # Open the file and write the LaTeX table
    with open(fp, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{ |" + " | ".join(['c'] * len(cols_include)) + "| }\n")
        f.write("\\hline\n")
        
        # Write the header row
        f.write(" & ".join(cols_include) + " \\\\\n")
        f.write("\\hline\n")
        
        # Write the data rows
        for _, row in df[cols_include].iterrows():
            f.write(" & ".join(row.astype(str)) + " \\\\\n")
            f.write("\\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("\\caption{Your Table Caption}\n")
        f.write("\\label{tab:your_table_label}\n")
        f.write("\\end{table}\n")
        
def get_comparison_table_by_key(all_dict_data, value_key, method_display_names, 
                                force_methods=None, fontsize=14):
    # get successful scenarios per method
    df = pd.DataFrame(all_dict_data)
    methods = df['method_name'].unique()
    method_to_scenarios = {m: list(df[df.method_name == m]['scenario_name'].unique()) for m in methods}
    
    # get intersection of scenarios
    scenario_sets = [set(method_to_scenarios[m]) for m in methods]
    scenarios_intersection = list( set.intersection(*scenario_sets) )
    
    # ipdb.set_trace()
    
    # compute average of succeses
    col_data = value_key[4:] + "_vec"
    list_data = []
    for m in methods:
        # ipdb.set_trace()
        avg_val, std_val, n_succ, avg_int, std_int = \
            compute_avg_std(df, col_data, m, scenarios_intersection)
        curr_dict = {
            "method_name": m,
            f"average: {value_key}": avg_val,
            f"std: {value_key}": std_val,
            "Number of successes": n_succ,
            f"average intersection: {value_key}": avg_int,
            f"std intersection: {value_key}": std_int,
        }
        list_data.append(curr_dict)
        # ipdb.set_trace()
        
    for m in force_methods:
        curr_dict = {
            "method_name": m,
            f"average: {value_key}": "No succeses",
            f"std: {value_key}": "No succeses",
            "Number of successes": "No succeses",
            f"average intersection: {value_key}": "No succeses",
            f"std intersection: {value_key}": "No succeses",
        }
        list_data.append(curr_dict)
    
        
    # create the table 
    df = pd.DataFrame(list_data)
    return df


def gen_comparison_table(fp_folder_aghf, fp_folder_croco, fp_folder_alig, 
                         fp_folder_store, fn_store, fp_folder_params, 
                         list_fn_params, method_display_names, 
                         force_methods):
    
    folders_aghf = [] if fp_folder_aghf is None else [ f for f in os.listdir(fp_folder_aghf) if f.endswith("csv") ]
    folders_croco = [] if fp_folder_croco is None else os.listdir(fp_folder_croco)
    folders_alig = [] if fp_folder_alig is None else os.listdir(fp_folder_alig)
    
    all_scenario_names = get_all_scenario_names(fp_folder_params, list_fn_params)
    all_dict_data = [ ]
    
    for scenario_name in all_scenario_names:
        res_aghf = group_results(scenario_name, fp_folder_aghf, folders_aghf)
        res_croco = group_results(scenario_name, fp_folder_croco, folders_croco)
        res_alig = group_results(scenario_name, fp_folder_alig, folders_alig)
        if res_aghf is not None:
            all_dict_data.append(res_aghf)
        if res_croco is not None:
            all_dict_data.append(res_croco)
        if res_alig is not None:
            all_dict_data.append(res_alig)
            
    fp_u2 = os.path.join(fp_folder_store, f"u2_{fn_store}.tex")
    fp_t_solve = os.path.join(fp_folder_store, f"t_solve_{fn_store}.tex")            
    
    df_u2 = get_comparison_table_by_key(all_dict_data=all_dict_data, value_key="avg_u2", 
                                method_display_names=method_display_names, force_methods=force_methods)
    df_t_solve = get_comparison_table_by_key(all_dict_data=all_dict_data, value_key="avg_t_solve", 
                                method_display_names=method_display_names, force_methods=force_methods)
    dataframe_to_custom_latex(df_u2, fp_u2, "avg_u2")
    dataframe_to_custom_latex(df_t_solve, fp_t_solve, "avg_t_solve")

def gen_unified_comparison_table(fp_folder_aghf, fp_folder_croco, 
                                 fp_folder_alig, fp_folder_store, fn_store,
                                 fp_folder_params, list_fn_params, 
                                 method_display_names, force_methods,
                                 scenario_names=None):
    
    folders_aghf = [] if fp_folder_aghf is None else [ f for f in os.listdir(fp_folder_aghf) if f.endswith("csv") ]
    folders_croco = [] if fp_folder_croco is None else os.listdir(fp_folder_croco)
    folders_alig = [] if fp_folder_alig is None else os.listdir(fp_folder_alig)
    
    if scenario_names is not None:
        all_scenario_names = scenario_names
    else:
        all_scenario_names = get_all_scenario_names(fp_folder_params, 
                                                    list_fn_params)

    all_dict_data = [ ]

    for scenario_name in all_scenario_names:
        res_alig = group_results(scenario_name, fp_folder_alig, folders_alig)
        res_aghf = group_results(scenario_name, fp_folder_aghf, folders_aghf)
        res_croco = group_results(scenario_name, fp_folder_croco, folders_croco)
        if res_aghf is not None:
            all_dict_data.append(res_aghf)
        if res_croco is not None:
            all_dict_data.append(res_croco)
        if res_alig is not None:
            all_dict_data.append(res_alig)
            
    
    df_u2 = get_comparison_table_by_key(all_dict_data=all_dict_data, value_key="avg_u2", 
                                method_display_names=method_display_names, force_methods=force_methods)
    df_t_solve = get_comparison_table_by_key(all_dict_data=all_dict_data, value_key="avg_t_solve", 
                                method_display_names=method_display_names, force_methods=force_methods)
    
    # ipdb.set_trace()
    fp_unified = os.path.join(fp_folder_store, f"{fn_store}_unified.tex")

    os.makedirs(os.path.dirname(fp_unified), exist_ok = True)

    df_merged = merge_and_prepare_dfs(df_u2, df_t_solve)
    save_df_to_latex(fp_unified, df_merged, df_merged.columns.tolist())

   
def merge_and_prepare_dfs(df_u2, df_t_solve):
    val_to_std = {
        "average: avg_u2": "std: avg_u2",
        "average intersection: avg_u2": "std intersection: avg_u2",
        "average: avg_t_solve": "std: avg_t_solve",
        "average intersection: avg_t_solve": "std intersection: avg_t_solve",
    }
    
    # Merge the dataframes
    df_merged = pd.merge(df_u2, df_t_solve, on=['method_name', 'Number of successes'])
    
    # Update the data column with significant numbers and append ±1 std value
    for avg_col, std_col in val_to_std.items():
        # Replace non-numeric values with NaN to avoid errors
        df_merged[avg_col] = pd.to_numeric(df_merged[avg_col], errors='coerce')
        df_merged[std_col] = pd.to_numeric(df_merged[std_col], errors='coerce')
        
        # Round values to 4 significant figures
        df_merged[avg_col] = df_merged[avg_col].apply(lambda x: f"{x:.4g}" if pd.notna(x) else "No successes")
        df_merged[std_col] = df_merged[std_col].apply(lambda x: f"{x:.4g}" if pd.notna(x) else "No successes")
        
        # Combine average with ±1 standard deviation
        df_merged[avg_col] = df_merged.apply(
            lambda row: f"{row[avg_col]} ± {row[std_col]}" if pd.notna(row[std_col]) and pd.notna(row[avg_col]) else row[avg_col], axis=1
        )

    # Drop the std columns as they are no longer needed
    df_merged.drop(columns=list(val_to_std.values()), inplace=True)
    
    # Improve column titles
    column_renaming = {
        "average: avg_u2": "u2 of succeses",
        "average intersection: avg_u2": "u2 of intersection",
        "average: avg_t_solve": "tsolve",
        "average intersection: avg_t_solve": "t solve of intersection"
    }
    
    df_merged.rename(columns=column_renaming, inplace=True)

    # Reorder columns
    cols_order = ['method_name', 'Number of successes', 'u2 of succeses', 
                  'u2 of intersection', 'tsolve', 't solve of intersection']
    
    df_merged = df_merged[cols_order]

    return df_merged

    
