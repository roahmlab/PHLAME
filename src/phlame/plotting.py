# provisionary functions to plot
import numpy as np
import matplotlib.pyplot as plt
from phlame.aghf import PostAghf
from phlame.utils import load_from_pickle
import ipdb
from matplotlib.colors import LinearSegmentedColormap


# Function to handle click events
def plot_single_state(data_mat, t, axs, label_str, data_type):
    """
    Plot position, velocity, and acceleration of all the joints in 3 different figures.
    Assumes that a figure already exists.

    Args:
        - data_mat (np.ndarray): (n_times, N), where N is the number of bodies. data_mat could be thought of as Q_mat, Qd_mat.
        - t (np.ndarray): Timespan, len(t) == n_times.
        - data_type (str): Type of data to plot. Options are 'position', 'velocity', or 'acceleration'.

    Returns:
        - None
    """
    N = data_mat.shape[1]
    plt.xlabel("Time")

    for ii in range(N):
        axs[ii].plot(t, data_mat[:, ii], label=label_str)
        axs[ii].legend()
        axs[ii].grid()
        axs[ii].set_ylabel(f"data_type={data_type}, index={ii+1}")
    # plt.legend()

def print_results(df, fp_str, s_query):
    """
    Prints and plots the results of the AGHF solution at a specific s value.

    Args:
        - df (pd.DataFrame): DataFrame containing the results.
        - fp_str (str): File path to the pickle file containing the AGHF solution.
        - s_query (float): The specific s value at which to evaluate the solution.

    Returns:
        - None

    This function loads the AGHF solution from a pickle file, extracts the solution at the specified s value,
    and computes the position, velocity, and acceleration profiles. It then plots the velocities and accelerations
    and prints the corresponding values.
    """
    result = load_from_pickle(fp_str)
    post_aghf = PostAghf(fp_str)
    curr_df = df[df.fp == fp_str]
    real_s_span = curr_df['real_s_span'].iloc[0]
    
    # obtain velocity difference as a vector
    idx = np.argmin( np.abs( real_s_span - s_query ) )
    print("Actual s: ", real_s_span[idx])
    ps_values = result.sol[idx, :]
    Xi = post_aghf.encode(ps_values)
    dXi = np.dot(post_aghf.D, Xi)
    ddXi = np.dot(post_aghf.D2, Xi)
    
    # the notation is bad and does not match the paper
    xp2 = Xi[:, post_aghf.N : ]
    xdp1 = dXi[:,  : post_aghf.N ]
    xdp2 = dXi[:, post_aghf.N : ]
    xddp1 = ddXi[:,  : post_aghf.N ]
    
    # to obtain cout
    plt.figure()
    plt.plot(xp2.T.flatten())
    plt.plot(xdp1.T.flatten())
    plt.title("Velocities")

    plt.figure()
    plt.plot(xdp2.T.flatten())
    plt.plot(xddp1.T.flatten())
    plt.title("Accelerations")

    plt.show()
    
    
    post_aghf.aghf_wrapper.update_AGHF(Xi, dXi, ddXi, post_aghf.p + 1, post_aghf.dXi_ds)
    d_ps_values_ds = post_aghf.decode(post_aghf.dXi_ds)

    print("Velocities")
    print(xp2)
    print(xdp1)
    print("Accelerations")
    print(xdp2)
    print(xddp1)
    print("d_ps_values_ds(called cout before)")
    print(d_ps_values_ds)
    
def plot_along_s_span_from_vecs(action_functional_vec, vel_dif_vec, 
                                accel_dif_vec, norm_dps_values_ds_vec, 
                                u_sq_vec, real_s_span):
    """
    Plots various metrics along the s-span.

    This function creates a figure with 5 subplots, each showing a different metric along the s-span.

    Args:
        - action_functional_vec (np.ndarray): Vector of action functional values.
        - vel_dif_vec (np.ndarray): Vector of velocity difference values.
        - accel_dif_vec (np.ndarray): Vector of acceleration difference values.
        - norm_dps_values_ds_vec (np.ndarray): Vector of norm of dps values ds.
        - u_sq_vec (np.ndarray): Vector of u squared values.
        - real_s_span (np.ndarray): Vector of s values.

    Returns:
        - None
    """

    fig, axs = plt.subplots(5, 1, sharex=True)  # 5 subplots in a vertical layout sharing the x-axis
    
    axs[0].plot(real_s_span, action_functional_vec)
    axs[0].set_ylabel("Action Functional")
    # make the y log scale
    axs[0].set_yscale('log')
    # add grid
    axs[0].grid(True)

    axs[1].plot(real_s_span, vel_dif_vec)
    axs[1].set_ylabel("Vel Dif")
    axs[1].grid(True)

    axs[2].plot(real_s_span, accel_dif_vec)
    axs[2].set_ylabel("Accel Dif")
    axs[2].grid(True)

    axs[3].plot(real_s_span, norm_dps_values_ds_vec)
    axs[3].set_ylabel("Norm PS Values DS")
    axs[3].grid(True)

    axs[4].plot(real_s_span, u_sq_vec)
    axs[4].set_ylabel("U squared")
    axs[4].grid(True)
    plt.xlabel("s-value")  # Common x-axis label
        
def plot_data(x, y, xlabel=None, ylabel=None, title=None, 
              font_size=12, ylog=False, xlog=False, 
              grid=False, color_ini='blue', color_end='red', 
              fp=None):
    """
    Plots the given x and y data with optional customization
    and saves it as a PDF if a filepath is provided.

    Args:
        - x (np.ndarray): Data for x-axis.
        - y (np.ndarray): Data for y-axis.
        - xlabel (str): Label for the x-axis (LaTeX code allowed).
        - ylabel (str): Label for the y-axis (LaTeX code allowed).
        - title (str): Title for the plot.
        - font_size (int): Font size for labels and title.
        - ylog (bool): Use logarithmic scale for y-axis (True/False).
        - xlog (bool): Use logarithmic scale for x-axis (True/False).
        - grid (bool): Enable grid lines (True/False).
        - color_ini (str): Starting color of the line.
        - color_end (str): Ending color of the line.
        - fp (str): File path to save the plot as a PDF.

    Returns:
        - None: Displays the plot and optionally saves it as a PDF.
    """
    
    plt.figure(figsize=(8, 6))
    
    # Create a color gradient between color_ini and color_end
    cmap = LinearSegmentedColormap.from_list('grad', [color_ini, 
                                                      color_end])
    num_points = len(x)
    colors = [cmap(i / num_points) for i in range(num_points)]

    # Plot the data with the color gradient
    for i in range(1, num_points):
        plt.plot(x[i-1:i+1], y[i-1:i+1], color=colors[i])

    # Set title, x-label, and y-label with the specified font size
    if title:
        plt.title(title, fontsize=font_size)
    if xlabel:
        plt.xlabel(f"${xlabel}$", fontsize=font_size)  # LaTeX support
    if ylabel:
        plt.ylabel(f"${ylabel}$", fontsize=font_size)  # LaTeX support
    
    # Set logarithmic scales if requested
    if ylog:
        plt.yscale('log')
    if xlog:
        plt.xscale('log')

    # Enable grid lines if requested
    if grid:
        plt.grid(True)
        plt.minorticks_on()  # Enable minor ticks for finer grid control
        plt.tick_params(axis='both', which='both', direction='in', 
                        top=True, right=True)  # Grid ticks all sides
    
    # Display tick values at each major tick
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)

    # Show the plot
    plt.tight_layout()

    # Save plot to a PDF if a file path is provided
    if fp:
        plt.savefig(fp, format='pdf')
        print(f"Plot saved to {fp}")
    
    # Display the plot
    plt.show()

def plot_along_s_span(df, fp_str, s_max=None, fp_urdf=None, fp_str_filter=None):
    """
    Plots various metrics along the s-span and the final state and control trajectories.

    This function loads the AGHF solution from a pickle file, extracts the solution at the specified s value,
    and computes the position, velocity, and acceleration profiles. It then plots the velocities and accelerations
    and prints the corresponding values.

    Args:
        - df (pd.DataFrame): DataFrame containing the results.
        - fp_str (str): File path to the pickle file containing the AGHF solution.
        - s_max (float, optional): The maximum s value to consider. Default is None.
        - fp_urdf (str, optional): File path to the URDF file. Default is None.
        - fp_str_filter (str, optional): File path filter for the DataFrame. Default is None.

    Returns:
        - None
    """
    
    if fp_str_filter is None:
        fp_str_filter = fp_str

    # load actual result
    result = load_from_pickle(fp_str)
    post_aghf = PostAghf(fp_str, fp_urdf)
     

    # compute quantities along the way
    curr_df = df[df.fp == fp_str_filter]
    
    action_functional_vec = curr_df['action_functional'].iloc[0]
    vel_dif_vec = curr_df['vel_dif'].iloc[0]
    accel_dif_vec = curr_df['accel_dif'].iloc[0]
    norm_dps_values_ds_vec = curr_df['norm_dps_values_ds'].iloc[0]
    u_sq_vec = curr_df['u_sq'].iloc[0]
    real_s_span = curr_df['real_s_span'].iloc[0]
    

    if s_max is not None:
        idx = np.argmin( np.abs( real_s_span - s_max ) )
        print("actual s_max to be used: ", real_s_span[idx] )
        real_s_span = real_s_span[ : idx]
        action_functional_vec = action_functional_vec[ : idx]
        vel_dif_vec = vel_dif_vec[ : idx]
        accel_dif_vec = accel_dif_vec[ : idx]
        norm_dps_values_ds_vec = norm_dps_values_ds_vec[ : idx]
        u_sq_vec = u_sq_vec[ : idx]
        Q_fin = curr_df['Q_mat'].iloc[0][idx, :, :]
        U_fin = curr_df['U_mat'].iloc[0][idx, :, :]
    else:
        Q_fin = curr_df['Q_mat'].iloc[0][-1, :, :]
        U_fin = curr_df['U_mat'].iloc[0][-1, :, :]

    
    
    fig, axs = plt.subplots(5, 1, sharex=True)  # 5 subplots in a vertical layout sharing the x-axis
    
    axs[0].plot(real_s_span, action_functional_vec)
    axs[0].set_ylabel("Action Functional")
    # make the y log scale
    axs[0].set_yscale('log')
    # add grid
    axs[0].grid(True)

    axs[1].plot(real_s_span, vel_dif_vec)
    axs[1].set_ylabel("Vel Dif")
    axs[1].grid(True)

    axs[2].plot(real_s_span, accel_dif_vec)
    axs[2].set_ylabel("Accel Dif")
    axs[2].grid(True)

    axs[3].plot(real_s_span, norm_dps_values_ds_vec)
    axs[3].set_ylabel("Norm PS Values DS")
    axs[3].grid(True)

    axs[4].plot(real_s_span, u_sq_vec)
    axs[4].set_ylabel("U squared")
    axs[4].grid(True)
    plt.xlabel("s-value")  # Common x-axis label
    
    # make a plot of Q_fin, as N subplots where N is the the dimension 2 of Q_fin
    # the x -axis is time between -1 and 1
    fig, axs = plt.subplots(Q_fin.shape[1], 1, sharex=True)  # 5 subplots in a vertical layout sharing the x-axis
    for i in range(Q_fin.shape[1]):
        axs[i].plot(Q_fin[:, i])
        axs[i].set_ylabel(f"q_{i}")
        axs[i].grid(True)
    plt.xlabel("time")  # Common x-axis label

    

    fig, axs = plt.subplots(U_fin.shape[1], 1, sharex=True)  # 5 subplots in a vertical layout sharing the x-axis
    for i in range(U_fin.shape[1]):
        axs[i].plot(U_fin[:, i])
        axs[i].set_ylabel(f"u_{i}")
        axs[i].grid(True)
    plt.xlabel("time")  # Common x-axis label

    plt.show(block=False)    
    


def plot_final_u():
    pass