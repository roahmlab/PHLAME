import numbers
import numpy as np
import os
import pickle


def is_scalar(value):
    return isinstance(value, numbers.Number)

def apply_func_to_vector(func, values):
    """
    Applies a given function to each element of a vector and returns the results as a matrix.

    Args:
        - func (Callable): The function to apply to each element of the vector.
        - values (np.ndarray): A 1-dimensional array of values (n,).

    Returns:
        - np.ndarray: A 2-dimensional array where each row corresponds to the result of applying the function to an element of the input vector.
    """
    temp_res = func(values[0])
    
    if is_scalar(temp_res):
        n_cols = 1
    else:
        n_cols = len(temp_res)

    matrix = np.zeros((len(values), n_cols))
    for ii in range(len(values)):
        matrix[ii, :] = np.squeeze(func(values[ii]))
    return matrix

def save_to_pickle(data, file_path):
    """
    Save data to a pickle file at the specified file path.

    Args:
        - data: The data to be saved.
        - file_path (str): The file path where the pickle file will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(file_path):
    """
    Load data from a pickle file located at the specified file path.

    Args:
        - file_path (str): The file path of the pickle file to load.

    Returns:
        - The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def interpolate_matrix(matrix, times, query_times):
    """
    Interpolate a matrix of values over time.

    Args:
        - matrix (numpy.ndarray): Matrix of values with dimensions (N_values, N_dimension).
        - times (numpy.ndarray): Time vector corresponding to the rows of the matrix.
        - query_times (numpy.ndarray): Times at which interpolation is required.

    Returns:
        - numpy.ndarray: Interpolated values at the specified query times with dimensions (len(query_times), N_values).
    """

    # Check if the input matrix and times have compatible dimensions
    if matrix.shape[0] != len(times):
        raise ValueError("The number of rows in the matrix must be equal to the length of the times vector.")
    
    if np.all(np.diff(times) < 0):
        raise ValueError("`times` HAS to be monotonically increasing. Read np.interp documentation")


    # Initialize an empty array to store the interpolated values
    interpolated_values = np.zeros((len(query_times), matrix.shape[1]))

    # Perform interpolation for each dimension separately
    for i in range(matrix.shape[1]):

        # Interpolate values at the specified query times for the i-th dimension
        interpolated_values[:, i] = np.interp(query_times, times, matrix[:, i])

    return interpolated_values

def cleanup_folder(fp_folder_store):
    """
    Cleanup the contents of a folder or create it if it doesn't exist.

    Args:
        - fp_folder_store (str): The absolute path of the folder to be cleaned up or created.

    Returns:
        None
    """
    if os.path.exists(fp_folder_store):
        # If folder exists, delete its contents
        for root, dirs, files in os.walk(fp_folder_store):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
    else:
        # If folder does not exist, create it
        os.makedirs(fp_folder_store)
        
def check_and_create_folder(fp_folder_store):
    """
   Create a folder if it doesn't exist.

    Args:
        - fp_folder_store (str): The absolute path of the folder to be created.

    Returns:
        None
    """
    if not os.path.exists(fp_folder_store):
        os.makedirs(fp_folder_store)

def generate_random_float(min_val: float, max_val: float) -> float:
    """
    Generate a random float number between min_val and max_val.

    Args:
        - min_val (float): The minimum value of the range.
        - max_val (float): The maximum value of the range.

    Returns:
        - float: A random float number sampled from a uniform distribution
                between min_val and max_val.
    """
    return np.random.uniform(min_val, max_val)

def extract_substring_after(main_string, substring):
    """
    Extracts the part of the main string that comes after the specified substring.

    Args:
        - main_string (str): The main string from which to extract the substring.
        - substring (str): The substring after which the extraction should occur.

    Returns:
        - str or None: The part of the main string after the substring, or None if the substring is not found.
    """
    # Find the position of the substring in the main string
    pos = main_string.find(substring)
    
    # Check if the substring is found
    if pos != -1:
        # Return the part of the string after the substring
        return main_string[pos + len(substring):]
    else:
        # Handle the case when the substring is not found
        return None
