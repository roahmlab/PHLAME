import pinocchio as pin
import numpy as np

def compute_u_matrix_static(q, qd, qdd, model, num_threads=1):
    """
    Args:
        - q (np.ndarray): (len(t), N)
        - qd (np.ndarray): (len(t), N)
        - qdd (np.ndarray): (len(t), N)
        - model (pin.Model): model of the robot
        - num_threads (int): number of threads to use in the computation

    Returns:
        - u (np.ndarray): (len(t), N)
    """
    # Pinocchio expects matrices to be (N, batch_size)
    q = q.T
    qd = qd.T
    qdd = qdd.T
    pool = pin.ModelPool(model)
    # num_threads = pin.omp_get_max_threads()
    num_threads = 1
    u = pin.rnea(num_threads, pool, q, qd, qdd)
    u = u.T
    return u

def compute_u_sq_doubleint(qdd, dt):
    """
    Args:
        - qdd (np.ndarray): (len(t), N)
        - dt (float)

    Returns:
        - u_sq (float): value of the riemann sum of the norm of the torques squared
    """
    # compute u_sq
    norm_u_real = np.sum(qdd * qdd, axis=1)
    # riemann sum to integrate
    u_sq = dt * np.sum(norm_u_real)
    return u_sq