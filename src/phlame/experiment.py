from .aghf import Aghf
from .aghf import PostAghf
from .constants import ResultType
from .error import CustomError
from .parameter_set import ParameterSetBase
from .result import ResultBase
from .utils import save_to_pickle

from concurrent.futures import TimeoutError
import ipdb
import os
from pebble import ProcessPool
from scikits.odes.odeint import odeint
from scipy.io import savemat
import time
from typing import List

class Experiment:
    def __init__(self, parameter_sets: List[ParameterSetBase], timeout, print_debug, folder_store):
        """
        Initializes the Experiment class.

        Args:
            - parameter_sets (List[ParameterSetBase]): A list of parameter sets.
            - timeout (int): The timeout value for the experiment.
            - print_debug (bool): Flag to enable or disable debug printing.
            - folder_store (str): The folder path to store experiment results.
        """
        self.parameter_sets = parameter_sets
        self.timeout = timeout
        self.print_debug = print_debug
        self.folder_store = folder_store
    
    def process_result(self, result, fp_file):
        """
        Processes the result of an experiment and saves relevant data to a .mat file.

        Args:
            - result (object): The result object containing the solution and other relevant data.
            - fp_file (str): The file path where the .mat file will be saved.

        Returns:
            - None
        """
        if len(result.sol) != 0:
            post_aghf = PostAghf(result)
            q_traj_ini, q_traj_fin = post_aghf.get_q_traj_ini_fin(result.sol)
            if hasattr(result, "obs_mat"):
                if len(result.sol > 0):
                    # If cvode 
                    data_matlab = {"q_traj_ini": q_traj_ini, "q_traj_fin": q_traj_fin, 
                                   "obs_mat": result.obs_mat, "fp_urdf": result.fp_urdf}
                    fp_mat = fp_file[:-4] + ".mat"
                    savemat(fp_mat, data_matlab)        
        

    def store_experiment(self, result):
        """
        Stores the experiment result to a pickle file and processes the result.

        Args:
            - result (dict or list of dict): The result of the experiment. It can be a single dictionary 
                           or a list of dictionaries if multiple trials were run per parameter set.

        Returns:
            - None
        """
        # result might be a dictionary or a list of dictionaries
        #   it is a list of dictionaries when we are running multiple trials per parameter set
        #   else it is just a dictionary

        if self.folder_store is None:
            return
        else:
            if isinstance(result, list):
                unique_name = result[0].generate_uniquename() + ".pkl"
            else:
                unique_name = result.generate_uniquename() + ".pkl"

            fp_file = os.path.join(self.folder_store, unique_name)
            save_to_pickle(result, fp_file)

            # Store a mat file that stores q_traj, obs_mat, fp_urdf
            # This logic could be incorporated into post_processing.py
            if isinstance(result, list):
                self.process_result(result[0], fp_file)
            else:
                self.process_result(result, fp_file)

    @staticmethod
    def run_single_static(pset, timeout, mode="general", print_debug=False, use_jacobian=True):
        """
        Solves the AGHF.

        Args:
            - pset (object): Parameter set containing initial values, method name, tolerances, and other configurations.
            - timeout (float): Maximum allowed time for the solver to run.
            - mode (str, optional): Mode of operation for the AGHF solver. Default is "general".
            - print_debug (bool, optional): Flag to enable or disable debug printing. Default is False.
            - use_jacobian (bool, optional): Flag to indicate whether to use the Jacobian matrix in the solver. Default is True.

        Returns:
            - result (ResultBase): An object containing the solution, time taken to solve, timeout value, result type, and parameter set.
        """
        aghf = Aghf(pset=pset, mode=mode, print_debug=print_debug, use_jacobian=use_jacobian)

        try:
            t_ini = time.time()
            if use_jacobian:
                res = odeint(rhsfun=aghf.rhs_pde_scikits, tout=pset.s_span, 
                            y0=pset.init_ps_values.reshape(-1), # because of scikits.odes
                                method=pset.method_name, max_steps=pset.max_steps, 
                                atol=pset.abs_tol, rtol=pset.rel_tol, old_api=False, 
                                jacfn=aghf.jac_rhs_pde_scikits)
            else:
                res = odeint(rhsfun=aghf.rhs_pde_scikits, tout=pset.s_span, 
                            y0=pset.init_ps_values.reshape(-1), # because of scikits.odes
                                method=pset.method_name, max_steps=pset.max_steps, 
                                atol=pset.abs_tol, rtol=pset.rel_tol, old_api=False)                
            t_solve = time.time() - t_ini
            sol = res.values.y
            type_result = ResultType.SUCCESS
        except Exception as err:
            print("exception: ", err)
            sol = []
            t_solve = timeout
            type_result = ResultType.CVODE_ERROR
        
        result = ResultBase(sol=sol, t_solve=t_solve, timeout=timeout, type_result=type_result, 
                            pset=pset)
        return result        

    def run_single(self, pset, mode="general", use_jacobian=True):
        """
        Solves the AGHF.

        Args:
            - pset (object): Parameter set containing initial values, method name, tolerances, and other configurations.
            - mode (str, optional): Mode of operation for the AGHF solver. Default is "general".
            - use_jacobian (bool, optional): Flag to indicate whether to use the Jacobian matrix in the solver. Default is True.

        Returns:
            - result (ResultBase): An object containing the solution, time taken to solve, timeout value, result type, and parameter set.
        """
        aghf = Aghf(pset=pset, mode=mode, use_jacobian=use_jacobian)

        try:
            t_ini = time.time()
            if use_jacobian:
                res = odeint(rhsfun=aghf.rhs_pde_scikits, tout=pset.s_span, 
                            y0=pset.init_ps_values.reshape(-1), # because of scikits.odes
                                method=pset.method_name, max_steps=pset.max_steps, 
                                atol=pset.abs_tol,
                                rtol=pset.rel_tol, old_api=False, jacfn=aghf.jac_rhs_pde_scikits)
            else:
                res = odeint(rhsfun=aghf.rhs_pde_scikits, tout=pset.s_span, 
                            y0=pset.init_ps_values.reshape(-1), # because of scikits.odes
                                method=pset.method_name, max_steps=pset.max_steps, 
                                atol=pset.abs_tol,
                                rtol=pset.rel_tol, old_api=False)
            t_solve = time.time() - t_ini
            sol = res.values.y
            type_result = ResultType.SUCCESS  
        except Exception as err:
            print("exception: ", err)
            sol = []
            t_solve = self.timeout
            type_result = ResultType.CVODE_ERROR
        
        result = ResultBase(sol=sol, t_solve=t_solve, timeout=self.timeout, 
                            type_result=type_result, pset=pset)
        return result

    def run_parallel(self, max_workers=1, max_tasks=1, use_jacobian=True, debug=False):
        """
        Solves multiple AGHFs in parallel.

        Args:
            - max_workers (int, optional): The maximum number of worker processes to use. Defaults to 1.
            - max_tasks (int, optional): The maximum number of tasks per worker. Defaults to 1.
            - use_jacobian (bool, optional): Whether to use the Jacobian in the computations. Defaults to True.
            - debug (bool, optional): If True, runs in single-threaded mode for debugging purposes. Defaults to False.
        """
        if debug:
            for pset in self.parameter_sets:
                res = self.run_single(pset, use_jacobian=use_jacobian)
        else:
            with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
                for pset in self.parameter_sets:
                    future = pool.schedule(self.run_single, args=(pset,), 
                                           kwargs={"use_jacobian": use_jacobian}, 
                                           timeout=self.timeout)
                    callback = self.create_callback_mp(pset)
                    future.add_done_callback(callback)

    def run_seq_and_store(self, use_jacobian=True, debug=False):
        """
        Runs a sequence of experiments using the provided parameter sets and stores the results.

        Args:
            - use_jacobian (bool, optional): Flag to indicate whether to use the Jacobian in the experiment. Defaults to True.
            - debug (bool, optional): Flag to indicate whether to run in debug mode. Defaults to False.
        """
        for pset in self.parameter_sets:
            result = self.run_single(pset, use_jacobian=use_jacobian)
            self.store_experiment(result)

    def create_callback_mp(self, pset):
        """
        Creates a callback function to handle the completion of a multiprocessing task.
        Args:
            - pset: A parameter set object containing various parameters for the task.
        Returns:
            - A callback function that processes the result of the task when it is done.
              The callback function handles different types of exceptions, including
              TimeoutError and general exceptions, and stores the result of the task
              using the `store_experiment` method.
        """
        def task_done(future):
            try:
                result = future.result()
            except TimeoutError as error:
                print("Timeout was triggered.")
                result = ResultBase(sol=[], t_solve=self.timeout, timeout=self.timeout, 
                                    type_result=ResultType.TIMEOUT, pset=pset)
            except Exception as error:
                print("Uncaught exception when trying to access `future.result()` raised %s" % error)
                print(error)  # traceback of the function
                print("k:", pset.k)
                print("p: ", pset.p)
                print("N: ", pset.N)
                print("s_max: ", pset.s_max)
                if hasattr(pset, 'c_cons'):
                    print("c_cons: ", pset.c_cons)
                    print("k_cons: ", pset.k_cons)
                result = ResultBase(sol=[], t_solve=self.timeout, timeout=self.timeout,
                                    type_result=ResultType.OTHER_ERROR, pset=pset)
            
            self.store_experiment(result)

        return task_done