import os

from .parameter_set import ParameterSetBase

class ResultBase:
    def __init__(self, sol, t_solve, timeout, type_result, pset: ParameterSetBase):
        """
        Initializes the ResultBase class.

        Args:
            - sol (np.ndarray): Solution array.
            - t_solve (float): Time taken to solve the problem.
            - timeout (float): Timeout value.
            - type_result (ResultType): Type of the result.
            - pset (ParameterSetBase): An instance of ParameterSetBase containing the parameters for the result.

        Returns:
            - None
        """
        self.sol = sol
        self.t_solve = t_solve
        self.timeout = timeout
        self.type_result = type_result

        # any parameter set will always have this values
        self.name = pset.name
        self.p = pset.p
        self.s_max = pset.s_max
        self.k = pset.k
        self.abs_tol = pset.abs_tol
        self.rel_tol = pset.rel_tol
        self.method_name = pset.method_name
        self.max_steps = pset.max_steps
        self.ns_points = pset.ns_points
        self.N = pset.N
        self.j_type = pset.j_type
        self.fp_urdf = pset.fp_urdf
        self.X0 = pset.X0
        self.Xf = pset.Xf
        self.init_ps_values = pset.init_ps_values
        self.cheb_t = pset.cheb_t
        self.D = pset.D
        self.D2 = pset.D2
        self.s_span = pset.s_span

        if hasattr(pset, "fp_world"):
            self.fp_world = pset.fp_world
            self.fp_trail = pset.fp_trail
            self.data_world = pset.data_world
            self.data_trail = pset.data_trail
            self.obstacles = pset.obstacles
            self.num_obstacles = pset.num_obstacles
            self.obs_tau = pset.obs_tau
        
        if hasattr(pset, "k_cons"):
            self.c_cons = pset.c_cons
            self.k_cons = pset.k_cons
        
        if hasattr(pset, "obs_mat"):
            self.obs_mat = pset.obs_mat
        
        if hasattr(pset, "scenario_name"):
            self.scenario_name = pset.scenario_name

    def generate_uniquename(self):
        """
        Generates a unique name for the result based on its attributes.

        This function creates a unique name string by concatenating the values of specific attributes of the result object.
        Args:
            - self (ResultBase): The instance of the ResultBase class.

        Returns:
            - str: A unique name string generated from the result's attributes.
        """

        data_include_base = ["name", "p", "s_max", "k", "abs_tol", "rel_tol", "method_name", "N",
                             "t_solve"]
        # data_include_extra = ["fp_urdf"]
        # data_include_obs = ["fp_world", "num_obstacles", "obs_tau"]

        elems = []
        for attr_name in data_include_base:
            if attr_name == "t_solve":
                str_val = f"{attr_name}={round(getattr(self, attr_name), 3) }"
                elems.append(str_val)
            else:
                str_val = f"{attr_name}={getattr(self, attr_name) }"
                elems.append(str_val)                
        
        fn_urdf = os.path.basename(self.fp_urdf)
        elems.append(f"fn_urdf={fn_urdf}")

        if hasattr(self, "fp_world"):
            fn_world = os.path.basename(self.fp_world)
            elems.append(f"fn_world={fn_world}")
            elems.append(f"num_obstacles={self.num_obstacles}")
            elems.append(f"obs_tau={self.obs_tau}")
        
        if hasattr(self, "c_cons"):
            elems.append(f"c_cons={self.c_cons}")
            elems.append(f"k_cons={self.k_cons}")

        if hasattr(self, "scenario_name"):
            elems.append(f"scenario_name={self.scenario_name}")

        uniquename = "-".join(elems)

        return uniquename            