
import example_robot_data
import numpy as np
import crocoddyl
from scipy.io import savemat
from scipy.io import loadmat
import os
import math
import time
import inspect
import timeit
import pickle
import ipdb
import pinocchio as pin

import sys
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), "../../src") ) )
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), ".." ) ) )
from phlame.cheb import Cheb

def gen_init_guess(X0, Xf, ic_type, num_times, N, p):
    """
    Generate initial guess for crocoddyl.
    NOTE: we assume that the initial guess for us is 0 for all time.
    Args:
    -----
    X0: np.array(2*N, 1): [q; qd] : initial condition
    Xf: np.array(2*N, 1): [q; qd] : final condition
    ic_type: str: type of initial condition
    num_times: int: number of nodes
    t_final: float: final time
    N: int: number of bodies
    """
    cheb_ins = Cheb(p=p, N=N, X0=X0, Xf=Xf)

    if ic_type == "line":
        init_ps_values = cheb_ins.generate_ps_traj_line()
    elif ic_type == "low_acc":
        init_ps_values = cheb_ins.generate_ps_traj_min_accel_v2()
    else:
        raise ValueError(f"ic_type: {ic_type} not recognized")
    
    # time from 0 to t_final, with num_times nodes
    t_x = np.linspace(-1, 1, num_times)

    result_traj, _, _ = cheb_ins.get_joints_trajectory(init_ps_values, t_x)
    inits_xs = [result_traj[i, :].reshape(-1, 1) for i in range(result_traj.shape[0])]
    inits_us = [np.zeros(shape=(N, 1)) for _ in range(num_times-1)]
    # ipdb.set_trace()
    return inits_xs, inits_us

def compute_quasi_static(model: pin.Model, x0, a, nq, nv):
    data = model.createData()
    q0 = x0[:nq]
    v0 = x0[nq : nq + nv]

    return pin.rnea(model, data, q0, v0, a)


def gen_init_guess_from_q_traj(q_traj, N, num_times, X0, Xf, mode_init_guess, model: pin.Model):
    """

    Input:
    -----
    q_traj: (n_samples, N), N: number of bodies
    N: int: number of bodies
    p: degree of the polynomial
    t_total: float: total time
    """

    n_samples_traj = q_traj.shape[0]
    cheb = Cheb(p=n_samples_traj-1, N=N, X0=X0, Xf=Xf)
    query_times = np.linspace(-1, 1, num_times+1)
    ps_values = cheb.generate_ps_traj_from_q_traj(q_traj)
    q_t, qd_t, qdd_t = cheb.get_q_qd_qdd(ps_values, query_times)
    result_traj = np.concatenate( (q_t, qd_t), axis=1)

    if mode_init_guess == "zeros":
        inits_xs = [ np.zeros((2*N, 1)) for _ in range(num_times + 1)]
        inits_us = [np.zeros(shape=(N, 1)) for _ in range(num_times)]
    elif mode_init_guess == "line_and_rnea":
        inits_xs = [result_traj[i, :].reshape(-1, 1) for i in range(result_traj.shape[0])]
        inits_us = [compute_quasi_static(model, result_traj[i, :].reshape(-1, 1), qdd_t[i, :], N, N, )  \
                    for i in range(num_times)]
    elif mode_init_guess == "x0_rnea":
        inits_xs = [X0 for _ in range(num_times+1)]
        inits_us = [compute_quasi_static(model, X0, np.zeros(N), N, N, )  for _ in range(num_times)]
    else:
        raise(f"`mode_init_guess` only possible values are: 'zeros', 'line_and_rnea', 'x0_rnea' but `f{mode_init_guess}` was given. ")

    return inits_xs, inits_us

def run_croco(x0, xf, weight_x=1e-2, weight_u=1e-3, weight_xf=1e3,
                       robot_name="custom_double_pendulum", dt=1e-2,
                       N=100, init_xs=None, init_us=None):
    # If the following line fails, read the Note at the beginning of the 
    # script
    """
    NB: number of bodies

    Input:
    ------ 
    x0: np.array(2*NB): [q; qd] : initial condition
    xf: np.array(2*NB): [q; qd] : final condition
    weight_x: float: weight on the running cost function of the state x ||x||^2
    weight_u: float: weight on the running cost function of the control u ||u||^2
    weight_xf: float: weight on the final state
    robot_name: str: specifies the name of the robot for `example_robot_data`
    dt: float: discretization of time
    N: int: number of nodes 

    Output:
    -------
    X
    U
    """
    
    #Determine whether to pass in the initial value
    if init_xs is not None and init_us is not None:
        pass_init_val = 1
    else:
        pass_init_val = 0
        
        

    robot_model = example_robot_data.load(robot_name).model
    state = crocoddyl.StateMultibody(robot_model)
    actuation = crocoddyl.ActuationModelFull(state)

    # Formulate the cost function
    nu = state.nv
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)
    # residual terms
    xResidual = crocoddyl.ResidualModelState(state, nu)
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    xfResidual = crocoddyl.ResidualModelState(state, xf, nu)
    # costs
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xfRegCost = crocoddyl.CostModelResidual(state, xfResidual)
    # add costs to cost model
    runningCostModel.addCost("uReg", uRegCost, weight_u)
    terminalCostModel.addCost("xfReg", xfRegCost, weight_xf)
    runningCostModel.addCost("xReg", xRegCost, weight_x) # So that the states are not huge

    # Next, we need to create an action model for running and terminal knots. The
    # forward dynamics (computed using ABA) are implemented
    # inside DifferentialActionModelFreeFwdDynamics.    
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        ),
        dt,
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        ),
        0.0,
    )

    # Generate the OCP
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * N, terminalModel)
    solver = crocoddyl.SolverFDDP(problem)
    
    init_xs_default = [val for val in solver.xs]
    init_us_default = [val for val in solver.us]
      
    # Set callbacks
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
    # Solve with either init_xs and init_us passed in or by using the default values to solve these
    if pass_init_val:
        t_ini = time.time()
        solver.solve(init_xs, init_us)
        t_solve = time.time() - t_ini
    else:
        t_ini = time.time()
        solver.solve()
        t_solve = time.time() - t_ini
       
    # Get solutions
    log = solver.getCallbacks()[0]
    U = np.array(log.us)
    X = np.array(log.xs)
    return X, U, t_solve

def store_result(dict_parameters, fp):
    savemat(fp, dict_parameters)

def make_dict_storage(x0, xf, weight_x, weight_u, weight_xf, robot_name, dt,
                       N, X, U, t_solve, cost_helper, init_xs=[], init_us=[]):
    dict_parameters = {
        'x0': x0,
        'xf': xf,
        'weight_x': weight_x,
        'weight_u': weight_u,
        'weight_xf': weight_xf,
        'robot_name': robot_name,
        'dt': dt,
        'N': N,
        'X': X,
        'U': U,
        't_solve': t_solve,
        'cost_helper': cost_helper,
        'init_xs': init_xs,
        'init_us': init_us,
    }
    return dict_parameters

def test_run_double_pendulum():
    fp = "storage/test_double_pendulum.mat"
    x0 = np.array([np.pi/2, 0.0, 0.0, 0.0])
    xf = np.array([-np.pi/2, 0.0, 0.0, 0.0])
    weight_x = 1e-2
    weight_u = 1e-3
    weight_xf = 1e3
    robot_name = "custom_double_pendulum"
    t_final = 1
    dt = 1e-4
    # N = t_final
    N = math.ceil(t_final / dt)
    X, U, t_solve = run_croco(x0, xf, weight_x=weight_x, weight_u=weight_u, 
                              weight_xf=weight_xf, robot_name=robot_name,
                              dt=dt, N=N)
    print("t_solve: ", t_solve)
    cost_helper = inspect.getsource(run_croco)
    dict_parameters = make_dict_storage(x0, xf, weight_x, weight_u, 
                                         weight_xf, robot_name, dt,
                                         N, X, U, t_solve, cost_helper)
    store_result(dict_parameters, fp)

def test_run_kinova():
    fp = "storage/test_kinova.mat"
    x0 = np.array([-1.0, -1.0, -1.0, -1.0,-1.0, -1.0, -1.0,  -2.48810803,   0.17484752,   5.44867024, -10.67093101,
            3.74048155,  -1.46462768,   2.42287559])
    xf = np.array([-4.66060394, -2.40998571,  6.21577519,  1.68012623,  5.866344  ,
        -2.22999715, -0.14888382, -3.10241871, -0.8535074 , -6.6003043 , -1.06186214,  3.59863419,
        -3.54590153,  3.28717275])    
    weight_x = 1e-2
    weight_u = 1e-3
    weight_xf = 1e3
    robot_name = "kinova_gen3"
    t_final = 2
    dt = 1e-4
    # N = t_final
    N = math.ceil(t_final / dt)
    X, U, t_solve = run_croco(x0, xf, weight_x=weight_x, weight_u=weight_u, 
                              weight_xf=weight_xf, robot_name=robot_name,
                              dt=dt, N=N)
    print("t_solve: ", t_solve)
    cost_helper = inspect.getsource(run_croco)
    dict_parameters = make_dict_storage(x0, xf, weight_x, weight_u, 
                                         weight_xf, robot_name, dt,
                                         N, X, U, t_solve, cost_helper)
    store_result(dict_parameters, fp) 

def run_scenario(fp_mat, weight_x, weight_u, weight_xf, robot_name, dt, t_final):
    data = loadmat(fp_mat)
    q_start = data['start'] # (7, 1)
    qd_start = data['goal']
    NB = q_start.shape[0]
    x0 = np.concatenate( ( q_start, np.zeros(shape=(NB, 1)) ) ).squeeze().astype(dtype=np.double, order='F')
    xf = np.concatenate( ( qd_start, np.zeros(shape=(NB, 1)) ) ).squeeze().astype(dtype=np.double, order='F')
    N = math.ceil(t_final / dt)
    X, U, t_solve = run_croco(x0, xf, weight_x=weight_x, weight_u=weight_u, 
        weight_xf=weight_xf, robot_name=robot_name, dt=dt, N=N)
    cost_helper = inspect.getsource(run_croco)
    res = make_dict_storage(x0, xf, weight_x, weight_u, weight_xf, robot_name, dt,
        N, X, U, t_solve, cost_helper)
    return res

def run_scenario_obstacles(weight_x, weight_u, weight_xf, weight_obs, weight_obs_fin, robot_name, dt, t_total, alpha_log,
                          frame_names, init_xs, init_us, obstacles, x0, xf, n_obstacles):
    """
    For each obstacle it assumes that its Rotation matrix is the identity.
    """

    if n_obstacles > 0:
        run_obstacle_weights = np.array([weight_obs] * n_obstacles)
        fin_obstacle_weights = np.array([weight_obs_fin] * n_obstacles)
    else:
        run_obstacle_weights = None
        fin_obstacle_weights = None

    X, U, t_solve = run_obstacle_avoidance(obstacles, frame_names, run_obstacle_weights, fin_obstacle_weights, robot_name, 
                           weight_x, weight_u, weight_xf, dt, t_total, x0, xf,
                           alpha_log, init_xs, init_us)
    return X, U, t_solve

def run_obstacle_avoidance(obstacles, frame_names, run_obstacle_weights, fin_obstacle_weights, robot_name, 
                           weight_x, weight_u, weight_xf, dt, t_total, x0, xf,
                           alpha_log, init_xs=None, init_us=None):
    """
    
    """
    
    """
    Inputs:
    -------
    obstacles: np.ndarray: (4, n_obstacles)
    frame_names: List[str]
    obstacle_weights: List[float]
    robot_name: str
    weight_x: float
    weight_u: float
    dt
    t_total
    x0
    init_xs
    init_us
    """
    if obstacles is None:
        n_obstacles = 0
    else:
        n_obstacles = obstacles.shape[1]

    robot_model = example_robot_data.load(robot_name).model
    state = crocoddyl.StateMultibody(robot_model)
    actuation = crocoddyl.ActuationModelFull(state)

    # Form the cost function
    nu = state.nv

    # create the cost function
    running_cost_model = crocoddyl.CostModelSum(state)
    terminal_cost_model = crocoddyl.CostModelSum(state)    

    # Create the residuals for the obstacles
    for ii in range(n_obstacles):
        pos_obs = obstacles[0 : 3, ii]
        radius_obs = obstacles[3, ii]
        run_weight_obs = run_obstacle_weights[ii]
        fin_weight_obs = fin_obstacle_weights[ii]
        for frame_name in frame_names:
            # residual
            # residual_obs = crocoddyl.ResidualModelFramePlacement(
            #     state,
            #     robot_model.getFrameId(frame_name),
            #     pin.SE3(rot_obs, pos_obs),
            #     nu
            # )
            if robot_name == "digit_pinned":
                residual_obs = crocoddyl.ResidualModelFrameTranslation(
                    state,
                    robot_model.getFrameId(frame_name, pin.FrameType.JOINT),
                    pos_obs,
                    nu
                )                
            else:
                residual_obs = crocoddyl.ResidualModelFrameTranslation(
                    state,
                    robot_model.getFrameId(frame_name),
                    pos_obs,
                    nu
                )
            # ipdb.set_trace()
            # 6 is nr, the dimension of the residual vector
            activation_obs = crocoddyl.ActivationModelQuadFlatLog(3, alpha_log)

            # cost model
            obs_cost = crocoddyl.CostModelResidual(state, activation_obs, residual_obs)

            # add costs from the obstacles
            running_cost_model.addCost(f"obs_{ii}_frame_{frame_name}", obs_cost, run_weight_obs)
            terminal_cost_model.addCost(f"obs_{ii}_frame_{frame_name}", obs_cost, fin_weight_obs)

    # other residuals
    x_residual = crocoddyl.ResidualModelState(state, nu)
    u_residual = crocoddyl.ResidualModelControl(state, nu)
    xf_residual = crocoddyl.ResidualModelState(state, xf, nu)
    # other costs
    x_reg_cost = crocoddyl.CostModelResidual(state, x_residual)
    u_reg_cost = crocoddyl.CostModelResidual(state, u_residual)
    xf_reg_cost = crocoddyl.CostModelResidual(state, xf_residual)
    # add other costs
    running_cost_model.addCost("x_reg", x_reg_cost, weight_x)
    running_cost_model.addCost("u_reg", u_reg_cost, weight_u)
    terminal_cost_model.addCost("xf_reg", xf_reg_cost, weight_xf)

    # create a running model
    running_model = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, running_cost_model
        ),
        dt,
    )

    # create a terminal model
    terminal_model = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminal_cost_model
        ),
        0.0,
    )

    # get number of nodes from dt and t_total
    N = math.ceil(t_total / dt)

    # Generate the OCP
    problem = crocoddyl.ShootingProblem(x0, [running_model] * N, terminal_model)

    solver = crocoddyl.SolverFDDP(problem)

    # ipdb.set_trace()

    # Set callbacks
    solver.setCallbacks(
        [
            crocoddyl.CallbackLogger(),
            # crocoddyl.CallbackVerbose(),
        ]
    )    
    if (init_xs is not None) and (init_us is not None):
        t_ini = time.time()
        solver.solve(init_xs, init_us)
        t_solve = time.time() - t_ini
        # print("init_xs[10]: ", init_xs[10])
    else:
        t_ini = time.time()
        solver.solve()
        t_solve = time.time() - t_ini        

    log = solver.getCallbacks()[0]
    U = np.array(log.us)
    X = np.array(log.xs)

    # print("solve time: ", t_solve)
    
    # print("solver.xs[-1]: ", solver.xs[-1])
    # print("X[-1, :]: ", X[-1, :])

    # ipdb.set_trace()
    return X, U, t_solve


def save_to_pickle(data, file_path):
    """
    Save data to a pickle file at the specified file path.

    Args:
        data: The data to be saved.
        file_path (str): The file path where the pickle file will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def run_all_scenarios(folder_scenarios, weight_x, weight_u, weight_xf, robot_name, n_runs, dt, t_final, folder_store):
    files = os.listdir(folder_scenarios)
    count_scenarios = 0
    for fn in files:
        print(f"scenarios: {count_scenarios} / {len(files)}")
        fp_mat = os.path.join(folder_scenarios, fn)
        curr_res = [] # contains all results for a single run        
        for ii in range(n_runs):
            res = run_scenario(fp_mat, weight_x, weight_u, weight_xf, robot_name, dt, t_final)
            res['scenario_name'] = fn[ : -4]
            t_solve = res['t_solve']
            log_str = f"Scenario: {fn} , Run: {ii+1} / {n_runs}, t_solve={t_solve}"
            print(log_str)
            curr_res.append(res)
        
        # store results for this scenario
        fn_store = fn[:-4] + "_crocoddyl_run.mat"
        fp_store = os.path.join(folder_store, fn_store)
        save_to_pickle(curr_res, fp_store)
        count_scenarios += 1


if __name__ == "__main__":
    # test_run_double_pendulum()
    test_run_kinova()
 