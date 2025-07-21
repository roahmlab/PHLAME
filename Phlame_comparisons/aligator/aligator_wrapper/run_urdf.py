
import example_robot_data as erd
import numpy as np

from scipy.io import savemat
from scipy.io import loadmat
import os
import math
import time
import inspect
import timeit
import pickle
import ipdb
import time

import pinocchio as pin
import crocoddyl
import aligator
from aligator import manifolds
from proxsuite_nlp import constraints
from aligator import dynamics


import sys
sys.path.append( os.path.abspath( os.path.join( os.getcwd(), "../../src") ) )

from phlame.cheb import Cheb

from aligator_wrapper.constants import ROT_NULL

def gen_init_guess(X0, Xf, ic_type, num_times, N, p):
    """
    NOTE: This function is repeated in Crocoddyl, work to improve this later.
    Generate initial guess.
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
    return inits_xs, inits_us


def compute_quasi_static(model: pin.Model, x0, a, nq, nv):
    data = model.createData()
    q0 = x0[:nq]
    v0 = x0[nq : nq + nv]
    return pin.rnea(model, data, q0, v0, a)

def gen_init_guess_from_q_traj(q_traj, N, num_times, X0, Xf, mode_init_guess, model: pin.Model, dt):
    """
    NOTE: This function is repeated in Crocoddyl, work to improve this later.
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

    # copied from aligator's examples
    space = manifolds.MultibodyPhaseSpace(model)
    B_mat = np.eye(N)
    ode = dynamics.MultibodyFreeFwdDynamics(space, B_mat)
    discrete_dynamics = dynamics.IntegratorSemiImplEuler(ode, dt)
    
    if mode_init_guess == "zeros":
        inits_xs = [ np.zeros((2*N, 1)) for _ in range(num_times + 1)]
        inits_us = [np.zeros(shape=(N, 1)) for _ in range(num_times)]
    elif mode_init_guess == "line_and_rnea":
        inits_xs = [result_traj[i, :].reshape(-1, 1) for i in range(result_traj.shape[0])]
        inits_us = [compute_quasi_static(model, result_traj[i, :].reshape(-1, 1), qdd_t[i, :], N, N, )  \
                    for i in range(num_times)]
    elif mode_init_guess == "zeros_u_rollout_x":
        inits_us = [np.zeros(shape=(N, 1)) for _ in range(num_times)]
        inits_xs = aligator.rollout(discrete_dynamics, X0, inits_us).tolist()
    elif mode_init_guess == "x0_rnea_rollout":
        inits_us = [compute_quasi_static(model, X0, np.zeros(N), N, N, )  for _ in range(num_times)]
        inits_xs = aligator.rollout(discrete_dynamics, X0, inits_us).tolist()
    else:
        raise(f"mode_init_guess={mode_init_guess} is not valid.")

    return inits_xs, inits_us

class   SphereObstacle(aligator.StageFunction):
    def __init__(self, nq, ndx, nu, center, radius, robot_model, frame_id, robot_data) -> None:
        super().__init__(ndx, nu, 1)
        self.ndx = ndx
        self.center = center.copy()
        self.radius = radius.copy()
        self.robot_model = robot_model
        self.nq = nq
        self.frame_id = frame_id
        self.robot_data = robot_data
    
    def evaluate(self, x, u, y, data): # distance function
        q = x[ : self.nq ]
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        M: pin.SE3 = pin.updateFramePlacement(self.robot_model, self.robot_data, self.frame_id)
        diff = M.translation[:3].reshape(-1) - self.center.reshape(-1)
        res = np.dot(diff, diff) - (self.radius**2)
        data.value[:] = -res  
    
    def computeJacobians(self, x, u, y, data):
        q = x[ : self.nq]
        J = pin.computeFrameJacobian(self.robot_model, self.robot_data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        M: pin.SE3 = pin.updateFramePlacement(self.robot_model, self.robot_data, self.frame_id)
        err = M.translation - self.center
        data.Jx[:self.nq] = -2 * J[0:3, :].T @ err

def run_single(init_xs, init_us, u0, x0, xf, obstacles, frame_names, robot_name, dt, t_total, w_x, w_u, w_xf, tol=1e-3,
                   max_threads=1, mu_init=1e-2, max_iters=200, verbose_level=aligator.VerboseLevel.VERBOSE):
    robot = erd.load(robot_name)
    solver, problem = prepare_run(u0, x0, xf, obstacles, frame_names, robot, dt, t_total, w_x, w_u, w_xf, tol,
                   max_threads, mu_init, max_iters, verbose_level)
    
    res, t_solve = run_aligator(solver, problem, init_xs, init_us)

    X = np.array(res.xs)
    U = np.array(res.us)

    return X, U, t_solve


def prepare_run(u0, x0, xf, obstacles, frame_names, robot, dt, t_total, w_x, w_u, w_xf, tol=1e-3,
                   max_threads=1, mu_init=1e-2, max_iters=200, verbose_level=aligator.VerboseLevel.VERBOSE):
    robot_model = robot.model
    robot_data = robot.data
    nq = robot_model.nq
    nv = robot_model.nv
    nu = robot_model.nq # fully actuated

    space = manifolds.MultibodyPhaseSpace(robot_model)
    ndx = space.ndx

    ode_dynamics = aligator.dynamics.MultibodyFreeFwdDynamics(space)
    n_steps = int(t_total / dt)

    dynamics_model = aligator.dynamics.IntegratorSemiImplEuler(ode_dynamics, dt)

    # stages
    stages = []

    for ii in range(n_steps):

        cost_stack = aligator.CostStack(space, nu)

        # reg cost
        x_reg_cost = aligator.QuadraticStateCost(space, nu, xf, w_x*np.eye(ndx))
        cost_stack.addCost(x_reg_cost)

        # u cost
        u_reg_cost = aligator.QuadraticControlCost(space, u0, w_u * np.eye(nu))
        cost_stack.addCost(u_reg_cost)

        stage = aligator.StageModel(cost_stack, dynamics_model)

        if obstacles is not None:
            for idx_obstacle in range(obstacles.shape[1]):
                pos_obs = obstacles[0 : 3, idx_obstacle]
                radius_obs = obstacles[3, idx_obstacle]
                for frame_name in frame_names:
                    
                    frame_id = robot_model.getFrameId(frame_name, pin.FrameType.JOINT)

                    sphere = SphereObstacle(nq, ndx, nu, pos_obs, radius_obs, robot_model, frame_id, robot_data)

                    stage.addConstraint(sphere, constraints.NegativeOrthant())

        stages.append(stage)


    
    # empty terminal cost
    # term_cost = aligator.CostStack(space, nu)
    term_cost = aligator.QuadraticStateCost(space, nu, xf, w_xf*np.eye(ndx))
 
    # problem
    problem = aligator.TrajOptProblem(x0, stages, term_cost)

    # add terminal constraint
    terminal_constraint = aligator.StageConstraint(
        aligator.StateErrorResidual(space, nu, xf),
        constraints.EqualityConstraintSet(),
    )
    problem.addTerminalConstraint(terminal_constraint)   

    # solver
    solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose_level)

    solver.max_iters = max_iters
    solver.setNumThreads( max_threads )
    solver.setup(problem)

    return solver, problem


def run_aligator(solver, problem, init_xs, init_us):
    t_ini = time.time()
    solver.run(problem, init_xs, init_us)
    t_solve = time.time() - t_ini
    res = solver.results
    return res, t_solve
