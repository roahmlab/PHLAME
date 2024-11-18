#ifndef ACTIVATED_SPHERE_AVOIDANCE_PENALTY_CPP
#define ACTIVATED_SPHERE_AVOIDANCE_PENALTY_CPP

#include "ActivatedSphereAvoidancePenalty.h"

ActivatedSphereAvoidancePenalty::ActivatedSphereAvoidancePenalty(
    const Model& model_inp,
    const Data& data_inp,
    const double k_cons_inp,
    const double c_cons_inp,
    const MatX& obstaclesInfo)
    : model(model_inp), data(data_inp)
{
  k_cons = k_cons_inp;
  c_cons = c_cons_inp;

  num_obstacles = obstaclesInfo.rows();

  if (obstaclesInfo.cols() != 4)
  {
    throw std::invalid_argument("obstaclesInfo has wrong row dimensions!");
  }

  obstacleCenters.resize(num_obstacles);
  obstacleRadii.resize(num_obstacles);

  for (size_t i = 0; i < num_obstacles; i++)
  {
    obstacleCenters(i) = obstaclesInfo.block(i, 0, 1, 3).transpose();
    obstacleRadii(i) = obstaclesInfo(i, 3);
  }

  activation_ins = std::make_unique<TanhActivation>(c_cons_inp);
  sphere_ineq_cons = std::make_unique<SphereInequalityConstraint>(model_inp);
}

double ActivatedSphereAvoidancePenalty::compute_value(const VecX& x)
{
  // Extract the joint positions from the input vector
  const VecX& q = x.head(model.nv);

  // Compute Forward Kinematics
  pino::forwardKinematics(model, data, q);
  pino::updateFramePlacements(model, data);

  // Initialize variables
  double total_penalty_value = 0;  // Total penalty value
  double curr_g = 0;               // Current inequality constraint value
  double curr_S = 0;               // Current activation function value

  // Loop through all the joints and compute the penalty value for each joint
  // for each obstacle
  for (int i = 1; i < model.joints.size(); i++)
  {
    auto joint_id = i;
    const Vec3 frame_position =
        data.oMi[joint_id].translation();  // Get the frame position

    // Loop through all the obstacles
    for (int j = 0; j < num_obstacles; j++)
    {
      // Compute the inequality constraint value and activation function value
      curr_g = sphere_ineq_cons->compute_value(
          obstacleRadii(j), frame_position, obstacleCenters(j));
      curr_S = activation_ins->compute_value(curr_g);

      // Accumulate the total penalty value
      total_penalty_value += k_cons * pow(curr_g, 2) * curr_S;
    }
  }

  return total_penalty_value;
}

Eigen::VectorXd ActivatedSphereAvoidancePenalty::compute_derivative(
    const VecX& x)
{
  // Extract the joint positions from the input vector
  const VecX& q = x.head(model.nv);

  // Compute the joint Jacobians and update the frame placements
  pino::computeJointJacobians(model, data, q);
  pino::updateFramePlacements(model, data);

  // Initialize variables
  Data::Matrix6x J(6, model.nv);              // FK Jacobian matrix
  MatX frame_position_jacobian(3, model.nv);  // Frame position Jacobian
  VecX total_penalty_derivative =
      VecX::Zero(x.size());           // Total penalty derivative
  double curr_g = 0;                  // Current inequality constraint value
  double curr_S = 0;                  // Current activation function value
  double dS_dg = 0;                   // Derivative of the activation function
  VecX dg_dx = VecX::Zero(x.size());  // Derivative of the inequality constraint

  // Loop through all the joints and compute the penalty derivative for each
  // joint for each obstacle
  for (int i = 1; i < model.joints.size(); i++)
  {
    auto joint_id = i;
    const Vec3 frame_position =
        data.oMi[joint_id].translation();  // Get the frame position

    // Compute the FK Jacobian
    J.setZero();
    pino::getJointJacobian(model, data, joint_id, pino::LOCAL_WORLD_ALIGNED, J);
    frame_position_jacobian = J.topRows(3);

    for (int j = 0; j < num_obstacles; j++)
    {
      // Compute the inequality constraint value and activation function value
      curr_g = sphere_ineq_cons->compute_value(
          obstacleRadii(j), frame_position, obstacleCenters(j));
      curr_S = activation_ins->compute_value(curr_g);

      // Compute the derivatives
      dg_dx = sphere_ineq_cons->compute_derivative(
          frame_position, obstacleCenters(j), frame_position_jacobian);
      dS_dg = activation_ins->compute_derivative(curr_g);

      // Accumulate the total penalty derivative
      total_penalty_derivative +=
          k_cons * (curr_g * (2 * dg_dx * curr_S + curr_g * dS_dg * dg_dx));
    }
  }

  return total_penalty_derivative;
}

std::pair<ActivatedPenalty::VecX, ActivatedPenalty::MatX>
ActivatedSphereAvoidancePenalty::compute_hessian(const VecX& x)
{
  // Extract the joint positions from the input vector
  const VecX& q = x.head(model.nv);

  // Compute the joint kinematic Hessians and update the frame placements
  pino::computeJointKinematicHessians(model, data, q);
  pino::updateFramePlacements(model, data);

  // Initialize variables
  MatX dFK_dx(6, model.nv);  // FK Jacobian matrix
  Eigen::Tensor<double, 3> ddFK_ddx(6, model.nv, model.nv);  // Hessian tensor
  Eigen::Tensor<double, 3> joint_position_hessian(
      3, model.nv, model.nv);  // Joint position Hessian
  joint_position_hessian.setZero();
  MatX joint_position_jacobian =
      MatX::Zero(3, model.nv);  // Joint position Jacobian
  MatX total_penalty_hessian =
      MatX::Zero(x.size(), x.size());  // Total penalty Hessian
  VecX total_penalty_derivative =
      VecX::Zero(x.size());  // Total penalty derivative

  double curr_g = 0;  // Current inequality constraint value
  double curr_S = 0;  // Current activation function value
  double dS_dg = 0;   // Derivative of the activation function

  // Derivative of the derivative of the activation function with respect to x
  VecX d__dS_dg__dx = VecX::Zero(x.size());

  VecX dg_dx = VecX::Zero(x.size());  // Derivative of the inequality constraint
  VecX deriv_term = VecX::Zero(x.size());  // Intermediate Derivative term

  // Second derivative of the inequality constraint
  MatX ddg_ddx = MatX::Zero(x.size(), x.size());

  // Loop through all the joints and compute the penalty hessian for each joint
  // for each obstacle
  for (int i = 1; i < model.joints.size(); i++)
  {
    auto jointId = i;
    const Vec3 joint_position =
        data.oMi[jointId].translation();  // Get the joint position

    // Compute FK Jacobian and Hessian
    dFK_dx.setZero();
    ddFK_ddx.setZero();
    pino::getJointJacobian(
        model, data, jointId, pino::LOCAL_WORLD_ALIGNED, dFK_dx);
    joint_position_jacobian = dFK_dx.topRows(3);
    pino::getJointKinematicHessian(
        model, data, jointId, pino::LOCAL_WORLD_ALIGNED, ddFK_ddx);

    // Slice the tensor to get the first 3 rows (3 x n x n)
    Eigen::array<Eigen::Index, 3> offsets = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> extents = {3, model.nv, model.nv};
    joint_position_hessian = ddFK_ddx.slice(offsets, extents);

    for (int j = 0; j < num_obstacles; j++)
    {
      // Compute the inequality constraint value and activation function value
      curr_g = sphere_ineq_cons->compute_value(
          obstacleRadii(j), joint_position, obstacleCenters(j));
      curr_S = activation_ins->compute_value(curr_g);

      // Compute the derivatives
      dg_dx = sphere_ineq_cons->compute_derivative(
          joint_position, obstacleCenters(j), joint_position_jacobian);
      ddg_ddx = sphere_ineq_cons->compute_hessian(
          joint_position, obstacleCenters(j), joint_position_jacobian,
          joint_position_hessian);
      dS_dg = activation_ins->compute_derivative(curr_g);
      d__dS_dg__dx = activation_ins->compute_hessian(curr_g) * dg_dx;

      // Compute the derivative term
      deriv_term = 2 * dg_dx * curr_S + curr_g * dS_dg * dg_dx;

      // Accumulate the total penalty derivative
      total_penalty_derivative += k_cons * (curr_g * deriv_term);

      // Accumulate the total penalty Hessian
      total_penalty_hessian +=
          k_cons *
          (deriv_term * dg_dx.transpose() +
           2 * curr_g *
               (ddg_ddx * curr_S + dg_dx * (dS_dg * dg_dx).transpose()) +
           curr_g * dS_dg * dg_dx * dg_dx.transpose() +
           curr_g * curr_g *
               (dg_dx * d__dS_dg__dx.transpose() + dS_dg * ddg_ddx));
    }
  }

  // Return the total penalty derivative and Hessian
  return std::make_pair(total_penalty_derivative, total_penalty_hessian);
}

#endif