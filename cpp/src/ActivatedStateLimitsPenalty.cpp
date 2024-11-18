#ifndef ACTIVATED_STATE_LIMITS_PENALTY_CPP
#define ACTIVATED_STATE_LIMITS_PENALTY_CPP

#include "ActivatedStateLimitsPenalty.h"

#include "LowerDifferenceInequalityConstraint.h"
#include "TanhActivation.h"
#include "UpperDifferenceInequalityConstraint.h"

ActivatedStateLimitsPenalty::ActivatedStateLimitsPenalty(
    const VecX& stateLimits_inp,
    const double c_cons_inp,
    const double k_cons_inp)
{
  stateLimits = stateLimits_inp;
  c_cons = c_cons_inp;
  k_cons = k_cons_inp;
  activation_ins = std::make_unique<TanhActivation>(c_cons_inp);

  if (stateLimits(0) < 0)
  {
    ineq_cons_ins = std::make_unique<LowerDifferenceInequalityConstraint>();
  }
  else
  {
    ineq_cons_ins = std::make_unique<UpperDifferenceInequalityConstraint>();
  }
}

double ActivatedStateLimitsPenalty::compute_value(const VecX& x)
{
  // Check if the size of the input state vector matches the size of the state
  // limits
  if (x.size() != stateLimits.size())
  {
    throw std::invalid_argument(
        "x has wrong row dimensions or state limits not initialized!");
  }

  // Initialize variables
  double total_penalty_value = 0;  // Total penalty value
  double curr_g = 0;               // Current inequality constraint value
  double curr_S = 0;               // Current activation function value

  // Loop through each element in the state vector
  for (int ii = 0; ii < x.size(); ii++)
  {
    // Compute the inequality constraint value and activation function value
    curr_g = ineq_cons_ins->compute_value(x(ii), stateLimits(ii));
    curr_S = activation_ins->compute_value(curr_g);

    // Accumulate the total penalty value
    total_penalty_value += k_cons * pow(curr_g, 2) * curr_S;
  }

  // Return the total penalty value
  return total_penalty_value;
}

Eigen::VectorXd ActivatedStateLimitsPenalty::compute_derivative(const VecX& x)
{
  // Check if the size of the input state vector matches the size of the state
  // limits
  if (x.size() != stateLimits.size())
  {
    throw std::invalid_argument(
        "x has wrong row dimensions or state limits not initialized!");
  }

  // Initialize variables
  VecX penalty_derivative = VecX::Zero(x.size());  // Total penalty derivative
  double curr_g = 0.0;  // Current inequality constraint value
  double curr_S = 0.0;  // Current activation function value
  double dg_dx = 0.0;   // Derivative of the inequality constraint
  double dS_dg = 0.0;   // Derivative of the activation function

  // Loop through each element in the state vector
  for (int ii = 0; ii < x.size(); ii++)
  {
    // Compute the inequality constraint value and activation function value
    curr_g = ineq_cons_ins->compute_value(x(ii), stateLimits(ii));
    curr_S = activation_ins->compute_value(curr_g);

    // Compute the derivatives
    dg_dx = ineq_cons_ins->compute_derivative(x(ii), stateLimits(ii));
    dS_dg = activation_ins->compute_derivative(curr_g);

    // Compute the penalty derivative for the current element
    penalty_derivative(ii) =
        k_cons * (curr_g * (2.0 * dg_dx * curr_S + curr_g * dS_dg * dg_dx));
  }

  // Return the total penalty derivative
  return penalty_derivative;
}

std::pair<ActivatedPenalty::VecX, ActivatedPenalty::MatX>
ActivatedStateLimitsPenalty::compute_hessian(const VecX& x)
{
  // Check if the size of the input state vector matches the size of the state
  // limits
  if (x.size() != stateLimits.size())
  {
    throw std::invalid_argument(
        "x has wrong row dimensions or state limits not initialized!");
  }

  // Initialize variables
  MatX total_penalty_hessian =
      MatX::Zero(x.size(), x.size());  // Total penalty Hessian
  MatX curr_penalty_hessian =
      MatX::Zero(x.size(), x.size());              // Current penalty Hessian
  VecX penalty_derivative = VecX::Zero(x.size());  // Total penalty derivative

  double curr_g = 0;  // Current inequality constraint value
  double curr_S = 0;  // Current activation function value
  double dS_dg = 0;   // Derivative of the activation function

  // Derivative of the inequality constraint for the current element
  double dg_dx_ii = 0.0;

  // Derivative of the derivative of the activation function with respect to x
  VecX d__dS_dg__dx = VecX::Zero(x.size());

  VecX dg_dx = VecX::Zero(x.size());  // Derivative of the inequality constraint
  VecX deriv_term = VecX::Zero(x.size());  // Intermediate derivative term

  // Second derivative of the inequality constraint
  MatX ddg_ddx = MatX::Zero(x.size(), x.size());

  // Loop through each element in the state vector
  for (int ii = 0; ii < x.size(); ii++)
  {
    // Reset the derivative of the inequality constraint
    dg_dx = VecX::Zero(x.size());

    // Compute the inequality constraint value and activation function value
    curr_g = ineq_cons_ins->compute_value(x(ii), stateLimits(ii));
    curr_S = activation_ins->compute_value(curr_g);

    // Compute the derivatives
    dg_dx_ii = ineq_cons_ins->compute_derivative(x(ii), stateLimits(ii));
    dg_dx(ii) = dg_dx_ii;
    dS_dg = activation_ins->compute_derivative(curr_g);
    d__dS_dg__dx = activation_ins->compute_hessian(curr_g) * dg_dx;

    // Compute the penalty derivative for the current element
    penalty_derivative(ii) =
        k_cons *
        (curr_g * (2.0 * dg_dx_ii * curr_S + curr_g * dS_dg * dg_dx_ii));

    // Compute the current penalty Hessian
    curr_penalty_hessian =
        k_cons *
        ((2 * dg_dx * curr_S + curr_g * dS_dg * dg_dx) * dg_dx.transpose() +
         2 * curr_g * (ddg_ddx * curr_S + dg_dx * (dS_dg * dg_dx).transpose()) +
         curr_g * dS_dg * dg_dx * dg_dx.transpose() +
         curr_g * curr_g *
             (dg_dx * d__dS_dg__dx.transpose() + dS_dg * ddg_ddx));

    // Accumulate the total penalty Hessian
    total_penalty_hessian += curr_penalty_hessian;
  }

  // Return the total penalty derivative and Hessian
  return std::make_pair(penalty_derivative, total_penalty_hessian);
}

#endif