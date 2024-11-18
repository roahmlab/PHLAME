#ifndef ACTIVATED_PENALTY_H
#define ACTIVATED_PENALTY_H

#include <Eigen/Dense>
#include <iostream>

// #include "Activation.h"
#include "InequalityConstraint.h"

// Base class for a penalty function that utilizes an activation function
// Mathematically, the penalty looks like k_cons * g_i(X_s)^2 * S( g_i(X_s) )
class ActivatedPenalty
{
 public:
  using Vec3 = Eigen::Vector3d;
  using VecX = Eigen::VectorXd;
  using MatX = Eigen::MatrixXd;

  ActivatedPenalty() = default;

  ~ActivatedPenalty() = default;

  virtual double compute_value(const VecX& x) = 0;

  virtual VecX compute_derivative(const VecX& x) = 0;
  virtual std::pair<VecX, MatX> compute_hessian(const VecX& x) = 0;

  // private:
  //     Activation activation_ins;
  //     InequalityConstraint ineq_cons_ins;
};

#endif  // ACTIVATED_PENALTY_H
