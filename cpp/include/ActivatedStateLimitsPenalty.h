#ifndef ACTIVATED_STATE_LIMITS_PENALTY_H
#define ACTIVATED_STATE_LIMITS_PENALTY_H

#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "ActivatedPenalty.h"
#include "Activation.h"
#include "StateInequalityConstraint.h"

class ActivatedStateLimitsPenalty : public ActivatedPenalty
{
 public:
  using Vec3 = Eigen::Vector3d;
  using VecX = Eigen::VectorXd;
  using MatX = Eigen::MatrixXd;

  ActivatedStateLimitsPenalty() = default;

  ActivatedStateLimitsPenalty(const VecX& stateLimits_inp,
                              const double c_cons_inp,
                              const double k_cons_inp);

  ~ActivatedStateLimitsPenalty() = default;

  double compute_value(const VecX& x) override;
  VecX compute_derivative(const VecX& x) override;
  std::pair<VecX, MatX> compute_hessian(const VecX& x) override;

  VecX stateLimits;
  double c_cons;
  double k_cons;

 private:
  // StateInequalityConstraint ineq_cons_ins;
  std::unique_ptr<StateInequalityConstraint> ineq_cons_ins;
  std::unique_ptr<Activation> activation_ins;
};

#endif  // ACTIVATED_STATE_PENALTY_H