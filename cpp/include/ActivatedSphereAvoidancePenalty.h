#ifndef ACTIVATED_SPHERE_AVOIDANCE_PENALTY
#define ACTIVATED_SPHERE_AVOIDANCE_PENALTY

// #include <Eigen/Dense>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/KroneckerProduct>

#include "ActivatedPenalty.h"
#include "SphereInequalityConstraint.h"
#include "TanhActivation.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

namespace pino = pinocchio;

class ActivatedSphereAvoidancePenalty : public ActivatedPenalty
{
 public:
  using Model = pino::Model;
  using Data = pino::Data;
  using Vec3 = Eigen::Vector3d;
  using VecX = Eigen::VectorXd;
  using MatX = Eigen::MatrixXd;

  ActivatedSphereAvoidancePenalty() = default;
  ActivatedSphereAvoidancePenalty(const Model& model_inp,
                                  const Data& data_inp,
                                  const double k_cons_inp,
                                  const double c_cons_inp,
                                  const MatX& obstaclesInfo);

  ~ActivatedSphereAvoidancePenalty() = default;

  double compute_value(const VecX& x) override;
  VecX compute_derivative(const VecX& x) override;
  std::pair<VecX, MatX> compute_hessian(
      const VecX& x) override;  // return both jacobian and hessian

  Model model;
  Data data;
  // SphereInequalityConstraint sphere_ineq_cons;
  std::unique_ptr<SphereInequalityConstraint> sphere_ineq_cons;
  std::unique_ptr<Activation> activation_ins;

  double c_cons;
  double k_cons;
  int num_obstacles = 0;
  Eigen::Array<Vec3, 1, Eigen::Dynamic> obstacleCenters;
  Eigen::Array<double, 1, Eigen::Dynamic> obstacleRadii;
};

#endif  // ACTIVATED_SPHERE_AVOIDANCE_PENALTY