#ifndef SPHERE_INEQUALITY_CONSTRAINT_H
#define SPHERE_INEQUALITY_CONSTRAINT_H

#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct> 
#include <unsupported/Eigen/CXX11/Tensor>

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

namespace pino = pinocchio;

class SphereInequalityConstraint {
public:
    using VecX = Eigen::VectorXd;
    using MatX = Eigen::MatrixXd;
    using Vec3 = Eigen::Vector3d;
    using Model = pino::Model;
    Model model;

    SphereInequalityConstraint() = default;

    SphereInequalityConstraint(const Model& model_inp);

    ~SphereInequalityConstraint() = default;

    double compute_value(double R, const VecX& FK, const VecX& obs_center);
    Eigen::MatrixXd TensorVectorMult_3Dwith1D(Eigen::Ref<VecX> vec1D,
                                            Eigen::Tensor<double, 3>& tensor3D,
                                            int tensor_mult_dim);
    VecX compute_derivative(const VecX& FK, 
                            const VecX& obs_center, 
                            const MatX& frame_position_jacobian);

    MatX compute_hessian(const VecX& FK, 
                         const VecX& obs_center, 
                         const MatX& frame_position_jacobian,
                         Eigen::Tensor<double, 3> frame_position_hessian);

};

#endif // SPHERE_INEQUALITY_CONSTRAINT_H