#ifndef TANH_ACTIVATION_H
#define TANH_ACTIVATION_H

#include "Activation.h"

#include <cmath>

class TanhActivation : public Activation {
public:
    using Vec3 = Eigen::Vector3d;
    using VecX = Eigen::VectorXd;
    using MatX = Eigen::MatrixXd;

    TanhActivation() = default;

    TanhActivation(const double c_inp);

    ~TanhActivation() = default;

    double compute_value(double x) override;

    double compute_derivative(double x) override;

    double compute_hessian(double x) override;

    double c_constant = 1.0; 
};

#endif // TANH_ACTIVATION_H