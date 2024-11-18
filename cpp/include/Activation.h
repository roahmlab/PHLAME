#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Eigen/Dense>
#include <iostream>

// Base class for the activation functions that are used to introduced an augmented lagrangian
// with soft constraints.
class Activation {

public:
    using VecX = Eigen::VectorXd;
    using MatX = Eigen::MatrixXd;

    Activation() = default;

    ~Activation() = default;

    virtual double compute_value(double x) = 0;

    virtual double compute_derivative(double x) = 0;
    virtual double compute_hessian(double x) = 0; //POTENTIAL

};


#endif // ACTIVATION_H