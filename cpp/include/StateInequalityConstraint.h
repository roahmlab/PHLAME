#ifndef STATE_INEQUALITY_CONSTRAINT_H
#define STATE_INEQUALITY_CONSTRAINT_H

#include <Eigen/Dense>
#include <iostream>

class StateInequalityConstraint {
public:
    using VecX = Eigen::VectorXd;
    using MatX = Eigen::MatrixXd;

    StateInequalityConstraint() = default;

    ~StateInequalityConstraint() = default;

    // For states
    virtual double compute_value(double x, double x_bound) = 0;
    virtual double compute_derivative(double x, double x_bound) = 0;
};

#endif // STATE_INEQUALITY_CONSTRAINT_H