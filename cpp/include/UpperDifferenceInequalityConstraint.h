#ifndef UPPER_DIFFERENCE_INEQUALITY_CONSTRAINT_H
#define UPPER_DIFFERENCE_INEQUALITY_CONSTRAINT_H

#include <Eigen/Dense>
#include <iostream>

#include "StateInequalityConstraint.h"

class UpperDifferenceInequalityConstraint : public StateInequalityConstraint {
public:
    using VecX = Eigen::VectorXd;
    using MatX = Eigen::MatrixXd;

    UpperDifferenceInequalityConstraint() = default;

    ~UpperDifferenceInequalityConstraint() = default;

    double compute_value(double x, double x_max) override;
    // This derivative is with respect to the state
    double compute_derivative(double x, double x_max) override;

};

// class UpperDifferenceInequalityConstraint {
// public:
//     using VecX = Eigen::VectorXd;
//     using MatX = Eigen::MatrixXd;

//     UpperDifferenceInequalityConstraint() = default;

//     ~UpperDifferenceInequalityConstraint() = default;

//     double compute_value(double x, double x_max);
//     // This derivative is with respect to the state
//     double compute_derivative(double x, double x_max);

// };


#endif // UPPER_DIFFERENCE_INEQUALITY_CONSTRAINT_H