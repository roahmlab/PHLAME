#ifndef LOWER_DIFFERENCE_INEQUALITY_CONSTRAINT_H
#define LOWER_DIFFERENCE_INEQUALITY_CONSTRAINT_H

#include <Eigen/Dense>
#include <iostream>

#include "StateInequalityConstraint.h"

class LowerDifferenceInequalityConstraint : public StateInequalityConstraint {
public:
    using VecX = Eigen::VectorXd;
    using MatX = Eigen::MatrixXd;

    LowerDifferenceInequalityConstraint() = default;

    ~LowerDifferenceInequalityConstraint() = default;

    double compute_value(double x, double x_min) override;
    // This derivative is with respect to the state
    double compute_derivative(double x, double x_min) override;

};

// class LowerDifferenceInequalityConstraint {
// public:
//     using VecX = Eigen::VectorXd;
//     using MatX = Eigen::MatrixXd;

//     LowerDifferenceInequalityConstraint() = default;

//     ~LowerDifferenceInequalityConstraint() = default;

//     double compute_value(double x, double x_min);
//     // This derivative is with respect to the state
//     double compute_derivative(double x, double x_min);

// };

#endif // LOWER_DIFFERENCE_INEQUALITY_CONSTRAINT_H