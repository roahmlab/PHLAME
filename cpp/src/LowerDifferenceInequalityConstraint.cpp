#ifndef LOWER_DIFFERENCE_INEQUALITY_CONSTRAINT_CPP
#define LOWER_DIFFERENCE_INEQUALITY_CONSTRAINT_CPP

#include <Eigen/Dense>
#include <iostream>

#include "LowerDifferenceInequalityConstraint.h"

double LowerDifferenceInequalityConstraint::compute_value(double x, double x_min) {
    return x_min - x;
}

double LowerDifferenceInequalityConstraint::compute_derivative(double x, double x_min) {
    return -1.0;
}

#endif // LOWER_DIFFERENCE_INEQUALITY_CONSTRAINT_CPP