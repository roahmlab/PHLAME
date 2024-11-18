#ifndef UPPER_DIFFERENCE_INEQUALITY_CONSTRAINT_CPP
#define UPPER_DIFFERENCE_INEQUALITY_CONSTRAINT_CPP

#include "UpperDifferenceInequalityConstraint.h"

#include <Eigen/Dense>
#include <iostream>

double UpperDifferenceInequalityConstraint::compute_value(double x,
                                                          double x_max)
{
  return x - x_max;
}

double UpperDifferenceInequalityConstraint::compute_derivative(double x,
                                                               double x_max)
{
  return 1;
}

#endif  // UPPER_DIFFERENCE_INEQUALITY_CONSTRAINT_CPP