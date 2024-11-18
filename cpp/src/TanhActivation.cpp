#ifndef TANH_ACTIVATION_CPP
#define TANH_ACTIVATION_CPP

#include "TanhActivation.h"

TanhActivation::TanhActivation(const double c_inp) : c_constant(c_inp) {}

double TanhActivation::compute_value(double x)
{
  double activation_value = 0.5 + 0.5 * tanh(c_constant * x);
  return activation_value;
}

double TanhActivation::compute_derivative(double x)
{
  double activation_derivative =
      (c_constant / 2.0) * (1 - pow(tanh(c_constant * x), 2));
  return activation_derivative;
}

double TanhActivation::compute_hessian(double x)
{
  double activation_hessian = -pow(c_constant, 2) *
                              (1 - pow(tanh(c_constant * x), 2)) *
                              tanh(c_constant * x);
  return activation_hessian;
}

#endif  // TANH_ACTIVATION_CPP