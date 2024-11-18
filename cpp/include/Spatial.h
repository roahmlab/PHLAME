#ifndef SPATIAL_H
#define SPATIAL_H

#include <Eigen/Dense>

typedef Eigen::Matrix<double, 6, 6> Matrix6;
typedef Eigen::Matrix<double, 6, 1> Vector6;

void jcalc(Matrix6& Xj, 
           Matrix6& dXjdt,
           Vector6& S, 
           const int jtyp,
           const double q,
           const double q_d);

void d_jcalc(Matrix6& Xj, 
           Matrix6& dXJdq,
           Matrix6& ddXJddq,
           Vector6& S, 
           const int jtyp,
           const double q);

Eigen::Matrix3d skew(const Eigen::Vector3d& v);

#endif // SPATIAL_H