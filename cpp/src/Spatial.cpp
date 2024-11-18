#ifndef SPATIAL_CPP
#define SPATIAL_CPP

#include "Spatial.h"

typedef Eigen::Matrix<double, 6, 6> Matrix6;
typedef Eigen::Matrix<double, 6, 1> Vector6;

void jcalc(Matrix6& Xj,
           Matrix6& dXjdt,
           Vector6& S,
           const int jtyp,
           const double q,
           const double q_d)
{
  Xj.setIdentity();
  dXjdt.setZero();
  S.setZero();

  double c = cos(q);
  double s = sin(q);
  double dcdt = -s * q_d;
  double dsdt = c * q_d;

  if (jtyp < 0)
  {  // reversed direction
    // c = cos(-q) = cos(q)
    s = -s;  // s = sin(-q) = -sin(q)
  }

  switch (jtyp)
  {
    case 1:  // revolute X axis 'Rx'
      Xj(1, 1) = c;
      Xj(1, 2) = s;
      Xj(2, 1) = -s;
      Xj(2, 2) = c;
      dXjdt(1, 1) = dcdt;
      dXjdt(1, 2) = dsdt;
      dXjdt(2, 1) = -dsdt;
      dXjdt(2, 2) = dcdt;
      S(0) = 1;
      break;
    case 2:  // revolute Y axis 'Ry'
      Xj(0, 0) = c;
      Xj(0, 2) = -s;
      Xj(2, 0) = s;
      Xj(2, 2) = c;
      dXjdt(0, 0) = dcdt;
      dXjdt(0, 2) = -dsdt;
      dXjdt(2, 0) = dsdt;
      dXjdt(2, 2) = dcdt;
      S(1) = 1;
      break;
    case 3:  // revolute Z axis 'Rz'
      Xj(0, 0) = c;
      Xj(0, 1) = s;
      Xj(1, 0) = -s;
      Xj(1, 1) = c;
      dXjdt(0, 0) = dcdt;
      dXjdt(0, 1) = dsdt;
      dXjdt(1, 0) = -dsdt;
      dXjdt(1, 1) = dcdt;
      S(2) = 1;
      break;
    case -1:  // reversed revolute X axis '-Rx'
      Xj(1, 1) = c;
      Xj(1, 2) = s;
      Xj(2, 1) = -s;
      Xj(2, 2) = c;
      dXjdt(1, 1) = dcdt;
      dXjdt(1, 2) = dsdt;
      dXjdt(2, 1) = -dsdt;
      dXjdt(2, 2) = dcdt;
      S(0) = -1;
      break;
    case -2:  // reversed revolute Y axis '-Ry'
      Xj(0, 0) = c;
      Xj(0, 2) = -s;
      Xj(2, 0) = s;
      Xj(2, 2) = c;
      dXjdt(0, 0) = dcdt;
      dXjdt(0, 2) = -dsdt;
      dXjdt(2, 0) = dsdt;
      dXjdt(2, 2) = dcdt;
      S(1) = -1;
      break;
    case -3:  // reversed revolute Z axis '-Rz'
      Xj(0, 0) = c;
      Xj(0, 1) = s;
      Xj(1, 0) = -s;
      Xj(1, 1) = c;
      dXjdt(0, 0) = dcdt;
      dXjdt(0, 1) = dsdt;
      dXjdt(1, 0) = -dsdt;
      dXjdt(1, 1) = dcdt;
      S(2) = -1;
      break;
    case 4:  // prismatic X axis 'Px'
      Xj(4, 2) = q;
      Xj(5, 1) = -q;
      dXjdt(4, 2) = q_d;
      dXjdt(5, 1) = -q_d;
      S(3) = 1;
      break;
    case 5:  // prismatic Y axis 'Py'
      Xj(3, 2) = -q;
      Xj(5, 0) = q;
      dXjdt(3, 2) = -q_d;
      dXjdt(5, 0) = q_d;
      S(4) = 1;
      break;
    case 6:  // prismatic Z axis 'Pz'
      Xj(3, 1) = q;
      Xj(4, 0) = -q;
      dXjdt(3, 1) = q_d;
      dXjdt(4, 0) = -q_d;
      S(5) = 1;
      break;
    case -4:  // reversed prismatic X axis '-Px'
      Xj(4, 2) = -q;
      Xj(5, 1) = q;
      dXjdt(4, 2) = -q_d;
      dXjdt(5, 1) = q_d;
      S(3) = -1;
      break;
    case -5:  // reversed prismatic Y axis '-Py'
      Xj(3, 2) = q;
      Xj(5, 0) = -q;
      dXjdt(3, 2) = q_d;
      dXjdt(5, 0) = -q_d;
      S(4) = -1;
      break;
    case -6:  // reversed prismatic Z axis '-Pz'
      Xj(4, 2) = q;
      Xj(5, 1) = -q;
      dXjdt(4, 2) = q_d;
      dXjdt(5, 1) = -q_d;
      S(5) = -1;
      break;
    default:
      throw std::invalid_argument("spatial.hpp: jcalc(): unknown joint type!");
      break;
  }

  if (fabs(jtyp) <= 3)
  {
    Xj.block(3, 3, 3, 3) = Xj.block(0, 0, 3, 3);
    dXjdt.block(3, 3, 3, 3) = dXjdt.block(0, 0, 3, 3);
  }
}

void d_jcalc(Matrix6& Xj,
             Matrix6& dXJdq,
             Matrix6& ddXJddq,
             Vector6& S,
             const int jtyp,
             const double q)
{
  Xj.setIdentity();
  dXJdq.setZero();
  ddXJddq.setZero();
  S.setZero();

  double c = cos(q);
  double s = sin(q);

  if (jtyp < 0)
  {  // reversed direction
    // c = cos(-q) = cos(q)
    s = -s;  // s = sin(-q) = -sin(q)
  }

  switch (jtyp)
  {
    case 1:  // revolute X axis 'Rx'
      Xj(1, 1) = c;
      Xj(1, 2) = s;
      Xj(2, 1) = -s;
      Xj(2, 2) = c;
      dXJdq(1, 1) = -s;
      dXJdq(1, 2) = c;
      dXJdq(2, 1) = -c;
      dXJdq(2, 2) = -s;
      ddXJddq(1, 1) = -c;
      ddXJddq(1, 2) = -s;
      ddXJddq(2, 1) = s;
      ddXJddq(2, 2) = -c;
      S(0) = 1;
      break;
    case 2:  // revolute Y axis 'Ry'
      Xj(0, 0) = c;
      Xj(0, 2) = -s;
      Xj(2, 0) = s;
      Xj(2, 2) = c;
      dXJdq(0, 0) = -s;
      dXJdq(0, 2) = -c;
      dXJdq(2, 0) = c;
      dXJdq(2, 2) = -s;
      ddXJddq(0, 0) = -c;
      ddXJddq(0, 2) = s;
      ddXJddq(2, 0) = -s;
      ddXJddq(2, 2) = -c;
      S(1) = 1;
      break;
    case 3:  // revolute Z axis 'Rz'
      Xj(0, 0) = c;
      Xj(0, 1) = s;
      Xj(1, 0) = -s;
      Xj(1, 1) = c;
      dXJdq(0, 0) = -s;
      dXJdq(0, 1) = c;
      dXJdq(1, 0) = -c;
      dXJdq(1, 1) = -s;
      ddXJddq(0, 0) = -c;
      ddXJddq(0, 1) = -s;
      ddXJddq(1, 0) = s;
      ddXJddq(1, 1) = -c;
      S(2) = 1;
      break;
    case -1:  // reversed revolute X axis '-Rx'
      Xj(1, 1) = c;
      Xj(1, 2) = -s;
      Xj(2, 1) = s;
      Xj(2, 2) = c;
      dXJdq(1, 1) = -s;
      dXJdq(1, 2) = -c;
      dXJdq(2, 1) = c;
      dXJdq(2, 2) = -s;
      ddXJddq(1, 1) = -c;
      ddXJddq(1, 2) = s;
      ddXJddq(2, 1) = -s;
      ddXJddq(2, 2) = -c;
      S(0) = -1;
      break;
    case -2:  // reversed revolute Y axis '-Ry'
      Xj(0, 0) = c;
      Xj(0, 2) = s;
      Xj(2, 0) = -s;
      Xj(2, 2) = c;
      dXJdq(0, 0) = -s;
      dXJdq(0, 2) = c;
      dXJdq(2, 0) = -c;
      dXJdq(2, 2) = -s;
      ddXJddq(0, 0) = -c;
      ddXJddq(0, 2) = -s;
      ddXJddq(2, 0) = s;
      ddXJddq(2, 2) = -c;
      S(1) = -1;
      break;
    case -3:  // reversed revolute Z axis '-Rz'
      Xj(0, 0) = c;
      Xj(0, 1) = -s;
      Xj(1, 0) = s;
      Xj(1, 1) = c;
      dXJdq(0, 0) = -s;
      dXJdq(0, 1) = -c;
      dXJdq(1, 0) = c;
      dXJdq(1, 1) = -s;
      ddXJddq(0, 0) = -c;
      ddXJddq(0, 1) = s;
      ddXJddq(1, 0) = -s;
      ddXJddq(1, 1) = -c;
      S(2) = -1;
      break;
    case 4:  // prismatic X axis 'Px'
      Xj(4, 2) = q;
      Xj(5, 1) = -q;
      dXJdq(4, 2) = 1;
      dXJdq(5, 1) = -1;
      S(3) = 1;
      break;
    case 5:  // prismatic Y axis 'Py'
      Xj(3, 2) = -q;
      Xj(5, 0) = q;
      dXJdq(3, 2) = -1;
      dXJdq(5, 0) = 1;
      S(4) = 1;
      break;
    case 6:  // prismatic Z axis 'Pz'
      Xj(3, 1) = q;
      Xj(4, 0) = -q;
      dXJdq(3, 1) = 1;
      dXJdq(4, 0) = -1;
      S(5) = 1;
      break;
    case -4:  // reversed prismatic X axis '-Px'
      Xj(4, 2) = -q;
      Xj(5, 1) = q;
      dXJdq(4, 2) = -1;
      dXJdq(5, 1) = 1;
      S(3) = -1;
      break;
    case -5:  // reversed prismatic Y axis '-Py'
      Xj(3, 2) = q;
      Xj(5, 0) = -q;
      dXJdq(3, 2) = 1;
      dXJdq(5, 0) = -1;
      S(4) = -1;
      break;
    case -6:  // reversed prismatic Z axis '-Pz'
      Xj(4, 2) = q;
      Xj(5, 1) = -q;
      dXJdq(4, 2) = 1;
      dXJdq(5, 1) = -1;
      S(5) = -1;
      break;
    default:
      throw std::invalid_argument("spatial.hpp: jcalc(): unknown joint type!");
      break;
  }

  if (fabs(jtyp) <= 3)
  {
    Xj.block(3, 3, 3, 3) = Xj.block(0, 0, 3, 3);
    dXJdq.block(3, 3, 3, 3) = dXJdq.block(0, 0, 3, 3);
    ddXJddq.block(3, 3, 3, 3) = ddXJddq.block(0, 0, 3, 3);
  }
}

Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
  Eigen::Matrix3d m;
  m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return m;
}

#endif  // SPATIAL_CPP