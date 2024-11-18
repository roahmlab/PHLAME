#ifndef PYBIND_WRAPPER_HPP
#define PYBIND_WRAPPER_HPP

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ActivatedSphereAvoidancePenalty.h"
#include "ActivatedStateLimitsPenalty.h"
#include "Spatial.h"
#include "tensor_utils.hpp"

// #include "ID_FO_AZA.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>

#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/bindings/python/parsers/urdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

namespace py = pybind11;
namespace pino = pinocchio;

class AghfPybindWrapper
{
 public:
  using VecX = Eigen::VectorXd;
  using MatX = Eigen::MatrixXd;

  pino::Model model;
  pino::Data data;
  double k = 1;
  int N = 1;
  Eigen::VectorXd j_type;

  int num_ps_nodes = 1;

  //   std::vector<std::unique_ptr<penalty>> penalty_vec;
  //   std::vector<std::unique_ptr<ActivatedPenalty>> activated_penalty_vec;
  //   double penalty_value;
  //   VecX d_penalty_dx;
  //   MatX dd_penalty_ddx;

  //   Eigen::VectorXd zero_vec;
  //   Eigen::MatrixXd zero_mat;
  //   std::pair<VecX, MatX> penalty_deriv_and_hessian;

  std::vector<std::unique_ptr<ActivatedPenalty>> activated_penalty_vec;
  double penalty_value;
  VecX d_penalty_dx;
  MatX dd_penalty_ddx;

  Eigen::VectorXd zero_vec;
  Eigen::MatrixXd zero_mat;
  std::pair<VecX, MatX> penalty_deriv_and_hessian;

  Eigen::Array<Matrix6, 1, Eigen::Dynamic> Xtree;
  Eigen::Array<Matrix6, 1, Eigen::Dynamic> I;
  Eigen::Array<Matrix6, 1, Eigen::Dynamic> Xup;
  Eigen::Array<Matrix6, 1, Eigen::Dynamic> dXupdt;
  Eigen::Array<Vector6, 1, Eigen::Dynamic> S;
  Eigen::Array<Matrix6, 1, Eigen::Dynamic> IC;
  Eigen::Array<Matrix6, 1, Eigen::Dynamic> dICdt;

  // Additional variables to compute dH_dq and ddH_ddq
  Eigen::Array<Matrix6, 1, Eigen::Dynamic> dXupdq;
  Eigen::Array<Matrix6, 1, Eigen::Dynamic> ddXupddq;
  Eigen::Array<Eigen::Tensor<double, 3>, 1, Eigen::Dynamic> dICdq;
  Eigen::Array<Eigen::Tensor<double, 4>, 1, Eigen::Dynamic> ddICddq;

  VecX xp1;
  VecX xp2;
  VecX xdp1;
  VecX xdp2;
  VecX xddp1;
  VecX xddp2;

  MatX k_diag;
  MatX k_inv_diag;

  VecX FD_0;  // acceleration corresponding to Forward Dynamics with tau = 0
  MatX dFD_dq;
  MatX dFD_dv;

  MatX H;
  MatX H_dot;
  MatX H_inv;

  VecX C;
  VecX C_dot;
  MatX dC_dq;
  MatX dC_dv;
  MatX dC_da;
  MatX dID_dq;
  MatX dID_dv;
  MatX dID_da;

  MatX G;
  MatX G_inv;
  VecX G_inv_dL_dxd;
  VecX G_inv_d_dt_dL_dxd;
  VecX G_inv_dL_dx;
  VecX G_inv_dLprime_dx;
  VecX G_inv_d_dt_dLprime_dxd;

  VecX Omega_1;
  VecX Omega_2;
  VecX Omega_3;
  VecX Omega_4;

  // Jacobian Objects
  MatX AGHF_jac;
  Eigen::Tensor<double, 3> dH_dq;
  Eigen::Tensor<double, 4> ddH_ddq;
  Eigen::Tensor<double, 3> ddC_ddq;
  Eigen::Tensor<double, 3> ddC_ddv;
  Eigen::Tensor<double, 3> ddC_dqdv;
  Eigen::Tensor<double, 3> ddC_dvdq;
  Eigen::Tensor<double, 3> du_dadq;
  Eigen::Tensor<double, 3> dHinv_dq;

  Eigen::Tensor<double, 3> ddID_ddq;
  Eigen::Tensor<double, 3> ddID_ddv;
  Eigen::Tensor<double, 3> ddID_dqdv;
  Eigen::Tensor<double, 3> ddID_dadq;

  Eigen::Tensor<double, 3> ddFD_ddq;
  Eigen::Tensor<double, 3> ddFD_ddv;
  Eigen::Tensor<double, 3> ddFD_dqdv;
  Eigen::Tensor<double, 3> ddFD_dudq;
  Eigen::Tensor<double, 3> ddFD_dvdq;

  MatX HinvTHinv;
  MatX HTH;
  Eigen::Tensor<double, 3> HinvTHinv_dxp1;
  Eigen::Tensor<double, 3> dGinv_dx;
  Eigen::Tensor<double, 3> dHdot_dq;

  MatX dCdot_dxp1;
  MatX dCdot_dxp2;

  VecX gamma;
  VecX alpha1;
  VecX alpha2;
  VecX H_xdp2_p_C;
  VecX Hdot_xdp2_p_H_xddp2_p_Cdot;
  MatX dgamma_dxp1;
  VecX H_2C_p_2H_xdp2;
  VecX twoC_p_2H_xdp2;
  MatX dH_dxp1_m_xdp2;

  MatX dOmega1_dxp1;    // d/dxp1 (Ginv*d/dt (dL/dxd) )
  MatX dOmega1_dxp2;    // d/dxp2 (Ginv*d/dt (dL/dxd) )
  MatX dOmega234_dxp1;  // d/dxp1 (Ginv * (dL/dx) )
  MatX dOmega234_dxp2;  // d/dxp2 (Ginv * (dL/dx) )

  Eigen::Tensor<double, 3> neg_ddFD_ddq_T;
  Eigen::Tensor<double, 3> neg_ddFD_dqdv_T;
  Eigen::Tensor<double, 3> neg_ddFD_ddv_T;
  Eigen::Tensor<double, 3> neg_ddFD_dvdq_T;

  MatX neg_dFD_dq_T;
  MatX neg_dFD_dv_T;

  MatX dalpha_b_block;
  MatX d_alpha1_dxp1;
  MatX d_alpha2_dxp1;
  MatX d_alpha1_dxp2;
  MatX d_alpha2_dxp2;
  MatX dH_dxp1_xp2_p_FD;
  VecX xdp2_p_FD;
  VecX GAMMA;
  Eigen::Tensor<double, 3> ddH_ddxp1_xdp2_p_FD;

  Eigen::Tensor<double, 3> d_GAMMA_block;
  Eigen::Tensor<double, 3> d_GAMMA_block_temp;
  MatX d_GAMMA_dxp1;
  MatX neg_dFD_dq;
  MatX neg_dFD_dv;
  VecX H_xdp2_p_FD;
  MatX dGAMMA_dxp1;
  MatX dGAMMA_dxp2;
  MatX dGAMMA_dxdp2;
  VecX xdp1_m_xp2;
  MatX two_H_dC_dxp2;

  Eigen::Tensor<double, 3> dH_dxp1_dFD_xp2_temp;
  Eigen::Tensor<double, 3> dH_dxp1_dFD_xp2;

  MatX dOmega_dxp1;
  MatX dOmega_dxp2;
  MatX dOmega_dxdp1;
  MatX dOmega_dxdp2;
  MatX dOmega_dxddp1;
  MatX dOmega_dxddp2;

  MatX dOmega_dx_temp;
  MatX dOmega_dx;
  MatX dOmega_dxd;
  MatX dOmega_dxdd;

  MatX dgamma_dxdp1;
  MatX dgamma_dxdp2;
  Eigen::Tensor<double, 3> dH_dxp1_I_temp;
  Eigen::Tensor<double, 3> dH_dxp1_I;

  MatX eye_modelnv;
  MatX eye_num_ps;

  MatX D_ps;   // small part of the differentiation matrix
  MatX D2_ps;  // small part of the differentiation matrix squared

  Eigen::RowVectorXd kron_ones;

  MatX AGHF_jac_x;
  MatX AGHF_jac_xd;
  MatX AGHF_jac_xdd;
  MatX temp_mat;

  MatX jac_aps;
  MatX jac_daps;
  MatX jac_ddaps;

  /**
   * @brief Constructor for double integrator system.
   * @param[in] k_val Penalty parameter.
   * @param[in] N_val Number of bodies in the system
   */
  AghfPybindWrapper(double k_val,
                    int N_val);  // constructor for double integrator system

  // Constructor for the AGHF where the lagrangian has been augmented with
  // inequality constraints that utilize an activation function \sum_i k_cons *
  // g_i(x)^2 * S (g_i(x))

  /**
   * @brief Constructor for AGHF with augmented Lagrangian using inequality
   * constraints.
   * @param[in] urdf_filename Path to the URDF file.
   * @param[in] k_val Penalty parameter.
   * @param[in] j_type_val Joint type values.
   */
  AghfPybindWrapper(const std::string& urdf_filename,
                    double k_val,
                    const Eigen::Ref<VecX> j_type_val);

  /**
   * @brief Constructor with URDF model, tuning parameter, and matrices.
   * @param[in] urdf_filename Path to the URDF file.
   * @param[in] k_val  Penalty parameter.
   * @param[in] j_type_val Joint type values.
   * @param[in] D_ps Pseudospectral First derivative matrix.
   * @param[in] D2_ps Pseudospectral Second derivative matrix.
   */
  AghfPybindWrapper(const std::string& urdf_filename,
                    double k_val,
                    const Eigen::Ref<VecX> j_type_val,
                    Eigen::Ref<MatX> D_ps,
                    Eigen::Ref<MatX> D2_ps);

  /**
   * @brief Multiplies a 3D tensor with a 2D matrix.
   * @details This function does 3D tensor - Matrix mulitplication in the
   * following way: Given tensor3D of dims (I,J,K)  and mat2D of dims (J,L) it
   * multiplies them based on tensor_mult_dim and matrix_mult_dim. e.g if
   * tensor_mult_dim = 1 and matrix_mult_dim = 0 then the multiplication of
   * tensor3D x mat2D will yield an Outputtensor3D of dims (I,L,K) where each
   * individual matrix multiplication would be of the form
   * tensor3D(:,:,idx)*mat2D which is (I,J) * (J,L) multiplication
   * @param[in] mat2D 2D matrix input.
   * @param[in] tensor3D 3D tensor input.
   * @param[in] tensor_mult_dim Tensor multiplication dimension.
   * @param[in] matrix_mult_dim Matrix multiplication dimension.
   * @return Resulting 3D tensor.
   */
  Eigen::Tensor<double, 3> TensorMatrixMult_3Dwith2D(
      Eigen::Ref<MatX> mat2D,
      Eigen::Tensor<double, 3>& tensor3D,
      int tensor_mult_dim,
      int matrix_mult_dim);

  /**
   * @brief Multiplies a 2D matrix with a 3D tensor.
   * @details This function does Matrix - 3D tensor mulitplication in the
   * following way: Given mat2D of dims (I,J)  and tensor3D of dims (J,K,L) it
   * multiplies them based on tensor_mult_dim and matrix_mult_dim. e.g if
   * tensor_mult_dim = 0 and  matrix_mult_dim = 1 then the multiplication of
   * mat2D x tensor3D will yield an output tensor3D of dims (I,K,L) where each
   * individual matrix multiplication would be of the form mat2D *
   * tensor3D(:,:,idx) which is (I,J) * (J,K) multiplication
   * @param[in] mat2D 2D matrix input.
   * @param[in] tensor3D 3D tensor input.
   * @param[in] tensor_mult_dim Tensor multiplication dimension.
   * @param[in] matrix_mult_dim Matrix multiplication dimension.
   * @return Resulting 3D tensor.
   */
  Eigen::Tensor<double, 3> TensorMatrixMult_2Dwith3D(
      Eigen::Ref<MatX> mat2D,
      Eigen::Tensor<double, 3>& tensor3D,
      int tensor_mult_dim,
      int matrix_mult_dim);

  /**
   * @brief Multiplies a 3D tensor with a 1D vector.
   * @details This function does 3D tensor - Vector mulitplication and returns a
   * matrix in the following way: Given tensor3D of dims (I,J,K)  and vec1D of
   * dims (J) it multiplies them based on tensor_mult_dim and the only dimension
   * in the vector. e.g if tensor_mult_dim = 1 then the multiplication of
   * tensor3D x vec1D will yield an OutputTensor2D of dims (I,K) where each
   * individual matrix-vector multiplication would be of the form
   * tensor3D(:,:,idx)*vec1D which is (I,J) * (J) multiplication
   * @param[in] vec1D 1D vector input.
   * @param[in] tensor3D 3D tensor input.
   * @param[in] tensor_mult_dim Tensor multiplication dimension.
   * @return Resulting 2D matrix.
   */
  Eigen::MatrixXd TensorVectorMult_3Dwith1D(Eigen::Ref<VecX> vec1D,
                                            Eigen::Tensor<double, 3>& tensor3D,
                                            int tensor_mult_dim);

  /**
   * @brief Sets the obstacle avoidance constraints for the optimization
   * problem.
   * @param[in] obstaclesInfo_inp Obstacle data.
   * @param[in] c_cons_inp Activation function sharpness parameter.
   * @param[in] k_cons_inp Constraint Penalty parameter.
   */
  void set_activated_obstacles(const py::array_t<double> obstaclesInfo_inp,
                               const double c_cons_inp,
                               const double k_cons_inp);

  /**
   * @brief Sets the state limit constraints for the optimization.
   * @param[in] stateLimitsLower_inp Lower state limits.
   * @param[in] stateLimitsUpper_inp Upper state limits.
   * @param[in] c_cons_inp Activation function sharpness parameter.
   * @param[in] k_cons_inp Constraint Penalty parameter.
   */
  void set_activated_state_limits(const Eigen::Ref<VecX> stateLimitsLower_inp,
                                  const Eigen::Ref<VecX> stateLimitsUpper_inp,
                                  const double c_cons_inp,
                                  const double k_cons_inp);
  /**
   * @brief Computes and updates the values of the mass matrix H and its
   * derivative H_dot using the chain rule in the Composite
   * Rigid Body Algorithm (CRBA).
   * @param[in] xp1 First N states (joint angles).
   * @param[in] xp2 Second N states (joint veclocities).
   * @note updates the variables H and H_dot
   */
  void CRBA_D(const Eigen::Ref<VecX> xp1, const Eigen::Ref<VecX> xp2);

  /**
   * @brief Computes partial derivatives of H using chain rule in the Composite
   * Rigid Body Algorithm (CRBA).
   * @param[in] xp1 First N states (joint angles).
   * @param[in] xp2 Second N states (joint veclocities).
   * @param[out] H Mass matrix.
   * @param[out] H_dot Time derivative of Mass Matrix.
   * @param[out] dH_dq Tensor of first partial derivatives of the Mass Matrix.
   * @param[out] ddH_ddq Tensor of second partial derivatives of the Mass
   * Matrix.
   */
  void CRBA_2D(const Eigen::Ref<VecX> xp1,
               const Eigen::Ref<VecX> xp2,
               Eigen::Ref<MatX> H,
               Eigen::Ref<MatX> H_dot,
               Eigen::Tensor<double, 3>& dH_dq,
               Eigen::Tensor<double, 4>& ddH_ddq);

  /**
   * @brief Computes second derivatives of the Forward Dynamics using ABA based
   * on https://arxiv.org/abs/2302.06001
   * @param[in] dFD_dq Partial derivative of Forward Dynamics wrt joint angles.
   * @param[in] dFD_dv Partial derivative of Forward Dynamics wrt joint
   * velocities.
   * @param[in] H_inv Inverse of Mass matrix.
   * @param[in] ddID_ddq Tensor of second partial derivates of Inverse Dynamics
   * wrt joint angles.
   * @param[in] ddID_ddv Tensor of second partial derivates of Inverse Dynamics
   * wrt joint velocities.
   * @param[in] ddID_dqdv Tensor of Mixed partial derivatives of Inverse
   * Dynamics wrt joint angles and joint velocities.
   * @param[in] ddID_dadq Tensor of Mixed partial derivatives of Inverse
   * Dynamics wrt joint acceleration and joint angles.
   * @param[out] ddFD_ddq Tensor of second partial derivatives of Forward
   * Dynamics wrt joint angles.
   * @param[out] ddFD_ddv Tensor of second partials derivatives of Forward
   * Dynamics wrt joint velocities.
   * @param[out] ddFD_dqdv Mixed partial derivatives of Forward Dynamics wrt
   * joint angles and joint velocities.
   * @param[out] ddFD_dudq Mixed partial derivatives of Forward Dynamics wrt
   * joint torques and joint angles.
   */
  void ABA_2D(Eigen::Ref<MatX> dFD_dq,
              Eigen::Ref<MatX> dFD_dv,
              Eigen::Ref<MatX> H_inv,
              Eigen::Tensor<double, 3>& ddID_ddq,
              Eigen::Tensor<double, 3>& ddID_ddv,
              Eigen::Tensor<double, 3>& ddID_dqdv,
              Eigen::Tensor<double, 3>& ddID_dadq,
              Eigen::Tensor<double, 3>& ddFD_ddq,
              Eigen::Tensor<double, 3>& ddFD_ddv,
              Eigen::Tensor<double, 3>& ddFD_dqdv,
              Eigen::Tensor<double, 3>& ddFD_dudq);

  /**
   * @brief Computes HTH terms and their derivatives.
   * @param[in] H Mass matrix.
   * @param[in] H_inv Inverse of the mass matrix.
   * @param[out] dHinv_dq Tensor of partial derivatives of the inverse mass
   * matrix w.r.t joint angles.
   * @param[out] HTH Product of H^T and H.
   * @param[out] HinvTHinv Product of H_inv^T and H_inv.
   * @param[out] HinvTHinv_dxp1 Tensor of partial derivatives of HinvTHinv w.r.t
   * the first N states.
   */
  void get_HTH_terms(Eigen::Ref<MatX> H,
                     Eigen::Ref<MatX> H_inv,
                     Eigen::Tensor<double, 3>& dHinv_dq,
                     Eigen::Ref<MatX> HTH,
                     Eigen::Ref<MatX> HinvTHinv,
                     Eigen::Tensor<double, 3>& HinvTHinv_dxp1);

  /**
   * @brief Computes partial derivative of the time derivative
   * of the Mass Matrix wrt joint positions and also returns product of ddH_ddq
   * * (xp2 - FD_0)
   * @param[in] xdp1 Vector of the derivative of the first N states.
   * @param[in] ddH_ddq Tensor of second partial derivatives of the Mass Matrix.
   * @param[in] xdp2_p_FD Vector of the derivative of the second N states minus
   * the acceleration FD_0
   * @param[in] contract_dims An array containing a pair of indices defining
   * which dimensions to do the tensor matrix multiplication along
   * @param[out] dHdot_dq Tensor of partial derivatives of the time derivative
   * of the Mass Matrix w.r.t the first N states.
   * @param[out] ddH_ddxp1_xdp2_p_FD Tensor constaining the 4D_tensor-matrix
   * multiplication: ddH_ddq * (xp2 - FD_0).
   */

  void get_Hdot_D(Eigen::Ref<VecX> xdp1,
                  Eigen::Tensor<double, 4>& ddH_ddq,
                  Eigen::Ref<VecX> xdp2_p_FD,
                  Eigen::array<Eigen::IndexPair<int>, 1>& contract_dims,
                  Eigen::Tensor<double, 3>& dHdot_dq,
                  Eigen::Tensor<double, 3>& ddH_ddxp1_xdp2_p_FD);

  /**
   * @brief Computes partial derivatives of the time derivative
   * of the grouped Coriolis and gravity term (C)
   * @param[in] xdp1 Vector of the derivative of the first N states.
   * @param[in] xdp2 Vector of the derivative of the second N states.
   * @param[in] ddC_ddq Tensor of second partial derivatives of the grouped
   * Coriolis and gravity term (C) wrt joint angles.
   * @param[in] ddC_ddv Tensor of second partial derivatives of the grouped
   * Coriolis and gravity term (C) wrt joint velocities.
   * @param[in] ddC_dqdv  Tensor of partial derivatives of the grouped
   * Coriolis and gravity term (C) wrt joint angles and joint
   * velocities.
   * @param[in] ddC_dvdq  Tensor of partial derivatives of the grouped
   * Coriolis and gravity term (C) wrt joint velocities and joint
   * angles.
   * @param[out] dCdot_dxp1 Partial derivative of the time derivative
   * of the grouped Coriolis and gravity term (C) w.r.t the first N states.
   * @param[out] dCdot_dxp2 Partial derivative of the time derivative
   * of the grouped Coriolis and gravity term (C) w.r.t the second N states.
   */

  void get_Cdot_D(Eigen::Ref<VecX> xdp1,
                  Eigen::Ref<VecX> xdp2,
                  Eigen::Tensor<double, 3>& ddC_ddq,
                  Eigen::Tensor<double, 3>& ddC_ddv,
                  Eigen::Tensor<double, 3>& ddC_dqdv,
                  Eigen::Tensor<double, 3>& ddC_dvdq,
                  Eigen::Ref<MatX> dCdot_dxp1,
                  Eigen::Ref<MatX> dCdot_dxp2);

  /**
   * @brief Computes the right-hand side (RHS) of the AGHF at num_samples
   * trajectory points
   * @param[in] x 2N vector of states.
   * @param[in] xd First derivatives of the states.
   * @param[in] xdd Second derivatives of the states.
   * @param[in] num_samples Number of trajectory points to evaluate the  at.
   * @param[out] AGHF_RHS Matrix storing the RHS of the AGHF, with dimensions
   * (num_samples, 2 * N).
   */
  void compute_AGHF_RHS(const Eigen::Ref<MatX> x,
                        const Eigen::Ref<MatX> xd,
                        const Eigen::Ref<MatX> xdd,
                        const int num_samples,
                        Eigen::Ref<MatX> AGHF_RHS);

  /**
   * @brief Computes the Jacobian of the right-hand side (RHS) of the AGHF at
   * all the pseudospectral nodes to pass to the ODE solver
   * @param[in] x 2N vector of states.
   * @param[in] xd First derivatives of the states.
   * @param[in] xdd Second derivatives of the states.
   * @param[in] num_samples Number of trajectory points to evaluate the AGHF RHS
   * @param[out] AGHF_jac Matrix storing the Jacobian of RHS of the AGHF at all
   * the pseudospectral nodes to be passed to the ODE solver, with dimensions
   * (2*model.nv*num_ps_nodes, 2*model.nv*num_ps_nodes).
   */
  void compute_PSAGHF_jac(const Eigen::Ref<MatX> x,
                          const Eigen::Ref<MatX> xd,
                          const Eigen::Ref<MatX> xdd,
                          const int num_samples,
                          Eigen::Ref<MatX> AGHF_jac);

  /**
   * @brief Computes the right-hand side (RHS) of the AGHF at num_samples
   * trajectory points for the double integrator system
   * @param[in] x  Vector of states.
   * @param[in] xd First derivatives of the states.
   * @param[in] xdd Second derivatives of the states.
   * @param[in] num_samples Number of trajectory points to evaluate the  at.
   * @param[out] AGHF_RHS Matrix storing the RHS of the AGHF, with dimensions
   * (num_samples, 2 * N).
   */
  void compute_AGHF_RHS_doubleint(const Eigen::Ref<MatX> x,
                                  const Eigen::Ref<MatX> xd,
                                  const Eigen::Ref<MatX> xdd,
                                  const int num_samples,
                                  Eigen::Ref<MatX> AGHF_RHS);

  /**
   * @brief Computes the right-hand side (RHS) of the AGHF at num_samples
   * trajectory points for the double integrator system with constraints on the
   * velocity
   * @param[in] x  Vector of states.
   * @param[in] xd First derivatives of the states.
   * @param[in] xdd Second derivatives of the states.
   * @param[in] num_samples Number of trajectory points to evaluate the  at.
   * @param[out] AGHF_RHS Matrix storing the RHS of the AGHF, with dimensions
   * (num_samples, 2 * N).
   */
  void compute_AGHF_RHS_doubleint_vel_cons(const Eigen::Ref<MatX> x,
                                           const Eigen::Ref<MatX> xd,
                                           const Eigen::Ref<MatX> xdd,
                                           const int num_samples,
                                           Eigen::Ref<MatX> AGHF_RHS);
};

#endif  // PYBIND_WRAPPER_HPP