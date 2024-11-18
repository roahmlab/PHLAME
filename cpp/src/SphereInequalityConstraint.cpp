#ifndef SPHERE_INEQUALITY_CONSTRAINT_CPP
#define SPHERE_INEQUALITY_CONSTRAINT_CPP

#include "SphereInequalityConstraint.h"

#include <Eigen/Dense>
#include <iostream>

SphereInequalityConstraint::SphereInequalityConstraint(const Model& model_inp)
    : model(model_inp)
{
}

Eigen::MatrixXd SphereInequalityConstraint::TensorVectorMult_3Dwith1D(
    Eigen::Ref<VecX> vec1D,
    Eigen::Tensor<double, 3>& tensor3D,
    int tensor_mult_dim)
{
  /* This function does 3D tensor - Vector mulitplication and returns a matrix
     in the following way: Given tensor3D of dims (I,J,K)  and vec1D of dims (J)
     it multiplies them based on tensor_mult_dim and the only dimension in the
     vector. e.g if tensor_mult_dim = 1 then the multiplication of tensor3D x
     vec1D will yield an OutputTensor2D of dims (I,K) where each individual
     matrix-vector multiplication would be of the form tensor3D(:,:,idx)*vec1D
     which is (I,J) * (J) multiplication*/

  if (tensor_mult_dim > 2)
  {
    throw std::invalid_argument(
        "Invalid tensor multiplication indices. The input mutliplication "
        "dimsension exceeds the size of the tensor ");
  }

  // convert vec1D to a 1D tensor for contraction
  Eigen::TensorMap<Eigen::Tensor<double, 1>> vec1D_tensor(
      vec1D.data(), vec1D.size());

  Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(
      tensor_mult_dim, 0)};  // Define the contraction mapping.
  Eigen::Tensor<double, 2> OutputTensor2D_temp = tensor3D.contract(
      vec1D_tensor,
      contract_dims);  // for Tenstensor_mult_dim = 1, tensor3D (I,J,K) x mat2D
                       // (J) = Outputtensor3D (I,K)
  Eigen::Map<Eigen::MatrixXd> OutputTensor2D(
      OutputTensor2D_temp.data(), model.nv,
      model.nv);  // convert 2D tensor to a matrix
  return OutputTensor2D;
}

double SphereInequalityConstraint::compute_value(double R,
                                                 const VecX& FK,
                                                 const VecX& obs_center)
{
  double ineq_val;
  VecX diff = FK - obs_center;
  ineq_val = pow(R, 2) - pow(diff.norm(), 2);
  return ineq_val;
}

Eigen::VectorXd SphereInequalityConstraint::compute_derivative(
    const VecX& FK, const VecX& obs_center, const MatX& frame_position_jacobian)
{
  Vec3 distance = FK - obs_center;

  VecX result = VecX::Zero(2 * model.nv);

  result.head(model.nv) = -2.0 * frame_position_jacobian.transpose() * distance;

  return result;
}

Eigen::MatrixXd SphereInequalityConstraint::compute_hessian(
    const VecX& FK,
    const VecX& obs_center,
    const MatX& frame_position_jacobian,
    Eigen::Tensor<double, 3> frame_position_hessian)
{
  Eigen::Tensor<double, 3> Outputtensor3D_temp = frame_position_hessian;
  Eigen::array<int, 3> shuffle_dims = {
      2, 0, 1};  // Define the permutation dimensions
  frame_position_hessian = Outputtensor3D_temp.shuffle(
      shuffle_dims);  // permute dimensions to get the tensor arranged in the
                      // right order (I,L,K)

  Vec3 distance = FK - obs_center;
  MatX d2FK_dx2_m_FK_m_c =
      TensorVectorMult_3Dwith1D(distance, frame_position_hessian, 1);

  MatX result = MatX::Zero(2 * model.nv, 2 * model.nv);

  result.block(0, 0, model.nv, model.nv) =
      -2.0 * (d2FK_dx2_m_FK_m_c +
              frame_position_jacobian.transpose() * frame_position_jacobian);

  return result;
}

#endif  // SPHERE_INEQUALITY_CONSTRAINT_CPP