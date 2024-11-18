
// Adapted from https://arxiv.org/abs/2302.06001
// Utilities for Eigen::Tensor slicing and accessing matrices, vectors
#ifndef __pinocchio_utils_tensor_utils_cpp__
#define __pinocchio_utils_tensor_utils_cpp__
#include "tensor_utils.hpp"

namespace pinocchio
{

// to get a matrix of a tensor with 3rd dim
void tensor_to_mat(Eigen::Tensor<double, 3>& hess,
                   Eigen::MatrixXd& mat,
                   int dim,
                   int r)
{
  for (int ii = 0; ii < dim; ii++)
  {
    for (int jj = 0; jj < dim; jj++)
    {
      mat(ii, jj) = hess(ii, jj, r);
    }
  }
}

void error_fun_SO(const Eigen::Tensor<double, 3>& ten_in,
                  const Eigen::Tensor<double, 3>& ten_ref,
                  Eigen::VectorXd& error_vec,
                  const int n1,
                  const int n2,
                  const int n3)
{
  Eigen::Tensor<double, 3> temp_diff(n1, n2, n3);
  temp_diff = ten_in - ten_ref;
  // Max Abs error
  Eigen::Tensor<double, 0> b = (temp_diff.abs()).maximum();
  error_vec[0] = b(0);
  double sum1 = 0.0;

  for (int k = 0; k < n3; k++)
  {
    for (int j = 0; j < n2; j++)
    {
      for (int i = 0; i < n1; i++)
      {
        sum1 += temp_diff(i, j, k) * temp_diff(i, j, k);
      }
    }
  }
  // RMS abs error
  error_vec[1] = sqrt(sum1 / (n1 * n2 * n3));

  // Relative error
  temp_diff.setZero();
  double sum2 = 0.0;

  for (int k = 0; k < n3; k++)
  {
    for (int j = 0; j < n2; j++)
    {
      for (int i = 0; i < n1; i++)
      {
        if (ten_in(i, j, k) == 0.0)
        {
          temp_diff(i, j, k) = 0.0;
        }
        else
        {
          temp_diff(i, j, k) = (ten_in(i, j, k) - ten_ref(i, j, k)) /
                               max(abs(ten_in(i, j, k)), 1.0);
        }
        sum2 += temp_diff(i, j, k) * temp_diff(i, j, k);
      }
    }
  }
  Eigen::Tensor<double, 0> c = (temp_diff.abs()).maximum();
  error_vec[2] = c(0);                         // max relative error
  error_vec[3] = sqrt(sum2 / (n1 * n2 * n3));  // rms relative error
}

double get_tens_diff_norm(Eigen::Tensor<double, 3>& ten1,
                          Eigen::Tensor<double, 3>& ten2,
                          int n)
{
  double tmp1 = 0.0;

  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      for (int k = 0; k < n; k++)
      {
        tmp1 = tmp1 + std::pow(ten1(k, j, i) - ten2(k, j, i), 2);
      }
    }
  }
  return tmp1;
}

double get_tens_diff_norm_v1(Eigen::Tensor<double, 3>& ten1,
                             Eigen::Tensor<double, 3>& ten2,
                             int n1,
                             int n2,
                             int n3)
{
  double tmp1 = 0.0;

  for (int i = 0; i < n1; i++)
  {
    for (int j = 0; j < n2; j++)
    {
      for (int k = 0; k < n3; k++)
      {
        tmp1 = tmp1 + std::pow(ten1(k, j, i) - ten2(k, j, i), 2);
      }
    }
  }
  return tmp1;
}

// Original
// VERY EXPENSIVE
// This method rotates a tensor along the 3rd dim, so that it's transposed along
// the 3rd dim dim of tens, rot_tens : nxnxn dim of mat : nxn
void tens_rot(const Eigen::Tensor<double, 3>& tens,
              Eigen::Tensor<double, 3>& rot_tens,
              int n)
{
  Eigen::MatrixXd temp_mat(n, n);
  for (int ii = 0; ii < n; ii++)
  {
    get_mat_from_tens1(tens, temp_mat, n, ii);
    hess_assign_fd(rot_tens, temp_mat, n, ii);
  }
}

// With col major- cuts down cost  in half
// VERY EXPENSIVE
// This method rotates a tensor along the 3rd dim, so that it's transposed along
// the 3rd dim dim of tens, rot_tens : nxnxn dim of mat : nxn
void tens_rot_v1(const Eigen::Tensor<double, 3>& tens,
                 Eigen::Tensor<double, 3>& rot_tens,
                 int n)
{
  Eigen::MatrixXd temp_mat(n, n);
  for (int ii = 0; ii < n; ii++)
  {
    get_mat_from_tens1_v1(tens, temp_mat, n, ii);
    hess_assign_fd_v1(rot_tens, temp_mat, n, ii);
  }
}

// function used to take the product of a matrix M, tensor T (m*T*m), along
// first and second dims of T
void mat_ten_mat(Eigen::Tensor<double, 3>& ten_in,
                 Eigen::Tensor<double, 3>& ten_out,
                 Eigen::MatrixXd& mat,
                 int n)
{
  Eigen::MatrixXd temp1(n, n);
  Eigen::MatrixXd temp2(n, n);

  for (int ii = 0; ii < n; ii++)
  {
    get_mat_from_tens3(ten_in, temp1, n, ii);
    temp2 = mat * temp1 * mat;
    hess_assign_fd(ten_out, temp2, n, ii);
  }
}

}  // namespace pinocchio

#endif  //__pinocchio_utils_tensor_utils_cpp__
