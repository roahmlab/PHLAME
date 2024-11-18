#ifndef PYBIND_WRAPPER_CPP
#define PYBIND_WRAPPER_CPP

#include "PybindWrapper.h"

AghfPybindWrapper::AghfPybindWrapper(double k_val, int N_val)
{
  k = k_val;
  N = N_val;
}

AghfPybindWrapper::AghfPybindWrapper(const std::string &urdf_filename,
                                     double k_val,
                                     const Eigen::Ref<VecX> j_type_val)
{
  // Initialization
  k = k_val;
  j_type = j_type_val;

  pino::urdf::buildModel(urdf_filename, model);

  data = pino::Data(model);

  // define gravity explicitly
  model.gravity.linear()(2) = -9.81;

  // Define a model to compute H_dot
  Xtree.resize(model.nv);
  I.resize(model.nv);
  Xup.resize(model.nv);
  dXupdt.resize(model.nv);
  S.resize(model.nv);
  IC.resize(model.nv);
  dICdt.resize(model.nv);
  for (size_t i = 0; i < model.nv; i++)
  {
    int pid = i + 1;
    Xtree(i).setZero();
    Xtree(i).topLeftCorner<3, 3>() =
        model.jointPlacements[pid].rotation().transpose();
    Xtree(i).bottomRightCorner<3, 3>() =
        model.jointPlacements[pid].rotation().transpose();
    Xtree(i).bottomLeftCorner<3, 3>() =
        -model.jointPlacements[pid].rotation().transpose() *
        skew(model.jointPlacements[pid].translation());

    I(i).setZero();
    Eigen::Matrix3d Cbar = skew(model.inertias[pid].lever());
    I(i).topLeftCorner<3, 3>() =
        model.inertias[pid].inertia().matrix() +
        model.inertias[pid].mass() * Cbar * Cbar.transpose();
    I(i).topRightCorner<3, 3>() = model.inertias[pid].mass() * Cbar;
    I(i).bottomLeftCorner<3, 3>() =
        model.inertias[pid].mass() * Cbar.transpose();
    I(i).bottomRightCorner<3, 3>() =
        model.inertias[pid].mass() * Eigen::Matrix3d::Identity();
  }

  // Preallocate memory for the intermediate variables
  zero_vec = VecX::Zero(model.nv);

  // Allocate joint configuration as well as joint velocity and torque
  xp1 = VecX::Zero(model.nv);
  xp2 = VecX::Zero(model.nv);
  xdp1 = VecX::Zero(model.nv);
  xdp2 = VecX::Zero(model.nv);
  xddp1 = VecX::Zero(model.nv);
  xddp2 = VecX::Zero(model.nv);

  k_diag = MatX::Identity(model.nv, model.nv) * k;
  k_inv_diag = MatX::Identity(model.nv, model.nv) / k;

  FD_0 = VecX::Zero(model.nv);
  dFD_dq = MatX::Zero(model.nv, model.nv);
  dFD_dv = MatX::Zero(model.nv, model.nv);

  C = VecX::Zero(model.nv);
  C_dot = VecX::Zero(model.nv);

  dC_dq = MatX::Zero(model.nv, model.nv);
  dC_dv = MatX::Zero(model.nv, model.nv);
  dC_da = MatX::Zero(model.nv, model.nv);

  dID_dq = MatX::Zero(model.nv, model.nv);
  dID_dv = MatX::Zero(model.nv, model.nv);
  dID_da = MatX::Zero(model.nv, model.nv);

  G = MatX::Zero(2 * model.nv, 2 * model.nv);
  G.topLeftCorner(model.nv, model.nv) = k_diag;
  G_inv = MatX::Zero(2 * model.nv, 2 * model.nv);
  G_inv.topLeftCorner(model.nv, model.nv) = k_inv_diag;
  G_inv_dL_dx = VecX::Zero(2 * model.nv);
  G_inv_dL_dxd = VecX::Zero(2 * model.nv);
  G_inv_d_dt_dL_dxd = VecX::Zero(2 * model.nv);
  G_inv_dLprime_dx = VecX::Zero(2 * model.nv);
  G_inv_d_dt_dLprime_dxd = VecX::Zero(2 * model.nv);

  // Omegas
  Omega_1 = VecX::Zero(2 * model.nv);
  Omega_2 = VecX::Zero(2 * model.nv);
  Omega_3 = VecX::Zero(2 * model.nv);
  Omega_4 = VecX::Zero(2 * model.nv);

  // penalty stuff
  activated_penalty_vec.clear();
  penalty_value = 1;
  d_penalty_dx = VecX::Zero(2 * model.nv);

  // Initialize Mass matrix related terms
  H = MatX::Zero(model.nv, model.nv);
  H_dot = MatX::Zero(model.nv, model.nv);
  H_inv = MatX::Zero(model.nv, model.nv);
  HinvTHinv = MatX::Zero(model.nv, model.nv);
  HTH = MatX::Zero(model.nv, model.nv);
}

AghfPybindWrapper::AghfPybindWrapper(const std::string &urdf_filename,
                                     double k_val,
                                     const Eigen::Ref<VecX> j_type_val,
                                     Eigen::Ref<MatX> D_ps,
                                     Eigen::Ref<MatX> D2_ps)
    : AghfPybindWrapper(urdf_filename, k_val, j_type_val)
{
  num_ps_nodes = D_ps.rows();

  // Add other model properties to compute dH/dq and ddH/ddq
  dXupdq.resize(model.nv);
  ddXupddq.resize(model.nv);

  // penalty stuff
  dd_penalty_ddx = MatX::Zero(2 * model.nv, 2 * model.nv);
  penalty_deriv_and_hessian = std::make_pair(zero_vec, zero_mat);

  zero_mat = Eigen::MatrixXd::Zero(model.nv, model.nv);

  // Mass Matrix Derivatives
  dH_dq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddH_ddq = Eigen::Tensor<double, 4>(model.nv, model.nv, model.nv, model.nv);
  dHinv_dq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);

  // Grouped Coriolis-Gravity term Second Derivatives
  ddC_ddq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddC_ddv = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddC_dqdv = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddC_dvdq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  du_dadq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);

  // Forward Dynamics Second Derivatives
  ddFD_ddq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddFD_ddv = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddFD_dqdv = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddFD_dudq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddFD_dvdq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);

  // Inverse Dynamics Second Derivatives
  ddID_ddq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddID_ddv = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddID_dqdv = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  ddID_dadq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);

  // Mass matrix inverse derivatives and G inverse derivatives
  HinvTHinv_dxp1 = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  dGinv_dx = Eigen::Tensor<double, 3>(2 * model.nv, 2 * model.nv, 2 * model.nv);
  dGinv_dx.setZero();

  // First derivatives of the Time derivative Grouped Coriolis-Gravity term
  dCdot_dxp1 = MatX::Zero(model.nv, model.nv);
  dCdot_dxp2 = MatX::Zero(model.nv, model.nv);

  // Jacobian intermediate variables
  gamma = VecX::Zero(model.nv);
  dHdot_dq = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  dgamma_dxp1 = MatX::Zero(model.nv, model.nv);
  H_xdp2_p_C = VecX::Zero(model.nv);
  Hdot_xdp2_p_H_xddp2_p_Cdot = VecX::Zero(model.nv);
  H_2C_p_2H_xdp2 = VecX::Zero(model.nv);
  twoC_p_2H_xdp2 = VecX::Zero(model.nv);
  dH_dxp1_m_xdp2 = MatX::Zero(model.nv, model.nv);

  dOmega1_dxp1 =
      MatX::Zero(2 * model.nv, model.nv);  // d/dxp1 (Ginv*d/dt (dL/dxd) )
  dOmega1_dxp2 =
      MatX::Zero(2 * model.nv, model.nv);  // d/dxp2 (Ginv*d/dt (dL/dxd) )
  dOmega234_dxp1 =
      MatX::Zero(2 * model.nv, model.nv);  // d/dxp1 (Ginv * (dL/dx) )
  dOmega234_dxp2 =
      MatX::Zero(2 * model.nv, model.nv);  // d/dxp2 (Ginv * (dL/dx) )

  neg_dFD_dq_T = MatX::Zero(model.nv, model.nv);
  neg_dFD_dv_T = MatX::Zero(model.nv, model.nv);

  neg_ddFD_ddq_T = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  neg_ddFD_dqdv_T = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  neg_ddFD_ddv_T = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  neg_ddFD_dvdq_T = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);

  // alpha terms
  alpha1 = VecX::Zero(model.nv);
  alpha2 = VecX::Zero(model.nv);
  dalpha_b_block =
      MatX::Zero(model.nv, model.nv);  // dH_dq(:,:,:) * twoC_p_2H_xdp2;

  // GAMMA terms
  xdp2_p_FD = VecX::Zero(model.nv);
  dH_dxp1_xp2_p_FD = MatX::Zero(model.nv, model.nv);
  GAMMA = VecX::Zero(model.nv);
  ddH_ddxp1_xdp2_p_FD = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);

  d_GAMMA_block_temp = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  d_GAMMA_block = Eigen::Tensor<double, 3>(
      model.nv, model.nv,
      model.nv);  // dH_dxp1 * d_FD_xp1 + ddH_ddxp1 * (xdp2 - FD_0)
  d_GAMMA_dxp1 = MatX::Zero(model.nv, model.nv);
  neg_dFD_dq = MatX::Zero(model.nv, model.nv);
  dGAMMA_dxp1 = MatX::Zero(model.nv, model.nv);
  dGAMMA_dxp2 = MatX::Zero(model.nv, model.nv);
  dGAMMA_dxdp2 = MatX::Zero(model.nv, model.nv);
  H_xdp2_p_FD = VecX::Zero(model.nv);

  xdp1_m_xp2 = VecX::Zero(model.nv);

  d_alpha1_dxp1 = MatX::Zero(model.nv, model.nv);
  d_alpha2_dxp1 = MatX::Zero(model.nv, model.nv);
  d_alpha1_dxp2 = MatX::Zero(model.nv, model.nv);
  d_alpha2_dxp2 = MatX::Zero(model.nv, model.nv);

  two_H_dC_dxp2 = VecX::Zero(model.nv);

  dOmega_dxp1 = MatX::Zero(2 * model.nv, model.nv);
  dOmega_dxp2 = MatX::Zero(2 * model.nv, model.nv);
  dOmega_dxdp1 = MatX::Zero(2 * model.nv, model.nv);
  dOmega_dxdp2 = MatX::Zero(2 * model.nv, model.nv);
  dOmega_dxddp1 = MatX::Zero(2 * model.nv, model.nv);
  dOmega_dxddp2 = MatX::Zero(2 * model.nv, model.nv);

  dOmega_dx_temp = MatX::Zero(2 * model.nv, 2 * model.nv);
  dOmega_dx = MatX::Zero(2 * model.nv, 2 * model.nv);
  dOmega_dxd = MatX::Zero(2 * model.nv, 2 * model.nv);
  dOmega_dxdd = MatX::Zero(2 * model.nv, 2 * model.nv);

  dgamma_dxdp1 = MatX::Zero(model.nv, model.nv);
  dgamma_dxdp2 = MatX::Zero(model.nv, model.nv);

  dH_dxp1_I_temp = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
  dH_dxp1_I = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);

  eye_modelnv = MatX::Identity(model.nv, model.nv);
  eye_num_ps = MatX::Identity(num_ps_nodes, num_ps_nodes);

  kron_ones = Eigen::RowVectorXd::Constant(num_ps_nodes, 1);

  AGHF_jac_x =
      MatX::Zero(2 * model.nv * num_ps_nodes, 2 * model.nv * num_ps_nodes);
  AGHF_jac_xd =
      MatX::Zero(2 * model.nv * num_ps_nodes, 2 * model.nv * num_ps_nodes);
  AGHF_jac_xdd =
      MatX::Zero(2 * model.nv * num_ps_nodes, 2 * model.nv * num_ps_nodes);

  jac_aps = eye_num_ps.replicate(2 * model.nv, 2 * model.nv);
  jac_daps = D_ps.replicate(2 * model.nv, 2 * model.nv);
  jac_ddaps = D2_ps.replicate(2 * model.nv, 2 * model.nv);
}

// Tensor Util functions
Eigen::Tensor<double, 3> AghfPybindWrapper::TensorMatrixMult_3Dwith2D(
    Eigen::Ref<MatX> mat2D,
    Eigen::Tensor<double, 3> &tensor3D,
    int tensor_mult_dim,
    int matrix_mult_dim)
{
  if (matrix_mult_dim > 1 || tensor_mult_dim > 2)
  {
    throw std::invalid_argument(
        "Invalid tensor multiplication indices. The input mutliplication "
        "dimsension exceeds the size of the tensor or matrix");
  }

  // convert mat2D to a 2D tensor for contraction
  Eigen::TensorMap<Eigen::Tensor<double, 2>> mat2D_tensor(
      mat2D.data(), mat2D.rows(), mat2D.cols());

  // Define the contraction mapping.
  Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
      Eigen::IndexPair<int>(tensor_mult_dim, matrix_mult_dim)};

  // for tensor_mult_dim = 1 and matrix_mult_dim = 0
  // tensor3D (I,J,K) x mat2D (J,L) = Outputtensor3D (I,K,L)
  Eigen::Tensor<double, 3> Outputtensor3D_temp =
      tensor3D.contract(mat2D_tensor, contract_dims);

  // Define the permutation dimensions
  Eigen::array<int, 3> shuffle_dims = {0, 2, 1};

  // permute dimensions to get the tensor arranged in the
  // right order (I,L,K)
  Eigen::Tensor<double, 3> Outputtensor3D =
      Outputtensor3D_temp.shuffle(shuffle_dims);

  return Outputtensor3D;
}

Eigen::Tensor<double, 3> AghfPybindWrapper::TensorMatrixMult_2Dwith3D(
    Eigen::Ref<MatX> mat2D,
    Eigen::Tensor<double, 3> &tensor3D,
    int tensor_mult_dim,
    int matrix_mult_dim)
{
  if (matrix_mult_dim > 1 || tensor_mult_dim > 2)
  {
    throw std::invalid_argument(
        "Invalid tensor multiplication indices. The input mutliplication "
        "dimsension exceeds the size of the tensor or matrix");
  }

  // convert mat2D to a 2D tensor for contraction
  Eigen::TensorMap<Eigen::Tensor<double, 2>> mat2D_tensor(
      mat2D.data(), mat2D.rows(), mat2D.cols());

  // Define the contraction mapping.
  Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
      Eigen::IndexPair<int>(matrix_mult_dim, tensor_mult_dim)};

  // for tensor_mult_dim = 0 and  matrix_mult_dim = 1,
  // mat2D (I,J) x tensor3D (J,K,L) = Outputtensor3D (I,K,L)
  Eigen::Tensor<double, 3> Outputtensor3D =
      mat2D_tensor.contract(tensor3D, contract_dims);

  return Outputtensor3D;
}

Eigen::MatrixXd AghfPybindWrapper::TensorVectorMult_3Dwith1D(
    Eigen::Ref<VecX> vec1D,
    Eigen::Tensor<double, 3> &tensor3D,
    int tensor_mult_dim)
{
  if (tensor_mult_dim > 2)
  {
    throw std::invalid_argument(
        "Invalid tensor multiplication indices. The input mutliplication "
        "dimsension exceeds the size of the tensor ");
  }

  // convert vec1D to a 1D tensor for contraction
  Eigen::TensorMap<Eigen::Tensor<double, 1>> vec1D_tensor(
      vec1D.data(), vec1D.size());

  // Define the contraction mapping.
  Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
      Eigen::IndexPair<int>(tensor_mult_dim, 0)};

  // for tensor_mult_dim = 1, tensor3D (I,J,K) x mat2D (J) = Outputtensor3D
  // (I,K)
  Eigen::Tensor<double, 2> OutputTensor2D_temp =
      tensor3D.contract(vec1D_tensor, contract_dims);

  // convert 2D tensor to a matrix
  Eigen::Map<Eigen::MatrixXd> OutputTensor2D(
      OutputTensor2D_temp.data(), tensor3D.dimension(0), tensor3D.dimension(1));

  return OutputTensor2D;
}

// AGHF functions
void AghfPybindWrapper::set_activated_obstacles(
    const py::array_t<double> obstaclesInfo_inp,
    const double c_cons_inp,
    const double k_cons_inp)
{
  py::buffer_info info = obstaclesInfo_inp.request();

  if (info.ndim != 2)
  {
    throw std::invalid_argument("obstaclesInfo has wrong row dimensions!");
  }

  if (info.shape[1] != 4)
  {
    throw std::invalid_argument("obstaclesInfo has wrong column dimensions!");
  }

  MatX obstaclesInfo =
      MatX::Map((double *)info.ptr, info.shape[0], info.shape[1]);

  activated_penalty_vec.push_back(
      std::make_unique<ActivatedSphereAvoidancePenalty>(
          model, data, k_cons_inp, c_cons_inp, obstaclesInfo));
}

void AghfPybindWrapper::set_activated_state_limits(
    const Eigen::Ref<VecX> stateLimitsLower_inp,
    const Eigen::Ref<VecX> stateLimitsUpper_inp,
    const double c_cons_inp,
    const double k_cons_inp)
{
  if (stateLimitsLower_inp.size() != model.nq + model.nv ||
      stateLimitsUpper_inp.size() != model.nq + model.nv)
  {
    throw std::invalid_argument(
        "lower state limits or upper state limits has wrong dimensions!");
  }

  if (c_cons_inp <= 0)
  {
    throw std::invalid_argument("c_cons_inp must be positive!");
  }

  if (k_cons_inp <= 0)
  {
    throw std::invalid_argument("k_cons_inp must be positive!");
  }

  // lower limit on the state
  activated_penalty_vec.push_back(std::make_unique<ActivatedStateLimitsPenalty>(
      stateLimitsLower_inp, c_cons_inp, k_cons_inp));

  // upper limit on the state
  activated_penalty_vec.push_back(std::make_unique<ActivatedStateLimitsPenalty>(
      stateLimitsUpper_inp, c_cons_inp, k_cons_inp));
}

void AghfPybindWrapper::CRBA_D(const Eigen::Ref<VecX> xp1,
                               const Eigen::Ref<VecX> xp2)
{
  Matrix6 XJ, dXJdq;
  for (size_t j = 0; j < model.nv; j++)
  {
    jcalc(XJ, dXJdq, S(j), j_type(j), xp1(j), xp2(j));
    Xup(j) = XJ * Xtree(j);
    dXupdt(j) = dXJdq * Xtree(j);
  }

  // Calculate composite inertia recursively
  IC = I;
  for (size_t j = 0; j < model.nv; j++)
  {
    dICdt(j).setZero();
  }

  for (int j = model.nv - 1; j >= 0; j--)
  {
    if (model.parents[j] >= 0 && model.parents[j] != j)
    {
      IC(model.parents[j]) += Xup(j).transpose() * IC(j) * Xup(j);
      dICdt(model.parents[j]) += Xup(j).transpose() * IC(j) * dXupdt(j) +
                                 Xup(j).transpose() * dICdt(j) * Xup(j) +
                                 dXupdt(j).transpose() * IC(j) * Xup(j);
    }
  }

  for (size_t j = 0; j < model.nv; j++)
  {
    Vector6 fh = IC(j) * S(j);
    Vector6 dfhdt = dICdt(j) * S(j);

    H(j, j) = S(j).transpose() * fh;
    H_dot(j, j) = S(j).transpose() * dfhdt;

    size_t k = j;
    while (model.parents[k] >= 0 && model.parents[k] != k)
    {
      dfhdt = dXupdt(k).transpose() * fh + Xup(k).transpose() * dfhdt;
      fh = Xup(k).transpose() * fh;
      k = model.parents[k];
      H(j, k) = S(j).transpose() * fh;
      H(k, j) = H(j, k);

      // Compute H_dot with time derivatives
      H_dot(j, k) = S(j).transpose() * dfhdt;
      H_dot(k, j) = H_dot(j, k);
    }
  }
}

void AghfPybindWrapper::CRBA_2D(const Eigen::Ref<VecX> xp1,
                                const Eigen::Ref<VecX> xp2,
                                Eigen::Ref<MatX> H,
                                Eigen::Ref<MatX> H_dot,
                                Eigen::Tensor<double, 3> &dH_dq,
                                Eigen::Tensor<double, 4> &ddH_ddq)
{
  Matrix6 XJ, dXJdq, ddXJddq;  // Derivatives of joint transforms
  for (size_t j = 0; j < model.nv; j++)
  {
    d_jcalc(XJ, dXJdq, ddXJddq, S(j), j_type(j), xp1(j));
    Xup(j) = XJ * Xtree(j);
    dXupdq(j) = dXJdq * Xtree(j);
    ddXupddq(j) = ddXJddq * Xtree(j);
  }

  // Set composite interia to be urdf inertia
  IC = I;

  // Resize the dICdq and ddICddq Arrays to have 'model.nv' elements
  dICdq.resize(model.nv);
  ddICddq.resize(model.nv);

  for (size_t j = 0; j < model.nv; j++)
  {
    dICdq(j) = Eigen::Tensor<double, 3>(6, 6, model.nv);
    ddICddq(j) = Eigen::Tensor<double, 4>(6, 6, model.nv, model.nv);
    dICdq(j).setZero();
    ddICddq(j).setZero();
  }

  // Initialize temporary matrices and tensors that will be overwritten during
  // recursion
  Eigen::Tensor<double, 2> dICdq_ii_jj_tens(6, 6);
  Eigen::MatrixXd dICdq_parent_ii_jj = Eigen::MatrixXd::Zero(6, 6);
  Eigen::Tensor<double, 2> ddICddq_ii_k1_k2_tens(6, 6);
  Eigen::MatrixXd ddICddq_parent_ii_k1_k2 = Eigen::MatrixXd::Zero(6, 6);

  // Calculate composite inertia and its derivatives recursively
  for (int ii = model.nv - 1; ii >= 0; ii--)
  {
    if (model.parents[ii] >= 0 && model.parents[ii] != ii)
    {
      IC(model.parents[ii]) += Xup(ii).transpose() * IC(ii) * Xup(ii);

      // Compute first derivative of composite inertia
      for (int jj = model.nv - 1; jj >= 0; jj--)
      {
        // convert necessary tensors to matrix for multiplication
        // line below equivalent to matlab dICdq(:,:,jj)
        dICdq_ii_jj_tens = dICdq(ii).chip<2>(jj);
        Eigen::Map<Eigen::MatrixXd> dICdq_ii_jj(dICdq_ii_jj_tens.data(), 6, 6);

        // Compute dICdq recursively
        if (ii == jj)
        {
          dICdq_parent_ii_jj = dXupdq(ii).transpose() * IC(ii) * Xup(ii) +
                               Xup(ii).transpose() * dICdq_ii_jj * Xup(ii) +
                               Xup(ii).transpose() * IC(ii) * dXupdq(ii);
        }
        else
        {
          dICdq_parent_ii_jj = Xup(ii).transpose() * dICdq_ii_jj * Xup(ii);
        }

        // Map matrix back to tensor for each links contribution to be added to
        // the derivative of composite inertia tensor
        Eigen::TensorMap<Eigen::Tensor<const double, 2>>
            dICdq_parent_ii_jj_tensor(dICdq_parent_ii_jj.data(), 6, 6);

        dICdq(model.parents[ii]).chip<2>(jj) += dICdq_parent_ii_jj_tensor;
      }

      /*#######################################################################*/
      // Compute second derivative of composite inertia
      for (int k1 = model.nv - 1; k1 >= 0; k1--)
      {
        for (int k2 = model.nv - 1; k2 >= k1; k2--)
        {
          // Convert indexed tensors to matrices for easy matrix multiplication
          // line below equivalent to matlab ddICddq(:,:,k1,k2)
          ddICddq_ii_k1_k2_tens = ddICddq(ii).chip<3>(k2).chip<2>(k1);
          Eigen::Map<Eigen::MatrixXd> ddICddq_ii_k1_k2(
              ddICddq_ii_k1_k2_tens.data(), 6, 6);

          // Compute ddICddq recursively
          if (k1 == k2)
          {
            if (k1 == ii)
            {
              dICdq_ii_jj_tens = dICdq(ii).chip<2>(k2);
              Eigen::Map<Eigen::MatrixXd> dICdq_ii_k2(
                  dICdq_ii_jj_tens.data(), 6, 6);

              ddICddq_parent_ii_k1_k2 =
                  ddXupddq(ii).transpose() * IC(ii) *
                      Xup(ii) +  // ddXupddq{i}'*IC{i}*Xup{i}
                  dXupdq(ii).transpose() * dICdq_ii_k2 *
                      Xup(ii) +  // dXupdq{i}'*dICdq{i}(:,:,k2)*Xup{i}
                  dXupdq(ii).transpose() * IC(ii) *
                      dXupdq(ii) +  // dXupdq{i}'*IC{i}*dXupdq{i}
                  dXupdq(ii).transpose() * dICdq_ii_k2 *
                      Xup(ii) +  // dXupdq{i}'*dICdq{i}(:,:,k1)*Xup{i} (since
                                 // k1=k2 just use k2 version)
                  Xup(ii).transpose() * ddICddq_ii_k1_k2 *
                      Xup(ii) +  // Xup{i}'*ddICddq{i}(:,:,k1,k2)*Xup{i}
                  Xup(ii).transpose() * dICdq_ii_k2 *
                      dXupdq(ii) +  // Xup{i}'*dICdq{i}(:,:,k1)*dXupdq{i} (since
                                    // k1=k2 just use k2 version)
                  dXupdq(ii).transpose() * IC(ii) *
                      dXupdq(ii) +  // dXupdq{i}'*IC{i}*dXupdq{i}
                  Xup(ii).transpose() * dICdq_ii_k2 *
                      dXupdq(ii) +  // Xup{i}'*dICdq{i}(:,:,k2)*dXupdq{i}
                  Xup(ii).transpose() * IC(ii) *
                      ddXupddq(ii);  // Xup{i}'*IC{i}*ddXupddq{i}
            }
            else
            {
              // Xup{i}'*ddICddq{i}(:,:,k1,k2)*Xup{i}
              ddICddq_parent_ii_k1_k2 =
                  Xup(ii).transpose() * ddICddq_ii_k1_k2 * Xup(ii);
            }

            // Map matrix back to tensor for each links contribution to be added
            // to the double derivative of composite inertia tensor
            Eigen::TensorMap<Eigen::Tensor<const double, 2>>
                ddICddq_parent_ii_k1_k2_tensor(
                    ddICddq_parent_ii_k1_k2.data(), 6, 6);
            ddICddq(model.parents[ii]).chip<3>(k2).chip<2>(k1) +=
                ddICddq_parent_ii_k1_k2_tensor;
          }
          else
          {
            if (k1 == ii)
            {
              // Convert indexed tensors to matrices for easy matrix
              // multiplication
              dICdq_ii_jj_tens = dICdq(ii).chip<2>(k2);
              Eigen::Map<Eigen::MatrixXd> dICdq_ii_k2(
                  dICdq_ii_jj_tens.data(), 6, 6);

              ddICddq_parent_ii_k1_k2 =
                  dXupdq(ii).transpose() * dICdq_ii_k2 *
                      Xup(ii) +  // dXupdq{i}'*dICdq{i}(:,:,k2)*Xup{i}
                  Xup(ii).transpose() * ddICddq_ii_k1_k2 *
                      Xup(ii) +  // Xup{i}'*ddICddq{i}(:,:,k1,k2)*Xup{i}
                  Xup(ii).transpose() * dICdq_ii_k2 *
                      dXupdq(ii);  // Xup{i}'*dICdq{i}(:,:,k2)*dXupdq{i}
            }
            else
            {
              if (k2 == ii)
              {
                dICdq_ii_jj_tens = dICdq(ii).chip<2>(k1);
                Eigen::Map<Eigen::MatrixXd> dICdq_ii_k1(
                    dICdq_ii_jj_tens.data(), 6, 6);
                ddICddq_parent_ii_k1_k2 =
                    dXupdq(ii).transpose() * dICdq_ii_k1 *
                        Xup(ii) +  // dXupdq{i}'*dICdq{i}(:,:,k1)*Xup{i}
                    Xup(ii).transpose() * ddICddq_ii_k1_k2 *
                        Xup(ii) +  // Xup{i}'*ddICddq{i}(:,:,k1,k2)*Xup{i}
                    Xup(ii).transpose() * dICdq_ii_k1 *
                        dXupdq(ii);  // Xup{i}'*dICdq{i}(:,:,k1)*dXupdq{i}
              }
              else
              {
                ddICddq_parent_ii_k1_k2 =
                    Xup(ii).transpose() * ddICddq_ii_k1_k2 *
                    Xup(ii);  // Xup{i}'*ddICddq{i}(:,:,k1,k2)*Xup{i}
              }

              // Map matrix back to tensor for each links contribution to be
              // added to the double derivative of composite inertia tensor
              Eigen::TensorMap<Eigen::Tensor<const double, 2>>
                  ddICddq_parent_ii_k1_k2_tensor(
                      ddICddq_parent_ii_k1_k2.data(), 6, 6);
              ddICddq(model.parents[ii]).chip<3>(k2).chip<2>(k1) +=
                  ddICddq_parent_ii_k1_k2_tensor;

              // Since the tensor is symmetric assign the values to the other
              // side
              ddICddq(model.parents[ii]).chip<3>(k1).chip<2>(k2) =
                  ddICddq(model.parents[ii]).chip<3>(k2).chip<2>(k1);
            }
          }
        }
      }
    }
  }

  // Initialize matrices and tensors to be used in the next stage
  Eigen::MatrixXd dfhdq = Eigen::MatrixXd::Zero(6, model.nv);
  Eigen::Tensor<double, 3> ddfhddq(6, model.nv, model.nv);
  Eigen::Tensor<double, 2> dfhdq_tensor(6, model.nv);
  dfhdq_tensor.setZero();
  ddfhddq.setZero();

  // initialize temporary objects for subsequent use
  Eigen::Tensor<double, 1> ddfhddq_ii_k1_k2_tensor(6);

  /*Recursively calculate spatial forces and compute the mass matrix and its
   * derivatives*/
  for (size_t ii = 0; ii < model.nv; ii++)
  {
    // Recursively calculate spatial forces and compute the diagonal terms of
    // the mass matrix and its derivatives
    Vector6 fh = IC(ii) * S(ii);
    H(ii, ii) = S(ii).transpose() * fh;

    // Define the contraction mapping.
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, 0)};

    // convert S(ii) to a 1D tensor for contraction
    Eigen::TensorMap<Eigen::Tensor<double, 1>> S_ii_tensor(S(ii).data(), 6);

    dfhdq_tensor = dICdq(ii).contract(S_ii_tensor, contract_dims);
    Eigen::Map<Eigen::MatrixXd> dfhdq(dfhdq_tensor.data(), 6, model.nv);

    ddfhddq = ddICddq(ii).contract(S_ii_tensor, contract_dims);

    // Populate diagonal terms of the mass matrix first derivative
    for (int k = 0; k < model.nv; k++)
    {
      dH_dq(ii, ii, k) = S(ii).transpose() * dfhdq.col(k);
    }

    // Populate diagonal terms of the mass matrix second derivative
    for (int k1 = 0; k1 < model.nv; k1++)
    {
      for (int k2 = 0; k2 <= k1; k2++)
      {
        ddfhddq_ii_k1_k2_tensor = ddfhddq.chip<2>(k2).chip<1>(k1);
        Eigen::Map<Eigen::VectorXd> ddfhddq_ii_k1_k2(
            ddfhddq_ii_k1_k2_tensor.data(), 6);
        ddH_ddq(ii, ii, k1, k2) = S(ii).transpose() * ddfhddq_ii_k1_k2;

        if (k1 != k2)
        {
          ddH_ddq(ii, ii, k2, k1) = ddH_ddq(ii, ii, k1, k2);
        }
      }
    }

    int jj = ii;

    while (model.parents[jj] >= 0 && model.parents[jj] != jj)
    {
      // Backwards pass to recursively calculate spatial forces for the
      // off-diagonal terms of the mass matrix and its derivatives
      for (int k1 = model.nv - 1; k1 >= 0; k1--)
      {
        for (int k2 = model.nv - 1; k2 >= k1; k2--)
        {
          ddfhddq_ii_k1_k2_tensor = ddfhddq.chip<2>(k2).chip<1>(k1);
          Eigen::Map<Eigen::VectorXd> ddfhddq_ii_k1_k2(
              ddfhddq_ii_k1_k2_tensor.data(), 6);

          // Compute ddfhddq recursively
          if (k1 == k2)
          {
            if (k1 == jj)
            {
              ddfhddq_ii_k1_k2 =
                  ddXupddq(jj).transpose() * fh +  // ddXupddq{j}' * fh
                  dXupdq(jj).transpose() *
                      dfhdq.col(k2) +  // dXupdq{j}' * dfhdq(:,k2)
                  dXupdq(jj).transpose() *
                      dfhdq.col(k1) +  // dXupdq{j}' * dfhdq(:,k2)
                  Xup(jj).transpose() *
                      ddfhddq_ii_k1_k2;  // Xup{j}' * ddfhddq(:,k1,k2)
            }
            else
            {
              ddfhddq_ii_k1_k2 =
                  Xup(jj).transpose() *
                  ddfhddq_ii_k1_k2;  // Xup{j}' * ddfhddq(:,k1,k2)
            }

            // Map matrix back to tensor for each links contribution to be added
            // to the double derivative of spatial force
            Eigen::TensorMap<Eigen::Tensor<const double, 1>>
                ddfhddq_ii_k1_k2_tensor(ddfhddq_ii_k1_k2.data(), 6);
            ddfhddq.chip<2>(k2).chip<1>(k1) = ddfhddq_ii_k1_k2_tensor;
          }
          else
          {
            if (k1 == jj)
            {
              ddfhddq_ii_k1_k2 =
                  dXupdq(jj).transpose() *
                      dfhdq.col(k2) +  // dXupdq{j}' * dfhdq(:,k2)
                  Xup(jj).transpose() *
                      ddfhddq_ii_k1_k2;  // Xup{j}' * ddfhddq(:,k1,k2)
            }
            else
            {
              if (k2 == jj)
              {
                ddfhddq_ii_k1_k2 =
                    dXupdq(jj).transpose() *
                        dfhdq.col(k1) +  // dXupdq{j}' * dfhdq(:,k1)
                    Xup(jj).transpose() *
                        ddfhddq_ii_k1_k2;  // Xup{j}' * ddfhddq(:,k1,k2)
              }
              else
              {
                ddfhddq_ii_k1_k2 =
                    Xup(jj).transpose() *
                    ddfhddq_ii_k1_k2;  // Xup{j}' * ddfhddq(:,k1,k2)
              }

              // Map matrix back to tensor for each links contribution to be
              // added to the double derivative of spatial force
              Eigen::TensorMap<Eigen::Tensor<const double, 1>>
                  ddfhddq_ii_k1_k2_tensor(ddfhddq_ii_k1_k2.data(), 6);
              ddfhddq.chip<2>(k2).chip<1>(k1) = ddfhddq_ii_k1_k2_tensor;

              // Since the tensor is symmetric assign the values to the other
              // side
              ddfhddq.chip<2>(k1).chip<1>(k2) = ddfhddq.chip<2>(k2).chip<1>(k1);
            }
          }
        }
      }

      for (int k = 0; k < model.nv; k++)
      {
        if (k == jj)
        {
          dfhdq.col(k) =
              dXupdq(jj).transpose() * fh + Xup(jj).transpose() * dfhdq.col(k);
        }
        else
        {
          dfhdq.col(k) = Xup(jj).transpose() * dfhdq.col(k);
        }
      }

      fh = Xup(jj).transpose() * fh;
      jj = model.parents[jj];

      // Compute mass matrix off diagonal terms
      H(jj, ii) = S(jj).transpose() * fh;
      H(ii, jj) = H(jj, ii);

      // Compute derivative of mass matrix off-diagonal terms
      for (int k = 0; k < model.nv; k++)
      {
        dH_dq(ii, jj, k) = S(jj).transpose() * dfhdq.col(k);
        dH_dq(jj, ii, k) = dH_dq(ii, jj, k);
      }

      // Populate off-diagonal terms of the mass matrix double derivative
      for (int k1 = 0; k1 < model.nv; k1++)
      {
        for (int k2 = 0; k2 <= k1; k2++)
        {
          ddfhddq_ii_k1_k2_tensor = ddfhddq.chip<2>(k2).chip<1>(k1);
          Eigen::Map<Eigen::VectorXd> ddfhddq_ii_k1_k2(
              ddfhddq_ii_k1_k2_tensor.data(), 6);

          ddH_ddq(ii, jj, k1, k2) = S(ii).transpose() * ddfhddq_ii_k1_k2;
          ddH_ddq(jj, ii, k1, k2) = ddH_ddq(ii, jj, k1, k2);

          if (k1 != k2)
          {
            ddH_ddq(ii, jj, k2, k1) = ddH_ddq(ii, jj, k1, k2);
            ddH_ddq(jj, ii, k2, k1) = ddH_ddq(jj, ii, k1, k2);
          }
        }
      }
    }
  }

  // Get H_dot
  H_dot = MatX::Zero(model.nv, model.nv);
  Eigen::Tensor<double, 2> dH_dq_ii_tensor(model.nv, model.nv);
  for (int ii = 0; ii < model.nv; ii++)
  {
    dH_dq_ii_tensor = dH_dq.chip<2>(ii);
    Eigen::Map<Eigen::MatrixXd> dH_dq_ii(
        dH_dq_ii_tensor.data(), model.nv, model.nv);
    H_dot += dH_dq_ii * xp2(ii);
  }
}

void AghfPybindWrapper::ABA_2D(Eigen::Ref<MatX> dFD_dq,
                               Eigen::Ref<MatX> dFD_dv,
                               Eigen::Ref<MatX> H_inv,
                               Eigen::Tensor<double, 3> &ddID_ddq,
                               Eigen::Tensor<double, 3> &ddID_ddv,
                               Eigen::Tensor<double, 3> &ddID_dqdv,
                               Eigen::Tensor<double, 3> &dH_dq,
                               Eigen::Tensor<double, 3> &ddFD_ddq,
                               Eigen::Tensor<double, 3> &ddFD_ddv,
                               Eigen::Tensor<double, 3> &ddFD_dqdv,
                               Eigen::Tensor<double, 3> &ddFD_dudq)
{
  // Adapted from code implementation of Second order derivatives of FD
  // (https://arxiv.org/abs/2302.06001)

  Eigen::Tensor<double, 3> prodq(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> prodv(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> prodqdd(model.nv, model.nv, model.nv);

  Eigen::MatrixXd mat_out(Eigen::MatrixXd::Zero(model.nv, model.nv));
  // Some temp variables here

  Eigen::VectorXd vec1(model.nv);
  Eigen::VectorXd vec2(model.nv);

  Eigen::MatrixXd mat1(model.nv, model.nv);
  Eigen::MatrixXd mat2(model.nv, model.nv);
  Eigen::MatrixXd mat3(model.nv, model.nv);
  Eigen::MatrixXd mat4(model.nv, model.nv);

  Eigen::MatrixXd term_in(model.nv, 4 * model.nv * model.nv);   // For DMM
  Eigen::MatrixXd term_out(model.nv, 4 * model.nv * model.nv);  // For DMM
  Eigen::MatrixXd mat_in_id_fo_aza(model.nv, 3 * model.nv);

  //--- For models N<30------------------//
  // Inner term Compute using DTM (or DMM)
  if (model.nv <= 30)
  {
    for (int u = 0; u < model.nv; u++)
    {
      pino::get_mat_from_tens3_v1(dH_dq, mat1, model.nv, u);
      for (int w = 0; w < model.nv; w++)
      {
        pino::get_mat_from_tens3_v1(dH_dq, mat2, model.nv, w);
        vec1 = mat1 * dFD_dq.col(w);
        vec2 = mat2 * dFD_dv.col(u);
        pino::hess_assign(prodq, vec1, 0, u, w, 1,
                          model.nv);  // slicing a vector in
        pino::hess_assign(prodv, vec2, 0, w, u, 1,
                          model.nv);  // slicing a vector in
      }
      mat3.noalias() = mat1 * H_inv;
      pino::hess_assign_fd2(prodqdd, mat3, model.nv, u);
    }
  }

  //--- For models N>30------------------//// Used for ATLAS and talos_full_v2
  // Inner term Compute using IDFOZA
  // else {
  //     mat_in_id_fo_aza << dFD_dq, dFD_dv, H_inv; // concatenating FO partial
  //     of FD wrt xp1 and qdot for (int ii = 0; ii < 3 * model.nv; ii++) {
  //         pino::computeID_FO_AZA(model, data, xp1, xp2,
  //         mat_in_id_fo_aza.col(ii), mat_out); if (ii < model.nv) {
  //             pino::hess_assign_fd_v1(prodq, mat_out, model.nv, ii);
  //         } else if (ii >= model.nv && ii < 2 * model.nv) {
  //             pino::hess_assign_fd_v1(prodv, mat_out, model.nv, ii -
  //             model.nv);
  //         } else {
  //             pino::hess_assign_fd_v1(prodqdd, mat_out, model.nv, ii - 2 *
  //             model.nv);
  //         }
  //     }
  // }

  // Inner term addition using single loop - overall cheaper than double loop
  // inner-term add
  for (int u = 0; u < model.nv; u++)
  {
    pino::get_mat_from_tens3_v1(ddID_ddq, mat1, model.nv, u);
    pino::get_mat_from_tens2(prodq, mat2, model.nv, u);
    pino::get_mat_from_tens3_v1(prodq, mat3, model.nv, u);
    mat1 += mat2 + mat3;

    term_in.middleCols(4 * u * model.nv, model.nv) = mat1;
    // partial w.r.t xp2
    pino::get_mat_from_tens3_v1(ddID_ddv, mat2, model.nv, u);
    term_in.middleCols((4 * u + 1) * model.nv, model.nv) = mat2;
    // partial w.r.t xp1/xp2
    pino::get_mat_from_tens3_v1(ddID_dqdv, mat3, model.nv, u);
    pino::get_mat_from_tens3_v1(prodv, mat2, model.nv, u);
    mat3 += mat2;
    term_in.middleCols((4 * u + 2) * model.nv, model.nv) = mat3;
    // partial w.r.t u/xp1
    pino::get_mat_from_tens2(prodqdd, mat1, model.nv, u);
    term_in.middleCols((4 * u + 3) * model.nv, model.nv) = mat1;
  }
  // outer term compute using DTM
  term_out = -H_inv * term_in;  // DMM here with -H_inv

  // final assign using double loop - overall cheaper than single loop assign
  for (int u = 0; u < model.nv; u++)
  {
    for (int w = 0; w < model.nv; w++)
    {
      pino::hess_assign(
          ddFD_ddq, term_out.col(4 * u * model.nv + w), 0, w, u, 1, model.nv);
      pino::hess_assign(ddFD_ddv, term_out.col((4 * u + 1) * model.nv + w), 0,
                        w, u, 1, model.nv);
      pino::hess_assign(ddFD_dqdv, term_out.col((4 * u + 2) * model.nv + w), 0,
                        w, u, 1, model.nv);
      pino::hess_assign(ddFD_dudq, term_out.col((4 * u + 3) * model.nv + w), 0,
                        w, u, 1, model.nv);
    }
  }
}

void AghfPybindWrapper::get_HTH_terms(Eigen::Ref<MatX> H,
                                      Eigen::Ref<MatX> H_inv,
                                      Eigen::Tensor<double, 3> &dHinv_dq,
                                      Eigen::Ref<MatX> HTH,
                                      Eigen::Ref<MatX> HinvTHinv,
                                      Eigen::Tensor<double, 3> &HinvTHinv_dxp1)
{
  HTH = H.transpose() * H;
  HinvTHinv = H_inv.transpose() * H_inv;

  HinvTHinv_dxp1 = TensorMatrixMult_3Dwith2D(H_inv, dHinv_dq, 1, 0) +
                   TensorMatrixMult_2Dwith3D(H_inv, dHinv_dq, 0, 1);
}

void AghfPybindWrapper::get_Hdot_D(
    Eigen::Ref<VecX> xdp1,
    Eigen::Tensor<double, 4> &ddH_ddq,
    Eigen::Ref<VecX> xdp2_p_FD,
    Eigen::array<Eigen::IndexPair<int>, 1> &contract_dims,
    Eigen::Tensor<double, 3> &dHdot_dq,
    Eigen::Tensor<double, 3> &ddH_ddxp1_xdp2_p_FD)
{
  // initialize temporary objects for subsequent use
  Eigen::Tensor<double, 2> ddH_ddq_slice1(model.nv, model.nv);
  Eigen::Tensor<double, 2> ddH_ddq_slice2(model.nv, model.nv);

  // convert xdp1 to a 1D tensor for contraction
  Eigen::TensorMap<Eigen::Tensor<double, 1>> xdp1_tensor(xdp1.data(), model.nv);
  Eigen::TensorMap<Eigen::Tensor<double, 1>> xdp2_p_FD_tensor(
      xdp2_p_FD.data(), model.nv);

  // Compute dHdot_dq
  for (int u = 0; u < model.nv; u++)
  {
    for (int w = 0; w < model.nv; w++)
    {
      // For H_dot computation
      ddH_ddq_slice1 = ddH_ddq.chip<3>(u).chip<1>(w);
      dHdot_dq.chip<2>(u).chip<1>(w) =
          ddH_ddq_slice1.contract(xdp1_tensor, contract_dims);

      // For ddH_ddxp1_xdp2_p_FD computation
      ddH_ddq_slice2 = ddH_ddq.chip<3>(w).chip<2>(u);
      ddH_ddxp1_xdp2_p_FD.chip<2>(w).chip<1>(u) =
          ddH_ddq_slice2.contract(xdp2_p_FD_tensor, contract_dims);
    }
  }
}

void AghfPybindWrapper::get_Cdot_D(Eigen::Ref<VecX> xdp1,
                                   Eigen::Ref<VecX> xdp2,
                                   Eigen::Tensor<double, 3> &ddC_ddq,
                                   Eigen::Tensor<double, 3> &ddC_ddv,
                                   Eigen::Tensor<double, 3> &ddC_dqdv,
                                   Eigen::Tensor<double, 3> &ddC_dvdq,
                                   Eigen::Ref<MatX> dCdot_dxp1,
                                   Eigen::Ref<MatX> dCdot_dxp2)
{
  dCdot_dxp1 = TensorVectorMult_3Dwith1D(xdp1, ddC_ddq, 1) +
               TensorVectorMult_3Dwith1D(xdp2, ddC_dvdq, 1);

  dCdot_dxp2 = TensorVectorMult_3Dwith1D(xdp1, ddC_dqdv, 1) +
               TensorVectorMult_3Dwith1D(xdp2, ddC_ddv, 1);
}

void AghfPybindWrapper::compute_AGHF_RHS(const Eigen::Ref<MatX> x,
                                         const Eigen::Ref<MatX> xd,
                                         const Eigen::Ref<MatX> xdd,
                                         const int num_samples,
                                         Eigen::Ref<MatX> AGHF_RHS)
{
  if (x.rows() != num_samples || x.cols() != 2 * model.nv)
  {
    throw std::invalid_argument("x has wrong dimensions!");
  }
  if (xd.rows() != num_samples || xd.cols() != 2 * model.nv)
  {
    throw std::invalid_argument("xd has wrong dimensions!");
  }
  if (xdd.rows() != num_samples || xdd.cols() != 2 * model.nv)
  {
    throw std::invalid_argument("xdd has wrong dimensions!");
  }
  if (AGHF_RHS.rows() != num_samples || AGHF_RHS.cols() != 2 * model.nv)
  {
    throw std::invalid_argument("AGHF_RHS has wrong dimensions!");
  }

  for (size_t i = 0; i < num_samples; i++)
  {
    // get x information at each sample
    xp1 = x.row(i).head(model.nv);
    xp2 = x.row(i).tail(model.nv);
    xdp1 = xd.row(i).head(model.nv);
    xdp2 = xd.row(i).tail(model.nv);
    xddp1 = xdd.row(i).head(model.nv);
    xddp2 = xdd.row(i).tail(model.nv);

    // compute inertia matrix related terms (H and H_dot)
    CRBA_D(xp1, xdp1);

    // compute the grouped Coriolis and gravity term C.
    // nonLinearEffects(xp1, xp2) is the same as RNEA(xp1, xp2, 0) = C
    nonLinearEffects(model, data, xp1, xp2);
    C = data.nle;

    // compute derivatives of the grouped Coriolis and gravity term
    pino::computeRNEADerivatives(
        model, data, xp1, xp2, zero_vec, dC_dq, dC_dv, dC_da);
    C_dot = dC_dq * xdp1 + dC_dv * xdp2;

    // Compute the first derivatives of the forward dynamics with u=0
    pino::computeABADerivatives(
        model, data, xp1, xp2, zero_vec, dFD_dq, dFD_dv, H_inv);

    // Also get the forward dynamics with u = 0 computed during
    // computeABADerivatives
    FD_0 = data.ddq;

    VecX Fd(2 * model.nv);
    Fd << xp2, FD_0;

    // compute Omega_1
    HTH = H.transpose() * H;
    HinvTHinv = H_inv.transpose() * H_inv;
    G_inv.bottomRightCorner(model.nv, model.nv) = HinvTHinv;

    Omega_1.head(model.nv) = 2 * (xddp1 - xdp2);
    Omega_1.tail(model.nv) =
        2 * HinvTHinv *
        (H_dot * H * xdp2 + H_dot * C + H.transpose() * H_dot * xdp2 +
         HTH * xddp2 + H.transpose() * C_dot);

    // compute Omega_2
    Omega_2.head(model.nv) =
        -2 * k_inv_diag * dFD_dq.transpose() * (H * C + HTH * xdp2);
    Omega_2.tail(model.nv) =
        -2 * HinvTHinv * dFD_dv.transpose() * (H * C + HTH * xdp2);

    // compute Omega_3
    Omega_3.tail(model.nv) = 2 * k * HinvTHinv * (xdp1 - xp2);

    // compute Omega_4
    model.gravity.setZero();  // temporarily set gravity to zero

    // Use computeRNEADerivatives to compute dID_dq which now stores d H*(xdp2 -
    // FD_0) dq
    pino::computeRNEADerivatives(
        model, data, xp1, zero_vec, xdp2 - FD_0, dID_dq, dID_dv, dID_da);

    Omega_4.head(model.nv) =
        k_inv_diag * (2 * dID_dq.transpose() * (H * (xdp2 - FD_0)));

    model.gravity.linear()(2) = -9.81;  // set the gravity back to normal

    // compute penalties
    d_penalty_dx = VecX::Zero(2 * model.nv);
    for (size_t bid = 0; bid < activated_penalty_vec.size(); bid++)
    {
      d_penalty_dx += activated_penalty_vec[bid]->compute_derivative(x.row(i));
    }

    // compute AGHF_RHS
    AGHF_RHS.row(i) =
        Omega_1 - (Omega_2 - Omega_3 + Omega_4) - G_inv * d_penalty_dx;
  }
}

void AghfPybindWrapper::compute_PSAGHF_jac(const Eigen::Ref<MatX> x,
                                           const Eigen::Ref<MatX> xd,
                                           const Eigen::Ref<MatX> xdd,
                                           const int num_samples,
                                           Eigen::Ref<MatX> AGHF_jac)
{
  // Corresponds to the formulation where the lagrangian is augmented by
  // multiplying it by a penalty function

  if (x.rows() != num_samples || x.cols() != 2 * model.nv)
  {
    throw std::invalid_argument("x has wrong dimensions!");
  }

  if (xd.rows() != num_samples || xd.cols() != 2 * model.nv)
  {
    throw std::invalid_argument("xd has wrong dimensions!");
  }

  if (xdd.rows() != num_samples || xdd.cols() != 2 * model.nv)
  {
    throw std::invalid_argument("xdd has wrong dimensions!");
  }

  if (AGHF_jac.rows() != 2 * model.nv * num_ps_nodes ||
      AGHF_jac.cols() != 2 * model.nv * num_ps_nodes)
  {
    throw std::invalid_argument("AGHF jacobian has wrong dimensions!");
  }

  for (size_t i = 0; i < num_samples; i++)
  {
    // get x information at each ps nodes
    xp1 = x.row(i).head(model.nv);
    xp2 = x.row(i).tail(model.nv);
    xdp1 = xd.row(i).head(model.nv);
    xdp2 = xd.row(i).tail(model.nv);
    xddp1 = xdd.row(i).head(model.nv);
    xddp2 = xdd.row(i).tail(model.nv);

    CRBA_2D(xp1, xdp1, H, H_dot, dH_dq, ddH_ddq);

    // compute generalized forces related terms (Coriolis, centrifugal, gravity)
    pino::computeABADerivatives(
        model, data, xp1, xp2, zero_vec, dFD_dq, dFD_dv, H_inv);

    // Compute C
    nonLinearEffects(model, data, xp1, xp2);
    C = data.nle;

    FD_0 = data.ddq;

    // compute derivatives of grouped Coriolis-Gravity Terms
    pino::computeRNEADerivatives(
        model, data, xp1, xp2, zero_vec, dC_dq, dC_dv, dC_da);
    C_dot = dC_dq * xdp1 + dC_dv * xdp2;

    // Define the contraction mapping.
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, 0)};

    // Compute dHdot_dq and ddH_ddq*(xp2 - FD_0)
    xdp2_p_FD = xdp2 - FD_0;
    get_Hdot_D(
        xdp1, ddH_ddq, xdp2_p_FD, contract_dims, dHdot_dq, ddH_ddxp1_xdp2_p_FD);

    // Use RNEA2D to compute double derivatives of grouped Coriolis-Gravity Term
    pino::ComputeRNEASecondOrderDerivatives(
        model, data, xp1, xp2, zero_vec, ddC_ddq, ddC_ddv, ddC_dqdv, du_dadq);

    // Define the permuation dimensions
    Eigen::array<int, 3> shuffle_dims = {0, 2, 1};
    ddC_dvdq = ddC_dqdv.shuffle(shuffle_dims);

    // Compute dHinv_dq
    Eigen::TensorMap<Eigen::Tensor<double, 2>> Hinv_tensor(
        H_inv.data(), model.nv, model.nv);
    Eigen::Tensor<double, 3> dHinv_dq_perm = -Hinv_tensor.contract(
        dH_dq.contract(Hinv_tensor, contract_dims), contract_dims);

    // permute dimesnions to get the tensor arranged in the right order
    dHinv_dq = dHinv_dq_perm.shuffle(shuffle_dims);

    // Use RNEA2D to compute double derivatives of ID
    // Note: Need to use FD_0 as the acceleration because you need the
    // acceleration that corresponds to FD with tau = 0, in order to get the
    // corresponding ID s.t. ID(FD) = I
    pino::ComputeRNEASecondOrderDerivatives(
        model, data, xp1, xp2, FD_0, ddID_ddq, ddID_ddv, ddID_dqdv, ddID_dadq);

    // Compute double derivatives of Forward dynamics using ID second
    // derivatives
    ABA_2D(dFD_dq, dFD_dv, H_inv, ddID_ddq, ddID_ddv, ddID_dqdv, dH_dq,
           ddFD_ddq, ddFD_ddv, ddFD_dqdv, ddFD_dudq);
    ddFD_dvdq = ddFD_dqdv.shuffle(shuffle_dims);

    get_HTH_terms(H, H_inv, dHinv_dq, HTH, HinvTHinv, HinvTHinv_dxp1);

    G.bottomRightCorner(model.nv, model.nv) = HTH;
    G_inv.bottomRightCorner(model.nv, model.nv) = HinvTHinv;

    // Compute gamma
    gamma = H_dot * (H * xdp2 + C) + H * (H_dot * xdp2 + H * xddp2 + C_dot);

    // Compute dCdot_dxp1 and dCdot_dxp2
    get_Cdot_D(xdp1, xdp2, ddC_ddq, ddC_ddv, ddC_dqdv, ddC_dvdq, dCdot_dxp1,
               dCdot_dxp2);

    // compute dgamma_dxp1
    H_xdp2_p_C = H * xdp2 + C;
    Hdot_xdp2_p_H_xddp2_p_Cdot = H_dot * xdp2 + H * xddp2 + C_dot;

    dH_dxp1_m_xdp2 = TensorVectorMult_3Dwith1D(xdp2, dH_dq, 1);
    dgamma_dxp1 =
        TensorVectorMult_3Dwith1D(H_xdp2_p_C, dHdot_dq, 1) +
        H_dot * (dH_dxp1_m_xdp2 + dC_dq) +
        TensorVectorMult_3Dwith1D(Hdot_xdp2_p_H_xddp2_p_Cdot, dH_dq, 1) +
        H * (TensorVectorMult_3Dwith1D(xdp2, dHdot_dq, 1) +
             TensorVectorMult_3Dwith1D(xddp2, dH_dq, 1) + dCdot_dxp1);

    // Compute the partials of Omega_1
    dOmega1_dxp1.block(model.nv, 0, model.nv, model.nv) =
        2 * (TensorVectorMult_3Dwith1D(gamma, HinvTHinv_dxp1, 1) +
             HinvTHinv * dgamma_dxp1);

    dOmega1_dxp2.block(model.nv, 0, model.nv, model.nv) =
        2 * HinvTHinv *
        (H_dot * dC_dv + H * dCdot_dxp2);  // 2 * HinvTHinv * dgamma_dxp2

    twoC_p_2H_xdp2 = 2 * (C + H * xdp2);
    H_2C_p_2H_xdp2 = H * twoC_p_2H_xdp2;

    // Compute alpha terms
    neg_dFD_dq_T = -dFD_dq.transpose();
    neg_dFD_dv_T = -dFD_dv.transpose();

    alpha1 = neg_dFD_dq_T * H_2C_p_2H_xdp2;
    alpha2 = neg_dFD_dv_T * H_2C_p_2H_xdp2;

    // Negate and transpose FD double derivatives for ease of subsequent
    // multiplication
    Eigen::array<int, 3> tensor_transpose_dims = {
        1, 0, 2};  // Define the tensor tranpose dimensions
    neg_ddFD_ddq_T = -ddFD_ddq.shuffle(tensor_transpose_dims);
    neg_ddFD_dqdv_T = -ddFD_dqdv.shuffle(tensor_transpose_dims);
    neg_ddFD_ddv_T = -ddFD_ddv.shuffle(tensor_transpose_dims);
    neg_ddFD_dvdq_T = -ddFD_dvdq.shuffle(tensor_transpose_dims);

    neg_dFD_dq = -dFD_dq;
    neg_dFD_dv = -dFD_dv;

    // Compute derivatives of alpha1
    dalpha_b_block = TensorVectorMult_3Dwith1D(twoC_p_2H_xdp2, dH_dq, 1) +
                     2 * H * (dC_dq + dH_dxp1_m_xdp2);

    two_H_dC_dxp2 = 2 * H * dC_dv;

    d_alpha1_dxp1 =
        TensorVectorMult_3Dwith1D(H_2C_p_2H_xdp2, neg_ddFD_ddq_T, 1) +
        neg_dFD_dq_T * dalpha_b_block;

    d_alpha1_dxp2 =
        TensorVectorMult_3Dwith1D(H_2C_p_2H_xdp2, neg_ddFD_dqdv_T, 1) +
        neg_dFD_dq_T * two_H_dC_dxp2;

    // Compute derivatives of alpha2
    d_alpha2_dxp1 =
        TensorVectorMult_3Dwith1D(H_2C_p_2H_xdp2, neg_ddFD_dvdq_T, 1) +
        neg_dFD_dv_T * dalpha_b_block;

    d_alpha2_dxp2 =
        TensorVectorMult_3Dwith1D(H_2C_p_2H_xdp2, neg_ddFD_ddv_T, 1) +
        neg_dFD_dv_T * two_H_dC_dxp2;

    // Compute GAMMA
    dH_dxp1_xp2_p_FD = TensorVectorMult_3Dwith1D(xdp2_p_FD, dH_dq, 1);
    H_xdp2_p_FD = H * xdp2_p_FD;

    GAMMA = 2 * k_inv_diag * dH_dxp1_xp2_p_FD.transpose() * H_xdp2_p_FD;

    // Compute dGAMMA_dxp1
    Eigen::array<int, 3> GAMMA_shuffle_dims = {
        2, 0, 1};  // Define the permuation dimensions
    d_GAMMA_block_temp = TensorMatrixMult_3Dwith2D(neg_dFD_dq, dH_dq, 1, 0) +
                         ddH_ddxp1_xdp2_p_FD;
    d_GAMMA_block = d_GAMMA_block_temp.shuffle(GAMMA_shuffle_dims);

    dGAMMA_dxp1 =
        TensorVectorMult_3Dwith1D(H_xdp2_p_FD, d_GAMMA_block, 1) +
        dH_dxp1_xp2_p_FD.transpose() * (dH_dxp1_xp2_p_FD + H * neg_dFD_dq);

    // Compute dGAMMA_dxp2
    dH_dxp1_dFD_xp2_temp = TensorMatrixMult_3Dwith2D(neg_dFD_dv, dH_dq, 1, 0);
    dH_dxp1_dFD_xp2 = dH_dxp1_dFD_xp2_temp.shuffle(GAMMA_shuffle_dims);
    dGAMMA_dxp2 = TensorVectorMult_3Dwith1D(H_xdp2_p_FD, dH_dxp1_dFD_xp2, 1) +
                  dH_dxp1_xp2_p_FD.transpose() * H * neg_dFD_dv;

    xdp1_m_xp2 = xdp1 - xp2;

    // Compute the partials of (Omega_2 - Omega_3 + Omega_4) wrt xp1 as
    // dOmega234_dxp1
    dOmega234_dxp1.block(0, 0, model.nv, model.nv) =
        k_inv_diag * d_alpha1_dxp1 +   // from dOmega2_dxp1
        2 * k_inv_diag * dGAMMA_dxp1;  // from dOmega4_dxp1

    dOmega234_dxp1.block(model.nv, 0, model.nv, model.nv) =
        TensorVectorMult_3Dwith1D(
            alpha2, HinvTHinv_dxp1, 1) +  // from dOmega2_dxp1
        HinvTHinv * d_alpha2_dxp1 +       // from dOmega2_dxp1
        -2 * k *
            TensorVectorMult_3Dwith1D(xdp1_m_xp2, HinvTHinv_dxp1,
                                      1);  // from dOmega3_dxp1

    // Compute the partials of (Omega_2 - Omega_3 + Omega_4) wrt xp2 as
    // dOmega234_dxp2
    dOmega234_dxp2.block(0, 0, model.nv, model.nv) =
        k_inv_diag * (d_alpha1_dxp2 +    // from dOmega2_dxp2
                      2 * dGAMMA_dxp2);  // from dOmega4_dxp2

    dOmega234_dxp2.block(model.nv, 0, model.nv, model.nv) =
        HinvTHinv * (d_alpha2_dxp2   // from dOmega2_dxp2
                     + 2 * k_diag);  // from dOmega3_dxp2

    dOmega_dxp1 = dOmega1_dxp1 - dOmega234_dxp1;
    dOmega_dxp2 = dOmega1_dxp2 - dOmega234_dxp2;

    // Compute dgamma_dxdp1
    dgamma_dxdp1 =
        TensorVectorMult_3Dwith1D(H_xdp2_p_C, dH_dq, 1) +  // dH_dq_H_xdp2_p_C
        H * (dH_dxp1_m_xdp2 + dC_dq);

    // Compute dgamma_dxdp2
    dgamma_dxdp2 = H_dot * H + H * (H_dot + dC_dv);

    // Compute dOmega_dxdp1
    dOmega_dxdp1.block(model.nv, 0, model.nv, model.nv) =
        2 * HinvTHinv * (dgamma_dxdp1 + k_diag);  // from dOmega1_dxdp1

    // Compute dGAMMA_dxdp2
    dH_dxp1_I_temp = TensorMatrixMult_3Dwith2D(eye_modelnv, dH_dq, 1, 0);
    dH_dxp1_I = dH_dxp1_I_temp.shuffle(GAMMA_shuffle_dims);

    dGAMMA_dxdp2 = TensorVectorMult_3Dwith1D(H_xdp2_p_FD, dH_dxp1_I, 1) +
                   dH_dxp1_xp2_p_FD.transpose() * H;

    dOmega_dxdp2.block(0, 0, model.nv, model.nv) =
        -2 * eye_modelnv +  // from dOmega1_dxdp1
        -2 * k_inv_diag *
            (neg_dFD_dq_T * HTH +  // from dOmega2_dxdp2 (dalpha1_dxdp2 term)
             dGAMMA_dxdp2);        // from dOmega4_dxdp2

    dOmega_dxdp2.block(model.nv, 0, model.nv, model.nv) =
        2 * HinvTHinv *
        (dgamma_dxdp2 +         // from dOmega1_dxdp2
         -neg_dFD_dv_T * HTH);  // from dOmega2_dxdp2 (dalpha2_dxdp2 term)

    dOmega_dxddp1.block(0, 0, model.nv, model.nv) = 2 * eye_modelnv;

    dOmega_dxddp2.block(model.nv, 0, model.nv, model.nv) = 2 * eye_modelnv;

    // Compute dGinv_dx
    Eigen::array<Eigen::Index, 3> offsets = {model.nv, model.nv, 0};
    Eigen::array<Eigen::Index, 3> extents = {model.nv, model.nv, model.nv};
    dGinv_dx.slice(offsets, extents) = HinvTHinv_dxp1;

    // Reset all penalty terms back to 0
    double penalty_val = 0;
    d_penalty_dx.setZero();
    dd_penalty_ddx.setZero();

    // compute penalties
    for (size_t bid = 0; bid < activated_penalty_vec.size(); bid++)
    {
      double penalty_val_i =
          activated_penalty_vec[bid]->compute_value(x.row(i));
      penalty_deriv_and_hessian =
          activated_penalty_vec[bid]->compute_hessian(x.row(i));
      penalty_val += penalty_val_i;
      d_penalty_dx += penalty_deriv_and_hessian.first;
      dd_penalty_ddx += penalty_deriv_and_hessian.second;
    }

    /* Build full jacobian matrices*/
    dOmega_dx_temp << dOmega_dxp1, dOmega_dxp2;
    dOmega_dxd << dOmega_dxdp1, dOmega_dxdp2;
    dOmega_dxdd << dOmega_dxddp1, dOmega_dxddp2;

    // Need last chain rule term. Do multiplication tensor with dGinv_dx
    dOmega_dx = dOmega_dx_temp - G_inv * dd_penalty_ddx -
                TensorVectorMult_3Dwith1D(d_penalty_dx, dGinv_dx, 1);

    // Build large matrices that have the jacobian at each of the ps nodes for
    // better vectorized computation
    for (int AGHF_RHS_idx = 0; AGHF_RHS_idx < 2 * model.nv; AGHF_RHS_idx++)
    {
      int jac_idx = AGHF_RHS_idx * num_ps_nodes + i;

      AGHF_jac_x.row(jac_idx) =
          Eigen::kroneckerProduct(dOmega_dx.row(AGHF_RHS_idx), kron_ones);
      AGHF_jac_xd.row(jac_idx) =
          Eigen::kroneckerProduct(dOmega_dxd.row(AGHF_RHS_idx), kron_ones);
      AGHF_jac_xdd.row(jac_idx) =
          Eigen::kroneckerProduct(dOmega_dxdd.row(AGHF_RHS_idx), kron_ones);
    }
  }

  // Compute the full jacobian using chain rule to account for mixing ps terms
  AGHF_jac = AGHF_jac_x.array() * jac_aps.array() +
             AGHF_jac_xd.array() * jac_daps.array() +
             AGHF_jac_xdd.array() * jac_ddaps.array();
}

void AghfPybindWrapper::compute_AGHF_RHS_doubleint(const Eigen::Ref<MatX> x,
                                                   const Eigen::Ref<MatX> xd,
                                                   const Eigen::Ref<MatX> xdd,
                                                   const int num_samples,
                                                   Eigen::Ref<MatX> AGHF_RHS)
{
  int N = x.cols() / 2;

  // Allocate joint configuration as well as joint velocity and torque
  xp1 = VecX::Zero(N);
  xp2 = VecX::Zero(N);
  xdp1 = VecX::Zero(N);
  xdp2 = VecX::Zero(N);
  xddp1 = VecX::Zero(N);
  xddp2 = VecX::Zero(N);

  VecX AGHF_RHS_i = VecX::Zero(2 * N);
  for (size_t i = 0; i < num_samples; i++)
  {
    // get x information at each sample
    xp1 = x.row(i).head(N);
    xp2 = x.row(i).tail(N);
    xdp1 = xd.row(i).head(N);
    xdp2 = xd.row(i).tail(N);
    xddp1 = xdd.row(i).head(N);
    xddp2 = xdd.row(i).tail(N);

    // compute AGHF_RHS
    AGHF_RHS_i << xdp2 * (-2.0) + xddp1 * 2.0,
        xddp2 * 2.0 - k * (xp2 - xdp1) * 2.0;
    AGHF_RHS.row(i) = AGHF_RHS_i;
  }
}

void AghfPybindWrapper::compute_AGHF_RHS_doubleint_vel_cons(
    const Eigen::Ref<MatX> x,
    const Eigen::Ref<MatX> xd,
    const Eigen::Ref<MatX> xdd,
    const int num_samples,
    Eigen::Ref<MatX> AGHF_RHS)
{
  int N = x.cols() / 2;
  // Allocate joint configuration as well as joint velocity and torque
  xp1 = VecX::Zero(N);
  xp2 = VecX::Zero(N);
  xdp1 = VecX::Zero(N);
  xdp2 = VecX::Zero(N);
  xddp1 = VecX::Zero(N);
  xddp2 = VecX::Zero(N);

  VecX AGHF_RHS_i = VecX::Zero(2 * N);
  for (size_t i = 0; i < num_samples; i++)
  {
    // get x information at each sample
    xp1 = x.row(i).head(N);
    xp2 = x.row(i).tail(N);
    xdp1 = xd.row(i).head(N);
    xdp2 = xd.row(i).tail(N);
    xddp1 = xdd.row(i).head(N);
    xddp2 = xdd.row(i).tail(N);

    // Calculate t2, t3, t4, and t5
    Eigen::VectorXd t2 = xp2.array() * 2.0;
    Eigen::VectorXd t3 = xp2.array() * 5.0e+1;
    Eigen::VectorXd t4 = t3.array() + 2.5e+2;
    Eigen::VectorXd t5 = t3.array() - 2.5e+2;

    // Calculate t6 and t7 using the hyperbolic tangent function (tanh)
    Eigen::VectorXd t6 = Eigen::tanh(t4.array());
    Eigen::VectorXd t7 = Eigen::tanh(t5.array());

    VecX a =
        xddp2.array() * 2.0 - k * (xp2 - xdp1).array() * 2.0 -
        k * (t6.array().pow(2) * 2.5e+1 - 2.5e+1) * (xp2.array() + 5.0).pow(2) +
        k * (t7.array().pow(2) * 2.5e+1 - 2.5e+1) * (xp2.array() - 5.0).pow(2) +
        k * (t6.array() / 2.0 - 1.0 / 2.0) * (t2.array() + 1.0e+1) -
        k * (t7.array() / 2.0 + 1.0 / 2.0) * (t2.array() - 1.0e+1);

    // compute AGHF_RHS
    AGHF_RHS_i << xdp2 * (-2.0) + xddp1 * 2.0, a;

    AGHF_RHS.row(i) = AGHF_RHS_i;
  }
}

#endif  // PYBIND_WRAPPER_CPP