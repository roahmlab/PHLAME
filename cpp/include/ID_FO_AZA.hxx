//
// Copyright (c) 2017-2020 CNRS INRIA
//

#ifndef __pinocchio_ID_FO_AZA_hxx__
#define __pinocchio_ID_FO_AZA_hxx__

#include <stdio.h>

#include <iostream>

#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/multibody/visitor.hpp"

namespace pinocchio
{

template <typename Scalar,
          int Options,
          template <typename, int>
          class JointCollectionTpl,
          typename ConfigVectorType,
          typename TangentVectorType1,
          typename TangentVectorType2>
struct ComputeID_FO_AZAForwardStep
    : public fusion::JointUnaryVisitorBase<
          ComputeID_FO_AZAForwardStep<Scalar,
                                      Options,
                                      JointCollectionTpl,
                                      ConfigVectorType,
                                      TangentVectorType1,
                                      TangentVectorType2> >
{
  typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
  typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

  typedef boost::fusion::vector<const Model &,
                                Data &,
                                const ConfigVectorType &,
                                const TangentVectorType1 &,
                                const TangentVectorType2 &>
      ArgsType;

  template <typename JointModel>
  static void algo(const JointModelBase<JointModel> &jmodel,
                   JointDataBase<typename JointModel::JointDataDerived> &jdata,
                   const Model &model,
                   Data &data,
                   const Eigen::MatrixBase<ConfigVectorType> &q,
                   const Eigen::MatrixBase<TangentVectorType1> &v,
                   const Eigen::MatrixBase<TangentVectorType2> &a)
  {
    typedef typename Model::JointIndex JointIndex;
    typedef typename Data::Motion Motion;
    typedef typename Data::Inertia Inertia;

    const JointIndex &i = jmodel.id();
    const JointIndex &parent = model.parents[i];
    Motion &oa = data.oa[i];
    Motion &vJ = data.vJ[i];

    jmodel.calc(jdata.derived(), q.derived(), v.derived());

    data.liMi[i] = model.jointPlacements[i] * jdata.M();

    if (parent > 0)
    {
      data.oMi[i] = data.oMi[parent] * data.liMi[i];
      oa = data.oa[parent];
    }
    else
    {
      data.oMi[i] = data.liMi[i];
      oa.setZero();
      // oa = -model.gravity;
    }

    typedef typename SizeDepType<JointModel::NV>::template ColsReturn<
        typename Data::Matrix6x>::Type ColsBlock;
    ColsBlock J_cols = jmodel.jointCols(data.J);
    ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
    // J
    J_cols.noalias() = data.oMi[i].act(jdata.S());
    // ddJ
    motionSet::motionAction(oa, J_cols, ddJ_cols);
    // velocity and accelaration finishing
    oa += data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(a));
    // oa += data.oMi[i].act( jdata.S() * jmodel.jointVelocitySelector(a) +
    // jdata.c() );

    // Composite rigid body inertia
    Inertia &oY = data.oYcrb[i];

    oY = data.oMi[i].act(model.inertias[i]);
    data.of[i] = oY * oa;
  }
};

template <typename Scalar,
          int Options,
          template <typename, int>
          class JointCollectionTpl,
          typename MatrixType1>
struct ComputeID_FO_AZABackwardStep
    : public fusion::JointUnaryVisitorBase<
          ComputeID_FO_AZABackwardStep<Scalar,
                                       Options,
                                       JointCollectionTpl,
                                       MatrixType1> >
{
  typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
  typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

  typedef boost::fusion::vector<const Model &, Data &, const MatrixType1 &>
      ArgsType;

  template <typename JointModel>
  static void algo(const JointModelBase<JointModel> &jmodel,
                   const Model &model,
                   Data &data,
                   const Eigen::MatrixBase<MatrixType1> &rnea_partial_dq)
  {
    typedef typename Model::JointIndex JointIndex;

    const JointIndex &i = jmodel.id();
    const JointIndex &parent = model.parents[i];

    typedef typename SizeDepType<JointModel::NV>::template ColsReturn<
        typename Data::Matrix6x>::Type ColsBlock;

    ColsBlock J_cols = jmodel.jointCols(data.J);
    ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);

    ColsBlock tmp1 = jmodel.jointCols(data.Ftmp1);
    ColsBlock tmp3 = jmodel.jointCols(data.Ftmp3);

    const Eigen::Index joint_idx = (Eigen::Index)jmodel.idx_v();
    const Eigen::Index joint_dofs = (Eigen::Index)jmodel.nv();
    const Eigen::Index subtree_dofs = (Eigen::Index)data.nvSubtree[i];
    const Eigen::Index successor_idx = joint_idx + joint_dofs;
    const Eigen::Index successor_dofs = subtree_dofs - joint_dofs;

    Inertia &oYcrb = data.oYcrb[i];  // IC in ground frame

    MatrixType1 &rnea_partial_dq_ =
        PINOCCHIO_EIGEN_CONST_CAST(MatrixType1, rnea_partial_dq);

    motionSet::inertiaAction(oYcrb, J_cols, tmp1);
    motionSet::inertiaAction<ADDTO>(oYcrb, ddJ_cols, tmp3);
    motionSet::act<ADDTO>(J_cols, data.of[i], tmp3);

    if (successor_dofs > 0)
    {
      rnea_partial_dq_
          .block(joint_idx, successor_idx, joint_dofs, successor_dofs)
          .noalias() = J_cols.transpose() *
                       data.Ftmp3.middleCols(successor_idx, successor_dofs);
    }
    rnea_partial_dq_.block(joint_idx, joint_idx, subtree_dofs, joint_dofs)
        .noalias() =
        data.Ftmp1.middleCols(joint_idx, subtree_dofs).transpose() * ddJ_cols;

    if (parent > 0)
    {
      data.oYcrb[parent] += data.oYcrb[i];
      data.of[parent] += data.of[i];
    }
  }
};

template <typename Scalar,
          int Options,
          template <typename, int>
          class JointCollectionTpl,
          typename ConfigVectorType,
          typename TangentVectorType1,
          typename TangentVectorType2,
          typename MatrixType1>
inline void computeID_FO_AZA(
    const ModelTpl<Scalar, Options, JointCollectionTpl> &model,
    DataTpl<Scalar, Options, JointCollectionTpl> &data,
    const Eigen::MatrixBase<ConfigVectorType> &q,
    const Eigen::MatrixBase<TangentVectorType1> &v,
    const Eigen::MatrixBase<TangentVectorType2> &a,
    const Eigen::MatrixBase<MatrixType1> &rnea_partial_dq)
{
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      q.size(), model.nq,
      "The joint configuration vector is not of right size");
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      v.size(), model.nv, "The joint velocity vector is not of right size");
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      a.size(), model.nv, "The joint acceleration vector is not of right size");
  PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dq.cols(), model.nv);
  PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dq.rows(), model.nv);
  assert(model.check(data) && "data is not consistent with model.");

  typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
  typedef typename Model::JointIndex JointIndex;

  typedef ComputeID_FO_AZAForwardStep<Scalar, Options, JointCollectionTpl,
                                      ConfigVectorType, TangentVectorType1,
                                      TangentVectorType2>
      Pass1;
  for (JointIndex i = 1; i < (JointIndex)model.njoints; ++i)
  {
    Pass1::run(model.joints[i], data.joints[i],
               typename Pass1::ArgsType(
                   model, data, q.derived(), v.derived(), a.derived()));
  }

  data.Ftmp1.setZero();
  data.Ftmp3.setZero();

  typedef ComputeID_FO_AZABackwardStep<Scalar, Options, JointCollectionTpl,
                                       MatrixType1>
      Pass2;
  for (JointIndex i = (JointIndex)(model.njoints - 1); i > 0; --i)
  {
    Pass2::run(model.joints[i],
               typename Pass2::ArgsType(
                   model, data,
                   PINOCCHIO_EIGEN_CONST_CAST(MatrixType1, rnea_partial_dq)));
  }
}

template <typename Scalar,
          int Options,
          template <typename, int>
          class JointCollectionTpl,
          typename ConfigVectorType,
          typename TangentVectorType1,
          typename TangentVectorType2,
          typename MatrixType1>
inline void computeID_FO_AZA(
    const ModelTpl<Scalar, Options, JointCollectionTpl> &model,
    DataTpl<Scalar, Options, JointCollectionTpl> &data,
    const Eigen::MatrixBase<ConfigVectorType> &q,
    const Eigen::MatrixBase<TangentVectorType1> &v,
    const Eigen::MatrixBase<TangentVectorType2> &a,
    const container::aligned_vector<ForceTpl<Scalar, Options> > &fext,
    const Eigen::MatrixBase<MatrixType1> &rnea_partial_dq)
{
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      q.size(), model.nq,
      "The joint configuration vector is not of right size");
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      v.size(), model.nv, "The joint velocity vector is not of right size");
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      a.size(), model.nv, "The joint acceleration vector is not of right size");
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      fext.size(), (size_t)model.njoints,
      "The size of the external forces is not of right size");
  PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dq.cols(), model.nv);
  PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dq.rows(), model.nv);

  assert(model.check(data) && "data is not consistent with model.");

  typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
  typedef typename Model::JointIndex JointIndex;

  typedef ComputeID_FO_AZAForwardStep<Scalar, Options, JointCollectionTpl,
                                      ConfigVectorType, TangentVectorType1,
                                      TangentVectorType2>
      Pass1;
  for (JointIndex i = 1; i < (JointIndex)model.njoints; ++i)
  {
    Pass1::run(model.joints[i], data.joints[i],
               typename Pass1::ArgsType(
                   model, data, q.derived(), v.derived(), a.derived()));
    data.of[i] -= data.oMi[i].act(fext[i]);
  }

  typedef ComputeID_FO_AZABackwardStep<Scalar, Options, JointCollectionTpl,
                                       MatrixType1>
      Pass2;
  for (JointIndex i = (JointIndex)(model.njoints - 1); i > 0; --i)
  {
    Pass2::run(model.joints[i],
               typename Pass2::ArgsType(
                   model, data,
                   PINOCCHIO_EIGEN_CONST_CAST(MatrixType1, rnea_partial_dq)));
  }
}

}  // namespace pinocchio

#endif  // ifndef
