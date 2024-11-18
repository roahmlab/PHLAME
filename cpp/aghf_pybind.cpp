#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include "PybindWrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(aghf_pybind, m)
{
  m.doc() = "pybind11 aghf_pybind plugin";

  py::class_<AghfPybindWrapper>(m, "AghfPybindWrapper")
      .def(py::init<const std::string&, double, Eigen::Ref<Eigen::VectorXd>>())
      .def(py::init<double, int>())
      .def(py::init<const std::string&, double, Eigen::Ref<Eigen::VectorXd>,
                    Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>>())
      .def("CRBA_D", &AghfPybindWrapper::CRBA_D)
      .def("CRBA_2D", &AghfPybindWrapper::CRBA_2D)
      .def("compute_PSAGHF_jac", &AghfPybindWrapper::compute_PSAGHF_jac)
      .def("compute_AGHF_RHS_doubleint",
           &AghfPybindWrapper::compute_AGHF_RHS_doubleint)
      .def("compute_AGHF_RHS_doubleint_vel_cons",
           &AghfPybindWrapper::compute_AGHF_RHS_doubleint_vel_cons)
      .def("compute_AGHF_RHS", &AghfPybindWrapper::compute_AGHF_RHS)
      .def("set_activated_state_limits",
           &AghfPybindWrapper::set_activated_state_limits)
      .def("set_activated_obstacles",
           &AghfPybindWrapper::set_activated_obstacles);
}