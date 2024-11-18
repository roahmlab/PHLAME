// #ifndef INEQUALITY_CONSTRAINT_H
// #define INEQUALITY_CONSTRAINT_H

// #include <Eigen/Dense>
// #include <iostream>

// class InequalityConstraint {
// public:
//     using VecX = Eigen::VectorXd;
//     using MatX = Eigen::MatrixXd;

//     InequalityConstraint() = default;

//     ~InequalityConstraint() = default;

//     // For states
//     virtual double compute_value(double x, double x_bound) = 0;
//     virtual double compute_derivative(double x, double x_bound) = 0;
    
//     // For sphere obstacles
//     virtual double compute_value(double R, const VecX& FK, const VecX& center) = 0;
//     virtual VecX compute_derivative(const VecX& FK, const VecX& center, const MatX& frame_position_jacobian) = 0;

// };

// #endif // INEQUALITY_CONSTRAINT_H