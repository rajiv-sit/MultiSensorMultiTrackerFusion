#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tracker {

  using Vec = Eigen::VectorXd;
  using Mat = Eigen::MatrixXd;

} // namespace tracker

