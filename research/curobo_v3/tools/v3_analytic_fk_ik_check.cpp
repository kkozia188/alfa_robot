#include "alfa_robot_analytic_ik/v3_redundant_analytic_ik.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

int main()
{
  using namespace alfa_robot::analytic_ik;
  std::cout << std::setprecision(17);
  int sample_id;
  int arm_side;
  while (std::cin >> sample_id >> arm_side) {
    V3RedundantArmAnalyticIk solver(
      arm_side == 0 ? V3RedundantArmModel::V322Left : V3RedundantArmModel::V322Right);
    V3RedundantIkRequest request;
    for (double& value : request.seed) std::cin >> value;
    for (int row = 0; row < 4; ++row)
      for (int column = 0; column < 4; ++column)
        std::cin >> request.target_in_arm_base.matrix()(row, column);
    const auto actual = solver.forwardInArmBase(request.seed);
    const double position_error =
      (actual.translation() - request.target_in_arm_base.translation()).norm();
    const double rotation_error = Eigen::AngleAxisd(
      actual.linear().transpose() * request.target_in_arm_base.linear()).angle();
    request.swivel_angle = solver.swivelAngle(request.seed);
    request.position_tolerance = 1e-6;
    request.orientation_tolerance = 1e-6;
    const auto started = std::chrono::steady_clock::now();
    const auto solutions = solver.solveInArmBase(request);
    const double solve_us = std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - started).count();
    std::cout << "sample " << sample_id << ' ' << arm_side << ' '
              << position_error << ' ' << rotation_error << ' '
              << solutions.size() << ' ' << solve_us << '\n';
    for (const auto& solution : solutions) {
      std::cout << "solution " << sample_id << ' ' << arm_side;
      for (const double value : solution.joints) std::cout << ' ' << value;
      std::cout << '\n';
    }
  }
}
