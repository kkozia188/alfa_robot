#include "alfa_robot_analytic_ik/v3_redundant_analytic_ik.hpp"
#include <algorithm>
#include <array>

using alfa_robot::analytic_ik::V3RedundantArmAnalyticIk;
using alfa_robot::analytic_ik::V3RedundantArmModel;
using alfa_robot::analytic_ik::V3RedundantIkRequest;

extern "C" double v3_swivel(int side, const double* joints)
{
  V3RedundantArmAnalyticIk solver(
    side == 0 ? V3RedundantArmModel::V322Left : V3RedundantArmModel::V322Right);
  std::array<double, 7> values;
  std::copy_n(joints, 7, values.begin());
  return solver.swivelAngle(values);
}

extern "C" int v3_solve(int side, const double* matrix, double swivel,
                         const double* seed, double* output, int capacity)
{
  V3RedundantArmAnalyticIk solver(
    side == 0 ? V3RedundantArmModel::V322Left : V3RedundantArmModel::V322Right);
  V3RedundantIkRequest request;
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 4; ++column)
      request.target_in_arm_base.matrix()(row, column) = matrix[row * 4 + column];
  request.swivel_angle = swivel;
  std::copy_n(seed, 7, request.seed.begin());
  request.position_tolerance = 1e-6;
  request.orientation_tolerance = 1e-6;
  const auto solutions = solver.solveInArmBase(request);
  const int count = std::min(capacity, static_cast<int>(solutions.size()));
  for (int index = 0; index < count; ++index)
    std::copy(solutions[index].joints.begin(), solutions[index].joints.end(), output + 7 * index);
  return count;
}
