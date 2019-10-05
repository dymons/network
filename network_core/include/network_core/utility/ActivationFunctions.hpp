#pragma once

#ifndef NETWORK_ACTIVATION_FUNCTIONS_HPP_
#define NETWORK_ACTIVATION_FUNCTIONS_HPP_

#include "network_core/Constants.hpp"

#include <cmath>
#include <algorithm>
#include <functional>

namespace network {
namespace computation {
  inline double differential(std::function<double(double)> t_func, const double& t_x) noexcept
  {
    return (t_func(t_x + Constants::ESP) - t_func(t_x)) / Constants::ESP;
  }

  inline double sigmoid(const double& t_x) noexcept
  {
    return 1 / (1 + exp(Constants::DEGREE_FUNCTION * (-t_x)));
  }

  inline double single_jump(const double& t_x) noexcept
  {
    return t_x >= Constants::TRESHOLD_SINGLE_JUMP ? 1.0 : 0.0;
  }
} // namespace computation
} // namespace network
#endif // NETWORK_ACTIVATION_FUNCTIONS_HPP_