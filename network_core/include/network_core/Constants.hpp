#pragma once

#ifndef NETWORK_CONSTANTS_HPP_
#define NETWORK_CONSTANTS_HPP_

#include <vector>
#include <memory>

namespace network {
  class Constants {
    public:
      static constexpr double WEIGHT_SYNAPSES_DEFAULT { 0.50 };
      static constexpr double OUTPUT_NEURON_DEFAULT   { 0.50 };
      static constexpr double LEARNING_RATE_DEFAULT   { 0.10 };
      static constexpr double ERROR_DEFAULT           { 0.50 };
      static constexpr double ESP                     { 0.01 };
      static constexpr double TRESHOLD_SINGLE_JUMP    { 10.0 };
      static constexpr double DEGREE_FUNCTION         { 1.00 };
    };
} /* namespace network */
#endif /* NETWORK_CONSTANTS_HPP_ */