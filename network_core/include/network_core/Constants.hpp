#pragma once

#ifndef NETWORK_CONSTANTS_HPP_
#define NETWORK_CONSTANTS_HPP_

namespace network {
  class Constants {
    public:
      static constexpr inline double WEIGHT_SYNAPSES_DEFAULT { 0.50 }; // Set value on default when create synapse between neuron
      static constexpr inline double OUTPUT_NEURON_DEFAULT   { 0.50 }; // Set output value neuron on default when create neuron with default constructor
      static constexpr inline double LEARNING_RATE_DEFAULT   { 0.10 }; // Coefficient for neural network training
      static constexpr inline double ERROR_DEFAULT           { 0.50 }; // Set error value neuron on default when create neuron with default constructor
      static constexpr inline double ESP                     { 0.01 }; // Differentiation step
      static constexpr inline double TRESHOLD_SINGLE_JUMP    { 10.0 }; // Coefficient for single_jump function
      static constexpr inline double DEGREE_FUNCTION         { 1.00 }; // Coefficient for sigmoid function
    };
} /* namespace network */
#endif /* NETWORK_CONSTANTS_HPP_ */