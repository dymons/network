#pragma once

#ifndef NETWORK_FORWARD_HPP_
#define NETWORK_FORWARD_HPP_

#include <memory>
#include <vector>

namespace network
{
  /* Network */
  class Network;
  using NetworkPtr = std::shared_ptr<Network>;
  using NetworkUPtr = std::unique_ptr<Network>;
  using NetworkConstPtr = std::shared_ptr<const Network>;
  using NetworkConstUPtr = std::unique_ptr<const Network>;

  /* Neuron */
  class Neuron;
  using NeuronPtr = std::shared_ptr<Neuron>;
  using NeuronUPtr = std::unique_ptr<Neuron>;
  using NeuronConstPtr = std::shared_ptr<const Neuron>;
  using NeuronConstUPtr = std::unique_ptr<const Neuron>;

  using Id = int64_t;
  using Ids = std::vector<Id>;
} // namespace network
#endif // NETWORK_FORWARD_HPP_