#pragma once

#ifndef NETWORK_IO_HPP_
#define NETWORK_IO_HPP_

#include "network_core/Forward.hpp"
#include "network_core/Network.hpp"
#include "network_core/Exeption.hpp"

// STL
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <optional>

// Boost
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/adaptor/indexed.hpp>

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

namespace network {
  using ErrorMessages = typename std::vector<std::string>;
  /**
   * @brief Load the neural network architecture.
   * @param config Path to the neural network configuration.
   * @return Returns a Network type object. The pointer is valid, otherwise an exception will be thrown.
   * @throws Network::IOError If the file was not found or the content of the file is incorrect.
   */
  std::optional<NetworkUPtr> load(const std::string& config, std::shared_ptr<ErrorMessages> errors = std::make_shared<ErrorMessages>());

  /**
   * @brief Load the neural network architecture.
   * @param config Path to the neural network configuration.
   * @param dataset Path to the neural network dataset.
   * @return Returns a Network type object. The pointer is valid, otherwise an exception will be thrown.
   * @throws Network::IOError If the file was not found or the content of the file is incorrect.
   */
  std::optional<NetworkUPtr> load(const std::string& dataset, const std::string& config, std::shared_ptr<ErrorMessages> errors = std::make_shared<ErrorMessages>());
} // namespace network
#endif // NETWORK_IO_HPP_