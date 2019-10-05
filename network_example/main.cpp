#include "network_core/Network.hpp"
#include "network_io/io.hpp"

// Boost
#include <boost/filesystem.hpp>

// STL
#include <optional>
#include <tuple>
#include <deque>
#include <memory>
#include <array>
#include <iostream>
#include <map>

namespace fs = boost::filesystem;

auto main(void) -> int
{
  auto getPathToDataSet = []() {
    std::string parent_path_ { fs::current_path().parent_path().string() };
    std::string dataset { parent_path_ + "/../network_example/dataset" };
    std::string config  { parent_path_ + "/../network_example/dataset/config.json" };
    return std::make_tuple(dataset, config);
  };

  std::string config {""}, dataset {""};
  std::tie(dataset, config) = getPathToDataSet();
  auto error = std::make_shared<network::ErrorMessages>();
  auto network = network::load(dataset, config, error);

  assert(error->empty());

  if(network) {
    std::cout << "\x1b[32m[INFO] Create network successfully.\x1b[0m" << std::endl;
    bool success = network->get()->education();

    if(success) {
      std::cout << "\x1b[32m[INFO] Network successfully educated.\x1b[0m" << std::endl;
    }
  }

  return 0;
}