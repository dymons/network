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

  auto [dataset, config] = getPathToDataSet();
  auto error = std::make_shared<network::ErrorMessages>();
  auto network = network::load(dataset, config, error);

  assert(error->empty());

  if(network) {
    std::cout << "\x1b[32m[INFO] Create network successfully.\x1b[0m" << std::endl;
    bool success = network->get()->education();

    if(success) {
      std::cout << "\x1b[32m[INFO] Network successfully educated.\x1b[0m" << std::endl;

      auto categorys_network = network->get()->getCategorys();
      std::string current_category {""};

      for (fs::recursive_directory_iterator it(dataset), end; it != end; ++it) {
        if(!boost::filesystem::is_regular_file(it->path())) {
          continue;
        }

        if(it->path().filename() == "config.json") {
          continue;
        }

        current_category = fs::path(it->path().string()).parent_path().filename().string();

        auto find_categorys_network_it = std::find_if(categorys_network.begin(), categorys_network.end(), [&](auto& category_network){
          return category_network == current_category;
        });

        if(find_categorys_network_it == categorys_network.end()) {
          continue;
        }

        std::vector<std::string> category_result = network->get()->perception(it->path().string());

        auto find_category_itr_ = std::find_if(category_result.begin(), category_result.end(), [&](auto& e){
          return e == current_category;
        });

        if(find_category_itr_ != category_result.end()) {
          std::cout << "\x1b[32m[INFO] Yes! " << it->path().filename() << " is correct.\x1b[0m" << std::endl;
        } else {
          std::cout << "\x1b[31m[ERROR] No! " << it->path().filename() << " isn't correct.\x1b[0m" << std::endl;
        }
      }

    } else {
        std::cout << "\x1b[31m[ERROR] Network successfully educated.\x1b[0m" << std::endl;
        return 1;
      }
  }

  return 0;
}