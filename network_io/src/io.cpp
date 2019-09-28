#include <network_io/io.hpp>

namespace network {
  std::optional<NetworkUPtr> load(const std::string& config, std::shared_ptr<ErrorMessages> errors)
  {
    if (!fs::exists(fs::path(config))) {
      errors->push_back("Could not find config file " + config);
      throw FileNotFoundError("Could not find config file " + config);
      return {};
    }

    pt::ptree root;
    pt::read_json(config, root);

    InputLayer  input  { };
    HiddenLayer hidden { };
    OutputLayer output { };

    auto getData = [&root](auto& layer, auto&& topic) mutable -> decltype(auto) {
      try {
        const pt::ptree data = root.get_child(topic);
        auto exist = data.empty();

        if(exist) {
          int size_ = root.get<int>(std::forward<decltype(topic)>(topic));
          if(size_<0) throw ParseError("size neurons in layer is < 0");
          layer.create(size_, std::conditional_t<is_single_layer<decltype(layer)>::value, std::true_type, std::false_type>{});
        } else {
          for(const auto& row : data | boost::adaptors::indexed(0)) {
            int size_ = row.value().second.get_value<int>();
            if(size_<0) throw ParseError("size neurons in layer is < 0");
            layer.create(std::move(size_), std::conditional_t<is_single_layer<decltype(layer)>::value, std::true_type, std::false_type>{});
          }
        }
      } catch(const pt::ptree_bad_path& e) {
        throw ParseError(e.what());
      }
    };

    try {
      getData(input, "topology.layers.input");
      getData(hidden, "topology.layers.hidden");
      getData(output, "topology.layers.output");
    } catch(const ParseError& e) {
      errors->push_back(e.what());
      return {};
    }

    auto network = std::make_unique<Network>(std::move(input), std::move(hidden), std::move(output));
    return network;
  }

  std::optional<NetworkUPtr> load(const std::string& dataset, const std::string& config, std::shared_ptr<ErrorMessages> errors)
  {
    std::optional<NetworkUPtr> network;

    if (!fs::exists(fs::path(dataset))) {
      errors->push_back("Could not find dataset folder " + dataset);
      throw FolderNotFoundError("Could not find dataset folder " + dataset);
      return network;
    }

    if (!fs::exists(fs::path(config))) {
      errors->push_back("Could not find config file " + config);
      throw FileNotFoundError("Could not find config file " + config);
      return network;
    }

    /* Считываем информацию о dimensions и category. */
    pt::ptree root;
    pt::read_json(config, root);

    const int width = root.get<int>("dimensions.width");
    const int height = root.get<int>("dimensions.height");

    std::vector<std::string> categorys { };
    const pt::ptree categorys_tree = root.get_child("category");

    if((categorys_tree.empty())||(width<0)||(height<0)) throw ParseError("config file is error: " + config);

    for(const auto& row : categorys_tree) {
      categorys.emplace_back(std::move(row.second.get_value<std::string>()));
    }

    /* Получаем информацию о входном слое и выходном. */
    const int size_nerons_input_  = root.get<int>("topology.layers.input");
    const int size_nerons_output_ = root.get<int>("topology.layers.output");

    if((size_nerons_input_<0)||(size_nerons_output_<0)) throw ParseError("config file is error: " + config);

    InputLayer  input  { };
    HiddenLayer hidden { };
    OutputLayer output { };

    {
      const int size_dimensions_ = width*height;
      if((size_dimensions_!=size_nerons_input_)) {
        std::cout << "\x1b[33m[WARN] dimensions.width * dimensions.height != topology.layers.input size, set to " << size_dimensions_ << "\x1b[0m" << std::endl;
      }

      input.create(size_dimensions_, std::conditional_t<is_single_layer<decltype(input)>::value, std::true_type, std::false_type>{});
    }

    {
      const int size_categorys_ = categorys.size();
      if(size_nerons_output_ < size_categorys_) {
        std::cout << "\x1b[33m[WARN] category.size() != topology.layers.output size, set to " << size_categorys_ << "\x1b[0m" << std::endl;
      }

      output.create(size_categorys_, std::conditional_t<is_single_layer<decltype(output)>::value, std::true_type, std::false_type>{});
    }

    /* Получаем информацию о скрытом слое. */
    {
      auto getData = [&root](auto& layer, auto&& topic) mutable -> decltype(auto) {
        try {
          const pt::ptree data = root.get_child(topic);
          auto exist = data.empty();

          if(exist) {
            int size_ = root.get<int>(std::forward<decltype(topic)>(topic));
            if(size_<0) throw ParseError("size neurons in layer is < 0");
            layer.create(size_, std::conditional_t<is_single_layer<decltype(layer)>::value, std::true_type, std::false_type>{});
          } else {
            for(const auto& row : data | boost::adaptors::indexed(0)) {
              int size_ = row.value().second.get_value<int>();
              if(size_<0) throw ParseError("size neurons in layer is < 0");
              layer.create(std::move(size_), std::conditional_t<is_single_layer<decltype(layer)>::value, std::true_type, std::false_type>{});
            }
          }
        } catch(const pt::ptree_bad_path& e) {
          throw ParseError(e.what());
        }
      };

      try {
        getData(hidden, "topology.layers.hidden");
      } catch(const ParseError& e) {
        errors->push_back(e.what());
        return network;
      }
    }

    /* Получаем количество эпох на обучение. */
    std::size_t epoch_ = root.get<std::size_t>("epoch");

    if(network = std::make_unique<Network>(std::move(input), std::move(hidden), std::move(output))) {
      (*network)->setDataset(dataset);
      (*network)->setCategorys(std::move(categorys));
      (*network)->setEpoch(std::move(epoch_));
    }

    return network;
  }
} // namespace network