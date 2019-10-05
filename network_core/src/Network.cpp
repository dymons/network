#include "network_core/Network.hpp"

namespace network {
  Network::Network(const InputLayer& t_input, const HiddenLayer& t_hidden, const OutputLayer& t_output) 
  : m_input_layer_(t_input), m_hidden_layer_(t_hidden), m_output_layer_(t_output)
  {
    // Check for a specific combination Network. If not satisfied, the network will not work correctly.
    assert( is_single_layer<decltype(m_input_layer_)>::value);
    assert(!is_single_layer<decltype(m_hidden_layer_)>::value);
    assert( is_single_layer<decltype(m_output_layer_)>::value);

    // Check initialize layers.
    if(!m_input_layer_.m_layers || !m_hidden_layer_.m_layers || !m_output_layer_.m_layers) {
      throw NotInitializeError("Not initialize layers in Network.");
    }

    // Create link between back hidden layer and output layer.
    if(auto layer_output_ptr_ = std::get_if<OutputLayer::PrimitiveTPtr>(&(*m_output_layer_.m_layers))) {
      if(auto layers_hidden_ptr_ = std::get_if<HiddenLayer::Layer>(&(*m_hidden_layer_.m_layers))) {
        if(!layer_output_ptr_->get() || layers_hidden_ptr_->empty()) {
          throw NotInitializeError("Not initialize output or hidden layer.");
        }

        (*layer_output_ptr_)->connect(layers_hidden_ptr_->back());

#ifdef NDEBUG
        const std::size_t size_hidden_layer_ = layers_hidden_ptr_->back()->size();
        for(const auto& neuron : *(*layer_output_ptr_)) {
          assert(neuron->size() == size_hidden_layer_);
        }
#endif
      }
    }

    // Create links between hidden layers.
    if(auto layers_hidden_ptr_ = std::get_if<HiddenLayer::Layer>(&(*m_hidden_layer_.m_layers))) {
      if(layers_hidden_ptr_->empty() ) {
        throw NotInitializeError("Not initialize hidden layer.");
      }

      const std::size_t size_hidden_layers_ = layers_hidden_ptr_->size() - 1;
      for(std::size_t i = 0; i < size_hidden_layers_; ++i) {
        layers_hidden_ptr_->at(i+1)->connect(layers_hidden_ptr_->at(i));
      }

#ifdef NDEBUG
      for(std::size_t i = 0; i < size_hidden_layers_; ++i) {
        const std::size_t size_hidden_layer_ = layers_hidden_ptr_->at(i)->size();
        for(const auto& neuron : *(layers_hidden_ptr_->at(i+1))) {
          assert(neuron->size() == size_hidden_layer_);
        }
      }
#endif
    }

    // Create link between input layer and front hidden layer.
    if(auto layer_input_ptr_ = std::get_if<InputLayer::PrimitiveTPtr>(&(*m_input_layer_.m_layers))) {
      if(auto layers_hidden_ptr_ = std::get_if<HiddenLayer::Layer>(&(*m_hidden_layer_.m_layers))) {
        if(!layer_input_ptr_->get() || layers_hidden_ptr_->empty()) {
          throw NotInitializeError("Not initialize input or hidden layer.");
        }

        layers_hidden_ptr_->front()->connect(*layer_input_ptr_);

#ifdef NDEBUG
        const std::size_t size_input_layer_ = (*layer_input_ptr_)->size();
        for(const auto& neuron : *layers_hidden_ptr_->front()) {
          assert(neuron->size() == size_input_layer_);
        }
#endif
      }
    }
  }

  void Network::setDataset(const std::string& t_dataset) noexcept
  {
    m_dataset = t_dataset;
  }

  void Network::setCategorys(const std::vector<std::string>& t_categorys) noexcept
  {
    m_categorys.clear();
    m_categorys.resize(t_categorys.size());
    std::copy(t_categorys.begin(), t_categorys.end(), m_categorys.begin());
  }

  void Network::setEpoch(const std::size_t& t_epoch) noexcept
  {
    m_epoch.emplace(t_epoch);
  }

  bool Network::education()
  {
    bool status { true };

    if(m_dataset.empty()) {
      throw FolderNotFoundError("Could not find dataset folder " + m_dataset);
      return (status = false);
    }

    if(!fs::exists(fs::path(m_dataset))) {
      throw FolderNotFoundError("Could not find dataset folder " + m_dataset);
      return (status = false);
    }

    if(!m_epoch || m_categorys.empty()) {
      throw NotInitializeError("Network isn't initialize.");
      return (status = false);
    }

    auto buffer = std::make_unique<std::unordered_map<std::string, std::vector<std::string>>>();

    // Save image paths to buffer.
    for (fs::recursive_directory_iterator it(m_dataset), end; it != end; ++it) {
      // Folder is a category.
      if(!boost::filesystem::is_regular_file(it->path())) {
        // Check that it is among our filters.
        auto exist_category_ = std::find_if(m_categorys.begin(), m_categorys.end(),
        [folder = it->path().filename().string()](auto& e){return e == folder;}) != m_categorys.end();

        if(exist_category_) {
          buffer->emplace(it->path().filename().string(), std::vector<std::string>());
        } else {
#ifdef NDEBUG
          std::cout << "\x1b[33m[WARN] This folder not found " << it->path().filename().string() << " in categorys: ";
          std::copy(m_categorys.begin(), m_categorys.end(), std::ostream_iterator<std::string>(std::cout, " "));
          std::cout << "\x1b[0m" << std::endl;
#endif
        }

        continue;
      } else {
        // So the file is checked for contents in the directory.
        auto exist_category_ = std::find_if(m_categorys.begin(), m_categorys.end(),
        [folder = it->path().parent_path().filename().string()](auto& e){return e == folder;}) != m_categorys.end();

        if(!exist_category_) {
          continue;
        }
      }

      // Check that the transferred file matches the formats being processed.
      const auto exist_extension_ = std::find_if(m_format.begin(), m_format.end(),
      [ex = it->path().extension()](auto e){return e == ex;}) != m_format.end();

      if (exist_extension_) {
        // TODO: Check for the desired image size.
        buffer->at(it->path().parent_path().filename().string()).push_back(it->path().filename().string());
      } else {
#ifdef NDEBUG
        std::cout << "\x1b[33m[WARN] image " << it->path().filename().string() << " is not correct format: ";
        std::copy(m_format.begin(), m_format.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << "\x1b[0m" << std::endl;
#endif
      }
    }

    // Get pointers on layers
    auto layer_input_ptr_   = std::get_if<InputLayer::PrimitiveTPtr>(&(*m_input_layer_.m_layers));
    auto layers_hidden_ptr_ = std::get_if<HiddenLayer::Layer>(&(*m_hidden_layer_.m_layers));
    auto layer_output_ptr_  = std::get_if<OutputLayer::PrimitiveTPtr>(&(*m_output_layer_.m_layers));

    if(buffer->size() != (*layer_output_ptr_)->size()) {
      throw std::out_of_range("");
      return (status = false);
    }

    // Set category for output layer
    for(auto&& row : *buffer | boost::adaptors::indexed(0)) {
      auto&& [category, collage] = row.value();
      (*layer_output_ptr_)->setCategory(static_cast<std::size_t>(row.index()), category);
    }

    for(std::size_t i = 0 ; i < (*m_epoch); ++i) {
      for(auto&& [category, collage] : *buffer) {
        for(auto& image : collage) {
          // Get image.
          cv::Mat data_input_ = cv::imread(m_dataset + '/' + category + '/' + image);

          // Check valid image.
          if(data_input_.empty()) { continue; }
          if(data_input_.type() != CV_8UC3) { /* TODO: convert */ }

          // Supply values to the input layer
          for(int r = 0; r < data_input_.rows; ++r) {
            for(int c = 0; c < data_input_.cols; ++c) {
              (*layer_input_ptr_)->set(static_cast<std::size_t>(c+(r*data_input_.cols)),
              static_cast<double>(data_input_.at<unsigned char>(r,c))/255);
            }
          }

          // Direct distribution Network
          {
            for(const auto& layer : *layers_hidden_ptr_) {
              layer.get()->calculate();
            }

            (*layer_output_ptr_)->calculate();
          }

// #ifdef NDEBUG
//           std::cout << "---" << category << std::endl;
//           for(const auto& neuron : **layer_output_ptr_) {
//             std::cout << neuron->getCategory() << ", value: " << neuron->getOutputValue() << ", Error: " << neuron->getError() << std::endl;
//           }
// #endif

          // Back distribution Network
          {
            // For output layer
            (*layer_output_ptr_)->update(category);

            // For hidden layer
            layers_hidden_ptr_->back()->update(*(*layer_output_ptr_));

            const std::size_t size_hidden_layers_ = layers_hidden_ptr_->size() - 1;
            for(std::size_t j = size_hidden_layers_; j != 0; --j) {
              layers_hidden_ptr_->at(j-1)->update(*layers_hidden_ptr_->at(j));
            }
          }

          // Update weight
          {
            for(const auto& layer : *layers_hidden_ptr_) {
              layer.get()->updateWeight();
            }

            (*layer_output_ptr_)->updateWeight();
          }
        }
      }
    }

    for(std::size_t i = 0 ; i < (*m_epoch); ++i) {
      for(auto&& [category, collage] : *buffer) {
        for(auto& image : collage) {
          // Get image.
          cv::Mat data_input_ = cv::imread(m_dataset + '/' + category + '/' + image);

          // Check valid image.
          if(data_input_.empty()) { continue; }
          if(data_input_.type() != CV_8UC3) { /* TODO: convert */ }

          // Supply values to the input layer
          for(int r = 0; r < data_input_.rows; ++r) {
            for(int c = 0; c < data_input_.cols; ++c) {
              (*layer_input_ptr_)->set(static_cast<std::size_t>(c+(r*data_input_.cols)),
              static_cast<double>(data_input_.at<unsigned char>(r,c))/255);
            }
          }

          // Direct distribution Network
          {
            for(const auto& layer : *layers_hidden_ptr_) {
              layer.get()->calculate();
            }

            (*layer_output_ptr_)->calculate();
          }

#ifdef NDEBUG
          std::cout << "---" << category << std::endl;
          for(const auto& neuron : **layer_output_ptr_) {
            // std::cout << neuron->getCategory() << ", value: " << neuron->getOutputValue() << ", Error: " << neuron->getError() << std::endl;
            std::cout << ", value: " << neuron->getOutputValue() << ", Error: " << neuron->getError() << std::endl;
          }
#endif

        }
      }
    }

    return status;
  }
} // namespace network