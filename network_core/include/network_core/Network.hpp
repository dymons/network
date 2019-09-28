#pragma once

#ifndef NETWORK_NETWORK_HPP_
#define NETWORK_NETWORK_HPP_

#include "network_core/primitives/Primitive.hpp"
#include "network_core/primitives/Neuron.hpp"
#include "network_core/Exeption.hpp"
#include "network_core/Forward.hpp"

// STL
#include <array>
#include <vector>
#include <memory>
#include <cassert>
#include <optional>
#include <iostream>
#include <type_traits>
#include <variant>

// Boost
#include <boost/filesystem.hpp>
#include <boost/range/adaptor/indexed.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

namespace fs = boost::filesystem;

namespace network {
  template<typename _Tp>
    class PrimitiveLayer {
      public:
        using Key = unsigned;
        using PrimitiveT = _Tp;
        using PrimitiveTPtr = std::shared_ptr<PrimitiveT>;
        using Layer = std::vector<PrimitiveTPtr>;

        void create(const std::size_t& t_size, std::true_type);
        void create(const std::size_t& t_size, std::false_type);

      private:
        friend class Network;

      private:
        std::optional<std::variant<Layer, PrimitiveTPtr>> m_layers { };
    };

  template<typename _Tp>
    void PrimitiveLayer<_Tp>::create(const std::size_t& t_size, std::true_type)
    {
      m_layers = std::make_shared<PrimitiveT>(t_size);
    }

  template<typename _Tp>
    void PrimitiveLayer<_Tp>::create(const std::size_t& t_size, std::false_type)
    {
      if(!m_layers) {
        m_layers = Layer();
      }

      if(auto layer_ptr_ = std::get_if<Layer>(&(*m_layers))) {
        layer_ptr_->emplace_back(std::move(std::make_shared<PrimitiveT>(t_size)));
      }
    }

  class InputLayer final : public PrimitiveLayer<primitives::Primitive<Neuron>> {
    public:
      using value = InputLayer;

      InputLayer() = default;
      ~InputLayer() = default;

    private:
      friend class Network;
  };

  class HiddenLayer final : public PrimitiveLayer<primitives::Primitive<Neuron>> {
    public:
      using value = HiddenLayer;

      HiddenLayer() = default;
      ~HiddenLayer() = default;

    private:
      friend class Network;
  };

  class OutputLayer final : public PrimitiveLayer<primitives::Primitive<Neuron>> {
    public:
      using value = HiddenLayer;

      OutputLayer() = default;
      ~OutputLayer() = default;

    private:
      friend class Network;
  };

  template< class T >
    struct is_single_layer : std::integral_constant< bool,
      std::is_same<InputLayer,  typename std::remove_cv<T>::type>::value ||
      std::is_same<OutputLayer, typename std::remove_cv<T>::type>::value > {};

  class Network {
    public:
      Network() noexcept = default;

      /**
       * @brief Construct from already initialized layers
       * @param new input layer
       * @param new hidden layers
       * @param new output layer
       *
       * Constructs a Network from its individual elements for the layers.
       */
      Network(const InputLayer& t_input, const HiddenLayer& t_hidden, const OutputLayer& t_output);

      Network(Network&& rhs) noexcept = default;
      Network& operator=(Network&& rhs) noexcept = default;
      Network(const Network& rhs) = delete;
      Network& operator=(const Network& rhs) = delete;
      ~Network() noexcept = default;

      /**
       * @brief Set path to dataset
       * @param new dataset
       */
      void setDataset(const std::string& t_dataset) noexcept;

      /**
       * @brief Set categorys
       * @param new categorys
       */
      void setCategorys(const std::vector<std::string>& t_categorys) noexcept;

      /**
       * @brief Set epoch for education
       * @param new epoch
       */
      void setEpoch(const std::size_t& t_epoch) noexcept;

      /**
       * @brief Start education Network
       */
      bool education();

      /**
       * @brief Get formats file
       */
      const std::array<std::string, 3> &formats_ref() {
        return m_format;
      }

    private:
      /**
       * @brief Network work with format image
       */
      static const std::array<std::string, 3> &formats() {
        static const std::array<std::string, 3> f = {{".png", ".jpeg", ".jpg"}};
        return f;
      }

    private:
      InputLayer  m_input_layer_  { };
      HiddenLayer m_hidden_layer_ { };
      OutputLayer m_output_layer_ { };

      std::string                 m_dataset   {""};
      std::vector<std::string>    m_categorys {""};
      std::optional<std::size_t>  m_epoch     { };

      const std::array<std::string, 3> &m_format = formats();
  };
} // namespace network
#endif // NETWORK_NETWORK_HPP_