#pragma once

#ifndef NETWORK_PRIMITIVE_HPP_
#define NETWORK_PRIMITIVE_HPP_

// #include "network_core/primitives/Neuron.hpp"
#include "network_core/Constants.hpp"
#include "network_core/Forward.hpp"

#include <atomic>
#include <iostream>
#include <vector>
#include <memory>

namespace {
  std::atomic<network::Id> unique_id_neuron_ { 0 };
} // namespace

inline network::Id getUniqueId() { return unique_id_neuron_++; }

namespace network {
  namespace primitives {
    template<typename _Tp>
      class Primitive {
        public:
          typedef typename std::vector<std::shared_ptr<_Tp>>::iterator       iterator;
          typedef typename std::vector<std::shared_ptr<_Tp>>::const_iterator const_iterator;

          explicit Primitive() = default;
          explicit Primitive(const std::size_t& size) noexcept;

          void connect(Primitive& t_layer)                  noexcept;
          void connect(std::shared_ptr<Primitive>& t_layer) noexcept;

          void set(const std::size_t& t_pose, const double& t_value);

          void calculate() noexcept;
          void update(const std::string& t_category)    noexcept;
          void update(Primitive& t_layer)               noexcept;

          void updateWeight() noexcept;

          inline std::size_t size() const noexcept { return m_neurons.size(); }

          const_iterator begin();
          const_iterator end();

          void setCategory(const std::size_t& t_pose, const std::string& t_category);

        protected:
          std::vector<std::shared_ptr<_Tp>> m_neurons { };
      };

    template<typename _Tp>
      Primitive<_Tp>::Primitive(const std::size_t& t_size) noexcept
      {
        m_neurons.reserve(t_size);
        for(std::size_t i = 0; i < t_size; ++i) {
          m_neurons.push_back(std::make_shared<_Tp>(getUniqueId()));
        }
      }

    template<typename _Tp>
      void Primitive<_Tp>::connect(Primitive& t_layer) noexcept
      {
        auto random = [](double min, double max) {
          return (double)(rand())/RAND_MAX*(max - min) + min;
        };

        static_assert(static_cast<bool>(Constants::WEIGHT_SYNAPSES_DEFAULT));
        for(auto& neuron : m_neurons) {
          for(auto& e : t_layer) {
            neuron->createSynapse(e, random(-0.5, 0.5));
          }
        }
      }

    template<typename _Tp>
      void Primitive<_Tp>::connect(std::shared_ptr<Primitive>& t_layer) noexcept
      {
        connect(*t_layer);
      }

    template<typename _Tp>
      void Primitive<_Tp>::set(const std::size_t& t_pose, const double& t_value)
      {
        if(t_pose > m_neurons.size()) {
          throw std::out_of_range("");
        }

        if(m_neurons[t_pose]) {
          m_neurons[t_pose]->setOutputValue(t_value);
        }
      }

    template<typename _Tp>
      void Primitive<_Tp>::calculate() noexcept
      {
        for(const auto& neuron : m_neurons) {
          neuron->computeOutputValue();
        }
      }

    template<typename _Tp>
      void Primitive<_Tp>::update(const std::string& t_category) noexcept
      {
        for(const auto& neuron : m_neurons) {
          neuron->computeError(t_category);
        }
      }

    template<typename _Tp>
      void Primitive<_Tp>::update(Primitive& t_layer) noexcept
      {
        for(auto& neuron_before : m_neurons) {
          typename _Tp::TypeValueNeuron acc { 0.0 };

          for(const auto& neuron_after : t_layer) {
            if(auto weight = neuron_after->getWeight(neuron_before)) {
              acc += (*weight) * neuron_after->getError();
            }
          }

          neuron_before->computeError(acc);
        }
      }
    
    template<typename _Tp>
      void Primitive<_Tp>::updateWeight() noexcept
      {
        for(const auto& neuron : m_neurons) {
          neuron->computeWeights();
        }
      }

    template<typename _Tp>
      typename Primitive<_Tp>::const_iterator Primitive<_Tp>::begin()
      {
        return m_neurons.begin();
      }

    template<typename _Tp>
      typename Primitive<_Tp>::const_iterator Primitive<_Tp>::end()
      {
        return m_neurons.end();
      }

    template<typename _Tp>
      void Primitive<_Tp>::setCategory(const std::size_t& t_pose, const std::string& t_category)
      {
        if(t_pose > m_neurons.size()) {
          throw std::out_of_range("pose > m_neurons.size()");
        }

        if(m_neurons[t_pose]) {
          m_neurons[t_pose]->setCategory(t_category);
        }
      }
  } // namespace primitives
} // namespace network
#endif // PRIMITIVE_HPP