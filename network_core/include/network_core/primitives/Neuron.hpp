#pragma once

#ifndef NETWORK_NEURON_HPP_
#define NETWORK_NEURON_HPP_

#include "network_core/Forward.hpp"
#include "network_core/Constants.hpp"
#include "network_core/utility/ActivationFunctions.hpp"

#include <optional>
#include <ostream>
#include <vector>
#include <memory>
#include <map>
#include <cmath>
#include <algorithm>

namespace network {
  class Neuron {
    public:
      using TypeWeight   = double;
      using TypeOutput   = double;
      using TypeSynapses = std::vector<std::pair<NeuronPtr, TypeWeight>>;

      Neuron()  { };
      ~Neuron() = default;

      template<typename... _Args>
        void create_link(_Args&&... __args);

      std::size_t size() const noexcept { return m_synapses.size(); }

      void setCategory(const std::string& t_category) noexcept;
      std::string getCategory() const noexcept;

      void setOutputValue(const TypeOutput& t_output) noexcept;
      TypeOutput getOutputValue()               const noexcept;

      TypeOutput getError() const noexcept;

      void transfer()             noexcept;

      void update(const std::string& t_category) noexcept;
      void update(const TypeOutput&  t_error)    noexcept;
      void updateWeight() noexcept;

      std::optional<TypeWeight> getWeight(const NeuronPtr& t_neuron) noexcept;

      friend inline std::ostream& operator<<(std::ostream& t_stream, Neuron& t_neuron);
      bool operator ==(const Neuron& t_neuron) noexcept;

    private:
      TypeSynapses m_synapses     { };
      std::string  m_category     { };
      TypeOutput   m_output       { Constants::OUTPUT_NEURON_DEFAULT };
      TypeOutput   m_error        { Constants::ERROR_DEFAULT  };
  };

  template<typename... Args>
    void Neuron::create_link(Args&&... t_args)
    {
      m_synapses.emplace_back(std::forward<Args>(t_args)...);
    }

  void Neuron::setCategory(const std::string& t_category) noexcept
  {
    m_category = t_category;
  }

  std::string Neuron::getCategory() const noexcept
  {
    return m_category;
  }

  Neuron::TypeOutput Neuron::getError() const noexcept
  {
    return m_error;
  }
  
  void Neuron::setOutputValue(const Neuron::TypeOutput& t_output) noexcept
  {
    m_output = t_output;
  }

  Neuron::TypeOutput Neuron::getOutputValue() const noexcept
  {
    return m_output;
  }

  void Neuron::transfer() noexcept
  {
    TypeOutput acc { 0.0 };
    for(const auto& synapse : m_synapses) {
      auto&& [neuron , weight] = synapse;
      acc += (neuron->getOutputValue() * weight);
    }

    m_output = computation::sigmoid(acc);
  }

  void Neuron::update(const std::string& t_category) noexcept
  {
    if(m_category.empty()) {
      return;
    }

    const TypeOutput expected = m_category == t_category ? static_cast<TypeOutput>(1.0) : static_cast<TypeOutput>(0.0);
    m_error = expected - m_output;
  }

  void Neuron::update(const Neuron::TypeOutput& t_error) noexcept
  {
    m_error = t_error;
  }

  void Neuron::updateWeight() noexcept
  {
    for(auto&& synapse : m_synapses) {
      auto&& [neuron , weight] = synapse;
      weight = weight + (Constants::LEARNING_RATE_DEFAULT * m_error * neuron->getOutputValue()
                      * computation::differential(std::bind(computation::sigmoid, std::placeholders::_1), m_output));
    }
  }

  std::optional<Neuron::TypeWeight> Neuron::getWeight(const NeuronPtr& t_neuron) noexcept
  {
    std::optional<Neuron::TypeWeight> result;

    if(!t_neuron) {
      return result;
    }

    auto it = std::find_if(m_synapses.begin(), m_synapses.end(), [&](auto& e) {
      return t_neuron == e.first;
    });

    if(it != m_synapses.end()) {
      result = it->second;
    }

    return result;
  }

  bool Neuron::operator ==(const Neuron& t_neuron) noexcept
  {
    return (this == &t_neuron);
  }

  inline std::ostream& operator<<(std::ostream& t_stream, Neuron& t_neuron)
  {
    t_stream << "Output: " << t_neuron.m_output;
    return t_stream;
  }
} // namespace network
#endif // NETWORK_NEURON_HPP_