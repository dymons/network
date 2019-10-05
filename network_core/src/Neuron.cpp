#include "network_core/primitives/Neuron.hpp"

namespace network {
  Neuron::Neuron(const Id& t_id, std::function<double(double)> t_func) : m_id(t_id), m_active_func(t_func) { }

  const Neuron::TypeValueCategory& Neuron::getCategory() const noexcept
  {
    return m_category;
  }

  std::size_t Neuron::size() const noexcept
  { 
    return m_synapses.size();
  }

  Neuron::TypeValueNeuron Neuron::getError() const noexcept
  {
    return m_error;
  }
  
  void Neuron::setOutputValue(const Neuron::TypeValueNeuron& t_output) noexcept
  {
    m_output = t_output;
  }

  Neuron::TypeValueNeuron Neuron::getOutputValue() const noexcept
  {
    return m_output;
  }

  void Neuron::computeOutputValue() noexcept
  {
    auto weightedSum = [&]() -> TypeValueNeuron {
      TypeValueNeuron acc { 0.0 };
      for(const auto& synapse : m_synapses) {
        auto&& [neuron , weight] = synapse;

        if(neuron) {
          acc += (neuron->getOutputValue() * weight);
        }
      }
      return acc;
    };

    if(m_active_func) {
      m_output = m_active_func(weightedSum());
    }
  }

  void Neuron::computeWeights() noexcept
  {
    for(auto&& synapse : m_synapses) {
      auto&& [neuron, weight] = synapse;

      if(neuron && m_active_func) {
        weight = weight + (Constants::LEARNING_RATE_DEFAULT * m_error * neuron->getOutputValue()
               * computation::differential(std::bind(m_active_func, std::placeholders::_1), m_output));
      }
    }
  }

  std::optional<Neuron::TypeValueNeuron> Neuron::getWeight(const NeuronPtr& t_neuron) noexcept
  {
    std::optional<Neuron::TypeValueNeuron> result;

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

  void Neuron::setActivationFunction(std::function<double(double)> t_func) noexcept
  {
    m_active_func = t_func;
  }

  bool Neuron::operator ==(const Neuron& t_neuron) noexcept
  {
    return (this == &t_neuron);
  }

  Id Neuron::getId() const noexcept
  {
    return m_id;
  }

  inline std::ostream& operator<<(std::ostream& t_stream, Neuron& t_neuron)
  {
    t_stream << "[id:" << t_neuron.m_id << ", synapses:";
    for(const auto& [neuron, weight] : t_neuron.m_synapses) {
      t_stream << "(id:" << neuron->getId() << ", weight:" << weight << ")";
    }
    t_stream << "]";
    return t_stream;
  }
} // namespace