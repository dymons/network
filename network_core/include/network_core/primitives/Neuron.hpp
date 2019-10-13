#pragma once

#ifndef NETWORK_NEURON_HPP_
#define NETWORK_NEURON_HPP_

#include "network_core/Forward.hpp"
#include "network_core/Constants.hpp"
#include "network_core/utility/ActivationFunctions.hpp"

// STL
#include <optional>
#include <ostream>
#include <memory>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <functional>

namespace network {
  namespace primitives {
    class Neuron {
      public:
        using TypeValueNeuron   = double;
        using TypeValueCategory = std::vector<std::string>;
        using TypeSynapses      = std::map<NeuronPtr, TypeValueNeuron>;

        explicit Neuron(const Id& t_id, std::function<double(double)> t_func = computation::sigmoid);

        Neuron()  = delete;
        ~Neuron() = default;

        /**
         * @brief Creates a synapses between neurons with a given weight.
         * @param t_args Pointer to the referring neuron and the weight of connection with it.
         * @return Returns a pointer to the created neuron, if the link to the neuron
         * already exists, will return a link to it.
         */
        template<typename... Args>
          std::optional<const NeuronPtr> createSynapse(Args&&... t_args) noexcept
          {
            std::optional<const NeuronPtr> opt_neuron_ptr_ { };

            const auto  [iterator, success] = m_synapses.try_emplace(std::forward<Args>(t_args)...);
            const auto& [neuron, weight] = *iterator;

            if(neuron) {
              opt_neuron_ptr_.emplace(neuron);
            }

            return opt_neuron_ptr_;
          }

        /**
         * @brief Neuron error calculation.
         * @param t_v Calculates the error based on the total error of the child layer or category.
         * @return Calculated neuron error.
         */
        template<typename T>
          std::optional<TypeValueNeuron> computeError(const T& t_v) noexcept
          {
            std::optional<TypeValueNeuron> error_to_send;

            if constexpr (std::is_same_v<T, TypeValueCategory::value_type>) {
              TypeValueNeuron error { 0.0 };
              if(auto itr (std::find_if(m_category.begin(), m_category.end(), [&](auto& c) {return c == t_v;})); itr != m_category.end()) {
                error = 1.0;
              }

              m_error = error - m_output;
            } else {
              m_error = static_cast<TypeValueNeuron>(t_v);
            }

            error_to_send.emplace(m_error);
            return error_to_send;
          }

        /**
         * @brief Getter for get error neuron.
         * @return error neuron.
         */
        TypeValueNeuron getError() const noexcept;

        /**
         * @brief Setter for defining categories in which a neuron will be trained (only for output neurons).
         * @param t_category List of categories for training.
         */
        template<typename T>
          void setCategory(const T& t_category) noexcept
          {
            if constexpr (std::is_same_v<TypeValueCategory, T>) {
              m_category.insert(m_category.end(), t_category.begin(), t_category.end());
            } else {
              m_category.push_back(t_category);
            }

            m_category.shrink_to_fit();
          }

        /**
         * @brief Getter for getting a list of categories.
         * @return Get a list of categories.
         */
        const Neuron::TypeValueCategory& getCategory() const noexcept;

        /**
         * @brief Calculation of the output value of a neuron based synapses.
         */
        void computeOutputValue() noexcept;

        /**
         * @brief Sets the output value of the neuron (used only for the input layer).
         * @param t_output Set output value neron.
         */
        void setOutputValue(const TypeValueNeuron& t_output) noexcept;

        /**
         * @brief Getter for getting a output value neuron.
         * @return Get output value neuron.
         */
        TypeValueNeuron getOutputValue() const noexcept;

        /**
         * @brief Calculates weight relationships based on the error layer of the child.
         */
        void computeWeights() noexcept;

        /**
         * @brief Returns the weight value if the received neuron exists among the existing links.
         * @param t_neuron The neuron with which the connection with the current neuron is formed.
         * @return output weight.
         */
        std::optional<TypeValueNeuron> getWeight(const NeuronPtr& t_neuron) noexcept;

        /**
         * @brief Set activation function for neuron.
         * @param t_func activation function.
         */
        void setActivationFunction(std::function<double(double)> t_func) noexcept;

        /**
         * @brief Get Id neuron
         * @return id neuron
         */
        Id getId() const noexcept;

        std::size_t size() const noexcept;
        friend inline std::ostream& operator<<(std::ostream& t_stream, Neuron& t_neuron);
        bool operator ==(const Neuron& t_neuron) noexcept;

      private:
        Id                m_id           { 0 };
        TypeSynapses      m_synapses     { };
        TypeValueCategory m_category     { };
        TypeValueNeuron   m_output       { Constants::OUTPUT_NEURON_DEFAULT };
        TypeValueNeuron   m_error        { Constants::ERROR_DEFAULT };

        std::function<double(double)> m_active_func { };
    };
  } // namespace primitives
} // namespace network
#endif // NETWORK_NEURON_HPP_