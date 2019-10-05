cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(network_core VERSION 1.0 LANGUAGES CXX)

add_library(${PROJECT_NAME}
  src/Network.cpp
  src/Neuron.cpp
)

add_library(network::network_core ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)