cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(network_io VERSION 1.0 LANGUAGES CXX)

add_library(${PROJECT_NAME}
  src/io.cpp
)

add_library(network::network_io ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
  network::network_core
)