cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(network_core VERSION 1.0 LANGUAGES CXX)

find_package(Boost 1.65.1 COMPONENTS system filesystem REQUIRED)
find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME}
  src/Network.cpp
  src/Neuron.cpp
)

add_library(network::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)