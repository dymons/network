cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(network_example VERSION 1.0 LANGUAGES CXX)

find_package(Boost 1.65.1 COMPONENTS system filesystem REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
  network::network_core
  network::network_io
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)