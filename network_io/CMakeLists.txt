cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(network_io VERSION 1.0 LANGUAGES CXX)

find_package(Boost 1.65.1 COMPONENTS system filesystem REQUIRED)
if(${Boost_FOUND})
  include_directories(${Boost_INCLUDE_DIRS})
endif()

find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME}
  src/io.cpp
)

add_library(network::network_io ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
  network::network_core
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)