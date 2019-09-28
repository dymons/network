cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(network LANGUAGES CXX)

add_definitions(-DDEBUG_NETWORK)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wshadow -Weffc++ -pthread -lboost_system")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(Boost_USE_STATIC_LIBS OFF)

find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_library(network_core SHARED network_core/src/Network.cpp)
target_include_directories(network_core PUBLIC ${PROJECT_SOURCE_DIR}/network_core/include ${OpenCV_LIBS})
target_link_libraries(network_core ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_library(network_io SHARED network_io/src/io.cpp)
target_include_directories(network_io PUBLIC ${PROJECT_SOURCE_DIR}/network_io/include)
target_link_libraries(network_io network_core ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_library(network_libs INTERFACE)
target_link_libraries(network_libs INTERFACE network_core network_io)

add_executable(network_example ${PROJECT_SOURCE_DIR}/network_example/main.cpp)

target_link_libraries(network_example PRIVATE network_libs ${Boost_LIBRARIES} ${OpenCV_LIBS})