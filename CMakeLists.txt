cmake_minimum_required(VERSION 3.14)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ALLSAN)
endif()

set(CMAKE_CXX_FLAGS_ASAN "-fsanitize=address -fno-omit-frame-pointer -g -O1")
set(CMAKE_CXX_FLAGS_UNDEFINED "-fsanitize=undefined -fno-omit-frame-pointer -g -O1")
set(CMAKE_CXX_FLAGS_ALLSAN "-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1")

project(test_asan)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED CONFIG)

add_executable(test_asan main.cpp)
target_link_libraries(test_asan
    xtensor
    xtensor-blas)
set_target_properties(test_asan PROPERTIES
CXX_STANDARD 14)