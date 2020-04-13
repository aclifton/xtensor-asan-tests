cmake_minimum_required(VERSION 3.10)
project(test_asan)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED CONFIG)

add_executable(test_asan main.cpp)
target_link_libraries(test_asan
    xtensor
    xtensor-blas)