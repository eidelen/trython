cmake_minimum_required(VERSION 4.0)
project(python_cpp)

set(CMAKE_CXX_STANDARD 20)

add_library(python_cpp SHARED library.cpp)
