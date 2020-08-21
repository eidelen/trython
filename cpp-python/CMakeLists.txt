cmake_minimum_required(VERSION 3.13)
project(cpp_python)

set(CMAKE_CXX_STANDARD 14)

# to get pybind library -> brew install pybind11
find_package(pybind11 REQUIRED)
find_package(PythonInterp 3)


add_executable(cpp_python main.cpp)
target_link_libraries(cpp_python PRIVATE pybind11::embed)