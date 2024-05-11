cmake_minimum_required(VERSION 3.15)
project(magicqoa CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(magicqoa magicqoa.cpp)
target_precompile_headers(magicqoa PRIVATE pch.hpp)