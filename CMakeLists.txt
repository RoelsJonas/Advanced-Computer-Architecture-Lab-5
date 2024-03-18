cmake_minimum_required(VERSION 3.21)
project(lab5 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(lab5 main.cu)

set_target_properties(lab5 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
