cmake_minimum_required(VERSION 3.10)
project(HPC_PROJECT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(include)

add_executable(lbm_solver src/main.cpp)
add_executable(lbm_unoptimal src/unoptimal.cpp)

# Link OpenMP to both executables
target_link_libraries(lbm_solver OpenMP::OpenMP_CXX)
target_link_libraries(lbm_unoptimal OpenMP::OpenMP_CXX)


set(CMAKE_BUILD_TYPE Simulation)

target_compile_options(lbm_solver PRIVATE
        -O3 -g -march=native -funroll-loops -ffast-math
        -mavx2 -mfma -fopenmp
)
message(STATUS "Building simulation with AVX2 + OpenMP")

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP support: ${OpenMP_FOUND}")
message(STATUS "AVX2 SIMD + OpenMP optimizations enabled for lbm_solver")