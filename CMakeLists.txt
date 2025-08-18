cmake_minimum_required(VERSION 3.10)
project(HPC_PROJECT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory for header files
include_directories(include)

# Create executable with the main source file
add_executable(lbm_solver src/main.cpp)

# Define the default flags for all builds, including optimization
# This applies to all build types unless overwritten
target_compile_options(lbm_solver PRIVATE
        -O3
        -march=native
        -funroll-loops
        -g
)

# Conditional flags for specific build types
if(CMAKE_BUILD_TYPE STREQUAL "O0_Perf")
    # A special build type for your unoptimized baseline with perf support
    target_compile_options(lbm_solver PRIVATE -O0 -g)
elseif(CMAKE_BUILD_TYPE STREQUAL "Gprof")
    # A specific build type for profiling with gprof
    target_compile_options(lbm_solver PRIVATE -pg -g)
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")