cmake_minimum_required(VERSION 3.10)
project(HPC_PROJECT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(lbm_solver src/main.cpp)
add_executable(lbm_unoptimal src/unoptimal.cpp)

# Benchmarking build
if(CMAKE_BUILD_TYPE STREQUAL "Baseline")
    # Stage 1: Unoptimised baseline with profiling
    target_compile_options(lbm_unoptimal PRIVATE
            -O0 -g -pg
            -fno-omit-frame-pointer  # Better profiling
    )
    target_link_options(lbm_unoptimal PRIVATE -pg)

    target_compile_options(lbm_solver PRIVATE
            -O0 -g -pg
            -fno-omit-frame-pointer
    )
    target_link_options(lbm_solver PRIVATE -pg)
    message(STATUS "Building BASELINE version (-O0 -pg)")

elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Stage 2: Optimisations without compiler optimization
    target_compile_options(lbm_unoptimal PRIVATE
            -O0 -g -pg
            -fno-omit-frame-pointer
    )
    target_link_options(lbm_unoptimal PRIVATE -pg)

    target_compile_options(lbm_solver PRIVATE
            -O0 -g -pg
            -fno-omit-frame-pointer
    )
    target_link_options(lbm_solver PRIVATE -pg)
    message(STATUS "Building DEBUG_OPTIMIZED version (-O0 -pg)")

elseif(CMAKE_BUILD_TYPE STREQUAL "Simulation")
    # Stage 3 & 4: Full optimisation
    target_compile_options(lbm_unoptimal PRIVATE
            -O3 -g
            -march=native
            -funroll-loops
            -ffast-math
    )

    target_compile_options(lbm_solver PRIVATE
            -O3 -g
            -march=native
            -funroll-loops
            -ffast-math
    )
    message(STATUS "Building SIMULATION version (-O3)")

elseif(CMAKE_BUILD_TYPE STREQUAL "Perf_Profile")
    # perf profiling (if needed)
    target_compile_options(lbm_unoptimal PRIVATE
            -O3 -g
            -march=native
            -fno-omit-frame-pointer
    )

    target_compile_options(lbm_solver PRIVATE
            -O3 -g
            -march=native
            -fno-omit-frame-pointer
    )
    message(STATUS "Building PERF_PROFILE version (-O3 -g)")

else()
    # Default to simulation build
    set(CMAKE_BUILD_TYPE Simulation)
    target_compile_options(lbm_unoptimal PRIVATE
            -O3 -g -march=native -funroll-loops -ffast-math
    )

    target_compile_options(lbm_solver PRIVATE
            -O3 -g -march=native -funroll-loops -ffast-math
    )
    message(STATUS "Building DEFAULT (Simulation) version")
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")