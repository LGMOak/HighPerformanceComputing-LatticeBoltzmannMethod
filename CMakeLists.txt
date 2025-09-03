cmake_minimum_required(VERSION 3.10)
project(HPC_PROJECT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

find_package(HDF5 QUIET COMPONENTS C CXX)
if(HDF5_FOUND)
    message(STATUS "HDF5 found - Animation support enabled")
    add_definitions(-DHDF5_AVAILABLE)
    include_directories(${HDF5_INCLUDE_DIRS})
else()
    message(STATUS "HDF5 not found - Animation support disabled")
    message(STATUS "To enable animation: sudo apt install libhdf5-dev (Ubuntu/Debian)")
    message(STATUS "                    brew install hdf5 (macOS)")
endif()

add_executable(lbm_solver src/main.cpp)
add_executable(lbm_unoptimal src/unoptimal.cpp)

if(HDF5_FOUND)
    target_link_libraries(lbm_solver ${HDF5_LIBRARIES})
    target_compile_definitions(lbm_solver PRIVATE HDF5_AVAILABLE)
endif()

# Benchmarking build configurations
if(CMAKE_BUILD_TYPE STREQUAL "Baseline")
    # Stage 1: Unoptimised baseline with profiling
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
message(STATUS "Animation support: ${HDF5_FOUND}")

# Create scripts directory and copy Python scripts
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/scripts)
file(COPY ${CMAKE_SOURCE_DIR}/scripts/time_evolution.py
        DESTINATION ${CMAKE_BINARY_DIR}/scripts/)