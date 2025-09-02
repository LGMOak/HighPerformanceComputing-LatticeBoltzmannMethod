# LBM HPC Project

High-performance Lattice Boltzmann Method solver for 2D channel flow simulation.

## Quick Start

1. **Setup project structure:**
   ```bash
   ./setup_project.sh
   ```

2. **Place header files in include/ directory:**
   - LBMConfig.h
   - LBMUtils.h  
   - LBMGrid.h
   - LBMSolver.h
   - LBMIO.h

3. **Build and run:**
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . -j$(nproc)
   ./bin/lbm_solver
   ```

4. **Visualize results:**
   ```bash
   cd .. # back to project root
   python scripts/visualize_results.py
   ```

## Performance Features

- Optimized memory layout for cache efficiency
- Loop unrolling and vectorization hints
- SIMD-friendly algorithms
- Compile-time optimizations

## Requirements

- CMake 3.16+
- C++20 compiler
- Python 3 + matplotlib, pandas, numpy (for visualization)
