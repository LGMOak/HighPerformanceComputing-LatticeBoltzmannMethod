#pragma once

#include <vector>
#include <array>

namespace LBM {

    // D2Q9 Lattice - open to generalisation
    constexpr int Q = 9;
    constexpr int D = 2;

    // velocity vectors
    constexpr std::array<std::array<int, 2>, Q> VELOCITIES = {
        {
            {0,0},
            {1,0},
            {0,1},
            {-1,0},
            {0,-1},
            {1,1},
            {-1,1},
            {-1,-1},
            {1,-1}
        }
    };

    // Weights used for equilibrium distribution
    constexpr std::array<double, Q> WEIGHTS = {
        4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
        };

    // Directions for bounce-back boundary conditions
    constexpr std::array<int, Q> OPPOSITE = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    struct SimulationParams {
        double tau = 0.6;
        double force_x = 5e-5;
        double force_y = 0.0;
        int nx = 256;
        int ny = 64;
        int num_timesteps = 10000;
        int output_frequency = 1000;


        // Derived parameters
        double nu() const { return (tau - 0.5) / 3.0; }
    };
}