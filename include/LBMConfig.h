#pragma once

#include <vector>
#include <array>

namespace LBM {

    // D2Q9 Lattice
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
        double tau = 0.7;
        // double force_x = 5e-5;
        // double force_y = 0.0;
        double inlet_velocity = 0.28;
        int nx = 1024;
        int ny = 256;
        int num_timesteps = 400000;
        int output_frequency = 1000;

        // cylinder parameters
        double cylinder_x = 0.2;
        double cylinder_y = 0.5;
        double cylinder_radius = 0.05;


        // Derived parameters
        double nu() const { return (tau - 0.5) / 3.0; }
        double reynolds() const {
            double D = 2.0 * cylinder_radius * ny;
            return (inlet_velocity * D) / nu();
        }

        // absolute cylinder position
        int get_cylinder_x() const { return static_cast<int>(cylinder_x * nx); }
        int get_cylinder_y() const { return static_cast<int>(cylinder_y * ny); }
        int get_cylinder_radius_cells() const {
            return static_cast<int>(cylinder_radius * ny);
        }
    };
}