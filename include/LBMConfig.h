#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>

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
        // Default grid parameters
        double tau = 0.6;
        double force_x = 2e-6;
        double force_y = 0.0;
        int nx = 256;
        int ny = 64;
        int num_timesteps = 5000;
        int output_frequency = 1000;

        double nu() const { return (tau - 0.5) / 3.0; }

        // Stability check helper
        double max_theoretical_velocity() const {
            const double H = ny - 1;
            return force_x * H * H / (8.0 * nu());
        }

        double mach_number() const {
            const double cs = 1.0 / std::sqrt(3.0);
            return max_theoretical_velocity() / cs;
        }

        bool is_stable() const {
            return mach_number() < 0.1;
        }

        void print_stability_info() const {
            // help guide simulation parameters
            const double u_max = max_theoretical_velocity();
            const double ma = mach_number();

            std::cout << "Stability Analysis:" << std::endl;
            std::cout << "  Max theoretical velocity: " << std::scientific << u_max << std::endl;
            std::cout << "  Mach number: " << std::fixed << std::setprecision(4) << ma << std::endl;
            std::cout << "  Stability: " << (is_stable() ? "STABLE" : "POTENTIALLY UNSTABLE") << std::endl;

            if (!is_stable()) {
                std::cout << "  WARNING: Consider reducing force_x or increasing tau" << std::endl;
                const double safe_force = 0.05 / std::sqrt(3.0) * 8.0 * nu() / ((ny-1)*(ny-1));
                std::cout << "  Suggested max force_x: " << std::scientific << safe_force << std::endl;
            }
        }
    };
}