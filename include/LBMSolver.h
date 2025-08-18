# pragma once

#include "LBMConfig.h"
#include "LBMUtils.h"
#include "LBMGrid.h"
#include <iostream>
#include <iomanip>
#include <immintrin.h>

namespace LBM {
    class Solver {
    private:
        SimulationParams params_;
        Grid grid_;

    public:
        explicit Solver(const SimulationParams& params) : params_(params) , grid_(params.nx, params.ny) {}

        void initialise() {
            std::cout << "LBM Parameters: nx=" << params_.nx << ", ny=" << params_.ny
                  << ", tau=" << params_.tau << ", nu=" << std::fixed << std::setprecision(4)
                  << params_.nu() << ", force_x=" << params_.force_x << std::endl;

            grid_.initialise();
        }

        bool run() {
            std::cout << "Starting LBM Channel Flow simulation (Poiseuille flow)..." << std::endl;

            for (int t = 0; t < params_.num_timesteps; ++t) {
                // Step 1: Collision
                collision_step();

                // Step 2: Streaming step;
                streaming_step();

                // Step 3: Boundary conditions
                apply_boundary_conditions();

                // Step 4: Check stability
                if (!grid_.check_stability()) {
                    std::cout << "LBM simulation failed to reach equilibrium at timestep " << t << std::endl;
                    return false;
                }

                // Progress output
                if (t % params_.output_frequency == 0) {
                    std::cout << "Timestep " << t << ": max_vel=" << std::fixed << std::setprecision(4) << grid_.max_velocity() << std::endl;
                }
            }

            return true;
        }

        const Grid& get_grid() const { return grid_; }
        const SimulationParams& get_params() const { return params_; }

    private:
        void collision_step() {
            const double tau_inv = 1.0 / params_.tau;

#pragma GCC ivdep
            for (int x = 0; x < params_.nx; ++x) {
                for (int y = 0; y < params_.ny; ++y) {
                    // Unroll loop and calculate macroscopic quantities
                    double rho_val = 0.0;
                    double ux_val = 0.0;
                    double uy_val = 0.0;

                    const double f0 = grid_.f(x, y, 0);
                    const double f1 = grid_.f(x, y, 1);
                    const double f2 = grid_.f(x, y, 2);
                    const double f3 = grid_.f(x, y, 3);
                    const double f4 = grid_.f(x, y, 4);
                    const double f5 = grid_.f(x, y, 5);
                    const double f6 = grid_.f(x, y, 6);
                    const double f7 = grid_.f(x, y, 7);
                    const double f8 = grid_.f(x, y, 8);

                    rho_val = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

                    ux_val = f1 - f3 + f5 - f6 - f7 + f8;
                    uy_val = f2 - f4 + f5 + f6 - f7 - f8;

                    // Avoid division by zero
                    const double rho_inv = (rho_val < 1e-9) ? 1.0/1e-9 : 1.0/rho_val;
                    ux_val *= rho_inv;
                    uy_val *= rho_inv;

                    // Update quantities
                    grid_.rho(x, y) = rho_val;
                    grid_.ux(x, y) = ux_val;
                    grid_.uy(x, y) = uy_val;

                    // BGK collision
                    grid_.f_temp(x, y, 0) = f0 - tau_inv * (f0 - equilibrium_with_force(0, rho_val, ux_val, uy_val, params_.force_x, params_.force_y));
                    grid_.f_temp(x,y, 1) = f1 - tau_inv * (f1 - equilibrium_with_force(1, rho_val, ux_val, uy_val, params_.force_x, params_.force_y));
                    grid_.f_temp(x,y,2) = f2 - tau_inv * (f2 - equilibrium_with_force(2, rho_val, ux_val, uy_val, params_.force_x, params_.force_y));
                    grid_.f_temp(x,y,3) = f3 - tau_inv * (f3 - equilibrium_with_force(3, rho_val, ux_val, uy_val, params_.force_x, params_.force_y));
                    grid_.f_temp(x,y,4) = f4 - tau_inv * (f4 - equilibrium_with_force(4, rho_val, ux_val, uy_val, params_.force_x, params_.force_y));
                    grid_.f_temp(x,y,5) = f5 - tau_inv * (f5 - equilibrium_with_force(5, rho_val, ux_val, uy_val, params_.force_x, params_.force_y));
                    grid_.f_temp(x,y,6) = f6 - tau_inv * (f6 - equilibrium_with_force(6, rho_val, ux_val, uy_val, params_.force_x, params_.force_y));
                    grid_.f_temp(x,y,7) = f7 - tau_inv * (f7 - equilibrium_with_force(7, rho_val, ux_val, uy_val, params_.force_x, params_.force_y));
                    grid_.f_temp(x,y,8) = f8 - tau_inv * (f8 - equilibrium_with_force(8, rho_val, ux_val, uy_val, params_.force_x, params_.force_y));
                }
            }
        }

        void streaming_step() {
            // Clear f to use as destination
            for (int x = 0; x < grid_.nx(); ++x) {
                for (int y = 0; y < grid_.ny(); ++y) {
                    for (int i = 0; i < Q; ++i) {
                        grid_.f(x, y, i) = 0.0;
                    }
                }
            }

            // Stream from f_temp (collision results) to f
            for (int x = 0; x < grid_.nx(); ++x) {
                for (int y = 0; y < grid_.ny(); ++y) {
                    for (int i = 0; i < Q; ++i) {
                        int x_dest = periodic_x(x + VELOCITIES[i][0], grid_.nx());
                        int y_dest = y + VELOCITIES[i][1];

                        if (y_dest >= 0 && y_dest < grid_.ny()) {
                            grid_.f(x_dest, y_dest, i) = grid_.f_temp(x, y, i);  // Read from f_temp!
                        }
                    }
                }
            }
        }

        void apply_boundary_conditions() {
            // Ceiling bounce back
            const int ceiling = grid_.ny() - 1;
#pragma GCC ivdep
            for (int x = 0; x < grid_.nx(); ++x) {
                grid_.f(x, ceiling, 4) = grid_.f_temp(x, ceiling - 1, 2); // Down from up
                grid_.f(x, ceiling, 7) = grid_.f_temp(x, ceiling - 1, 5); // Down-left from up-right
                grid_.f(x, ceiling, 8) = grid_.f_temp(x, ceiling - 1, 6); // Down-right from up-left
            }
            // Floor bounce back
#pragma GCC ivdep
            for (int x = 0; x < grid_.nx(); ++x) {
                grid_.f(x, 0, 2) = grid_.f_temp(x, 1, 4); // Up from down
                grid_.f(x, 0, 5) = grid_.f_temp(x, 1, 8); // Up-right from down-left
                grid_.f(x, 0, 6) = grid_.f_temp(x, 1, 7); // Up-left from down-right
            }
        }
    };
}