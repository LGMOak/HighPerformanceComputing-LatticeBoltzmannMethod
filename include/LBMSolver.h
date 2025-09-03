#pragma once

#include "LBMConfig.h"
#include "LBMUtils.h"
#include "LBMGrid.h"
#include "LBMTimeEvolution.h"
#include <iostream>
#include <iomanip>
#include <memory>

namespace LBM {
    class Solver {
    private:
        SimulationParams params_;
        Grid grid_;
        std::unique_ptr<AnimationWriter> animation_writer_;

        // Animation parameters
        bool enable_animation_;
        int animation_frequency_;

    public:
        explicit Solver(const SimulationParams& params,
                       bool enable_animation = false,
                       int animation_frequency = 100)
            : params_(params), grid_(params.nx, params.ny),
              enable_animation_(enable_animation),
              animation_frequency_(animation_frequency) {}

        void initialise() {
            std::cout << "LBM Parameters: nx=" << params_.nx << ", ny=" << params_.ny
                  << ", tau=" << params_.tau << ", nu=" << std::fixed << std::setprecision(5)
                  << params_.nu() << ", force_x=" << params_.force_x << std::endl;

            // Initialize animation writer if enabled
            if (enable_animation_) {
                std::string filename = "animation_data_" +
                                     std::to_string(params_.nx) + "x" +
                                     std::to_string(params_.ny) + ".h5";
                animation_writer_ = std::make_unique<AnimationWriter>(filename, grid_);
                std::cout << "Animation enabled. Writing to: " << filename << std::endl;
                std::cout << "Animation frequency: every " << animation_frequency_ << " timesteps" << std::endl;
            }

            grid_.initialise();
        }

        bool run() {
            std::cout << "Starting LBM Channel Flow simulation (Poiseuille flow)..." << std::endl;

            // Write initial state if animation enabled
            if (enable_animation_) {
                animation_writer_->write_frame(0);
            }

            for (int t = 1; t <= params_.num_timesteps; ++t) {
                // Step 1: Collision
                collision_step();

                // Step 2: Streaming
                streaming_step();

                // Step 3: Boundary conditions
                apply_boundary_conditions();

                // Step 4: Check stability
                if (!grid_.check_stability()) {
                    std::cout << "LBM simulation failed at timestep " << t << std::endl;
                    return false;
                }

                // Step 5: Write animation frame if needed
                if (enable_animation_ && (t % animation_frequency_ == 0)) {
                    animation_writer_->write_frame(t);
                }

                // Progress output
                if (t % params_.output_frequency == 0) {
                    std::cout << "Timestep " << t << ": max_vel="
                              << std::fixed << std::setprecision(4)
                              << grid_.max_velocity() << std::endl;
                }
            }

            return true;
        }

        const Grid& get_grid() const { return grid_; }
        const SimulationParams& get_params() const { return params_; }

    private:
        void collision_step() {
            const double tau_inv = 1.0 / params_.tau;

            for (int x = 0; x < params_.nx; ++x) {
                for (int y = 0; y < params_.ny; ++y) {
                    // ... (same collision implementation as before)
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

                    const double rho_inv = (rho_val < 1e-9) ? 1.0/1e-9 : 1.0/rho_val;
                    ux_val *= rho_inv;
                    uy_val *= rho_inv;

                    grid_.rho(x, y) = rho_val;
                    grid_.ux(x, y) = ux_val;
                    grid_.uy(x, y) = uy_val;

                    // BGK collision (unrolled)
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
            for (int x = 0; x < grid_.nx(); ++x) {
                for (int y = 0; y < grid_.ny(); ++y) {
                    for (int i = 0; i < Q; ++i) {
                        grid_.f(x, y, i) = 0.0;
                    }
                }
            }

            for (int x = 0; x < grid_.nx(); ++x) {
                for (int y = 0; y < grid_.ny(); ++y) {
                    for (int i = 0; i < Q; ++i) {
                        int x_dest = periodic_x(x + VELOCITIES[i][0], grid_.nx());
                        int y_dest = y + VELOCITIES[i][1];

                        if (y_dest >= 0 && y_dest < grid_.ny()) {
                            grid_.f(x_dest, y_dest, i) = grid_.f_temp(x, y, i);
                        }
                    }
                }
            }
        }

        void apply_boundary_conditions() {
            const int ceiling = grid_.ny() - 1;

            for (int x = 0; x < grid_.nx(); ++x) {
                grid_.f(x, ceiling, 4) = grid_.f_temp(x, ceiling - 1, 2);
                grid_.f(x, ceiling, 7) = grid_.f_temp(x, ceiling - 1, 5);
                grid_.f(x, ceiling, 8) = grid_.f_temp(x, ceiling - 1, 6);
            }

            for (int x = 0; x < grid_.nx(); ++x) {
                grid_.f(x, 0, 2) = grid_.f_temp(x, 1, 4);
                grid_.f(x, 0, 5) = grid_.f_temp(x, 1, 8);
                grid_.f(x, 0, 6) = grid_.f_temp(x, 1, 7);
            }
        }
    };
}