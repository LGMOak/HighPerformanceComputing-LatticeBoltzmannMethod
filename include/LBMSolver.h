# pragma once

#include "LBMConfig.h"
#include "LBMUtils.h"
#include "LBMGrid.h"
#include "LBMTimeEvolution.h"
#include <iostream>
#include <iomanip>

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
                  << params_.nu() << ", force_x=" <<  std::scientific << params_.force_x << std::endl;

            if (enable_animation_) {
                std::string filename = "animation_data_" +
                                     std::to_string(params_.nx) + "x" +
                                     std::to_string(params_.ny) + ".h5";
                animation_writer_ = std::make_unique<AnimationWriter>(filename, grid_);
                std::cout << "Animation enabled. Writing to: " << filename << std::endl;
            }


            params_.print_stability_info();

            if (!params_.is_stable()) {
                std::cout << "WARNING: Simulation may be numerically unstable!" << std::endl;
            }

            grid_.initialise();
        }

        bool run() {
            std::cout << "Starting LBM Channel Flow simulation (Poiseuille flow)..." << std::endl;

            // Write initial state if animation enabled
            if (enable_animation_) {
                animation_writer_->write_frame(0);
            }

            for (int t = 1; t < params_.num_timesteps; ++t) {
                // Clear next timestep array
                grid_.clear_next();

                // Step 1: Collision + Streaming
                collision_and_streaming_step();

                // Step 2: Apply boundary conditions to next array
                apply_boundary_conditions();

                // Step 3: Swap arrays (next becomes current)
                grid_.swap_distributions();

                // Step 4: Check stability
                if (!grid_.check_stability()) {
                    std::cout << "LBM simulation failed - instability at timestep " << t << std::endl;
                    return false;
                }

                // Step 5: Write animation frame if needed
                if (enable_animation_ && (t % animation_frequency_ == 0)) {
                    animation_writer_->write_frame(t);
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
        void collision_and_streaming_step() {
            const double tau_inv = 1.0 / params_.tau;

            // Process each lattice site
            for (int x = 0; x < params_.nx; ++x) {
                for (int y = 0; y < params_.ny; ++y) {

                    // Calculate macroscopic quantities from current distributions
                    double rho_val = 0.0;
                    double ux_val = 0.0;
                    double uy_val = 0.0;

                    // Unroll for performance
                    const double f0 = grid_.f_current(x, y, 0);
                    const double f1 = grid_.f_current(x, y, 1);
                    const double f2 = grid_.f_current(x, y, 2);
                    const double f3 = grid_.f_current(x, y, 3);
                    const double f4 = grid_.f_current(x, y, 4);
                    const double f5 = grid_.f_current(x, y, 5);
                    const double f6 = grid_.f_current(x, y, 6);
                    const double f7 = grid_.f_current(x, y, 7);
                    const double f8 = grid_.f_current(x, y, 8);

                    rho_val = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
                    ux_val = f1 - f3 + f5 - f6 - f7 + f8;
                    uy_val = f2 - f4 + f5 + f6 - f7 - f8;

                    // Avoid division by zero
                    const double rho_inv = (rho_val < 1e-9) ? 1.0/1e-9 : 1.0/rho_val;
                    ux_val *= rho_inv;
                    uy_val *= rho_inv;

                    // Update macroscopic quantities
                    grid_.rho(x, y) = rho_val;
                    grid_.ux(x, y) = ux_val;
                    grid_.uy(x, y) = uy_val;

                    // Collision: compute post-collision distributions
                    const double f_eq[Q] = {
                        equilibrium_with_force(0, rho_val, ux_val, uy_val, params_.force_x, params_.force_y),
                        equilibrium_with_force(1, rho_val, ux_val, uy_val, params_.force_x, params_.force_y),
                        equilibrium_with_force(2, rho_val, ux_val, uy_val, params_.force_x, params_.force_y),
                        equilibrium_with_force(3, rho_val, ux_val, uy_val, params_.force_x, params_.force_y),
                        equilibrium_with_force(4, rho_val, ux_val, uy_val, params_.force_x, params_.force_y),
                        equilibrium_with_force(5, rho_val, ux_val, uy_val, params_.force_x, params_.force_y),
                        equilibrium_with_force(6, rho_val, ux_val, uy_val, params_.force_x, params_.force_y),
                        equilibrium_with_force(7, rho_val, ux_val, uy_val, params_.force_x, params_.force_y),
                        equilibrium_with_force(8, rho_val, ux_val, uy_val, params_.force_x, params_.force_y)
                    };

                    const double f_post_collision[Q] = {
                        f0 - tau_inv * (f0 - f_eq[0]),
                        f1 - tau_inv * (f1 - f_eq[1]),
                        f2 - tau_inv * (f2 - f_eq[2]),
                        f3 - tau_inv * (f3 - f_eq[3]),
                        f4 - tau_inv * (f4 - f_eq[4]),
                        f5 - tau_inv * (f5 - f_eq[5]),
                        f6 - tau_inv * (f6 - f_eq[6]),
                        f7 - tau_inv * (f7 - f_eq[7]),
                        f8 - tau_inv * (f8 - f_eq[8])
                    };

                    // Streaming: propagate to destination sites
                    for (int i = 0; i < Q; ++i) {
                        int x_dest = periodic_x(x + VELOCITIES[i][0], grid_.nx());
                        int y_dest = y + VELOCITIES[i][1];

                        // Only stream to valid y destinations
                        if (y_dest >= 0 && y_dest < grid_.ny()) {
                            grid_.f_next(x_dest, y_dest, i) = f_post_collision[i];
                        }
                    }
                }
            }
        }

        void apply_boundary_conditions() {
            const int ceiling = grid_.ny() - 1;

            // Top wall bounce-back (y = ceiling)
            for (int x = 0; x < grid_.nx(); ++x) {
                // Bounce back particles that hit the top wall
                grid_.f_next(x, ceiling, 4) = grid_.f_next(x, ceiling, 2); // Down from up
                grid_.f_next(x, ceiling, 7) = grid_.f_next(x, ceiling, 5); // Down-left from up-right
                grid_.f_next(x, ceiling, 8) = grid_.f_next(x, ceiling, 6); // Down-right from up-left
            }

            // Bottom wall bounce-back (y = 0)
            for (int x = 0; x < grid_.nx(); ++x) {
                // Bounce back particles that hit the bottom wall
                grid_.f_next(x, 0, 2) = grid_.f_next(x, 0, 4); // Up from down
                grid_.f_next(x, 0, 5) = grid_.f_next(x, 0, 7); // Up-right from down-left
                grid_.f_next(x, 0, 6) = grid_.f_next(x, 0, 8); // Up-left from down-right
            }
        }
    };
}