#pragma once

#include "LBMConfig.h"
#include "LBMUtils.h"
#include "LBMGrid.h"
#include "LBMIO.h"
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <algorithm>

namespace LBM {
    class Solver {
    private:
        SimulationParams params_;
        Grid grid_;

    public:
        explicit Solver(const SimulationParams& params)
            : params_(params), grid_(params.nx, params.ny) {}

        void initialise() {
            if (grid_.mpi_rank() == 0) {
                std::cout << "Cylinder Flow LBM Parameters:\n";
                std::cout << "  Domain: " << params_.nx << "×" << params_.ny << "\n";
                std::cout << "  tau = " << params_.tau << ", nu = " << params_.nu() << "\n";
                std::cout << "  Inlet velocity = " << params_.inlet_velocity << "\n";
                std::cout << "  Reynolds number = " << params_.reynolds() << std::endl;
            }
            grid_.setup_geometry(params_);
            grid_.initialise(params_.inlet_velocity);
        }

        bool run(IOManager& io_manager) {
            if (grid_.mpi_rank() == 0) {
                std::cout << "Starting LBM cylinder flow simulation..." << std::endl;
            }

            for (int t = 0; t < params_.num_timesteps; ++t) {
                collision_step();

                // Record forces AFTER collision, as it reads from the f_next array
                if (t % params_.output_frequency == 0) {
                    io_manager.record_forces(t, grid_, params_);
                }

                grid_.exchange_ghost_cells();
                streaming_step();
                apply_boundary_conditions();

                if (!grid_.check_stability()) {
                    if (grid_.mpi_rank() == 0)
                        fprintf(stderr, "Simulation unstable at timestep %d\n", t);
                    return false;
                }

                if (t > 0 && t % params_.output_frequency == 0) {
                    double max_vel = grid_.max_velocity();
                    if (grid_.mpi_rank() == 0) {
                        std::cout << "Timestep " << t << ": max_vel="
                                  << std::fixed << std::setprecision(6) << max_vel << std::endl;
                    }
                }
            }
            return true;
        }

        const Grid& get_grid() const { return grid_; }
        const SimulationParams& get_params() const { return params_; }

    private:
        void collision_step() {
            const double tau_inv = 1.0 / params_.tau;
            const int GHOST = 1;

#pragma omp parallel for schedule(static) collapse(2)
            for (int y = 0; y < grid_.local_ny(); ++y) {
                for (int x = 0; x < grid_.local_nx(); ++x) {
                    // Skip solid cells - they don't participate in collision
                    if (grid_.is_solid(x, y)) continue;

                    int gx = x + GHOST;
                    int gy = y + GHOST;

                    double* f_curr = grid_.f_current_ptr(gx, gy);
                    double* f_next = grid_.f_next_ptr(gx, gy);

                    // Calculate macroscopic quantities
                    double rho_val = 0.0, ux_val = 0.0, uy_val = 0.0;
                    for(int i = 0; i < Q; ++i) {
                        rho_val += f_curr[i];
                        ux_val += VELOCITIES[i][0] * f_curr[i];
                        uy_val += VELOCITIES[i][1] * f_curr[i];
                    }

                    ux_val /= rho_val;
                    uy_val /= rho_val;

                    // Store macroscopic quantities
                    grid_.rho(x, y) = rho_val;
                    grid_.ux(x, y) = ux_val;
                    grid_.uy(x, y) = uy_val;

                    // BGK collision
                    double u_sq = ux_val * ux_val + uy_val * uy_val;
                    for(int i = 0; i < Q; ++i) {
                        double ci_u = VELOCITIES[i][0] * ux_val + VELOCITIES[i][1] * uy_val;
                        double f_eq_i = WEIGHTS[i] * rho_val *
                                       (1.0 + 3.0 * ci_u + 4.5 * ci_u * ci_u - 1.5 * u_sq);
                        f_next[i] = f_curr[i] - tau_inv * (f_curr[i] - f_eq_i);
                    }
                }
            }
        }

        void streaming_step() {
            const int GHOST = 1;

#pragma omp parallel for schedule(static) collapse(2)
            for (int y = 0; y < grid_.local_ny(); ++y) {
                for (int x = 0; x < grid_.local_nx(); ++x) {
                    int gx = x + GHOST;
                    int gy = y + GHOST;

                    // Pull scheme: read from upstream neighbors
                    for (int i = 0; i < Q; ++i) {
                        int src_x = gx - VELOCITIES[i][0];
                        int src_y = gy - VELOCITIES[i][1];
                        grid_.f_current(gx, gy, i) = grid_.f_next(src_x, src_y, i);
                    }
                }
            }
        }

        void apply_boundary_conditions() {
            const int GHOST = 1;

#pragma omp parallel
            {
                // 1. Wall boundaries (bounce-back)
                if (grid_.is_bottom_boundary()) {
#pragma omp for schedule(static) nowait
                    for (int x = 0; x < grid_.local_nx(); ++x) {
                        if (grid_.is_solid(x, 0)) continue;
                        int gx = x + GHOST;
                        int gy = GHOST;
                        // Bounce back: reverse direction indices
                        grid_.f_current(gx, gy, 2) = grid_.f_current(gx, gy, 4);  // N ← S
                        grid_.f_current(gx, gy, 5) = grid_.f_current(gx, gy, 7);  // NE ← SW
                        grid_.f_current(gx, gy, 6) = grid_.f_current(gx, gy, 8);  // NW ← SE
                    }
                }

                if (grid_.is_top_boundary()) {
#pragma omp for schedule(static) nowait
                    for (int x = 0; x < grid_.local_nx(); ++x) {
                        if (grid_.is_solid(x, grid_.local_ny() - 1)) continue;
                        int gx = x + GHOST;
                        int gy = GHOST + grid_.local_ny() - 1;  // Last interior cell explicitly
                        grid_.f_current(gx, gy, 4) = grid_.f_current(gx, gy, 2);  // S ← N
                        grid_.f_current(gx, gy, 7) = grid_.f_current(gx, gy, 5);  // SW ← NE
                        grid_.f_current(gx, gy, 8) = grid_.f_current(gx, gy, 6);  // SE ← NW
                    }
                }

                // 2. Inlet boundary (Zou-He velocity BC)
                if (grid_.is_left_boundary()) {
#pragma omp for schedule(static) nowait
                    for (int y = 0; y < grid_.local_ny(); ++y) {
                        if (grid_.is_solid(0, y)) continue;

                        int gx = GHOST;
                        int gy = y + GHOST;
                        double* f = grid_.f_current_ptr(gx, gy);

                        double u_in = params_.inlet_velocity;
                        double v_in = 0.0;

                        // Zou-He: calculate density from known populations
                        // After streaming, we know: f[0], f[2], f[4] (stationary & vertical)
                        // and incoming: f[3], f[6], f[7] (from the left/west)
                        double rho_bc = (f[0] + f[2] + f[4] + 2.0 * (f[3] + f[6] + f[7]))
                                       / (1.0 - u_in);

                        // Set unknown populations (pointing right/east)
                        f[1] = f[3] + (2.0/3.0) * rho_bc * u_in;
                        f[5] = f[7] - 0.5 * (f[2] - f[4]) + (1.0/6.0) * rho_bc * u_in;
                        f[8] = f[6] + 0.5 * (f[2] - f[4]) + (1.0/6.0) * rho_bc * u_in;

                        // Update macroscopic quantities
                        grid_.rho(0, y) = rho_bc;
                        grid_.ux(0, y) = u_in;
                        grid_.uy(0, y) = v_in;
                    }
                }

                // 3. Outlet boundary (Zou-He pressure BC or convective)
                if (grid_.is_right_boundary()) {
#pragma omp for schedule(static) nowait
                    for (int y = 0; y < grid_.local_ny(); ++y) {
                        if (grid_.is_solid(grid_.local_nx() - 1, y)) continue;

                        int gx = GHOST + grid_.local_nx() - 1;
                        int gy = y + GHOST;
                        double* f = grid_.f_current_ptr(gx, gy);

                        // Zou-He pressure outlet (rho = 1.0)
                        double rho_out = 1.0;

                        // Calculate velocity from known populations
                        double u_out = -1.0 + (f[0] + f[2] + f[4] +
                                               2.0 * (f[1] + f[5] + f[8])) / rho_out;
                        double v_out = 0.0;  // or calculate from transverse balance

                        // Set unknown populations (pointing left/west)
                        f[3] = f[1] - (2.0/3.0) * rho_out * u_out;
                        f[6] = f[8] - 0.5 * (f[2] - f[4]) - (1.0/6.0) * rho_out * u_out;
                        f[7] = f[5] + 0.5 * (f[2] - f[4]) - (1.0/6.0) * rho_out * u_out;

                        grid_.rho(grid_.local_nx() - 1, y) = rho_out;
                        grid_.ux(grid_.local_nx() - 1, y) = u_out;
                        grid_.uy(grid_.local_nx() - 1, y) = v_out;
                    }
                }

                // 4. Cylinder bounce-back (must be done AFTER all other BCs)
                // Note: Can't use collapse(2) here due to temp array
#pragma omp for schedule(static)
                for (int y = 0; y < grid_.local_ny(); ++y) {
                    for (int x = 0; x < grid_.local_nx(); ++x) {
                        if (!grid_.is_solid(x, y)) continue;

                        int gx = x + GHOST;
                        int gy = y + GHOST;

                        // Store current state temporarily
                        double f_temp[Q];
                        for (int i = 0; i < Q; ++i) {
                            f_temp[i] = grid_.f_current(gx, gy, i);
                        }

                        // Bounce back: f_i -> f_opposite(i)
                        for (int i = 0; i < Q; ++i) {
                            grid_.f_current(gx, gy, i) = f_temp[OPPOSITE[i]];
                        }

                        // Solid cells have zero velocity
                        grid_.ux(x, y) = 0.0;
                        grid_.uy(x, y) = 0.0;
                    }
                }
            } // end omp parallel
        }
    };
}