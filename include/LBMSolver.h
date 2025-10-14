#pragma once

#include "LBMConfig.h"
#include "LBMUtils.h"
#include "LBMGrid.h"
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <omp.h>

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
                std::cout << "LBM Parameters: nx=" << params_.nx << ", ny=" << params_.ny
                      << ", tau=" << params_.tau << ", nu=" << std::fixed
                      << std::setprecision(7) << params_.nu()
                      << ", force_x=" << std::fixed << std::setprecision(7) << params_.force_x << std::endl;
            }
            grid_.initialise();
        }

        bool run() {
            if (grid_.mpi_rank() == 0) {
                std::cout << "Starting LBM simulation (Final Corrected Decoupled Pull Scheme)..." << std::endl;
            }

            for (int t = 0; t < params_.num_timesteps; ++t) {
                // unfused and reorded
                collision_step();
                grid_.exchange_ghost_cells();
                streaming_step();
                apply_physical_boundary_conditions();

                if (!grid_.check_stability()) {
                    if (grid_.mpi_rank() == 0) fprintf(stderr, "Simulation unstable at timestep %d\n", t);
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

            // even workload - combines nested loops
#pragma omp parallel for schedule(static) collapse(2)
            for (int y = 0; y < grid_.local_ny(); ++y) {
                for (int x = 0; x < grid_.local_nx(); ++x) {
                    int gx = x + GHOST;
                    int gy = y + GHOST;

                    double* f_curr = grid_.f_current_ptr(gx, gy);
                    double* f_next = grid_.f_next_ptr(gx, gy);

                    // calculate density and macroscopic momentum
                    double rho_val = 0.0, ux_val = 0.0, uy_val = 0.0;
                    for(int i = 0; i < Q; ++i) {
                        rho_val += f_curr[i];
                        ux_val += VELOCITIES[i][0] * f_curr[i];
                        uy_val += VELOCITIES[i][1] * f_curr[i];
                    }

                    // apply the force to momentum
                    ux_val += 0.5 * params_.force_x;
                    uy_val += 0.5 * params_.force_y;

                    // final velocity
                    ux_val /= rho_val;
                    uy_val /= rho_val;

                    // update the grid with the new macroscopic values
                    grid_.rho(x, y) = rho_val;
                    grid_.ux(x, y) = ux_val;
                    grid_.uy(x, y) = uy_val;

                    // Compute equilibrium distribution with guo forcing
                    double u_sq = ux_val * ux_val + uy_val * uy_val;
                    for(int i = 0; i < Q; ++i) {
                        double ci_u = VELOCITIES[i][0] * ux_val + VELOCITIES[i][1] * uy_val;

                        // standard equilibrium
                        double f_eq_i = WEIGHTS[i] * rho_val * (1.0 + 3.0 * ci_u + 4.5 * ci_u * ci_u - 1.5 * u_sq);

                        // Gup force term
                        double force_term = 3.0 * WEIGHTS[i] * (VELOCITIES[i][0] * params_.force_x + VELOCITIES[i][1] * params_.force_y);

                        // BGK collision with force
                        f_next[i] = f_curr[i] - tau_inv * (f_curr[i] - f_eq_i - force_term);
                    }
                }
            }
        }

        void streaming_step() {
            const int GHOST = 1;
            // even workload - combines nested loops
            #pragma omp parallel for schedule(static) collapse(2)
            for (int y = 0; y < grid_.local_ny(); ++y) {
                for (int x = 0; x < grid_.local_nx(); ++x) {
                    int gx = x + GHOST;
                    int gy = y + GHOST;
                    for (int i = 0; i < Q; ++i) {
                        int src_x = gx - VELOCITIES[i][0];
                        int src_y = gy - VELOCITIES[i][1];
                        grid_.f_current(gx, gy, i) = grid_.f_next(src_x, src_y, i);
                    }
                }
            }
        }

        // Boundary condition step (on f_current)
        void apply_physical_boundary_conditions() {
            const int GHOST = 1;
            #pragma omp parallel
            {
                if (grid_.is_bottom_boundary()) {
                    // boundary conditions are independent so threads can continue
                    #pragma omp for schedule(static) nowait
                    for (int x = 0; x < grid_.local_nx(); ++x) {
                        int gx = x + GHOST;
                        int gy = GHOST;
                        grid_.f_current(gx, gy, 2) = grid_.f_current(gx, gy, 4);
                        grid_.f_current(gx, gy, 5) = grid_.f_current(gx, gy, 7);
                        grid_.f_current(gx, gy, 6) = grid_.f_current(gx, gy, 8);
                    }
                }
                if (grid_.is_top_boundary()) {
                    #pragma omp for schedule(static) nowait
                    for (int x = 0; x < grid_.local_nx(); ++x) {
                        int gx = x + GHOST;
                        int gy = grid_.local_ny();
                        grid_.f_current(gx, gy, 4) = grid_.f_current(gx, gy, 2);
                        grid_.f_current(gx, gy, 7) = grid_.f_current(gx, gy, 5);
                        grid_.f_current(gx, gy, 8) = grid_.f_current(gx, gy, 6);
                    }
                }
            }
        }
    };
}