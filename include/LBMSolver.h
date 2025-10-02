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
        explicit Solver(const SimulationParams& params) : params_(params) , grid_(params.nx, params.ny) {}

        void initialise() {
            if (grid_.mpi_rank() == 0) {
                std::cout << "LBM Parameters: nx=" << params_.nx << ", ny=" << params_.ny
                      << ", tau=" << params_.tau << ", nu=" << std::fixed << std::setprecision(5)
                      << params_.nu() << ", force_x=" << params_.force_x << std::endl;
            }
            grid_.initialise();
        }

        bool run() {
            if (grid_.mpi_rank() == 0) {
                std::cout << "Starting LBM Channel Flow simulation (Poiseuille flow)..." << std::endl;
            }

            for (int t = 0; t < params_.num_timesteps; ++t) {
                // combined collision-streaming step into f_next()
                collision_streaming_step();

                // Exchange ghost cells in f_next
                grid_.start_halo_exchange();
                grid_.complete_halo_exchange();

                // apply boundary conditions on f_next
                apply_physical_boundary_conditions();

                // f_current now has the post-algorithm data
                grid_.swap_f_arrays();

                if (!grid_.check_stability()) {
                    // It's helpful to know which rank failed
                    fprintf(stderr, "Rank %d: LBM simulation failed to reach equilibrium at timestep %d\n", grid_.mpi_rank(), t);
                    return false;
                }

                if (t % params_.output_frequency == 0) {
                    double max_vel = grid_.max_velocity();
                    if (grid_.mpi_rank() == 0) {
                        std::cout << "Timestep " << t << ": max_vel=" << std::fixed << std::setprecision(4) << max_vel << std::endl;
                    }
                }
            }
            return true;
        }

        const Grid& get_grid() const { return grid_; }
        const SimulationParams& get_params() const { return params_; }

    private:
        void collision_streaming_step() {
            const double tau_inv = 1.0 / params_.tau;
            const __m256d tau_inv_vec = _mm256_set1_pd(tau_inv);

            // Parallelise over outer y-loop for better work distribution
            // avoid memory contention in streaming step
            #pragma omp parallel
            {
                // Thread-local variables to avoid false sharing
                alignas(32) double f_eq_temp[8];
                alignas(32) double f_post_collision[Q];

                #pragma omp for schedule(dynamic, 4)
                for (int y = 0; y < grid_.local_ny(); ++y) {
                    #pragma GCC ivdep
                    for (int x = 0; x < grid_.local_nx(); ++x) {
                        double* f_current_ptr = grid_.f_current_ptr(x, y);

                        const __m256d f_1234 = _mm256_loadu_pd(&f_current_ptr[1]);
                        const __m256d f_5678 = _mm256_loadu_pd(&f_current_ptr[5]);

                        double rho_val = f_current_ptr[0];
                        double ux_val = 0.0;
                        double uy_val = 0.0;

                        alignas(32) double f_1234_arr[4], f_5678_arr[4];
                        _mm256_store_pd(f_1234_arr, f_1234);
                        _mm256_store_pd(f_5678_arr, f_5678);

                        rho_val += f_1234_arr[0] + f_1234_arr[1] + f_1234_arr[2] + f_1234_arr[3] +
                                  f_5678_arr[0] + f_5678_arr[1] + f_5678_arr[2] + f_5678_arr[3];

                        ux_val = f_1234_arr[0] - f_1234_arr[2] + f_5678_arr[0] - f_5678_arr[1] - f_5678_arr[2] + f_5678_arr[3];
                        uy_val = f_1234_arr[1] - f_1234_arr[3] + f_5678_arr[0] + f_5678_arr[1] - f_5678_arr[2] - f_5678_arr[3];

                        ux_val += 0.5 * params_.force_x;
                        uy_val += 0.5 * params_.force_y;

                        const double rho_inv = (rho_val < 1e-9) ? 1.0/1e-9 : 1.0/rho_val;
                        ux_val *= rho_inv;
                        uy_val *= rho_inv;

                        grid_.rho(x, y) = rho_val;
                        grid_.ux(x, y) = ux_val;
                        grid_.uy(x, y) = uy_val;

                        const double f0_eq = equilibrium_with_force_scalar(rho_val, ux_val, uy_val, params_.force_x, params_.force_y);
                        f_post_collision[0] = f_current_ptr[0] - tau_inv * (f_current_ptr[0] - f0_eq);

                        equilibrium_with_force_simd(rho_val, ux_val, uy_val, params_.force_x, params_.force_y, f_eq_temp);
                        const __m256d f_eq_1234 = _mm256_loadu_pd(&f_eq_temp[0]);
                        const __m256d f_eq_5678 = _mm256_loadu_pd(&f_eq_temp[4]);

                        const __m256d diff_1234 = _mm256_sub_pd(f_1234, f_eq_1234);
                        const __m256d diff_5678 = _mm256_sub_pd(f_5678, f_eq_5678);

                        const __m256d collision_1234 = _mm256_sub_pd(f_1234, _mm256_mul_pd(tau_inv_vec, diff_1234));
                        const __m256d collision_5678 = _mm256_sub_pd(f_5678, _mm256_mul_pd(tau_inv_vec, diff_5678));

                        _mm256_storeu_pd(&f_post_collision[1], collision_1234);
                        _mm256_storeu_pd(&f_post_collision[5], collision_5678);

                        // propagate to neighbours
                        for (int i = 0; i < Q; ++i) {
                            // int x_dest = periodic_x(x + VELOCITIES[i][0], grid_.local_nx());
                            int x_dest = (x + VELOCITIES[i][0] + grid_.local_nx()) % grid_.local_nx();
                            int y_dest = y + VELOCITIES[i][1];

                            // Particles streaming across rank boundaries are handled by halo exchange
                            if (y_dest >= 0 && y_dest < grid_.local_ny()) {
                                grid_.f_next(x_dest, y_dest, i) = f_post_collision[i];
                            }
                        }
                    }
                }
            }
        }

        void apply_physical_boundary_conditions() {
#pragma omp parallel
            {
                // Bottom wall (y=0) is on rank 0
                if (grid_.mpi_rank() == 0) {
#pragma omp for schedule(static) nowait
                    for (int x = 0; x < grid_.local_nx(); ++x) {
                        // Explicit bounce-back for bottom wall on f_current
                        grid_.f_next(x, 0, 2) = grid_.f_next(x, 0, 4); // N from S
                        grid_.f_next(x, 0, 5) = grid_.f_next(x, 0, 7); // NE from SW
                        grid_.f_next(x, 0, 6) = grid_.f_next(x, 0, 8); // NW from SE
                    }
                }

                // Top wall (y=global_ny-1) is on the last rank
                if (grid_.mpi_rank() == grid_.mpi_size() - 1) {
                    const int top_y = grid_.local_ny() - 1;
#pragma omp for schedule(static) nowait
                    for (int x = 0; x < grid_.local_nx(); ++x) {
                        // Explicit bounce-back for top wall on f_current
                        grid_.f_next(x, top_y, 4) = grid_.f_next(x, top_y, 2); // S from N
                        grid_.f_next(x, top_y, 7) = grid_.f_next(x, top_y, 5); // SW from NE
                        grid_.f_next(x, top_y, 8) = grid_.f_next(x, top_y, 6); // SE from NW
                    }
                }
            }
        }
    };
}