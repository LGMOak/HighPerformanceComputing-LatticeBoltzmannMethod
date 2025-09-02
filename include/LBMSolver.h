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
                  << ", tau=" << params_.tau << ", nu=" << std::fixed << std::setprecision(5)
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
            const __m256d tau_inv_vec = _mm256_set1_pd(tau_inv);

            alignas(32) double f_eq_temp[8];
            alignas(32) double f_vals[8];

#pragma GCC ivdep
            for (int x = 0; x < params_.nx; ++x) {
                for (int y = 0; y < params_.ny; ++y) {
                    double* f_ptr_current = grid_.f_ptr(x, y);
                    double* f_temp_ptr_current = grid_.f_temp_ptr(x, y);

                    // Load distribution functions using SIMD
                    const __m256d f_1234 = _mm256_loadu_pd(&f_ptr_current[1]);
                    const __m256d f_5678 = _mm256_loadu_pd(&f_ptr_current[5]);

                    // Calculate macroscopic quantities - start with f0 (scalar)
                    double rho_val = f_ptr_current[0];
                    double ux_val = 0.0;
                    double uy_val = 0.0;

                    // Sum up directions 1-8 using SIMD
                    alignas(32) double f_1234_arr[4], f_5678_arr[4];
                    _mm256_store_pd(f_1234_arr, f_1234);
                    _mm256_store_pd(f_5678_arr, f_5678);

                    // Manual unroll for density
                    rho_val += f_1234_arr[0] + f_1234_arr[1] + f_1234_arr[2] + f_1234_arr[3] +
                              f_5678_arr[0] + f_5678_arr[1] + f_5678_arr[2] + f_5678_arr[3];

                    // Manual unroll for velocities (directions 1-8)
                    ux_val = f_1234_arr[0] - f_1234_arr[2] + f_5678_arr[0] - f_5678_arr[1] - f_5678_arr[2] + f_5678_arr[3];
                    uy_val = f_1234_arr[1] - f_1234_arr[3] + f_5678_arr[0] + f_5678_arr[1] - f_5678_arr[2] - f_5678_arr[3];

                    // Avoid division by zero
                    const double rho_inv = (rho_val < 1e-9) ? 1.0/1e-9 : 1.0/rho_val;
                    ux_val *= rho_inv;
                    uy_val *= rho_inv;

                    // Update quantities
                    grid_.rho(x, y) = rho_val;
                    grid_.ux(x, y) = ux_val;
                    grid_.uy(x, y) = uy_val;

                    // BGK collision - scalar for direction 0
                    const double f0_eq = equilibrium_with_force_scalar(rho_val, ux_val, uy_val, params_.force_x, params_.force_y);
                    f_temp_ptr_current[0] = f_ptr_current[0] - tau_inv * (f_ptr_current[0] - f0_eq);

                    // BGK collision - SIMD for directions 1-8
                    equilibrium_with_force_simd(rho_val, ux_val, uy_val, params_.force_x, params_.force_y, f_eq_temp);

                    // Load equilibrium values
                    const __m256d f_eq_1234 = _mm256_loadu_pd(&f_eq_temp[0]);
                    const __m256d f_eq_5678 = _mm256_loadu_pd(&f_eq_temp[4]);

                    // BGK collision: f_new = f - (1/tau) * (f - f_eq)
                    const __m256d diff_1234 = _mm256_sub_pd(f_1234, f_eq_1234);
                    const __m256d diff_5678 = _mm256_sub_pd(f_5678, f_eq_5678);

                    const __m256d collision_1234 = _mm256_sub_pd(f_1234, _mm256_mul_pd(tau_inv_vec, diff_1234));
                    const __m256d collision_5678 = _mm256_sub_pd(f_5678, _mm256_mul_pd(tau_inv_vec, diff_5678));

                    // Store results
                    _mm256_storeu_pd(&f_temp_ptr_current[1], collision_1234);
                    _mm256_storeu_pd(&f_temp_ptr_current[5], collision_5678);
                }
            }
        }

        void streaming_step() {
            // Clear f array using SIMD
            const __m256d zero = _mm256_setzero_pd();
            const size_t total_f_elements = grid_.nx() * grid_.ny() * Q;
            const size_t simd_end = (total_f_elements / 4) * 4;

            double* f_data = grid_.f_ptr(0, 0);
            for (size_t i = 0; i < simd_end; i += 4) {
                _mm256_storeu_pd(&f_data[i], zero);
            }
            // Handle remaining elements
            for (size_t i = simd_end; i < total_f_elements; ++i) {
                f_data[i] = 0.0;
            }

            // Stream from f_temp to f
            for (int x = 0; x < grid_.nx(); ++x) {
                for (int y = 0; y < grid_.ny(); ++y) {
                    double* f_temp_src = grid_.f_temp_ptr(x, y);

                    // Direction 0 (rest particle) - no streaming
                    grid_.f(x, y, 0) = f_temp_src[0];

                    // Directions 1-8 - streaming
                    for (int i = 1; i < Q; ++i) {
                        int x_dest = periodic_x(x + VELOCITIES[i][0], grid_.nx());
                        int y_dest = y + VELOCITIES[i][1];

                        if (y_dest >= 0 && y_dest < grid_.ny()) {
                            grid_.f(x_dest, y_dest, i) = f_temp_src[i];
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
                grid_.f(x, ceiling, 4) = grid_.f_temp(x, ceiling, 2); // Down from up
                grid_.f(x, ceiling, 7) = grid_.f_temp(x, ceiling, 5); // Down-left from up-right
                grid_.f(x, ceiling, 8) = grid_.f_temp(x, ceiling, 6); // Down-right from up-left
            }

            // Floor bounce back
#pragma GCC ivdep
            for (int x = 0; x < grid_.nx(); ++x) {
                grid_.f(x, 0, 2) = grid_.f_temp(x, 0, 4); // Up from down
                grid_.f(x, 0, 5) = grid_.f_temp(x, 0, 7); // Up-right from down-left
                grid_.f(x, 0, 6) = grid_.f_temp(x, 0, 8); // Up-left from down-right
            }
        }
    };
}