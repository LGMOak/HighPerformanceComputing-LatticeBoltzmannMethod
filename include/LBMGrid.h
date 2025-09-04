#pragma once

#include "LBMConfig.h"
#include "LBMUtils.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <immintrin.h>

namespace LBM {
    class Grid {
    private:
        int nx_, ny_;

        // Aligned memory
        alignas(32) std::vector<double> f_data_;
        double* f_current_;
        double* f_next_;

        // Macroscopic quantities
        alignas(32) std::vector<double> rho_;
        alignas(32) std::vector<double> ux_;
        alignas(32) std::vector<double> uy_;

    public:
        Grid(int nx, int ny) : nx_(nx), ny_(ny) {
            const size_t grid_size = static_cast<size_t>(nx_) * ny_;
            const size_t f_grid_size = grid_size * Q;

            f_data_.resize(f_grid_size * 2);
            f_current_ = f_data_.data();
            f_next_ = f_current_ + f_grid_size;

            rho_.resize(grid_size, 1.0);
            ux_.resize(grid_size, 0.0);
            uy_.resize(grid_size, 0.0);
        }

        inline double& f_current(int x, int y, int i) {
            return f_current_[(static_cast<size_t>(y) * nx_ + x) * Q + i];
        }
        inline const double& f_current(int x, int y, int i) const {
            return f_current_[(static_cast<size_t>(y) * nx_ + x) * Q + i];
        }

        inline double& f_next(int x, int y, int i) {
            return f_next_[(static_cast<size_t>(y) * nx_ + x) * Q + i];
        }

        inline double* f_current_ptr(int x, int y) {
            return &f_current_[(static_cast<size_t>(y) * nx_ + x) * Q];
        }

        inline double* f_next_ptr(int x, int y) {
            return &f_next_[(static_cast<size_t>(y) * nx_ + x) * Q];
        }

        inline double& rho(int x, int y) {
            return rho_[(static_cast<size_t>(y) * nx_ + x)];
        }
        inline const double& rho(int x, int y) const {
            return rho_[(static_cast<size_t>(y) * nx_ + x)];
        }

        inline double& ux(int x, int y) {
            return ux_[(static_cast<size_t>(y) * nx_ + x)];
        }
        inline const double& ux(int x, int y) const {
            return ux_[(static_cast<size_t>(y) * nx_ + x)];
        }

        inline double& uy(int x, int y) {
            return uy_[(static_cast<size_t>(y) * nx_ + x)];
        }
        inline const double& uy(int x, int y) const {
            return uy_[(static_cast<size_t>(y) * nx_ + x)];
        }

        int nx() const { return nx_; }
        int ny() const { return ny_; }

        void initialise() {
            alignas(32) double f_eq_temp[8];

            #pragma GCC ivdep
            for (int x = 0; x < nx_; ++x) {
                for (int y = 0; y < ny_; ++y) {
                    const double rho_val = rho(x, y);
                    const double ux_val = ux(x, y);
                    const double uy_val = uy(x, y);

                    double* f_ptr_local = f_current_ptr(x, y);

                    f_ptr_local[0] = equilibrium_scalar(rho_val, ux_val, uy_val);
                    equilibrium_simd(rho_val, ux_val, uy_val, f_eq_temp);

                    _mm256_storeu_pd(&f_ptr_local[1], _mm256_loadu_pd(&f_eq_temp[0]));
                    _mm256_storeu_pd(&f_ptr_local[5], _mm256_loadu_pd(&f_eq_temp[4]));
                }
            }
        }

        void swap_f_arrays() {
            std::swap(f_current_, f_next_);
        }

        bool check_stability() const {
            const __m256d max_val = _mm256_set1_pd(1e5);
            const __m256d min_val = _mm256_set1_pd(-1e5);

            const size_t total_size = static_cast<size_t>(nx_) * ny_ * Q;
            const size_t simd_end = (total_size / 4) * 4;
            const double* f_data = f_current_;

            for (size_t i = 0; i < simd_end; i += 4) {
                __m256d values = _mm256_loadu_pd(&f_data[i]);
                __m256d is_finite = _mm256_cmp_pd(values, values, _CMP_EQ_OQ);
                if (_mm256_movemask_pd(is_finite) != 0xF) {
                    return false;
                }
                __m256d too_large = _mm256_cmp_pd(values, max_val, _CMP_GT_OQ);
                __m256d too_small = _mm256_cmp_pd(values, min_val, _CMP_LT_OQ);
                if (_mm256_movemask_pd(too_large) || _mm256_movemask_pd(too_small)) {
                    return false;
                }
            }

            for (size_t i = simd_end; i < total_size; ++i) {
                if (!is_stable(f_data[i])) {
                    return false;
                }
            }
            return true;
        }

        double max_velocity() const {
            __m256d max_vel_sq_vec = _mm256_setzero_pd();
            const size_t total_elements = static_cast<size_t>(nx_) * ny_;
            const size_t simd_end = (total_elements / 4) * 4;

            for (size_t i = 0; i < simd_end; i += 4) {
                __m256d ux_vec = _mm256_loadu_pd(&ux_[i]);
                __m256d uy_vec = _mm256_loadu_pd(&uy_[i]);
                __m256d vel_sq = _mm256_add_pd(_mm256_mul_pd(ux_vec, ux_vec), _mm256_mul_pd(uy_vec, uy_vec));
                max_vel_sq_vec = _mm256_max_pd(max_vel_sq_vec, vel_sq);
            }

            alignas(32) double max_vals[4];
            _mm256_store_pd(max_vals, max_vel_sq_vec);
            double max_vel_sq = std::max({max_vals[0], max_vals[1], max_vals[2], max_vals[3]});

            for (size_t i = simd_end; i < total_elements; ++i) {
                const double vel_sq = ux_[i] * ux_[i] + uy_[i] * uy_[i];
                max_vel_sq = std::max(max_vel_sq, vel_sq);
            }

            return std::sqrt(max_vel_sq);
        }
    };
}