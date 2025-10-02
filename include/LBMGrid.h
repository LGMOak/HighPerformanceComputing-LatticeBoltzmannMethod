#pragma once

#include "LBMConfig.h"
#include "LBMUtils.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

namespace LBM {
    class Grid {
    private:
        int global_nx_, global_ny_;
        int local_nx_, local_ny_;
        int ghost_nodes_;

        // MPI info
        int mpi_rank_, mpi_size_;
        int y_start_, y_end_;

        // total rows including ghost nodes
        int total_local_y_;

        // Aligned memory
        alignas(32) std::vector<double> f_data_;
        double* f_current_;
        double* f_next_;

        // Macroscopic quantities
        alignas(32) std::vector<double> rho_;
        alignas(32) std::vector<double> ux_;
        alignas(32) std::vector<double> uy_;

        // Communication buffers (aligned)
        alignas(32) std::vector<double> send_buffer_up_;
        alignas(32) std::vector<double> send_buffer_down_;
        alignas(32) std::vector<double> recv_buffer_up_;
        alignas(32) std::vector<double> recv_buffer_down_;

        // MPI requests for non-blocking communication
        std::vector<MPI_Request> mpi_requests_;

    public:
        Grid(int nx, int ny) : global_nx_(nx), global_ny_(ny), ghost_nodes_(1) {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

            // simple 1D decomposition
            local_nx_ = global_nx_; // x not decomposed
            int base_local_ny = global_ny_ / mpi_size_;
            int remainder = global_ny_ % mpi_size_;

            if (mpi_rank_ < remainder) {
                local_ny_ = base_local_ny + 1;
                y_start_ = mpi_rank_ * local_ny_;
            } else {
                local_ny_ = base_local_ny;
                y_start_ = remainder * (base_local_ny + 1) + (mpi_rank_ - remainder) * base_local_ny;
            }
            y_end_ = y_start_ + local_ny_ - 1;

            // Total local size including ghost nodes
            int total_local_ny = local_ny_ + 2 * ghost_nodes_;
            const size_t grid_size = static_cast<size_t>(local_nx_) * total_local_ny;
            const size_t f_grid_size = grid_size * Q;

            // distribution functions aligned memory
            f_data_.resize(f_grid_size * 2);
            f_current_ = f_data_.data();
            f_next_ = f_current_ + f_grid_size;

            // macroscopic quantities
            rho_.resize(grid_size, 1.0);
            ux_.resize(grid_size, 0.0);
            uy_.resize(grid_size, 0.0);

            // Initialize communication buffers
            int buffer_size = local_nx_ * Q;
            send_buffer_up_.resize(buffer_size);
            send_buffer_down_.resize(buffer_size);
            recv_buffer_up_.resize(buffer_size);
            recv_buffer_down_.resize(buffer_size);
            mpi_requests_.resize(4);

            if (mpi_rank_ == 0) {
                printf("Optimized MPI Grid Setup:\n");
                printf("  Global: %dx%d\n", global_nx_, global_ny_);
                printf("  MPI ranks: %d\n", mpi_size_);
                printf("  Local NY: %d (remainder: %d)\n", base_local_ny, remainder);
                printf("  OpenMP threads per rank: %d\n", omp_get_max_threads());
            }
        }

        // Fast inline index calculation
        inline size_t get_index(int x, int local_y) const {
            return (static_cast<size_t>(local_y + ghost_nodes_) * local_nx_ + x);
        }

        inline size_t get_f_index(int x, int local_y, int i) const {
            return get_index(x, local_y) * Q + i;
        }

        // accessors for distribution functions
        inline double& f_current(int x, int local_y, int i) {
            return f_current_[get_f_index(x, local_y, i)];
        }

        inline const double& f_current(int x, int local_y, int i) const {
            return f_current_[get_f_index(x, local_y, i)];
        }

        inline double& f_next(int x, int local_y, int i) {
            return f_next_[get_f_index(x, local_y, i)];
        }

        // Pointer access for SIMD operations
        inline double* f_current_ptr(int x, int local_y) {
            return &f_current_[get_f_index(x, local_y, 0)];
        }

        inline double* f_next_ptr(int x, int local_y) {
            return &f_next_[get_f_index(x, local_y, 0)];
        }

        // Macroscopic quantity accessors
        inline double& rho(int x, int local_y) {
            return rho_[get_index(x, local_y)];
        }

        inline const double& rho(int x, int local_y) const {
            return rho_[get_index(x, local_y)];
        }

        inline double& ux(int x, int local_y) {
            return ux_[get_index(x, local_y)];
        }

        inline const double& ux(int x, int local_y) const {
            return ux_[get_index(x, local_y)];
        }

        inline double& uy(int x, int local_y) {
            return uy_[get_index(x, local_y)];
        }

        inline const double& uy(int x, int local_y) const {
            return uy_[get_index(x, local_y)];
        }

        void swap_f_arrays() {
            std::swap(f_current_, f_next_);
        }

        // Getters
        int local_nx() const { return local_nx_; }
        int local_ny() const { return local_ny_; }
        int global_nx() const { return global_nx_; }
        int global_ny() const { return global_ny_; }
        int mpi_rank() const { return mpi_rank_; }
        int mpi_size() const { return mpi_size_; }
        int y_start() const { return y_start_; }
        int y_end() const { return y_end_; }

        void initialise() {
            alignas(32) double f_eq_temp[8];

            #pragma omp parallel for schedule(static) private(f_eq_temp)
            for (int x = 0; x < local_nx_; ++x) {
#pragma GCC ivdep
                for (int y = 0; y < local_ny_; ++y) {
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

        // Non-blocking halo exchange
        void start_halo_exchange() {
            const int buffer_size = local_nx_ * Q;
            int req_idx = 0;

            // Pack boundary data efficiently
            pack_boundary_data();

            // Non-blocking sends and receives
            if (mpi_rank_ < mpi_size_ - 1) {
                MPI_Isend(send_buffer_up_.data(), buffer_size, MPI_DOUBLE,
                         mpi_rank_ + 1, 0, MPI_COMM_WORLD, &mpi_requests_[req_idx++]);
                MPI_Irecv(recv_buffer_up_.data(), buffer_size, MPI_DOUBLE,
                         mpi_rank_ + 1, 1, MPI_COMM_WORLD, &mpi_requests_[req_idx++]);
            }

            if (mpi_rank_ > 0) {
                MPI_Isend(send_buffer_down_.data(), buffer_size, MPI_DOUBLE,
                         mpi_rank_ - 1, 1, MPI_COMM_WORLD, &mpi_requests_[req_idx++]);
                MPI_Irecv(recv_buffer_down_.data(), buffer_size, MPI_DOUBLE,
                         mpi_rank_ - 1, 0, MPI_COMM_WORLD, &mpi_requests_[req_idx++]);
            }
        }

        void complete_halo_exchange() {
            int num_requests = 0;
            if (mpi_rank_ < mpi_size_ - 1) num_requests += 2;
            if (mpi_rank_ > 0) num_requests += 2;

            if (num_requests > 0) {
                MPI_Waitall(num_requests, mpi_requests_.data(), MPI_STATUSES_IGNORE);
            }

            // Unpack received data
            unpack_boundary_data();
        }

        bool check_stability() const {
            const __m256d max_val = _mm256_set1_pd(1e5);
            const __m256d min_val = _mm256_set1_pd(-1e5);

            bool local_stable = true;
            // const size_t total_size = static_cast<size_t>(local_nx_) * local_ny_ * Q;
            const size_t total_size = static_cast<size_t>(local_nx_) * (static_cast<size_t>(local_ny_) + 2 * ghost_nodes_) * Q;
            const size_t simd_end = (total_size / 4) * 4;

#pragma omp parallel for reduction(&& : local_stable)
            for (size_t i = 0; i < simd_end; i += 4) {
                if (!local_stable) continue;

                __m256d values = _mm256_loadu_pd(&f_current_[i]);
                __m256d is_finite = _mm256_cmp_pd(values, values, _CMP_EQ_OQ);
                if (_mm256_movemask_pd(is_finite) != 0xF) {
                    local_stable = false;
                    continue;
                }

                __m256d too_large = _mm256_cmp_pd(values, max_val, _CMP_GT_OQ);
                __m256d too_small = _mm256_cmp_pd(values, min_val, _CMP_LT_OQ);
                if (_mm256_movemask_pd(too_large) || _mm256_movemask_pd(too_small)) {
                    local_stable = false;
                }
            }

            // Check remaining elements
            for (size_t i = simd_end; i < total_size && local_stable; ++i) {
                if (!is_stable(f_current_[i])) {
                    local_stable = false;
                }
            }

            int local_stable_int = local_stable ? 1 : 0;
            int global_stable_int;
            MPI_Allreduce(&local_stable_int, &global_stable_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);


            return global_stable_int == 1;
        }


        double max_velocity() const {
            double local_max_sq = 0.0;
#pragma omp parallel reduction(max : local_max_sq)
            {
                __m256d thread_max_vec = _mm256_setzero_pd(); // Each thread has its own SIMD max vector

                // Decompose the loop to be thread-safe
#pragma omp for nowait
                for (size_t i = 0; i < (static_cast<size_t>(local_nx_) * local_ny_ / 4) * 4; i += 4) {
                    __m256d ux_vec = _mm256_loadu_pd(&ux_[i]);
                    __m256d uy_vec = _mm256_loadu_pd(&uy_[i]);
                    __m256d vel_sq = _mm256_add_pd(_mm256_mul_pd(ux_vec, ux_vec),
                                                  _mm256_mul_pd(uy_vec, uy_vec));
                    thread_max_vec = _mm256_max_pd(thread_max_vec, vel_sq);
                }

                // After the loop, find the max value within the thread's SIMD vector
                alignas(32) double max_vals[4];
                _mm256_store_pd(max_vals, thread_max_vec);
                double thread_max_scalar = std::max({max_vals[0], max_vals[1], max_vals[2], max_vals[3]});

                // The reduction now happens on the scalar 'local_max_sq'
                local_max_sq = std::max(local_max_sq, thread_max_scalar);
            }

            // Handle the few remaining elements outside the parallel region (simpler)
            const size_t total_elements = static_cast<size_t>(local_nx_) * local_ny_;
            const size_t simd_end = (total_elements / 4) * 4;
            for (size_t i = simd_end; i < total_elements; ++i) {
                const double vel_sq = ux_[i] * ux_[i] + uy_[i] * uy_[i];
                local_max_sq = std::max(local_max_sq, vel_sq);
            }

            // Now, perform the MPI reduction on the final scalar result
            double global_max_sq;
            MPI_Allreduce(&local_max_sq, &global_max_sq, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            return std::sqrt(global_max_sq);
        }

    private:
        void pack_boundary_data() {
            // Pack top boundary (last physical row) for sending to the rank above
            if (mpi_rank_ < mpi_size_ - 1) {
                const int top_y = local_ny_ - 1;
                int buf_idx = 0;
                for (int x = 0; x < local_nx_; ++x) {
                    for (int i = 0; i < Q; ++i) {
                        send_buffer_up_[buf_idx++] = f_next(x, top_y, i);
                    }
                }
            }

            // Pack bottom boundary (first physical row) for sending to the rank below
            if (mpi_rank_ > 0) {
                int buf_idx = 0;
                for (int x = 0; x < local_nx_; ++x) {
                    for (int i = 0; i < Q; ++i) {
                        send_buffer_down_[buf_idx++] = f_next(x, 0, i);
                    }
                }
            }
        }

        void unpack_boundary_data() {
            // Receive and apply particles that streamed from neighbor

            if (mpi_rank_ < mpi_size_ - 1) {
                // Receive from upper neighbor (their bottom row streams down to us)
                int buf_idx = 0;
                for (int x = 0; x < local_nx_; ++x) {
                    for (int i = 0; i < Q; ++i) {
                        // Only accept particles moving DOWNWARD
                        if (VELOCITIES[i][1] < 0) {
                            f_next(x, local_ny_ - 1, i) = recv_buffer_up_[buf_idx];
                        }
                        buf_idx++;
                    }
                }
            }

            if (mpi_rank_ > 0) {
                // Receive from lower neighbor (their top row streams up to us)
                int buf_idx = 0;
                for (int x = 0; x < local_nx_; ++x) {
                    for (int i = 0; i < Q; ++i) {
                        // Only accept particles moving UPWARD
                        if (VELOCITIES[i][1] > 0) {
                            f_next(x, 0, i) = recv_buffer_down_[buf_idx];
                        }
                        buf_idx++;
                    }
                }
            }
        }
    };
}