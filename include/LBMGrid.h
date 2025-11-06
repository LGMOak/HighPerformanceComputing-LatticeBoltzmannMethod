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

        // total size including ghost cells
        int total_nx_, total_ny_;
        static constexpr int GHOST_LAYERS = 1;

        MPI_Comm cart_comm_;
        int mpi_rank_, mpi_size_;
        int rank_x_, rank_y_;
        int px_, py_;

        int north_, south_, east_, west_;

        // Global coordinates of first interior cell
        int x_start_, y_start_;

        // f_current_ holds the post-streaming/BC state
        // f_next_ holds the post-collision state
        alignas(32) std::vector<double> f_data_;
        double* f_current_;
        double* f_next_;

        // Macroscopic quantities (interior cells only)
        alignas(32) std::vector<double> rho_;
        alignas(32) std::vector<double> ux_;
        alignas(32) std::vector<double> uy_;

        // Communication buffers for ghost cell exchange
        alignas(32) std::vector<double> send_buffer_north_;
        alignas(32) std::vector<double> send_buffer_south_;
        alignas(32) std::vector<double> send_buffer_east_;
        alignas(32) std::vector<double> send_buffer_west_;
        alignas(32) std::vector<double> recv_buffer_north_;
        alignas(32) std::vector<double> recv_buffer_south_;
        alignas(32) std::vector<double> recv_buffer_east_;
        alignas(32) std::vector<double> recv_buffer_west_;

        std::vector<MPI_Request> mpi_requests_;
        std::vector<bool> is_solid_;

    public:
        Grid(int nx, int ny) : global_nx_(nx), global_ny_(ny) {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

            initialise_2d_topology();

            total_nx_ = local_nx_ + 2 * GHOST_LAYERS;
            total_ny_ = local_ny_ + 2 * GHOST_LAYERS;

            const size_t total_grid_size = static_cast<size_t>(total_nx_) * total_ny_;
            const size_t f_total_size = total_grid_size * Q;
            f_data_.resize(f_total_size * 2);
            f_current_ = f_data_.data();
            f_next_ = f_current_ + f_total_size;

            const size_t interior_size = static_cast<size_t>(local_nx_) * local_ny_;
            rho_.resize(interior_size, 1.0);
            ux_.resize(interior_size, 0.0);
            uy_.resize(interior_size, 0.0);
            is_solid_.resize(interior_size, false);

            // allocate communication buffers for full exchange (all 9 populations per cell)
            const int EW_BUF_SIZE = local_ny_ * Q;
            const int NS_BUF_SIZE = local_nx_ * Q;
            send_buffer_north_.resize(NS_BUF_SIZE);
            recv_buffer_north_.resize(NS_BUF_SIZE);
            send_buffer_south_.resize(NS_BUF_SIZE);
            recv_buffer_south_.resize(NS_BUF_SIZE);
            send_buffer_east_.resize(EW_BUF_SIZE);
            recv_buffer_east_.resize(EW_BUF_SIZE);
            send_buffer_west_.resize(EW_BUF_SIZE);
            recv_buffer_west_.resize(EW_BUF_SIZE);

            mpi_requests_.resize(8);

            if (mpi_rank_ == 0) {
                printf("2D MPI + OpenMP Grid\n");
                printf("  Global domain: %dx%d\n", global_nx_, global_ny_);
                printf("  MPI ranks: %d (%dx%d grid)\n", mpi_size_, px_, py_);
                printf("  Local interior per rank: %dx%d\n", local_nx_, local_ny_);
                printf("  Local with ghosts: %dx%d\n", total_nx_, total_ny_);
                printf("  Ghost layers: %d\n", GHOST_LAYERS);
                printf("  OpenMP threads per rank: %d\n", omp_get_max_threads());
                printf("  Memory per rank: %.2f MB\n",
                       (f_total_size * 2 * sizeof(double)) / (1024.0 * 1024.0));
            }
        }

        inline size_t get_f_index(int x, int y, int i) const {
            return (static_cast<size_t>(y) * total_nx_ + x) * Q + i;
        }

        inline size_t get_interior_index(int x, int y) const {
            return static_cast<size_t>(y) * local_nx_ + x;
        }

        // accessors for distribution functions
        // non-const allows writing over values
        inline double& f_current(int x, int y, int i) { return f_current_[get_f_index(x, y, i)]; }
        inline const double& f_current(int x, int y, int i) const { return f_current_[get_f_index(x, y, i)]; }
        inline double& f_next(int x, int y, int i) { return f_next_[get_f_index(x, y, i)]; }
        inline const double& f_next(int x, int y, int i) const { return f_next_[get_f_index(x, y, i)]; }

        inline double* f_current_ptr(int x, int y) { return &f_current_[get_f_index(x, y, 0)]; }
        inline double* f_next_ptr(int x, int y) { return &f_next_[get_f_index(x, y, 0)]; }

        // accessors for macroscopic quantities
        inline double& rho(int x, int y) { return rho_[get_interior_index(x, y)]; }
        inline const double& rho(int x, int y) const { return rho_[get_interior_index(x, y)]; }
        inline double& ux(int x, int y) { return ux_[get_interior_index(x, y)]; }
        inline const double& ux(int x, int y) const { return ux_[get_interior_index(x, y)]; }
        inline double& uy(int x, int y) { return uy_[get_interior_index(x, y)]; }
        inline const double& uy(int x, int y) const { return uy_[get_interior_index(x, y)]; }

        // getters that are needed
        int x_start() const { return x_start_; }
        int y_start() const { return y_start_; }
        int local_nx() const { return local_nx_; }
        int local_ny() const { return local_ny_; }
        int total_nx() const { return total_nx_; }
        int total_ny() const { return total_ny_; }
        int mpi_rank() const { return mpi_rank_; }
        bool is_bottom_boundary() const { return rank_y_ == 0; }
        bool is_top_boundary() const { return rank_y_ == py_ - 1; }
        int global_nx() const { return global_nx_; }
        int global_ny() const { return global_ny_; }
        int mpi_size() const { return mpi_size_; }

        inline bool is_solid(int x, int y) const {
            return is_solid_[get_interior_index(x, y)];
        }

        bool is_left_boundary() const { return rank_x_ == 0; }
        bool is_right_boundary() const { return rank_x_ == px_ - 1; }

        void setup_geometry(const SimulationParams& params) {
            int cyl_x = params.get_cylinder_x();
            int cyl_y = params.get_cylinder_y();
            int cyl_r = params.get_cylinder_radius_cells();

            int local_solid_count = 0;
#pragma omp parallel for schedule(static) reduction(+:local_solid_count)
            for (int y = 0; y < local_ny_; ++y) {
                for (int x = 0; x < local_nx_; ++x) {
                    int global_x = x_start_ + x;
                    int global_y = y_start_ + y;

                    double dx = global_x - cyl_x;
                    double dy = global_y - cyl_y;
                    double dist_sq = dx*dx + dy*dy;

                    if (dist_sq <= cyl_r * cyl_r) {
                        is_solid_[get_interior_index(x, y)] = true;
                        local_solid_count++;
                    }
                }
            }
            int global_solid_count;
            MPI_Reduce(&local_solid_count, &global_solid_count, 1, MPI_INT,
                      MPI_SUM, 0, MPI_COMM_WORLD);

            if (mpi_rank_ == 0) {
                printf("  Cylinder: center=(%d,%d), radius=%d cells\n",
                       cyl_x, cyl_y, cyl_r);
                printf("  Solid cells: %d\n", global_solid_count);
            }
        }

        void initialise(double inlet_u) {
            alignas(32) double f_eq_temp[8];

            // Initialise all cells (interior + ghosts) for both buffers
            // This prevents garbage values in ghost cells on first timestep
        #pragma omp parallel for schedule(static) private(f_eq_temp)
            for (int gy = 0; gy < total_ny_; ++gy) {
                for (int gx = 0; gx < total_nx_; ++gx) {
                    // initialise with uniform inlet flow everywhere
                    double rho_init = 1.0;
                    double ux_init = inlet_u;
                    double uy_init = 0.0;

                    // Get pointers to both buffers
                    double* f_curr = f_current_ptr(gx, gy);
                    double* f_nxt = f_next_ptr(gx, gy);

                    // Compute equilibrium once
                    f_curr[0] = equilibrium_scalar(rho_init, ux_init, uy_init);
                    equilibrium_simd(rho_init, ux_init, uy_init, f_eq_temp);
                    _mm256_storeu_pd(&f_curr[1], _mm256_loadu_pd(&f_eq_temp[0]));
                    _mm256_storeu_pd(&f_curr[5], _mm256_loadu_pd(&f_eq_temp[4]));

                    // Copy to f_next
                    for (int i = 0; i < Q; ++i) {
                        f_nxt[i] = f_curr[i];
                    }
                }
            }

            // Set interior macroscopic quantities and handle solid cells
        #pragma omp parallel for schedule(static) collapse(2)
            for (int y = 0; y < local_ny_; ++y) {
                for (int x = 0; x < local_nx_; ++x) {
                    if (!is_solid(x, y)) {
                        ux(x, y) = inlet_u;
                        uy(x, y) = 0.0;
                        rho(x, y) = 1.0;
                    } else {
                        // Solid cells: zero velocity
                        ux(x, y) = 0.0;
                        uy(x, y) = 0.0;
                        rho(x, y) = 1.0;

                        // Initialise solid cell distributions to zero velocity equilibrium
                        int gx = x + GHOST_LAYERS;
                        int gy = y + GHOST_LAYERS;
                        double* f_curr = f_current_ptr(gx, gy);
                        double* f_nxt = f_next_ptr(gx, gy);

                        f_curr[0] = equilibrium_scalar(1.0, 0.0, 0.0);
                        equilibrium_simd(1.0, 0.0, 0.0, f_eq_temp);
                        _mm256_storeu_pd(&f_curr[1], _mm256_loadu_pd(&f_eq_temp[0]));
                        _mm256_storeu_pd(&f_curr[5], _mm256_loadu_pd(&f_eq_temp[4]));

                        for (int i = 0; i < Q; ++i) {
                            f_nxt[i] = f_curr[i];
                        }
                    }
                }
            }
        }

        // exchanges 9 velocity vectors in ghost cells
        void exchange_ghost_cells() {
            pack_ghost_cells();

            int req_idx = 0;

            // east-west communication
            MPI_Isend(send_buffer_east_.data(), send_buffer_east_.size(), MPI_DOUBLE,
                     east_, 0, cart_comm_, &mpi_requests_[req_idx++]);
            MPI_Irecv(recv_buffer_west_.data(), recv_buffer_west_.size(), MPI_DOUBLE,
                     west_, 0, cart_comm_, &mpi_requests_[req_idx++]);
            MPI_Isend(send_buffer_west_.data(), send_buffer_west_.size(), MPI_DOUBLE,
                     west_, 1, cart_comm_, &mpi_requests_[req_idx++]);
            MPI_Irecv(recv_buffer_east_.data(), recv_buffer_east_.size(), MPI_DOUBLE,
                     east_, 1, cart_comm_, &mpi_requests_[req_idx++]);

            // north-south communication
            if (north_ != MPI_PROC_NULL) {
                MPI_Isend(send_buffer_north_.data(), send_buffer_north_.size(), MPI_DOUBLE,
                         north_, 2, cart_comm_, &mpi_requests_[req_idx++]);
                MPI_Irecv(recv_buffer_north_.data(), recv_buffer_north_.size(), MPI_DOUBLE,
                         north_, 3, cart_comm_, &mpi_requests_[req_idx++]);
            }
            if (south_ != MPI_PROC_NULL) {
                MPI_Isend(send_buffer_south_.data(), send_buffer_south_.size(), MPI_DOUBLE,
                         south_, 3, cart_comm_, &mpi_requests_[req_idx++]);
                MPI_Irecv(recv_buffer_south_.data(), recv_buffer_south_.size(), MPI_DOUBLE,
                         south_, 2, cart_comm_, &mpi_requests_[req_idx++]);
            }

            if (req_idx > 0) {
                MPI_Waitall(req_idx, mpi_requests_.data(), MPI_STATUSES_IGNORE);
            }

            unpack_ghost_cells();
        }

        bool check_stability() const {
            const __m256d max_val = _mm256_set1_pd(1e5);
            const __m256d min_val = _mm256_set1_pd(-1e5);

            bool local_stable = true;
            const size_t total_size = static_cast<size_t>(total_nx_) * total_ny_ * Q;
            const size_t simd_end = (total_size / 4) * 4;

            // combines boolean results from all threads on logical AND
            #pragma omp parallel for schedule(static) reduction(&& : local_stable)
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
            for (size_t i = simd_end; i < total_size && local_stable; ++i) {
                if (!is_stable(f_current_[i])) local_stable = false;
            }

            int local_stable_int = local_stable ? 1 : 0;
            int global_stable_int;
            MPI_Allreduce(&local_stable_int, &global_stable_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            return global_stable_int == 1;
        }

        double max_velocity() const {
            double local_max_sq = 0.0;
            // max operator across all threads
            #pragma omp parallel reduction(max : local_max_sq)
            {
                __m256d thread_max_vec = _mm256_setzero_pd();
                // ignore barrier - threads continue working without waiting for other threads to finish
                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < (static_cast<size_t>(local_nx_) * local_ny_ / 4) * 4; i += 4) {
                    __m256d ux_vec = _mm256_loadu_pd(&ux_[i]);
                    __m256d uy_vec = _mm256_loadu_pd(&uy_[i]);
                    __m256d vel_sq = _mm256_add_pd(_mm256_mul_pd(ux_vec, ux_vec), _mm256_mul_pd(uy_vec, uy_vec));
                    thread_max_vec = _mm256_max_pd(thread_max_vec, vel_sq);
                }
                alignas(32) double max_vals[4];
                _mm256_store_pd(max_vals, thread_max_vec);
                local_max_sq = std::max({max_vals[0], max_vals[1], max_vals[2], max_vals[3]});
            }
            const size_t total = static_cast<size_t>(local_nx_) * local_ny_;
            for (size_t i = (total / 4) * 4; i < total; ++i) {
                local_max_sq = std::max(local_max_sq, ux_[i] * ux_[i] + uy_[i] * uy_[i]);
            }
            double global_max_sq;
            MPI_Allreduce(&local_max_sq, &global_max_sq, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            return std::sqrt(global_max_sq);
        }

    private:
        void initialise_2d_topology() {
            find_optimal_decomposition(mpi_size_, global_nx_, global_ny_, px_, py_);
            int dims[2] = {px_, py_};
            int periods[2] = {0, 0};  // no more periodic boundary conditions
            int reorder = 1;
            MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm_);
            MPI_Comm_rank(cart_comm_, &mpi_rank_);
            int coords[2];
            MPI_Cart_coords(cart_comm_, mpi_rank_, 2, coords);
            rank_x_ = coords[0];
            rank_y_ = coords[1];
            local_nx_ = global_nx_ / px_;
            local_ny_ = global_ny_ / py_;
            x_start_ = rank_x_ * local_nx_;
            y_start_ = rank_y_ * local_ny_;
            MPI_Cart_shift(cart_comm_, 0, 1, &west_, &east_);
            MPI_Cart_shift(cart_comm_, 1, 1, &south_, &north_);
        }

        void find_optimal_decomposition(int nprocs, int nx, int ny, int& px_out, int& py_out) {
            // lowest suface-to-volume ratio
            // maintains global aspect ratio
            double aspect_ratio = static_cast<double>(nx) / ny;
            double best_score = 1e9;
            int best_px = 1, best_py = nprocs;
            for (int px = 1; px <= nprocs; ++px) {
                if (nprocs % px != 0) continue;
                int py = nprocs / px;
                if (nx % px != 0 || ny % py != 0) continue;
                int lnx = nx / px;
                int lny = ny / py;
                double surface = 2.0 * (lnx + lny);
                double volume = lnx * lny;
                double local_aspect = static_cast<double>(lnx) / lny;
                // heuristic - log is symmetric about 0
                double aspect_penalty = std::abs(std::log(local_aspect / aspect_ratio));
                double score = surface / std::sqrt(volume) + aspect_penalty;
                if (score < best_score) {
                    best_score = score;
                    best_px = px;
                    best_py = py;
                }
            }
            px_out = best_px;
            py_out = best_py;
        }

        // packs 9 vectors from boundary cells
        void pack_ghost_cells() {
            const int G = GHOST_LAYERS;
            int buf_idx;

            // east boundary
            buf_idx = 0;
            for (int y = 0; y < local_ny_; ++y) {
                int gx = local_nx_ - 1 + G;
                int gy = y + G;
                for (int i = 0; i < Q; ++i) {
                    send_buffer_east_[buf_idx++] = f_next(gx, gy, i);
                }
            }

            // west boundary
            buf_idx = 0;
            for (int y = 0; y < local_ny_; ++y) {
                int gx = G;
                int gy = y + G;
                for (int i = 0; i < Q; ++i) {
                    send_buffer_west_[buf_idx++] = f_next(gx, gy, i);
                }
            }

            // north boundary
            if (north_ != MPI_PROC_NULL) {
                buf_idx = 0;
                for (int x = 0; x < local_nx_; ++x) {
                    int gx = x + G;
                    int gy = local_ny_ - 1 + G;
                    for (int i = 0; i < Q; ++i) {
                        send_buffer_north_[buf_idx++] = f_next(gx, gy, i);
                    }
                }
            }

            // south boundary
            if (south_ != MPI_PROC_NULL) {
                buf_idx = 0;
                for (int x = 0; x < local_nx_; ++x) {
                    int gx = x + G;
                    int gy = G;
                    for (int i = 0; i < Q; ++i) {
                        send_buffer_south_[buf_idx++] = f_next(gx, gy, i);
                    }
                }
            }
        }

        void unpack_ghost_cells() {
            const int G = GHOST_LAYERS;
            int buf_idx;

            // west ghost layer
            buf_idx = 0;
            for (int y = 0; y < local_ny_; ++y) {
                int gx = 0;
                int gy = y + G;
                for (int i = 0; i < Q; ++i) {
                    f_next(gx, gy, i) = recv_buffer_west_[buf_idx++];
                }
            }

            // east ghost layer
            buf_idx = 0;
            for (int y = 0; y < local_ny_; ++y) {
                int gx = total_nx_ - 1;
                int gy = y + G;
                for (int i = 0; i < Q; ++i) {
                    f_next(gx, gy, i) = recv_buffer_east_[buf_idx++];
                }
            }

            // south ghost layer
            if (south_ != MPI_PROC_NULL) {
                buf_idx = 0;
                for (int x = 0; x < local_nx_; ++x) {
                    int gx = x + G;
                    int gy = 0;
                    for (int i = 0; i < Q; ++i) {
                        f_next(gx, gy, i) = recv_buffer_south_[buf_idx++];
                    }
                }
            }

            // north ghost layer
            if (north_ != MPI_PROC_NULL) {
                buf_idx = 0;
                for (int x = 0; x < local_nx_; ++x) {
                    int gx = x + G;
                    int gy = total_ny_ - 1;
                    for (int i = 0; i < Q; ++i) {
                        f_next(gx, gy, i) = recv_buffer_north_[buf_idx++];
                    }
                }
            }
        }
    };
}