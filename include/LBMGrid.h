#pragma once

#include "LBMConfig.h"
#include "LBMUtils.h"
#include <vector>
#include <algorithm>
#include <cstring>

namespace LBM {
    class Grid {
    private:
        int nx_, ny_;

        // Two distribution arrays
        std::vector<double> f_current_;     // current timestep distributions
        std::vector<double> f_next_;        // next timestep distributions

        // Macroscopic quantities
        std::vector<double> rho_;           // density
        std::vector<double> ux_;            // x velocity
        std::vector<double> uy_;            // y velocity

    public:
        Grid(int nx, int ny) : nx_(nx), ny_(ny) {
            const size_t grid_size = nx_ * ny_;
            f_current_.resize(grid_size * Q, 0.0);
            f_next_.resize(grid_size * Q, 0.0);
            rho_.resize(grid_size, 1.0);
            ux_.resize(grid_size, 0.0);
            uy_.resize(grid_size, 0.0);
        }

        // access current timestep distributions
        inline double& f_current(int x, int y, int i) {
            return f_current_[(y * nx_ + x) * Q + i];
        }
        inline const double& f_current(int x, int y, int i) const {
            return f_current_[(y * nx_ + x) * Q + i];
        }

        // access next timestep distributions
        inline double& f_next(int x, int y, int i) {
            return f_next_[(y * nx_ + x) * Q + i];
        }
        inline const double& f_next(int x, int y, int i) const {
            return f_next_[(y * nx_ + x) * Q + i];
        }

        // macroscopic quantities
        inline double& rho(int x, int y) {
            return rho_[(y * nx_ + x)];
        }
        inline const double& rho(int x, int y) const {
            return rho_[(y * nx_ + x)];
        }

        inline double& ux(int x, int y) {
            return ux_[(y * nx_ + x)];
        }
        inline const double& ux(int x, int y) const {
            return ux_[(y * nx_ + x)];
        }

        inline double& uy(int x, int y) {
            return uy_[(y * nx_ + x)];
        }
        inline const double& uy(int x, int y) const {
            return uy_[(y * nx_ + x)];
        }

        int nx() const { return nx_; }
        int ny() const { return ny_; }

        // Initialise equilibrium distributions in current array
        void initialise() {
            for (int x = 0; x < nx_; ++x) {
                for (int y = 0; y < ny_; ++y) {
                    const double rho_val = rho(x, y);
                    const double ux_val = ux(x, y);
                    const double uy_val = uy(x, y);

                    // Unroll loop
                    f_current(x, y, 0) = equilibrium(0, rho_val, ux_val, uy_val);
                    f_current(x, y, 1) = equilibrium(1, rho_val, ux_val, uy_val);
                    f_current(x, y, 2) = equilibrium(2, rho_val, ux_val, uy_val);
                    f_current(x, y, 3) = equilibrium(3, rho_val, ux_val, uy_val);
                    f_current(x, y, 4) = equilibrium(4, rho_val, ux_val, uy_val);
                    f_current(x, y, 5) = equilibrium(5, rho_val, ux_val, uy_val);
                    f_current(x, y, 6) = equilibrium(6, rho_val, ux_val, uy_val);
                    f_current(x, y, 7) = equilibrium(7, rho_val, ux_val, uy_val);
                    f_current(x, y, 8) = equilibrium(8, rho_val, ux_val, uy_val);
                }
            }
        }

        // Swap current and next array
        void swap_distributions() {
            std::swap(f_current_, f_next_);
        }

        // Check stability across current distributions
        bool check_stability() const {
            for (double val : f_current_) {
                if (!is_stable(val)) {
                    return false;
                }
            }
            return true;
        }

        // Calculate maximum velocity magnitude
        double max_velocity() const {
            double max_vel_sq = 0.0;
            for (int idx = 0; idx < nx_ * ny_; ++idx) {
                const double vel_sq = ux_[idx] * ux_[idx] + uy_[idx] * uy_[idx];
                max_vel_sq = std::max(max_vel_sq, vel_sq);
            }
            return std::sqrt(max_vel_sq);
        }

        // Clear next timestep array
        void clear_next() {
            std::fill(f_next_.begin(), f_next_.end(), 0.0);
        }
    };
}