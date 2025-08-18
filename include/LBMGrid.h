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

        // contigious memory allocation
        std::vector<double> f_;             // Distribution function
        std::vector<double> f_temp_;        // Temporary array
        std::vector<double> rho_;           // Density
        std::vector<double> ux_;            // x velocity
        std::vector<double> uy_;            // y velocity

    public:
        Grid(int nx, int ny) : nx_(nx), ny_(ny) {
            const size_t grid_size = nx_ * ny_;
            f_.resize(grid_size * Q, 0.0);
            f_temp_.resize(grid_size * Q, 0.0);
            rho_.resize(grid_size, 1.0);
            ux_.resize(grid_size, 0.0);
            uy_.resize(grid_size, 0.0);
        }

        inline double& f(int x, int y, int i) {
            return f_[(y * nx_ + x) * Q + i];
        }
        inline const double& f(int x, int y, int i) const {
            return f_[(y * nx_ + x) * Q + i];
        }

        inline double& f_temp(int x, int y, int i) {
            return f_temp_[(y * nx_ + x) * Q + i];
        }

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

        // Initialise equilibrium distributions
        void initialise() {
            #pragma GCC ivdep
            for (int x = 0; x < nx_; ++x) {
                for (int y = 0; y < ny_; ++y) {
                    const double rho_val = rho(x, y);
                    const double ux_val = ux(x, y);
                    const double uy_val = uy(x, y);

                    // Unroll loop
                    f(x,y,0) = equilibrium(0, rho_val, ux_val, uy_val);
                    f(x,y,1) = equilibrium(1, rho_val, ux_val, uy_val);
                    f(x,y,2) = equilibrium(2, rho_val, ux_val, uy_val);
                    f(x,y,3) = equilibrium(3, rho_val, ux_val, uy_val);
                    f(x,y,4) = equilibrium(4, rho_val, ux_val, uy_val);
                    f(x,y,5) = equilibrium(5, rho_val, ux_val, uy_val);
                    f(x,y,6) = equilibrium(6, rho_val, ux_val, uy_val);
                    f(x,y,7) = equilibrium(7, rho_val, ux_val, uy_val);
                    f(x,y,8) = equilibrium(8, rho_val, ux_val, uy_val);
                }
            }
        }

        // swap f and f_temp arrays
        void swap_f_arrays() {
            std::swap(f_, f_temp_);
        }

        // check stability across the entire grid
        bool check_stability () const {
            for (double i : f_) {
                if (!is_stable(i)) {
                    return false;
                }
            }
            return true;
        }

        // Calculate maximum velocity magnitude
        double max_velocity() const {
            double max_vel_sq = 0.0;
            #pragma GCC ivdep
            for (int idx = 0; idx < nx_ * ny_; ++idx) {
                const double vel_sq = ux_[idx] * ux_[idx] + uy_[idx] * uy_[idx];
                max_vel_sq = std::max(max_vel_sq, vel_sq);
            }
            return std::sqrt(max_vel_sq);
        }
    };
}
