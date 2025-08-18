#pragma once

#include "LBMConfig.h"
#include <cmath>

namespace LBM {
    // Optimised equilibrium distribution calculation
    inline double equilibrium(int i, double rho, double ux, double uy) {
        const auto& c= VELOCITIES[i];
        const double ci_u = c[0] * ux + c[1] * uy;
        const double u_sq = ux * ux + uy * uy;

        return WEIGHTS[i] * rho * (1.0 + 3.0 * ci_u + 4.5 * ci_u * ci_u - 1.5 * u_sq);
    }

    // Optimised equilibrium with force term
    inline double equilibrium_with_force(int i, double rho, double ux, double uy, double force_x, double force_y) {
        const auto& c= VELOCITIES[i];
        const double ci_u = c[0] * ux + c[1] * uy;
        const double u_sq = ux * ux + uy * uy;
        const double body_force_term = 3.0 * WEIGHTS[i] * (c[0] * force_x + c[1] * force_y);

        return WEIGHTS[i] * rho * (1.0 + 3.0 * ci_u + 4.5 * ci_u * ci_u - 1.5 * u_sq) + body_force_term;
    }

    // Periodic boundary conditions
    inline int periodic_x(int x, int nx) {
        // Appeal to profiling to determine the better implementation
        // return (x >= nx) ? x - nx : ((x < 0) ? x + nx : x);
        return (x + nx) % nx;
    }

    // Check for stability
    inline bool is_stable(const double value) {
        return std::isfinite(value) && std::abs(value) < 1e5;
    }
}