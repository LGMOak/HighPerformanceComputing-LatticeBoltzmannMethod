#pragma once

#include "LBMConfig.h"
#include <cmath>
#include <immintrin.h>

namespace LBM {
    // Scalar equilibrium for direction 0 (rest particle)
    inline double equilibrium_scalar(double rho, double ux, double uy) {
        const double u_sq = ux * ux + uy * uy;
        return WEIGHTS[0] * rho * (1.0 - 1.5 * u_sq);
    }

    // Scalar equilibrium with force for direction 0
    inline double equilibrium_with_force_scalar(double rho, double ux, double uy, double force_x, double force_y) {
        const double u_sq = ux * ux + uy * uy;
        // Direction 0 has velocity (0,0), so force term is zero
        return WEIGHTS[0] * rho * (1.0 - 1.5 * u_sq);
    }

    // SIMD equilibrium for orthogonal directions 1-4 and diagonal directions 5-8
    inline void equilibrium_simd(double rho, double ux, double uy, double* f_eq) {
        // Broadcast macroscopic quantities to SIMD vectors
        const __m256d rho_vec = _mm256_set1_pd(rho);
        const __m256d ux_vec = _mm256_set1_pd(ux);
        const __m256d uy_vec = _mm256_set1_pd(uy);
        const __m256d u_sq = _mm256_set1_pd(ux * ux + uy * uy);

        // Common constants for the equilibrium equation
        const __m256d three = _mm256_set1_pd(3.0);
        const __m256d four_half = _mm256_set1_pd(4.5);
        const __m256d one_half = _mm256_set1_pd(1.5);
        const __m256d one = _mm256_set1_pd(1.0);

        // Group 1: Orthogonal directions 1-4 (East, North, West, South)
        const __m256d cx_orth = _mm256_setr_pd(1.0, 0.0, -1.0, 0.0);
        const __m256d cy_orth = _mm256_setr_pd(0.0, 1.0, 0.0, -1.0);
        const __m256d weights_orth = _mm256_set1_pd(1.0/9.0);

        const __m256d ci_u_orth = _mm256_add_pd(_mm256_mul_pd(cx_orth, ux_vec), _mm256_mul_pd(cy_orth, uy_vec));
        const __m256d ci_u_sq_orth = _mm256_mul_pd(ci_u_orth, ci_u_orth);
        const __m256d term1_orth = _mm256_mul_pd(three, ci_u_orth);
        const __m256d term2_orth = _mm256_mul_pd(four_half, ci_u_sq_orth);
        const __m256d term3 = _mm256_mul_pd(one_half, u_sq);

        const __m256d bracket_orth = _mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(one, term1_orth), term3), term2_orth);
        const __m256d result_orth = _mm256_mul_pd(_mm256_mul_pd(weights_orth, rho_vec), bracket_orth);

        _mm256_storeu_pd(&f_eq[0], result_orth);

        // Group 2: Diagonal directions 5-8 (NE, NW, SW, SE)
        const __m256d cx_diag = _mm256_setr_pd(1.0, -1.0, -1.0, 1.0);
        const __m256d cy_diag = _mm256_setr_pd(1.0, 1.0, -1.0, -1.0);
        const __m256d weights_diag = _mm256_set1_pd(1.0/36.0);

        const __m256d ci_u_diag = _mm256_add_pd(_mm256_mul_pd(cx_diag, ux_vec), _mm256_mul_pd(cy_diag, uy_vec));
        const __m256d ci_u_sq_diag = _mm256_mul_pd(ci_u_diag, ci_u_diag);
        const __m256d term1_diag = _mm256_mul_pd(three, ci_u_diag);
        const __m256d term2_diag = _mm256_mul_pd(four_half, ci_u_sq_diag);

        const __m256d bracket_diag = _mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(one, term1_diag), term3), term2_diag);
        const __m256d result_diag = _mm256_mul_pd(_mm256_mul_pd(weights_diag, rho_vec), bracket_diag);

        _mm256_storeu_pd(&f_eq[4], result_diag);
    }

    // SIMD equilibrium with force for orthogonal and diagonal directions
    inline void equilibrium_with_force_simd(double rho, double ux, double uy, double force_x, double force_y, double* f_eq) {
        // Broadcast macroscopic quantities to SIMD vectors
        const __m256d rho_vec = _mm256_set1_pd(rho);
        const __m256d ux_vec = _mm256_set1_pd(ux);
        const __m256d uy_vec = _mm256_set1_pd(uy);
        const __m256d u_sq = _mm256_set1_pd(ux * ux + uy * uy);
        const __m256d fx_vec = _mm256_set1_pd(force_x);
        const __m256d fy_vec = _mm256_set1_pd(force_y);

        // Common constants for the equilibrium equation
        const __m256d three = _mm256_set1_pd(3.0);
        const __m256d four_half = _mm256_set1_pd(4.5);
        const __m256d one_half = _mm256_set1_pd(1.5);
        const __m256d one = _mm256_set1_pd(1.0);

        // Orthogonal directions 1-4 (East, North, West, South)
        const __m256d cx_orth = _mm256_setr_pd(1.0, 0.0, -1.0, 0.0);
        const __m256d cy_orth = _mm256_setr_pd(0.0, 1.0, 0.0, -1.0);
        const __m256d weights_orth = _mm256_set1_pd(1.0/9.0);

        const __m256d ci_u_orth = _mm256_add_pd(_mm256_mul_pd(cx_orth, ux_vec), _mm256_mul_pd(cy_orth, uy_vec));
        const __m256d force_dot_orth = _mm256_add_pd(_mm256_mul_pd(cx_orth, fx_vec), _mm256_mul_pd(cy_orth, fy_vec));

        const __m256d ci_u_sq_orth = _mm256_mul_pd(ci_u_orth, ci_u_orth);
        const __m256d term1_orth = _mm256_mul_pd(three, ci_u_orth);
        const __m256d term2_orth = _mm256_mul_pd(four_half, ci_u_sq_orth);
        const __m256d term3 = _mm256_mul_pd(one_half, u_sq);

        const __m256d bracket_orth = _mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(one, term1_orth), term3), term2_orth);
        const __m256d eq_term_orth = _mm256_mul_pd(_mm256_mul_pd(weights_orth, rho_vec), bracket_orth);
        const __m256d body_force_orth = _mm256_mul_pd(_mm256_mul_pd(three, weights_orth), force_dot_orth);
        const __m256d result_orth = _mm256_add_pd(eq_term_orth, body_force_orth);

        _mm256_storeu_pd(&f_eq[0], result_orth);

        // Diagonal directions 5-8 (NE, NW, SW, SE)
        const __m256d cx_diag = _mm256_setr_pd(1.0, -1.0, -1.0, 1.0);
        const __m256d cy_diag = _mm256_setr_pd(1.0, 1.0, -1.0, -1.0);
        const __m256d weights_diag = _mm256_set1_pd(1.0/36.0);

        const __m256d ci_u_diag = _mm256_add_pd(_mm256_mul_pd(cx_diag, ux_vec), _mm256_mul_pd(cy_diag, uy_vec));
        const __m256d force_dot_diag = _mm256_add_pd(_mm256_mul_pd(cx_diag, fx_vec), _mm256_mul_pd(cy_diag, fy_vec));

        const __m256d ci_u_sq_diag = _mm256_mul_pd(ci_u_diag, ci_u_diag);
        const __m256d term1_diag = _mm256_mul_pd(three, ci_u_diag);
        const __m256d term2_diag = _mm256_mul_pd(four_half, ci_u_sq_diag);

        const __m256d bracket_diag = _mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(one, term1_diag), term3), term2_diag);
        const __m256d eq_term_diag = _mm256_mul_pd(_mm256_mul_pd(weights_diag, rho_vec), bracket_diag);
        const __m256d body_force_diag = _mm256_mul_pd(_mm256_mul_pd(three, weights_diag), force_dot_diag);
        const __m256d result_diag = _mm256_add_pd(eq_term_diag, body_force_diag);

        _mm256_storeu_pd(&f_eq[4], result_diag);
    }

    // Periodic boundary conditions
    inline int periodic_x(int x, int nx) {
        return (x + nx) % nx;
    }

    // Check for stability
    inline bool is_stable(const double value) {
        return std::isfinite(value) && std::abs(value) < 1e5;
    }
}