#pragma once

#include "LBMConfig.h"
#include "LBMGrid.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <mpi.h>

namespace LBM {

class IOManager {
public:
    static void write_results(const Grid& grid, const SimulationParams& params) {
        if (grid.mpi_rank() == 0) {
            std::cout << "\nGathering results and writing output data..." << std::endl;
        }

        // --- Gather macroscopic data (ux, uy) to rank 0 ---
        std::vector<double> global_ux, global_uy;
        if (grid.mpi_rank() == 0) {
            global_ux.resize(grid.global_nx() * grid.global_ny());
            global_uy.resize(grid.global_nx() * grid.global_ny());
        }

        // Each rank prepares its local data for sending (excluding ghost cells)
        std::vector<double> local_ux_flat(grid.local_nx() * grid.local_ny());
        std::vector<double> local_uy_flat(grid.local_nx() * grid.local_ny());
        for (int y = 0; y < grid.local_ny(); ++y) {
            for (int x = 0; x < grid.local_nx(); ++x) {

                local_ux_flat[y * grid.local_nx() + x] = grid.ux(x, y);
                local_uy_flat[y * grid.local_nx() + x] = grid.uy(x, y);
            }
        }

        // --- Prepare for MPI_Gatherv ---
        std::vector<int> recvcounts(grid.mpi_size());
        std::vector<int> displs(grid.mpi_size());

        int local_size = grid.local_nx() * grid.local_ny();
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (grid.mpi_rank() == 0) {
            displs[0] = 0;
            for (int i = 1; i < grid.mpi_size(); ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
        }

        // gather all information from ranks
        MPI_Gatherv(local_ux_flat.data(), local_size, MPI_DOUBLE,
                    global_ux.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(local_uy_flat.data(), local_size, MPI_DOUBLE,
                    global_uy.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);


        // --- Rank 0 performs all file I/O ---
        if (grid.mpi_rank() == 0) {
            write_velocity_field(global_ux, global_uy, params);
            write_velocity_profile(global_ux, params);
            write_simulation_params(global_ux, params);

            std::cout << "Files: velocity_field.csv, velocity_profile.csv, simulation_params.csv" << std::endl;
            std::cout << "Run: python3 visualise_results.py" << std::endl;
        }
    }

private:
    // Helper function to get global 1D index
    static size_t global_idx(int x, int y, int nx) { return static_cast<size_t>(y) * nx + x; }

    static void write_velocity_field(const std::vector<double>& ux_g, const std::vector<double>& uy_g, const SimulationParams& p) {
        std::ofstream file("velocity_field.csv");
        if (!file) { std::cerr << "Error: velocity_field.csv\n"; return; }
        file << "x,y,ux,uy,velocity_magnitude\n";
        for (int y = 0; y < p.ny; ++y) {
            for (int x = 0; x < p.nx; ++x) {
                size_t idx = global_idx(x, y, p.nx);
                const double ux_val = ux_g[idx];
                const double uy_val = uy_g[idx];
                const double vel_mag = std::sqrt(ux_val * ux_val + uy_val * uy_val);
                file << x << "," << y << "," << std::fixed << std::setprecision(6)
                     << ux_val << "," << uy_val << "," << vel_mag << "\n";
            }
        }
    }

    static void write_velocity_profile(const std::vector<double>& ux_g, const SimulationParams& p) {
        std::ofstream file("velocity_profile.csv");
        if (!file) { std::cerr << "Error: velocity_profile.csv\n"; return; }
        file << "y,ux\n";
        const int mid_x = p.nx / 2;
        for (int y = 0; y < p.ny; ++y) {
            file << y << "," << std::fixed << std::setprecision(6) << ux_g[global_idx(mid_x, y, p.nx)] << "\n";
        }
    }

    static void write_simulation_params(const std::vector<double>& ux_g, const SimulationParams& p) {
        std::ofstream file("simulation_params.csv");
        if (!file) { std::cerr << "Error: simulation_params.csv\n"; return; }

        const double kinematic_viscosity = p.nu();
        const double channel_height = static_cast<double>(p.ny);
        const double body_force = p.force_x;
        const double u_max_theory = (body_force * channel_height * channel_height) / (8.0 * kinematic_viscosity);
        const double u_avg_theory = (2.0 / 3.0) * u_max_theory;

        double sum_ux = 0.0, max_ux_sim = 0.0;
        const int mid_x = p.nx / 2;
        for (int y = 0; y < p.ny; ++y) {
            const double current_ux = ux_g[global_idx(mid_x, y, p.nx)];
            sum_ux += current_ux;
            if (current_ux > max_ux_sim) max_ux_sim = current_ux;
        }
        const double u_avg_sim = sum_ux / static_cast<double>(p.ny);
        const double reynolds_number = (u_avg_sim * channel_height) / kinematic_viscosity;

        double sum_squared_error = 0.0;
        for (int y = 0; y < p.ny; ++y) {
            const double y_pos = static_cast<double>(y);
            const double u_theory = (body_force / (2.0 * kinematic_viscosity)) * y_pos * (channel_height - 1.0 - y_pos);
            const double error = ux_g[global_idx(mid_x, y, p.nx)] - u_theory;
            sum_squared_error += error * error;
        }
        const double rmse = std::sqrt(sum_squared_error / static_cast<double>(p.ny));

        file << "parameter,value,analytical_value,error\n"
             << "nx," << p.nx << ",\n" << "ny," << p.ny << ",\n"
             << "tau," << p.tau << ",\n" << "nu," << p.nu() << ",\n"
             << "force_x," << p.force_x << ",\n"
             << "num_timesteps," << p.num_timesteps << ",\n"
             << "max_velocity," << std::fixed << std::setprecision(6) << max_ux_sim << "," << u_max_theory << "," << (max_ux_sim - u_max_theory) << "\n"
             << "avg_velocity," << u_avg_sim << "," << u_avg_theory << "," << (u_avg_sim - u_avg_theory) << "\n"
             << "reynolds_number," << reynolds_number << ",\n"
             << "rmse_from_theory," << rmse << ",\n";
    }
};

} // namespace LBM