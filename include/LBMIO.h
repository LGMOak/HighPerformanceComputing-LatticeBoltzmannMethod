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
            std::cout << "\nGathering results..." << std::endl;
        }

        // Gather velocity and density data
        std::vector<double> global_ux, global_uy, global_rho;
        if (grid.mpi_rank() == 0) {
            global_ux.resize(grid.global_nx() * grid.global_ny(), 0.0);
            global_uy.resize(grid.global_nx() * grid.global_ny(), 0.0);
            global_rho.resize(grid.global_nx() * grid.global_ny(), 1.0);
        }

        struct RankInfo {
            int x_start, y_start, local_nx, local_ny;
        };

        RankInfo my_info;
        my_info.local_nx = grid.local_nx();
        my_info.local_ny = grid.local_ny();
        my_info.x_start = grid.x_start();
        my_info.y_start = grid.y_start();

        std::vector<RankInfo> all_info(grid.mpi_size());
        MPI_Gather(&my_info, sizeof(RankInfo), MPI_BYTE,
                   all_info.data(), sizeof(RankInfo), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        // Prepare flat arrays
        std::vector<double> local_ux_flat(grid.local_nx() * grid.local_ny());
        std::vector<double> local_uy_flat(grid.local_nx() * grid.local_ny());
        std::vector<double> local_rho_flat(grid.local_nx() * grid.local_ny());

        for (int y = 0; y < grid.local_ny(); ++y) {
            for (int x = 0; x < grid.local_nx(); ++x) {
                int idx = y * grid.local_nx() + x;
                local_ux_flat[idx] = grid.ux(x, y);
                local_uy_flat[idx] = grid.uy(x, y);
                local_rho_flat[idx] = grid.rho(x, y);
            }
        }

        // Gather
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

        std::vector<double> recv_ux, recv_uy, recv_rho;
        if (grid.mpi_rank() == 0) {
            int total_size = 0;
            for (int c : recvcounts) total_size += c;
            recv_ux.resize(total_size);
            recv_uy.resize(total_size);
            recv_rho.resize(total_size);
        }

        MPI_Gatherv(local_ux_flat.data(), local_size, MPI_DOUBLE,
                    recv_ux.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(local_uy_flat.data(), local_size, MPI_DOUBLE,
                    recv_uy.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(local_rho_flat.data(), local_size, MPI_DOUBLE,
                    recv_rho.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // Reconstruct on rank 0
        if (grid.mpi_rank() == 0) {
            for (int rank = 0; rank < grid.mpi_size(); ++rank) {
                const RankInfo& info = all_info[rank];
                int offset = displs[rank];

                for (int ly = 0; ly < info.local_ny; ++ly) {
                    for (int lx = 0; lx < info.local_nx; ++lx) {
                        int global_x = info.x_start + lx;
                        int global_y = info.y_start + ly;
                        int global_idx = global_y * grid.global_nx() + global_x;
                        int local_idx = ly * info.local_nx + lx;

                        global_ux[global_idx] = recv_ux[offset + local_idx];
                        global_uy[global_idx] = recv_uy[offset + local_idx];
                        global_rho[global_idx] = recv_rho[offset + local_idx];
                    }
                }
            }

            write_velocity_field(global_ux, global_uy, global_rho, params);
            write_simulation_params(global_ux, global_uy, params);

            std::cout << "Files written: velocity_field.csv, simulation_params.csv" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

private:
    static size_t global_idx(int x, int y, int nx) {
        return static_cast<size_t>(y) * nx + x;
    }

    static void write_velocity_field(const std::vector<double>& ux_g,
                                     const std::vector<double>& uy_g,
                                     const std::vector<double>& rho_g,
                                     const SimulationParams& p) {
        std::ofstream file("velocity_field.csv");
        if (!file) {
            std::cerr << "ERROR: Cannot write velocity_field.csv\n";
            return;
        }

        file << "x,y,ux,uy,rho,velocity_magnitude\n";

        for (int y = 0; y < p.ny; ++y) {
            for (int x = 0; x < p.nx; ++x) {
                size_t idx = global_idx(x, y, p.nx);
                const double ux_val = ux_g[idx];
                const double uy_val = uy_g[idx];
                const double rho_val = rho_g[idx];
                const double vel_mag = std::sqrt(ux_val * ux_val + uy_val * uy_val);

                file << x << "," << y << ","
                     << std::fixed << std::setprecision(8)
                     << ux_val << "," << uy_val << ","
                     << rho_val << "," << vel_mag << "\n";
            }
        }
        file.close();
        std::cout << "  velocity_field.csv written\n";
    }

    static void write_simulation_params(const std::vector<double>& ux_g,
                                        const std::vector<double>& uy_g,
                                        const SimulationParams& p) {
        std::ofstream file("simulation_params.csv");
        if (!file) {
            std::cerr << "ERROR: Cannot write simulation_params.csv\n";
            return;
        }

        // Calculate statistics
        double max_vel = 0.0;
        double avg_vel = 0.0;
        int count = 0;

        for (int y = 0; y < p.ny; ++y) {
            for (int x = 0; x < p.nx; ++x) {
                size_t idx = global_idx(x, y, p.nx);
                double vel = std::sqrt(ux_g[idx] * ux_g[idx] + uy_g[idx] * uy_g[idx]);
                max_vel = std::max(max_vel, vel);
                avg_vel += vel;
                count++;
            }
        }
        avg_vel /= count;

        file << "parameter,value\n"
             << "nx," << p.nx << "\n"
             << "ny," << p.ny << "\n"
             << "tau," << p.tau << "\n"
             << "nu," << std::fixed << std::setprecision(8) << p.nu() << "\n"
             << "inlet_velocity," << p.inlet_velocity << "\n"
             << "num_timesteps," << p.num_timesteps << "\n"
             << "reynolds_number," << p.reynolds() << "\n"
             << "cylinder_x," << p.get_cylinder_x() << "\n"
             << "cylinder_y," << p.get_cylinder_y() << "\n"
             << "cylinder_radius," << p.get_cylinder_radius_cells() << "\n"
             << "max_velocity," << max_vel << "\n"
             << "avg_velocity," << avg_vel << "\n";

        file.close();
    }
};

} // namespace LBM