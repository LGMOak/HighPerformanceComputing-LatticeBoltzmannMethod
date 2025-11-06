/*
* LBMIO.h
 *
 * This file contains the IOManager class responsible for handling
 * parallel I/O, force calculations, and final data aggregation using MPI.
 *
 * Note: AI assistance was used in the development and debugging
 * of the MPI-Gatherv logic and file-writing routines in this class.
 */

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
private:
    std::ofstream force_file_;
    int mpi_rank_;
    bool force_file_open_;

    static size_t global_idx(int x, int y, int nx) {
        return static_cast<size_t>(y) * nx + x;
    }

public:
    IOManager() : mpi_rank_(-1), force_file_open_(false) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        if (mpi_rank_ == 0) {
            force_file_.open("forces.csv");
            if (force_file_) {
                force_file_ << "timestep,drag_force,lift_force,drag_coeff,lift_coeff\n";
                force_file_open_ = true;
            } else {
                std::cerr << "ERROR: Could not open forces.csv\n";
            }
        }
    }

    // Destructor
    ~IOManager() {
        if (mpi_rank_ == 0 && force_file_open_) {
            force_file_.close();
        }
    }

    static void write_vtk_timestep(const std::vector<double>& ux_g,
                                const std::vector<double>& uy_g,
                                const std::vector<double>& rho_g,
                                const SimulationParams& p,
                                int timestep) {
        char filename[256];
        snprintf(filename, sizeof(filename), "vtk_output/lbm_%06d.vtk", timestep);

        std::ofstream file(filename);
        if (!file) {
            std::cerr << "ERROR: Cannot write " << filename << "\n";
            return;
        }

        // VTK Legacy format header
        file << "# vtk DataFile Version 3.0\n";
        file << "LBM Flow Timestep " << timestep << "\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << p.nx << " " << p.ny << " 1\n";
        file << "ORIGIN 0 0 0\n";
        file << "SPACING 1 1 1\n";
        file << "POINT_DATA " << (p.nx * p.ny) << "\n";

        // Velocity vectors
        file << "VECTORS velocity double\n";
        for (int y = 0; y < p.ny; ++y) {
            for (int x = 0; x < p.nx; ++x) {
                size_t idx = global_idx(x, y, p.nx);
                file << std::fixed << std::setprecision(8)
                     << ux_g[idx] << " " << uy_g[idx] << " 0.0\n";
            }
        }

        // Velocity magnitude
        file << "\nSCALARS velocity_magnitude double\n";
        file << "LOOKUP_TABLE default\n";
        for (int y = 0; y < p.ny; ++y) {
            for (int x = 0; x < p.nx; ++x) {
                size_t idx = global_idx(x, y, p.nx);
                double vel_mag = std::sqrt(ux_g[idx] * ux_g[idx] + uy_g[idx] * uy_g[idx]);
                file << std::fixed << std::setprecision(8) << vel_mag << "\n";
            }
        }

        // Density
        file << "\nSCALARS density double\n";
        file << "LOOKUP_TABLE default\n";
        for (int y = 0; y < p.ny; ++y) {
            for (int x = 0; x < p.nx; ++x) {
                size_t idx = global_idx(x, y, p.nx);
                file << std::fixed << std::setprecision(8) << rho_g[idx] << "\n";
            }
        }

        file.close();
    }

    // Call this after collision but before streamiing
    void record_forces(int timestep, const Grid& grid, const SimulationParams& params) {
        double local_fx = 0.0;
        double local_fy = 0.0;
        const int GHOST = 1;

        // Momentum exchange method:
        // For each solid cell, sum the momentum of populations that would stream
        // from fluid neighbours into this solid cell

        for (int y = 0; y < grid.local_ny(); ++y) {
            for (int x = 0; x < grid.local_nx(); ++x) {
                // Only process solid cells
                if (!grid.is_solid(x, y)) continue;

                int gx = x + GHOST;
                int gy = y + GHOST;

                // For each direction, check if there's a fluid neighbour
                // that would stream into this solid cell
                for (int i = 1; i < Q; ++i) {
                    // Look in the OPPOSITE direction to find the fluid cell
                    // that has a population pointing toward this solid cell
                    int opp_i = OPPOSITE[i];

                    // Fluid neighbour location (in local coordinates)
                    int fluid_x = x - VELOCITIES[i][0];
                    int fluid_y = y - VELOCITIES[i][1];

                    // Check if this neighbour is in bounds and is fluid
                    if (fluid_x >= 0 && fluid_x < grid.local_nx() &&
                        fluid_y >= 0 && fluid_y < grid.local_ny() &&
                        !grid.is_solid(fluid_x, fluid_y))
                    {
                        // The fluid neighbour at (fluid_x, fluid_y) has a population
                        // in direction 'i' pointing toward this solid cell
                        int fluid_gx = fluid_x + GHOST;
                        int fluid_gy = fluid_y + GHOST;

                        // Read the post-collision population (f_next) that will bounce back
                        double f_i = grid.f_next(fluid_gx, fluid_gy, i);

                        // Momentum transfer: 2 * c_i * f_i
                        // (factor of 2 because momentum reverses)
                        local_fx += 2.0 * VELOCITIES[i][0] * f_i;
                        local_fy += 2.0 * VELOCITIES[i][1] * f_i;
                    }
                }
            }
        }

        // Sum forces across all MPI ranks
        double global_fx = 0.0;
        double global_fy = 0.0;
        MPI_Reduce(&local_fx, &global_fx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_fy, &global_fy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Rank 0 writes to file
        if (mpi_rank_ == 0 && force_file_open_) {
            double rho_ref = 1.0;
            double U_ref = params.inlet_velocity;
            double D_ref = 2.0 * params.get_cylinder_radius_cells();

            double q_ref = 0.5 * rho_ref * U_ref * U_ref * D_ref;
            double C_D = (q_ref > 1e-12) ? global_fx / q_ref : 0.0;
            double C_L = (q_ref > 1e-12) ? global_fy / q_ref : 0.0;

            force_file_ << timestep << ","
                        << std::fixed << std::setprecision(8)
                        << global_fx << ","
                        << global_fy << ","
                        << C_D << ","
                        << C_L << "\n";

            // Flush periodically for monitoring
            if (timestep % 10000 == 0) {
                force_file_.flush();
            }
        }
    }

    void write_final_results(const Grid& grid, const SimulationParams& params) {
        if (mpi_rank_ == 0) {
            std::cout << "\nGathering final results..." << std::endl;
        }

        std::vector<double> global_ux, global_uy, global_rho;
        if (mpi_rank_ == 0) {
            global_ux.resize(grid.global_nx() * grid.global_ny());
            global_uy.resize(grid.global_nx() * grid.global_ny());
            global_rho.resize(grid.global_nx() * grid.global_ny());
        }

        gather_and_reconstruct_field(grid, global_ux, global_uy, global_rho);

        if (mpi_rank_ == 0) {
            write_velocity_field(global_ux, global_uy, global_rho, params);
            write_simulation_params(global_ux, global_uy, params);

            // Calculate time-averaged drag from forces.csv
            calculate_time_averaged_drag();

            std::cout << "Files written: velocity_field.csv, simulation_params.csv, forces.csv" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

private:
    /*
     * Rebuilds the grid on a single process
     */
    static void gather_and_reconstruct_field(const Grid& grid,
                                            std::vector<double>& global_ux,
                                            std::vector<double>& global_uy,
                                            std::vector<double>& global_rho) {
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        struct RankInfo { int x_start, y_start, local_nx, local_ny; };
        RankInfo my_info = {grid.x_start(), grid.y_start(), grid.local_nx(), grid.local_ny()};

        std::vector<RankInfo> all_info(mpi_size);
        MPI_Gather(&my_info, sizeof(RankInfo), MPI_BYTE,
                   all_info.data(), sizeof(RankInfo), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        std::vector<double> local_ux_flat(my_info.local_nx * my_info.local_ny);
        std::vector<double> local_uy_flat(my_info.local_nx * my_info.local_ny);
        std::vector<double> local_rho_flat(my_info.local_nx * my_info.local_ny);

        for (int y = 0; y < my_info.local_ny; ++y) {
            for (int x = 0; x < my_info.local_nx; ++x) {
                int idx = y * my_info.local_nx + x;
                local_ux_flat[idx] = grid.ux(x, y);
                local_uy_flat[idx] = grid.uy(x, y);
                local_rho_flat[idx] = grid.rho(x, y);
            }
        }

        std::vector<int> recvcounts(mpi_size);
        std::vector<int> displs(mpi_size);
        int local_size = my_info.local_nx * my_info.local_ny;
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < mpi_size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }

        std::vector<double> recv_ux, recv_uy, recv_rho;
        if (mpi_rank == 0) {
            int total = 0;
            for (int c : recvcounts) total += c;
            recv_ux.resize(total);
            recv_uy.resize(total);
            recv_rho.resize(total);
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

        if (mpi_rank == 0) {
            for(int rank = 0; rank < mpi_size; ++rank) {
                const auto& info = all_info[rank];
                int offset = displs[rank];
                for (int ly = 0; ly < info.local_ny; ++ly) {
                    for (int lx = 0; lx < info.local_nx; ++lx) {
                        int g_idx = (info.y_start + ly) * grid.global_nx() + (info.x_start + lx);
                        int l_idx = ly * info.local_nx + lx;
                        global_ux[g_idx] = recv_ux[offset + l_idx];
                        global_uy[g_idx] = recv_uy[offset + l_idx];
                        global_rho[g_idx] = recv_rho[offset + l_idx];
                    }
                }
            }
        }
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
                size_t idx = static_cast<size_t>(y) * p.nx + x;
                double vel_mag = std::sqrt(ux_g[idx] * ux_g[idx] + uy_g[idx] * uy_g[idx]);
                file << x << "," << y << ","
                     << std::fixed << std::setprecision(8)
                     << ux_g[idx] << "," << uy_g[idx] << ","
                     << rho_g[idx] << "," << vel_mag << "\n";
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

        // Calculate velocity statistics
        double max_vel = 0.0;
        double avg_vel = 0.0;
        for (int y = 0; y < p.ny; ++y) {
            for (int x = 0; x < p.nx; ++x) {
                size_t idx = static_cast<size_t>(y) * p.nx + x;
                double vel = std::sqrt(ux_g[idx] * ux_g[idx] + uy_g[idx] * uy_g[idx]);
                max_vel = std::max(max_vel, vel);
                avg_vel += vel;
            }
        }
        avg_vel /= (p.nx * p.ny);

        file << "parameter,value\n"
             << "nx," << p.nx << "\n"
             << "ny," << p.ny << "\n"
             << "tau," << std::fixed << std::setprecision(8) << p.tau << "\n"
             << "nu," << p.nu() << "\n"
             << "inlet_velocity," << p.inlet_velocity << "\n"
             << "num_timesteps," << p.num_timesteps << "\n"
             << "reynolds_number," << p.reynolds() << "\n"
             << "cylinder_x," << p.get_cylinder_x() << "\n"
             << "cylinder_y," << p.get_cylinder_y() << "\n"
             << "cylinder_radius," << p.get_cylinder_radius_cells() << "\n"
             << "max_velocity," << max_vel << "\n"
             << "avg_velocity," << avg_vel << "\n";

        file.close();
        std::cout << "  simulation_params.csv written\n";
    }

    static void calculate_time_averaged_drag() {
        std::ifstream forces_in("forces.csv");
        if (!forces_in) {
            std::cerr << "Warning: Could not read forces.csv for averaging\n";
            return;
        }

        std::string header;
        std::getline(forces_in, header);  // Skip header

        double sum_cd = 0.0, sum_cl = 0.0;
        double max_cd = -1e9, min_cd = 1e9;
        double max_cl = -1e9, min_cl = 1e9;
        int count = 0;
        int skip_initial = 1000;  // Skip initial transient

        std::string line;
        while (std::getline(forces_in, line)) {
            int timestep;
            double fx, fy, cd, cl;
            if (sscanf(line.c_str(), "%d,%lf,%lf,%lf,%lf",
                      &timestep, &fx, &fy, &cd, &cl) == 5) {
                if (timestep > skip_initial) {
                    sum_cd += cd;
                    sum_cl += cl;
                    max_cd = std::max(max_cd, cd);
                    min_cd = std::min(min_cd, cd);
                    max_cl = std::max(max_cl, cl);
                    min_cl = std::min(min_cl, cl);
                    count++;
                }
            }
        }
        forces_in.close();

        if (count > 0) {
            double avg_cd = sum_cd / count;
            double avg_cl = sum_cl / count;

            std::cout << "\n=== Time-Averaged Force Coefficients ===\n";
            std::cout << "  Mean C_D = " << std::fixed << std::setprecision(6) << avg_cd << "\n";
            std::cout << "  C_D range: [" << min_cd << ", " << max_cd << "]\n";
            std::cout << "  Mean C_L = " << avg_cl << "\n";
            std::cout << "  C_L range: [" << min_cl << ", " << max_cl << "]\n";
            std::cout << "  (Averaged over " << count << " samples)\n";
        }
    }
};

} // namespace LBM