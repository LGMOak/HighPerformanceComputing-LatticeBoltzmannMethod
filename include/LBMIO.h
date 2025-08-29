#pragma once

#include "LBMConfig.h"
#include "LBMGrid.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace LBM {

class IOManager {
public:
    static void write_results(const Grid& grid, const SimulationParams& params) {
        std::cout << "\nWriting output data..." << std::endl;

        write_velocity_field(grid);
        write_velocity_profile(grid);
        write_simulation_params(grid, params);

        std::cout << "Files: velocity_field.csv, velocity_profile.csv, simulation_params.csv" << std::endl;
        std::cout << "Run: python3 visualise_results.py" << std::endl;
    }

private:
    static void write_velocity_field(const Grid& grid) {
        std::ofstream file("velocity_field.csv");
        if (!file) { std::cerr << "Error: velocity_field.csv\n"; return; }

        file << "x,y,ux,uy,velocity_magnitude\n";

        for (int x = 0; x < grid.nx(); ++x) {
            for (int y = 0; y < grid.ny(); ++y) {
                const double ux_val = grid.ux(x, y);
                const double uy_val = grid.uy(x, y);
                const double vel_mag = std::sqrt(ux_val * ux_val + uy_val * uy_val);

                file << x << "," << y << ","
                     << std::fixed << std::setprecision(6)
                     << ux_val << "," << uy_val << "," << vel_mag << "\n";
            }
        }
    }

    static void write_velocity_profile(const Grid& grid) {
        std::ofstream file("velocity_profile.csv");
        if (!file) { std::cerr << "Error: velocity_profile.csv\n"; return; }

        file << "y,ux\n";
        const int mid_x = grid.nx() / 2;
        for (int y = 0; y < grid.ny(); ++y) {
            file << y << "," << std::fixed << std::setprecision(6)
                 << grid.ux(mid_x, y) << "\n";
        }
    }

    static void write_simulation_params(const Grid& grid, const SimulationParams& params) {
        std::ofstream file("simulation_params.csv");
        if (!file) { std::cerr << "Error: simulation_params.csv\n"; return; }

        // Analytical Values
        const double kinematic_viscosity = params.nu();
        const double channel_height = static_cast<double>(params.ny);
        const double body_force = params.force_x;

        const double u_max_theory = (body_force * channel_height * channel_height) / (8.0 * kinematic_viscosity);
        const double u_avg_theory = (2.0 / 3.0) * u_max_theory;

        // Simulation Values
        double sum_ux = 0.0;
        double max_ux_sim = 0.0;
        const int mid_x = params.nx / 2;
        for (int y = 0; y < params.ny; ++y) {
            const double current_ux = grid.ux(mid_x, y);
            sum_ux += current_ux;
            if (current_ux > max_ux_sim) {
                max_ux_sim = current_ux;
            }
        }
        const double u_avg_sim = sum_ux / static_cast<double>(params.ny);
        const double flow_rate_sim = u_avg_sim * channel_height;

        // RMSE Calculation
        double sum_squared_error = 0.0;
        for (int y = 0; y < params.ny; ++y) {
            const double y_pos = static_cast<double>(y);
            const double u_theory = (body_force / (2.0 * kinematic_viscosity)) * y_pos * (channel_height - 1.0 - y_pos);
            const double error = grid.ux(mid_x, y) - u_theory;
            sum_squared_error += error * error;
        }
        const double rmse = std::sqrt(sum_squared_error / static_cast<double>(params.ny));

        // Reynolds Number
        // Using average velocity and channel height as characteristic length
        const double reynolds_number = (u_avg_sim * channel_height) / kinematic_viscosity;

        file << "parameter,value,analytical_value,error\n"
             << "nx," << params.nx << ",\n"
             << "ny," << params.ny << ",\n"
             << "tau," << params.tau << ",\n"
             << "nu," << params.nu() << ",\n"
             << "force_x," << params.force_x << ",\n"
             << "num_timesteps," << params.num_timesteps << ",\n"
             << "max_velocity," << std::fixed << std::setprecision(6) << max_ux_sim << "," << u_max_theory << "," << (max_ux_sim - u_max_theory) << "\n"
             << "avg_velocity," << u_avg_sim << "," << u_avg_theory << "," << (u_avg_sim - u_avg_theory) << "\n"
             << "flow_rate," << flow_rate_sim << ",\n"
             << "reynolds_number," << reynolds_number << ",\n"
             << "rmse_from_theory," << rmse << ",\n";
    }
};

}