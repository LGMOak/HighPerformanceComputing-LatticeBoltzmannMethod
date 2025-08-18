#pragma once

#include "LBMConfig.h"
#include "LBMGrid.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <filesystem>

namespace LBM {

class IOManager {
public:
    static void write_results(const Grid& grid, const SimulationParams& params) {
        // Create data directory (assumed)
        // std::filesystem::create_directories("data");

        std::cout << "\nWriting output data to CSV files in 'data' directory..." << std::endl;

        write_velocity_field(grid);
        write_velocity_profile(grid);
        write_simulation_params(params);

        std::cout << "Data written to:" << std::endl;
        std::cout << "  - data/velocity_field.csv (full velocity field)" << std::endl;
        std::cout << "  - data/velocity_profile.csv (profile at x=" << grid.nx()/2 << ")" << std::endl;
        std::cout << "  - data/simulation_params.csv (simulation parameters)" << std::endl;
        std::cout << "\nRun 'python3 visualise_results.py' to generate plots." << std::endl;
    }

private:
    static void write_velocity_field(const Grid& grid) {
        std::ofstream file("data/velocity_field.csv");
        if (!file.is_open()) {
            std::cerr << "Error: Could not open data/velocity_field.csv for writing" << std::endl;
            return;
        }

        file << "x,y,ux,uy,velocity_magnitude\n";

        for (int x = 0; x < grid.nx(); ++x) {
            for (int y = 0; y < grid.ny(); ++y) {
                const double ux_val = grid.ux(x, y);
                const double uy_val = grid.uy(x, y);
                const double vel_mag = std::sqrt(ux_val * ux_val + uy_val * uy_val);

                file << x << "," << y << ","
                     << std::fixed << std::setprecision(8)
                     << ux_val << "," << uy_val << "," << vel_mag << "\n";
            }
        }

        file.close();
    }

    static void write_velocity_profile(const Grid& grid) {
        std::ofstream file("data/velocity_profile.csv");
        if (!file.is_open()) {
            std::cerr << "Error: Could not open data/velocity_profile.csv for writing" << std::endl;
            return;
        }

        file << "y,ux\n";

        const int mid_x = grid.nx() / 2;
        for (int y = 0; y < grid.ny(); ++y) {
            file << y << "," << std::fixed << std::setprecision(8)
                 << grid.ux(mid_x, y) << "\n";
        }

        file.close();
    }

    static void write_simulation_params(const SimulationParams& params) {
        std::ofstream file("data/simulation_params.csv");
        if (!file.is_open()) {
            std::cerr << "Error: Could not open data/simulation_params.csv for writing" << std::endl;
            return;
        }

        file << "parameter,value\n";
        file << "nx," << params.nx << "\n";
        file << "ny," << params.ny << "\n";
        file << "tau," << params.tau << "\n";
        file << "nu," << params.nu() << "\n";
        file << "force_x," << params.force_x << "\n";
        file << "num_timesteps," << params.num_timesteps << "\n";

        file.close();
    }
};

}