#include "../include/LBMConfig.h"
#include "../include/LBMSolver.h"
#include "../include/LBMIO.h"
#include <iostream>

int main() {
    try {
        // Configure simulation parameters
        LBM::SimulationParams params;
        params.tau = 1;
        params.force_x = 1e-5;
        params.force_y = 0.0;

        // Local parameters
        params.nx = 200;
        params.ny = 50;
        params.num_timesteps = 10000;
        params.output_frequency = 1000;

        // HPC parameters
        // params.nx = 4096;
        // params.ny = 1024;
        // params.num_timesteps = 50000;
        // params.output_frequency = 10000;

        // Create solver
        LBM::Solver solver(params);
        solver.initialise();

        // Run simulation
        if (!solver.run()) {
            std::cerr << "LBM simulation failed to reach equilibrium" << std::endl;
            return 1;
        }

        // Write output files
        LBM::IOManager::write_results(solver.get_grid(), solver.get_params());

        std::cout << "\nSimulation completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}