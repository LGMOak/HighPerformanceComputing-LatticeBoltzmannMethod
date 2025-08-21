#include "../include/LBMConfig.h"
#include "../include/LBMSolver.h"
#include "../include/LBMIO.h"
#include <iostream>

int main() {
    try {
        // Configure simulation parameters
        LBM::SimulationParams params;

        // Benchmark parameters
        params.nx = 1024;
        params.ny = 256;
        params.num_timesteps = 5000;
        params.output_frequency = 100;

        // Scalability parameters
        // params.nx = 2048;
        // params.ny = 512;
        // params.num_timesteps = 10000;
        // params.output_frequency = 2000;

        // simulation parameters
        // params.nx = 4096;
        // params.ny = 1024;
        // params.num_timesteps = 20000;
        // params.output_frequency = 5000;

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