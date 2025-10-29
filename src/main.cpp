#include "../include/LBMConfig.h"
#include "../include/LBMSolver.h"
#include "../include/LBMIO.h"
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    // initialise MPI
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        // Configure simulation parameters
        LBM::SimulationParams params;

        // Simulation parameters
        params.nx = 1024;
        params.ny = 256;
        params.num_timesteps = 500000;
        params.output_frequency = 2000;
        params.tau = 0.6;
        params.force_x = 4e-7;

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

        if (rank == 0) {
            std::cout << "\nSimulation completed successfully!" << std::endl;
        }
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
