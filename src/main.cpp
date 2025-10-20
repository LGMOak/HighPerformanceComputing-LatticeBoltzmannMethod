#include "../include/LBMConfig.h"
#include "../include/LBMSolver.h"
#include "../include/LBMIO.h"
#include <iostream>
#include <mpi.h>

#include "../include/LBMConfig.h"
#include "../include/LBMSolver.h"
#include "../include/LBMIO.h" // Make sure this is included
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    try {
        LBM::SimulationParams params;
        LBM::Solver solver(params);
        LBM::IOManager io_manager; // 1. Create the IOManager instance here

        solver.initialise();

        // 2. Pass the IOManager to the run method
        bool success = solver.run(io_manager);

        if (success) {
            // 3. Call the final write method after the simulation is complete
            io_manager.write_final_results(solver.get_grid(), solver.get_params());
            if (solver.get_grid().mpi_rank() == 0) {
                std::cout << "\nSimulation completed successfully!" << std::endl;
            }
        } else {
            if (solver.get_grid().mpi_rank() == 0) {
                std::cerr << "LBM simulation failed." << std::endl;
            }
        }
    } catch (const std::exception& e) {
        if (MPI::Is_initialized()) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                std::cerr << "An exception occurred: " << e.what() << std::endl;
            }
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}