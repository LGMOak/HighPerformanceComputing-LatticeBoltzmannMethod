#include "../include/LBMConfig.h"
#include "../include/LBMSolver.h"
#include "../include/LBMIO.h"
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    try {
        LBM::SimulationParams params;
        LBM::Solver solver(params, true);
        LBM::IOManager io_manager;

        solver.initialise();

        bool success = solver.run(io_manager);

        if (success) {
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