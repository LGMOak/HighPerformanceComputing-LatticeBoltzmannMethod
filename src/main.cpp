#include "../include/LBMConfig.h"
#include "../include/LBMSolver.h"
#include "../include/LBMIO.h"
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    try {
        // Configure simulation parameters
        LBM::SimulationParams params;

        // Animation settings
        bool enable_animation = true;
        int animation_frequency = 500;

        // Simulation parameters
        params.nx = 1024;
        params.ny = 256;
        params.num_timesteps = 80000;
        params.output_frequency = 5000;
        params.tau = 0.6;
        params.force_x = 1.2e-7;

        std::cout << "\nSimulation Configuration:" << std::endl;
        std::cout << "  Grid size: " << params.nx << " x " << params.ny << std::endl;
        std::cout << "  Timesteps: " << params.num_timesteps << std::endl;
        std::cout << "  Animation: " << (enable_animation ? "enabled" : "disabled") << std::endl;

        std::cout << std::endl;

        LBM::Solver solver(params, enable_animation, animation_frequency);
        solver.initialise();

        // Run simulation
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!solver.run()) {
            std::cerr << "LBM simulation failed to reach equilibrium" << std::endl;
            return 1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\nSimulation completed in " << duration.count() << " ms" << std::endl;

        // Write final results (static output)
        std::cout << "Writing final results..." << std::endl;
        LBM::IOManager::write_results(solver.get_grid(), solver.get_params());

        if (enable_animation) {
            std::cout << "\nAnimation data written. To create animation, run:" << std::endl;
            std::string data_file = "animation_data_" + std::to_string(params.nx) +
                                   "x" + std::to_string(params.ny) + ".h5";
            std::cout << "  python3 scripts/animate_lbm.py " << data_file << std::endl;
            std::cout << "  # or to save as video:" << std::endl;
            std::cout << "  python3 scripts/animate_lbm.py " << data_file
                      << " --output lbm_animation.mp4" << std::endl;
        }

        std::cout << "\nSimulation completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}