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
        bool enable_animation = false;
        int animation_frequency = 100;  // Save every 100 timesteps

        // Parse command line arguments for animation
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--animation" || arg == "-a") {
                enable_animation = true;
                std::cout << "Animation enabled" << std::endl;
            } else if (arg == "--anim-freq" || arg == "-f") {
                if (i + 1 < argc) {
                    animation_frequency = std::stoi(argv[++i]);
                    std::cout << "Animation frequency set to: " << animation_frequency << std::endl;
                }
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n";
                std::cout << "Options:\n";
                std::cout << "  --animation, -a          Enable animation output\n";
                std::cout << "  --anim-freq, -f <freq>   Animation save frequency (default: 100)\n";
                std::cout << "  --help, -h               Show this help message\n";
                return 0;
            }
        }

        // O0 flag parameters
        params.nx = 256;
        params.ny = 64;
        params.num_timesteps = 5000;
        params.output_frequency = 1000;

        // Benchmark parameters
        // params.nx = 1024;
        // params.ny = 256;
        // params.num_timesteps = 5000;
        // params.output_frequency = 100;

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

        std::cout << "\nSimulation Configuration:" << std::endl;
        std::cout << "  Grid size: " << params.nx << " x " << params.ny << std::endl;
        std::cout << "  Timesteps: " << params.num_timesteps << std::endl;
        std::cout << "  Animation: " << (enable_animation ? "enabled" : "disabled") << std::endl;
        if (enable_animation) {
            std::cout << "  Animation frequency: every " << animation_frequency << " timesteps" << std::endl;
            int expected_frames = params.num_timesteps / animation_frequency + 1;
            std::cout << "  Expected frames: ~" << expected_frames << std::endl;

            // Estimate file size (rough calculation)
            double data_per_frame_mb = (params.nx * params.ny * 3 * sizeof(double)) / (1024.0 * 1024.0);
            double estimated_size_mb = data_per_frame_mb * expected_frames;
            std::cout << "  Estimated file size: ~" << std::fixed << std::setprecision(1)
                      << estimated_size_mb << " MB" << std::endl;

            if (estimated_size_mb > 1000) {
                std::cout << "  WARNING: Large file size expected. Consider increasing animation_frequency." << std::endl;
            }
        }
        std::cout << std::endl;

        // Create solver with animation support
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