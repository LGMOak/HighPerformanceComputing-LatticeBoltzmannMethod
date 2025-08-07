#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <fstream>

int main() {
    // Channel dimensions
    constexpr int nx = 200;
    constexpr int ny = 50;

    // Relaxation time (>0.5)
    constexpr double tau = 1.0;

    // Constant external force
    constexpr double force_x = 1e-5;
    constexpr double force_y = 0.0;

    // Number of simulation iterations
    constexpr int num_timesteps = 10000;

    // Derived kinematic viscosity
    constexpr double nu = (tau - 0.5) / 3.0;

    std::cout << "LBM Parameters: nx=" << nx << ", ny=" << ny
              << ", tau=" << tau << ", nu=" << std::fixed << std::setprecision(4) << nu
              << ", force_x=" << force_x << std::endl;

    // Discrete velocity vectors (D2Q9 model)
    const std::vector<std::vector<int>> c = {
        {0,0},   // 0
        {1,0},   // 1
        {0,1},   // 2
        {-1,0},  // 3
        {0,-1},  // 4
        {1,1},   // 5
        {-1,1},  // 6
        {-1,-1}, // 7
        {1,-1}   // 8
    };

    // Weights for equilibrium distribution
    const std::vector<double> weights = {
        4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
    };

    // Initialise density (rho) and macroscopic velocities
    std::vector<std::vector<double>> rho(nx, std::vector<double>(ny, 1.0));
    std::vector<std::vector<double>> ux(nx, std::vector<double>(ny, 0.0));
    std::vector<std::vector<double>> uy(nx, std::vector<double>(ny, 0.0));

    // Distribution function f_i
    std::vector<std::vector<std::vector<double>>> f(nx, std::vector<std::vector<double>>(ny, std::vector<double>(9, 0.0)));

    // Temporary arrays
    std::vector<std::vector<std::vector<double>>> f_star(nx, std::vector<std::vector<double>>(ny, std::vector<double>(9, 0.0)));

    // Calculate initial equilibrium populations for all cells
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            double u_sq = ux[x][y] * ux[x][y] + uy[x][y] * uy[x][y];
            for (int i = 0; i < 9; ++i) {
                double ci_u = c[i][0] * ux[x][y] + c[i][1] * uy[x][y];
                f[x][y][i] = weights[i] * rho[x][y] * (1.0 + 3.0 * ci_u + 4.5 * ci_u * ci_u - 1.5 * u_sq);
            }
        }
    }

    std::cout << "Starting LBM Channel Flow simulation (Poiseuille flow)..." << std::endl;

    // Simulation loop
    for (int t = 0; t < num_timesteps; ++t) {
        // Step 1: Collision step - calculate macroscopic properties
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                // Calculate density and velocity
                rho[x][y] = 0.0;
                ux[x][y] = 0.0;
                uy[x][y] = 0.0;

                for (int i = 0; i < 9; ++i) {
                    rho[x][y] += f[x][y][i];
                    ux[x][y] += f[x][y][i] * c[i][0];
                    uy[x][y] += f[x][y][i] * c[i][1];
                }

                // Avoid division by zero
                if (rho[x][y] < 1e-9) {
                    rho[x][y] = 1e-9;
                }

                ux[x][y] /= rho[x][y];
                uy[x][y] /= rho[x][y];

                // Compute equilibrium and apply collision
                double u_sq = ux[x][y] * ux[x][y] + uy[x][y] * uy[x][y];

                for (int i = 0; i < 9; ++i) {
                    double ci_u = c[i][0] * ux[x][y] + c[i][1] * uy[x][y];
                    double body_force_term = 3.0 * weights[i] * (c[i][0] * force_x + c[i][1] * force_y);
                    double f_eq = weights[i] * rho[x][y] * (1.0 + 3.0 * ci_u + 4.5 * ci_u * ci_u - 1.5 * u_sq) + body_force_term;

                    // BGK collision
                    f_star[x][y][i] = f[x][y][i] - (1.0 / tau) * (f[x][y][i] - f_eq);
                }
            }
        }

        // Step 2: Streaming step
        std::vector<std::vector<std::vector<double>>> f_streamed(nx, std::vector<std::vector<double>>(ny, std::vector<double>(9, 0.0)));

        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int i = 0; i < 9; ++i) {
                    // Calculate destination after streaming
                    int x_dest = (x + c[i][0] + nx) % nx;  // Periodic in x
                    int y_dest = y + c[i][1];              // Handle y boundaries separately

                    // Only stream if destination is within bounds
                    if (y_dest >= 0 && y_dest < ny) {
                        f_streamed[x_dest][y_dest][i] = f_star[x][y][i];
                    }
                }
            }
        }

        // Step 3: Update f with streamed populations
        f = f_streamed;

        // Step 4: Apply boundary conditions
        // Top wall (y = ny-1): bounce-back
        for (int x = 0; x < nx; ++x) {
            f[x][ny-1][4] = f_star[x][ny-2][2];  // DOWN from UP
            f[x][ny-1][7] = f_star[x][ny-2][5];  // DOWN-LEFT from UP-RIGHT
            f[x][ny-1][8] = f_star[x][ny-2][6];  // DOWN-RIGHT from UP-LEFT
        }

        // Bottom wall (y = 0): bounce-back
        for (int x = 0; x < nx; ++x) {
            f[x][0][2] = f_star[x][1][4];  // UP from DOWN
            f[x][0][5] = f_star[x][1][8];  // UP-RIGHT from DOWN-LEFT (FIXED)
            f[x][0][6] = f_star[x][1][7];  // UP-LEFT from DOWN-RIGHT (FIXED)
        }

        // Stability checks
        bool has_nan_inf = false;
        double min_f = 0.0;

        for (int x = 0; x < nx && !has_nan_inf; ++x) {
            for (int y = 0; y < ny && !has_nan_inf; ++y) {
                for (int i = 0; i < 9; ++i) {
                    if (std::isnan(f[x][y][i]) || std::isinf(f[x][y][i])) {
                        has_nan_inf = true;
                        break;
                    }
                    min_f = std::min(min_f, f[x][y][i]);
                }
            }
        }

        if (has_nan_inf) {
            std::cout << "Error: NaN/Inf detected at timestep " << t << ". Exiting." << std::endl;
            return -1;
        }

        if (min_f < -1e-5) {
            std::cout << "Error: Significant negative f detected at timestep " << t
                      << ". Min f: " << std::scientific << min_f << ". Exiting." << std::endl;
            return -1;
        }

        // Progress output
        if (t % 1000 == 0) {
            // Calculate max velocity for progress reporting
            double max_vel_sq = 0.0;
            for (int x = 0; x < nx; ++x) {
                for (int y = 0; y < ny; ++y) {
                    double vel_sq = ux[x][y] * ux[x][y] + uy[x][y] * uy[x][y];
                    max_vel_sq = std::max(max_vel_sq, vel_sq);
                }
            }
            std::cout << "Timestep " << t << ", Max velocity: "
                      << std::fixed << std::setprecision(4) << std::sqrt(max_vel_sq) << std::endl;
        }
    }

    std::cout << "\nWriting output data to CSV files..." << std::endl;

    // 1. Write velocity field data
    std::ofstream velocity_file("velocity_field.csv");
    velocity_file << "x,y,ux,uy,velocity_magnitude\n";
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            double vel_mag = std::sqrt(ux[x][y] * ux[x][y] + uy[x][y] * uy[x][y]);
            velocity_file << x << "," << y << ","
                         << std::fixed << std::setprecision(8)
                         << ux[x][y] << "," << uy[x][y] << "," << vel_mag << "\n";
        }
    }
    velocity_file.close();

    // 2. Write velocity profile at mid-channel
    std::ofstream profile_file("velocity_profile.csv");
    profile_file << "y,ux\n";
    int mid_x = nx / 2;
    for (int y = 0; y < ny; ++y) {
        profile_file << y << "," << std::fixed << std::setprecision(8) << ux[mid_x][y] << "\n";
    }
    profile_file.close();

    // 3. Write parameters for Python script
    std::ofstream params_file("simulation_params.csv");
    params_file << "parameter,value\n";
    params_file << "nx," << nx << "\n";
    params_file << "ny," << ny << "\n";
    params_file << "tau," << tau << "\n";
    params_file << "nu," << nu << "\n";
    params_file << "force_x," << force_x << "\n";
    params_file << "num_timesteps," << num_timesteps << "\n";
    params_file.close();

    std::cout << "Data written to:" << std::endl;
    std::cout << "  - velocity_field.csv (full velocity field)" << std::endl;
    std::cout << "  - velocity_profile.csv (profile at x=" << mid_x << ")" << std::endl;
    std::cout << "  - simulation_params.csv (simulation parameters)" << std::endl;
    std::cout << "\nRun 'python visualize_results.py' to generate plots." << std::endl;

    return 0;
}