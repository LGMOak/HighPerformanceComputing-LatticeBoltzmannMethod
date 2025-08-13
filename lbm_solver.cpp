#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <fstream>

// --- Data Structures and Constants ---
constexpr int nx = 200;
constexpr int ny = 50;
constexpr double tau = 1.0;
constexpr double force_x = 1e-5;
constexpr double force_y = 0.0;
constexpr int num_timesteps = 10000;
constexpr double nu = (tau - 0.5) / 3.0;


// discrete velocity vectors (D2Q9 model)
const std::vector<std::vector<int>> c = {
    {0,0}, {1,0}, {0,1}, {-1,0}, {0,-1},
    {1,1}, {-1,1}, {-1,-1}, {1,-1}
};

// weights for equlibrium distribution
const std::vector<double> weights = {
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// Typedefs for clarity
using Grid = std::vector<std::vector<double>>;
using Distribution = std::vector<std::vector<std::vector<double>>>;

// Function Declarations
void initialise(Distribution& f, Grid& rho, Grid& ux, Grid& uy);
void collision_step(Distribution& f, Distribution& f_star, Grid& rho, Grid& ux, Grid& uy);
void streaming_step(Distribution& f, const Distribution& f_star);
void apply_boundary_conditions(Distribution& f, const Distribution& f_star);
void stability_check(const Distribution& f, int t);
void write_output_data(const Grid& ux, const Grid& uy);
void print_progress(int t, const Grid& ux, const Grid& uy);


int main() {
    std::cout << "LBM Parameters: nx=" << nx << ", ny=" << ny
              << ", tau=" << tau << ", nu=" << std::fixed << std::setprecision(4) << nu
              << ", force_x=" << force_x << std::endl;

    // Data structures
    Grid rho(nx, std::vector<double>(ny, 1.0));
    Grid ux(nx, std::vector<double>(ny, 0.0));
    Grid uy(nx, std::vector<double>(ny, 0.0));
    Distribution f(nx, std::vector<std::vector<double>>(ny, std::vector<double>(9, 0.0)));
    Distribution f_star(nx, std::vector<std::vector<double>>(ny, std::vector<double>(9, 0.0)));

    initialise(f, rho, ux, uy);

    std::cout << "Starting LBM Channel Flow simulation (Poiseuille flow)..." << std::endl;

    // Simulation loop
    for (int t = 0; t < num_timesteps; ++t) {
        collision_step(f, f_star, rho, ux, uy);
        streaming_step(f, f_star);
        apply_boundary_conditions(f, f_star);
        stability_check(f, t);
        print_progress(t, ux, uy);
    }

    std::cout << "\nSimulation finished. Writing output data..." << std::endl;
    write_output_data(ux, uy);
    std::cout << "Data written successfully." << std::endl;

    return 0;
}

void initialise(Distribution& f, Grid& rho, Grid& ux, Grid& uy) {
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            double u_sq = ux[x][y] * ux[x][y] + uy[x][y] * uy[x][y];
            for (int i = 0; i < 9; ++i) {
                double ci_u = c[i][0] * ux[x][y] + c[i][1] * uy[x][y];
                f[x][y][i] = weights[i] * rho[x][y] * (1.0 + 3.0 * ci_u + 4.5 * ci_u * ci_u - 1.5 * u_sq);
            }
        }
    }
}

void collision_step(Distribution& f, Distribution& f_star, Grid& rho, Grid& ux, Grid& uy) {
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            // Calculate macroscopic properties
            rho[x][y] = 0.0;
            ux[x][y] = 0.0;
            uy[x][y] = 0.0;
            for (int i = 0; i < 9; ++i) {
                rho[x][y] += f[x][y][i];
                ux[x][y] += f[x][y][i] * c[i][0];
                uy[x][y] += f[x][y][i] * c[i][1];
            }
            if (rho[x][y] < 1e-9) rho[x][y] = 1e-9;
            ux[x][y] /= rho[x][y];
            uy[x][y] /= rho[x][y];

            // Compute equilibrium and apply BGK collision
            double u_sq = ux[x][y] * ux[x][y] + uy[x][y] * uy[x][y];
            for (int i = 0; i < 9; ++i) {
                double ci_u = c[i][0] * ux[x][y] + c[i][1] * uy[x][y];
                double body_force_term = 3.0 * weights[i] * (c[i][0] * force_x + c[i][1] * force_y);
                double f_eq = weights[i] * rho[x][y] * (1.0 + 3.0 * ci_u + 4.5 * ci_u * ci_u - 1.5 * u_sq) + body_force_term;
                f_star[x][y][i] = f[x][y][i] - (1.0 / tau) * (f[x][y][i] - f_eq);
            }
        }
    }
    // After collision, f_star is the post-collision state.
    // The next step (streaming) will update f based on f_star.
}

void streaming_step(Distribution& f, const Distribution& f_star) {
    // This function implements the streaming and updates f directly
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int i = 0; i < 9; ++i) {
                int x_dest = (x + c[i][0] + nx) % nx;
                int y_dest = y + c[i][1];

                if (y_dest >= 0 && y_dest < ny) {
                    f[x_dest][y_dest][i] = f_star[x][y][i];
                }
            }
        }
    }
}

void apply_boundary_conditions(Distribution& f, const Distribution& f_star) {
    // Top wall (y = ny-1): bounce-back
    for (int x = 0; x < nx; ++x) {
        f[x][ny-1][4] = f_star[x][ny-1][2]; // 2 (UP) -> 4 (DOWN)
        f[x][ny-1][7] = f_star[x][ny-1][5]; // 5 (UP-RIGHT) -> 7 (DOWN-LEFT)
        f[x][ny-1][8] = f_star[x][ny-1][6]; // 6 (UP-LEFT) -> 8 (DOWN-RIGHT)
    }

    // Bottom wall (y = 0): bounce-back
    for (int x = 0; x < nx; ++x) {
        f[x][0][2] = f_star[x][0][4]; // 4 (DOWN) -> 2 (UP)
        f[x][0][5] = f_star[x][0][7]; // 7 (DOWN-LEFT) -> 5 (UP-RIGHT)
        f[x][0][6] = f_star[x][0][8]; // 8 (DOWN-RIGHT) -> 6 (UP-LEFT)
    }
}

void stability_check(const Distribution& f, int t) {
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
        exit(-1);
    }
    if (min_f < -1e-5) {
        std::cout << "Error: Significant negative f detected at timestep " << t
                  << ". Min f: " << std::scientific << min_f << ". Exiting." << std::endl;
        exit(-1);
    }
}

void print_progress(int t, const Grid& ux, const Grid& uy) {
    if (t % 1000 == 0) {
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

void write_output_data(const Grid& ux, const Grid& uy) {
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
}
