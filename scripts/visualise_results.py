import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    """
    Visualisation script for numerical fluid dynamics simulation results (Lattice Boltzmann method)
    """

    if not os.path.exists('../data/velocity_field.csv'):
        print("Error: ../data/velocity_field.csv not found. Run the simulation first.")
        return

    # Read data from data directory (one level up from scripts)
    velocity_data = pd.read_csv('../data/velocity_field.csv')
    profile_data = pd.read_csv('../data/velocity_profile.csv')
    params = pd.read_csv('../data/simulation_params.csv')

    # Get parameters
    nx = int(params[params['parameter'] == 'nx']['value'].values[0])
    ny = int(params[params['parameter'] == 'ny']['value'].values[0])

    print(f"Grid dimensions: {nx} x {ny}")
    print(f"Data points: {len(velocity_data)}")

    ux = velocity_data['ux'].values.reshape((nx, ny))
    uy = velocity_data['uy'].values.reshape((nx, ny))
    vel_mag = velocity_data['velocity_magnitude'].values.reshape((nx, ny))

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Velocity magnitude contour
    # Transpose for proper orientation (x=horizontal, y=vertical)
    im1 = axes[0, 0].contourf(vel_mag.T, levels=50, cmap='viridis', origin='lower')
    axes[0, 0].set_title('Velocity Magnitude')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])

    # Create 1D coordinate arrays
    x_1d = np.arange(nx)
    y_1d = np.arange(ny)

    X, Y = np.meshgrid(x_1d, y_1d, indexing='xy')

    ux_plot = ux.T
    uy_plot = uy.T
    vel_mag_plot = vel_mag.T

    strm = axes[0, 1].streamplot(X, Y, ux_plot, uy_plot, density=2, color=vel_mag_plot, cmap='plasma')
    axes[0, 1].set_title('Streamlines')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')

    # Velocity profile
    axes[1, 0].plot(profile_data['ux'], profile_data['y'], 'b-', linewidth=2, label='Numerical Simulation (LBM)')
    axes[1, 0].set_title('Velocity Profile at Mid-Channel')
    axes[1, 0].set_xlabel('Velocity (ux)')
    axes[1, 0].set_ylabel('y position')
    axes[1, 0].grid(True, alpha=0.3)

    # Compare with analytical solution
    y_theory = np.linspace(0, ny-1, 100)
    H = ny - 1
    nu = float(params[params['parameter'] == 'nu']['value'].values[0])
    force_x = float(params[params['parameter'] == 'force_x']['value'].values[0])

    # For channel flow with walls at y=0 and y=ny-1:
    # Theoretical parabolic profile: u = (force_x / (2*nu)) * y * (H - y)
    # This gives zero velocity at walls (y=0 and y=H)
    u_theory = (force_x / (2 * nu)) * y_theory * (H - y_theory)

    axes[1, 0].plot(u_theory, y_theory, 'r--', linewidth=2, label='Theoretical', alpha=0.7)
    axes[1, 0].legend()

    # Horizontal velocity contour
    im2 = axes[1, 1].contourf(ux.T, levels=50, cmap='RdBu_r', origin='lower')
    axes[1, 1].set_title('Horizontal Velocity (ux)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1, 1])

    # Print some diagnostics
    print(f"Max velocity: {np.max(vel_mag):.6f}")
    print(f"Max ux at center: {np.max(ux[nx//2, :]):.6f}")
    print(f"Theoretical max ux: {(force_x * H**2) / (8 * nu):.6f}")

    plt.tight_layout()
    plt.savefig('../lbm_results.png', dpi=300, bbox_inches='tight')
    print("Visualisation saved as '../lbm_results.png'")
    plt.show()

if __name__ == '__main__':
    main()