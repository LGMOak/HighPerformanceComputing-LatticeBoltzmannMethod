import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    """LBM visualisation"""

    # Load data
    velocity_data = pd.read_csv('velocity_field.csv')
    profile_data = pd.read_csv('velocity_profile.csv')
    params = pd.read_csv('simulation_params.csv')

    # Extract parameters
    nx = int(params[params['parameter'] == 'nx']['value'].values[0])
    ny = int(params[params['parameter'] == 'ny']['value'].values[0])
    nu = float(params[params['parameter'] == 'nu']['value'].values[0])
    force_x = float(params[params['parameter'] == 'force_x']['value'].values[0])
    tau = float(params[params['parameter'] == 'tau']['value'].values[0])

    # Reshape velocity field data
    ux = velocity_data['ux'].values.reshape((ny, nx))
    uy = velocity_data['uy'].values.reshape((ny, nx))
    vel_mag = velocity_data['velocity_magnitude'].values.reshape((ny, nx))

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'LBM Poiseuille Flow: {nx}×{ny} grid, τ={tau}', fontsize=14)

    # 1. Velocity Profile Validation
    ax = axes[0]
    ax.plot(profile_data['ux'], profile_data['y'], 'b-', linewidth=3, label='LBM')

    # Analytical solution
    y_theory = np.linspace(0, ny-1, 100)
    H = ny - 1
    u_theory = (force_x / (2 * nu)) * y_theory * (H - y_theory)
    ax.plot(u_theory, y_theory, 'r--', linewidth=2, label='Theory')

    ax.set_title('Velocity Profile')
    ax.set_xlabel('ux')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Velocity Magnitude Field
    ax = axes[1]
    im = ax.contourf(vel_mag, levels=30, cmap='viridis', origin='lower')
    ax.set_title('Velocity Magnitude')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='|u|')

    # 3. Streamlines
    ax = axes[2]
    x_coords = np.arange(nx)
    y_coords = np.arange(ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')

    # Subsample for clean streamlines
    step = max(1, min(nx, ny) // 40)
    ax.streamplot(X[::step, ::step], Y[::step, ::step],
                  ux[::step, ::step], uy[::step, ::step],
                  density=1.5, color=vel_mag[::step, ::step],
                  cmap='plasma', linewidth=2)
    ax.set_title('Flow Streamlines')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('lbm_figure.png', dpi=300, bbox_inches='tight')
    print(f"Max velocity: {np.max(vel_mag):.6f}")

    plt.show()

if __name__ == '__main__':
    main()