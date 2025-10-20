import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def main():
    """
    Comprehensive LBM Cylinder Flow Visualization
    Generates a 4-panel figure for analysis:
    1. Velocity Magnitude Contour
    2. Streamlines
    3. Vorticity Field
    4. Pressure Field
    """
    try:
        velocity_data = pd.read_csv('velocity_field.csv')
        params = pd.read_csv('simulation_params.csv', index_col='parameter')['value']
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Ensure simulation has run successfully.", file=sys.stderr)
        sys.exit(1)
    except (KeyError, IndexError) as e:
        print(f"Error parsing data: {e}. Check CSV file format.", file=sys.stderr)
        sys.exit(1)

    # --- Extract Parameters ---
    nx = int(params['nx'])
    ny = int(params['ny'])
    re = float(params['reynolds_number'])
    cyl_x = int(params['cylinder_x'])
    cyl_y = int(params['cylinder_y'])
    cyl_r = int(params['cylinder_radius'])

    # --- Reshape Data for 2D Plotting ---
    x_grid = velocity_data['x'].values.reshape(ny, nx)
    y_grid = velocity_data['y'].values.reshape(ny, nx)
    ux_grid = velocity_data['ux'].values.reshape(ny, nx)
    uy_grid = velocity_data['uy'].values.reshape(ny, nx)
    vel_mag_grid = velocity_data['velocity_magnitude'].values.reshape(ny, nx)
    rho_grid = velocity_data['rho'].values.reshape(ny, nx) if 'rho' in velocity_data else np.ones((ny, nx))

    # --- Calculate Vorticity and Pressure ---
    # Use numpy.gradient to compute derivatives for vorticity
    dudy, dudx = np.gradient(ux_grid)
    dvdy, dvdx = np.gradient(uy_grid)
    vorticity = dvdx - dudy

    # Pressure is related to density in LBM: p = c_s^2 * rho (where c_s^2 = 1/3)
    # We plot the deviation from the average density as the pressure field
    pressure = (rho_grid - np.mean(rho_grid)) / 3.0

    # --- Create Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'LBM Cylinder Flow Analysis (Re ≈ {re:.1f})', fontsize=20, fontweight='bold')

    # --- 1. Velocity Magnitude Plot ---
    ax = axes[0, 0]
    contour = ax.contourf(x_grid, y_grid, vel_mag_grid, levels=100, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Velocity Magnitude')
    ax.set_title('Velocity Magnitude Field', fontsize=14)
    draw_cylinder(ax, cyl_x, cyl_y, cyl_r)

    # --- 2. Streamlines Plot ---
    ax = axes[0, 1]
    step = max(1, ny // 40)
    ax.streamplot(x_grid[::step, ::step], y_grid[::step, ::step],
                  ux_grid[::step, ::step], uy_grid[::step, ::step],
                  color=vel_mag_grid[::step, ::step], cmap='autumn',
                  density=2.0, linewidth=1.0, arrowsize=1.0)
    ax.set_title('Flow Streamlines', fontsize=14)
    draw_cylinder(ax, cyl_x, cyl_y, cyl_r)
    ax.set_facecolor('lightgray') # Add background for contrast

    # --- 3. Vorticity Plot ---
    ax = axes[1, 0]
    # Use a diverging colormap for vorticity (positive/negative spin)
    vort_limit = np.max(np.abs(vorticity)) * 0.5 # Clip color range for better contrast
    contour = ax.contourf(x_grid, y_grid, vorticity, levels=100, cmap='RdBu_r',
                          vmin=-vort_limit, vmax=vort_limit)
    fig.colorbar(contour, ax=ax, label='Vorticity (ω)')
    ax.set_title('Vorticity Field', fontsize=14)
    draw_cylinder(ax, cyl_x, cyl_y, cyl_r)

    # --- 4. Pressure Plot ---
    ax = axes[1, 1]
    pressure_limit = np.max(np.abs(pressure))
    contour = ax.contourf(x_grid, y_grid, pressure, levels=100, cmap='coolwarm',
                          vmin=-pressure_limit, vmax=pressure_limit)
    fig.colorbar(contour, ax=ax, label='Pressure (p - p_avg)')
    ax.set_title('Pressure Field', fontsize=14)
    draw_cylinder(ax, cyl_x, cyl_y, cyl_r)

    # --- Final Touches ---
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel('x-coordinate')
            ax.set_ylabel('y-coordinate')
            ax.set_aspect('equal')
            ax.margins(x=0, y=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('cylinder_flow_analysis.png', dpi=300)
    print("Generated comprehensive analysis plot: cylinder_flow_analysis.png")
    plt.show()

def draw_cylinder(ax, x, y, r):
    """Helper function to draw the cylinder on an axis."""
    cylinder = plt.Circle((x, y), r, color='black', zorder=10)
    ax.add_artist(cylinder)

if __name__ == '__main__':
    main()

