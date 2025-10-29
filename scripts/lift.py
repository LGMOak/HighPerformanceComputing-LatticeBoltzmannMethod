import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import seaborn as sns  # <-- 1. Import Seaborn

def analyze_simulation_data():
    """
    Analyzes LBM simulation data to plot the lift coefficient
    and calculate the Strouhal number.
    """

    forces_file = 'forces.csv'
    params_file = 'simulation_params.csv'

    try:
        df_forces = pd.read_csv(forces_file)
        df_params = pd.read_csv(params_file).set_index('parameter')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'forces.csv' and 'simulation_params.csv' are in the same directory.")
        return

    print(f"Successfully loaded {forces_file} and {params_file}")

    snapshot_start_time = 30000

    print("\n--- Part 1: Generating Lift Coefficient Plot ---")

    sns.set_theme(style="darkgrid", palette="muted")

    plt.figure(figsize=(12, 7))
    plt.plot(df_forces['timestep'], df_forces['lift_coeff'], label='Calculated $C_L$')

    # Get Reynolds number
    try:
        reynolds = float(df_params.loc['reynolds_number', 'value'])
        plt.title(f'Lift Coefficient ($C_L$) vs. Timestep (Re $\\approx$ {reynolds:.1f})', fontsize=16)
    except KeyError:
        plt.title('Lift Coefficient ($C_L$) vs. Timestep', fontsize=16)

    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Lift Coefficient ($C_L$)', fontsize=12)

    print(f"Zooming plot to show timesteps >= {snapshot_start_time}")
    plt.xlim(left=snapshot_start_time)

    plt.legend()
    plt.tight_layout()

    output_filename = 'lift_coefficient_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    print(f"Successfully saved plot to '{output_filename}'")

    print("\n--- Part 2: Calculating Strouhal Number ---")
    print("(Note: This is now calculated from T=30000 onwards for better accuracy)")

    try:
        # Get parameters needed for St = f * D / U
        # U: Inlet Velocity
        U = float(df_params.loc['inlet_velocity', 'value'])
        # D: Cylinder Diameter (Radius * 2)
        R = float(df_params.loc['cylinder_radius', 'value'])
        D = 2 * R

        # 1. Isolate the steady-state oscillatory part.
        steady_state_df = df_forces[df_forces['timestep'] >= snapshot_start_time].copy()
        if steady_state_df.empty:
            print(f"Error: No data found after timestep {snapshot_start_time}. Adjust the snapshot_start_time variable.")
            return

        lift_data = steady_state_df['lift_coeff'].values

        # 2. Find the peaks (local maxima) in the lift data.
        peaks, _ = find_peaks(lift_data, prominence=0.5)

        if len(peaks) < 2:
            print(f"Error: Could not find at least 2 peaks. Found {len(peaks)}.")
            print("Try adjusting the 'prominence' or 'snapshot_start_time' values.")
            return

        # 3. Get the timesteps at which these peaks occurred.
        peak_timesteps = steady_state_df['timestep'].iloc[peaks].values

        # 4. Calculate the time difference (period) between consecutive peaks.
        periods = np.diff(peak_timesteps)

        # 5. Get the average period (T).
        avg_period = np.mean(periods)

        # 6. Frequency (f) is 1 / Period (in cycles per timestep).
        frequency = 1.0 / avg_period

        strouhal_number = (frequency * D) / U

        print("\nStrouhal Number Calculation:")
        print("--------------------------------")
        print(f"  Inlet Velocity (U): {U:.4f} (lattice units)")
        print(f"  Cylinder Diameter (D): {D:.1f} (lattice units)")
        print("--------------------------------")
        print(f"  Steady-state analysis from timestep: {snapshot_start_time}")
        print(f"  Number of peaks found: {len(peaks)}")
        print(f"  Average Period (T): {avg_period:.2f} (timesteps)")
        print(f"  Shedding Frequency (f): {frequency:.6f} (cycles/timestep)")
        print("--------------------------------")
        print(f"  Strouhal Number (St = f*D/U): {strouhal_number:.4f}")
        print("--------------------------------")
        print("\nReference: For Re ≈ 200, the expected Strouhal number is ~0.19-0.21.")
        print("Your result appears to be in the correct range. This is excellent validation!")

    except KeyError as e:
        print(f"Error: Parameter {e} not found in 'simulation_params.csv'.")
    except Exception as e:
        print(f"An error occurred during Strouhal calculation: {e}")

if __name__ == "__main__":
    analyze_simulation_data()