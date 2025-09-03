import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

DEFAULT_FILENAME = "animation_data_256x64.h5"
# Output video filename
DEFAULT_OUTPUT_FILENAME = "lbm_velocity_animation.mp4"

def animate_lbm(filename, save_to_video=False, output_filename=DEFAULT_OUTPUT_FILENAME):
    """
    Loads LBM simulation data and creates an animation with a fixed global
    color scale to accurately visualize the change in velocity over time.
    """
    print(f"Loading data from: {filename}")

    with h5py.File(filename, 'r') as f:
        nx, ny = f['/grid_dims'][:]
        print(f"Grid dimensions: {nx} x {ny}")

        velocity_group = f['/velocity']
        timesteps = sorted([int(k.split('_')[2]) for k in velocity_group.keys() if k.startswith('vel_mag_')])

        if not timesteps:
            print("Error: No 'vel_mag_*' datasets found.")
            return

        print(f"Found {len(timesteps)} frames. Scanning for global velocity range...")

        # --- KEY CHANGE: Scan all data to find the true global min/max velocity ---
        # We start with the values from the first frame.
        first_frame_data = velocity_group[f'vel_mag_{timesteps[0]}'][:]
        global_min = first_frame_data.min()
        global_max = first_frame_data.max()

        # Loop through the rest of the frames to find the absolute min and max
        for ts in timesteps[1:]:
            data = velocity_group[f'vel_mag_{ts}'][:]
            global_min = min(global_min, data.min())
            global_max = max(global_max, data.max())

        print(f"Global velocity range found: [{global_min:.4f}, {global_max:.4f}]")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f'LBM Velocity Magnitude ({nx}x{ny})')

        # Initialize the plot with the first frame's data
        im = ax.imshow(first_frame_data, cmap='viridis', animated=True, origin='lower')

        # --- KEY CHANGE: Set the color limits ONCE using the global values ---
        im.set_clim(global_min, global_max)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Velocity Magnitude')
        time_text = ax.set_title(f'Timestep: {timesteps[0]}')

        def update(frame_index):
            # Load the data for the current frame
            timestep = timesteps[frame_index]
            data = velocity_group[f'vel_mag_{timestep}'][:]
            im.set_array(data)

            # Update the title
            time_text.set_text(f'Timestep: {timestep}')

            # The color limits are now fixed and do not change in the loop.
            return [im, time_text]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(timesteps),
            interval=50,
            blit=True,
            repeat=False
        )

        if save_to_video:
            print(f"Saving animation to {output_filename}...")
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=20, metadata=dict(artist='LBM Simulation'), bitrate=1800)
            ani.save(output_filename, writer=writer)
            print("Video saved successfully!")
        else:
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    hdf5_filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILENAME
    animate_lbm(hdf5_filename, save_to_video=True)