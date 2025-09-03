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
    Loads LBM simulation data from an HDF5 file and creates an animation.
    Optionally saves the animation to a video file.
    """
    print(f"Loading data from: {filename}")

    with h5py.File(filename, 'r') as f:
        nx, ny = f['/grid_dims'][:]
        print(f"Grid dimensions: {nx} x {ny}")

        velocity_group = f['/velocity']
        timesteps = []
        for key in velocity_group.keys():
            if key.startswith('vel_mag_'):
                timesteps.append(int(key.split('_')[2]))
        timesteps.sort()

        if not timesteps:
            print("Error: No 'vel_mag_*' datasets found.")
            return

        print(f"Found {len(timesteps)} frames.")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f'LBM Velocity Magnitude ({nx}x{ny})')

        first_frame_data = velocity_group[f'vel_mag_{timesteps[0]}'][:]
        if first_frame_data.shape != (ny, nx):
            print(f"WARNING: Shape mismatch! Expected ({ny}, {nx}) but got {first_frame_data.shape}")

        im = ax.imshow(first_frame_data, cmap='viridis', animated=True, origin='lower')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Velocity Magnitude')
        time_text = ax.set_title(f'Timestep: {timesteps[0]}')

        def update(frame_index):
            timestep = timesteps[frame_index]
            data = velocity_group[f'vel_mag_{timestep}'][:]
            im.set_array(data)
            time_text.set_text(f'Timestep: {timestep}')
            im.set_clim(data.min(), data.max())
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
            # Set up the writer. You can adjust the fps (frames per second).
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

            # Save the animation. This can take a moment.
            ani.save(output_filename, writer=writer)
            print("Video saved successfully!")
        else:
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    hdf5_filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILENAME

    animate_lbm(hdf5_filename, save_to_video=True)