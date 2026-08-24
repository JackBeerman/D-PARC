# /utilities/viz.py
import matplotlib.pyplot as plt
import numpy as np
import imageio


def save_gifs_for_channels(
    predictions,
    channels,
    cmaps,
    filename_prefix="predictions",
    interval=0.1,
    batch_idx=0,
):
    """
    Save a sequence of predictions as separate GIFs for each channel with a specified colormap.

    This corrected version ensures consistent color scaling across all frames and uses an
    efficient plotting method to improve performance.

    Args:
        predictions (torch.Tensor): The predictions tensor of shape (timesteps, batch_size, channels, height, width).
        channels (list of str): List of channel names.
        cmaps (list of str): List of colormaps for each channel.
        filename_prefix (str): Prefix for the output GIF filenames.
        interval (float): Time between frames in seconds.
        batch_idx (int): Index of the sample in the batch to visualize.
    """
    # Select the batch sample to visualize
    prediction_sequence = predictions[:, batch_idx]  # Shape: (timesteps, channels, height, width)

    # Loop over each channel
    for i, channel_name in enumerate(channels):
        cmap = cmaps[i]
        frames = []

        # --- FIX 1: Determine global min/max for consistent color scaling ---
        channel_data = prediction_sequence[:, i, :, :]
        vmin = channel_data.min().item()
        vmax = channel_data.max().item()

        # --- FIX 2: Create figure and axes only ONCE per channel ---
        fig, ax = plt.subplots()
        
        # Create an initial image object and a single colorbar
        initial_frame = channel_data[0, :, :].cpu().numpy()
        im = ax.imshow(initial_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)
        ax.axis("off")

        for t in range(prediction_sequence.shape[0]):
            # Get the frame data
            frame = channel_data[t, :, :].cpu().numpy()
            
            # --- FIX 2 (cont.): Update the data of the existing image object ---
            im.set_data(frame)
            ax.set_title(f"{channel_name} - Timestep {t+1}")

            # Save the current frame as an image in memory
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            frames.append(image)
        
        # Close the plot after processing all frames for the current channel
        plt.close(fig)

        # Save frames as a gif for the current channel
        gif_filename = f"{filename_prefix}_{channel_name}.gif"
        imageio.mimsave(gif_filename, frames, duration=interval * 1000, loop=0) # duration is in ms for some versions
        print(f"GIF saved to {gif_filename}")


def visualize_channels(
    ic,
    t0,
    t1,
    target,
    channel_names=None,
    channel_cmaps=None,
    sample_index=0,
    batch_size=1,
    figsize=(25, 20),
):
    """
    Visualizes the channels of the initial condition and target timesteps with customizable color maps.

    Args:
        ic (torch.Tensor): Initial condition tensor of shape (batch_size, channels, height, width).
        t0 (torch.Tensor): Scalar tensor (0.0), not used in visualization but kept for compatibility.
        t1 (torch.Tensor): Tensor of shape (future_steps,), not used directly in visualization.
        target (torch.Tensor): Target tensor of shape (future_steps, batch_size, channels, height, width).
        channel_names (list of str, optional): List of channel names for labeling. If None, channels will be unnamed.
        channel_cmaps (list of str or matplotlib.colors.Colormap, optional):
            List of color maps for each channel. If None, 'viridis' is used for all channels.
        sample_index (int, optional): Index of the sample in the batch to visualize. Defaults to 0.
        batch_size (int, optional): Total number of samples in the batch. Needed if `sample_index` is to be visualized.
        figsize (tuple, optional): Figure size for the plots. Defaults to (25, 20).

    Raises:
        ValueError: If `sample_index` is out of bounds or if `channel_cmaps` length doesn't match number of channels.
    """
    # Validate sample_index
    if sample_index >= batch_size or sample_index < 0:
        raise ValueError(
            f"sample_index {sample_index} is out of bounds for batch size {batch_size}."
        )

    # Select the specific sample from the batch
    ic_sample = ic[sample_index]  # Shape: (channels, height, width)
    target_sample = target[
        :, sample_index
    ]  # Shape: (future_steps, channels, height, width)

    num_channels = ic_sample.shape[0]
    future_steps = target_sample.shape[0]

    # Handle channel_cmaps
    if channel_cmaps is None:
        # Use 'viridis' for all channels by default
        channel_cmaps = ["viridis"] * num_channels
    else:
        if not isinstance(channel_cmaps, list):
            raise TypeError(
                "channel_cmaps must be a list of color map names or Colormap objects."
            )
        if len(channel_cmaps) != num_channels:
            raise ValueError(
                f"Length of channel_cmaps ({len(channel_cmaps)}) does not match number of channels ({num_channels}). "
                f"Ensure you provide a color map for each channel or leave it as None to use default."
            )

    # Debugging: Print min and max of each channel
    print("Channel Data Statistics:")
    for idx in range(num_channels):
        channel_data_ic = ic_sample[idx].cpu().numpy()
        channel_data_target = target_sample[:, idx].cpu().numpy()
        print(
            f"Channel {idx}: IC min={channel_data_ic.min()}, IC max={channel_data_ic.max()}"
        )
        for step in range(future_steps):
            print(
                f"  Step {step + 1}: min={channel_data_target[step].min()}, max={channel_data_target[step].max()}"
            )

    # Determine subplot grid size
    cols = num_channels
    rows = future_steps + 1  # +1 for the initial condition

    # Create a figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle axes array shape
    if num_channels == 1:
        axes = axes.reshape(-1, 1)
    if rows == 1:
        axes = axes.reshape(1, -1)

    # Plot Initial Condition
    for channel in range(num_channels):
        ax = axes[0, channel]
        data = ic_sample[channel].cpu().numpy()
        cmap = channel_cmaps[channel]
        im = ax.imshow(data, cmap=cmap)
        if channel_names:
            ax.set_title(f"IC - {channel_names[channel]}")
        else:
            ax.set_title(f"IC - Channel {channel}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot Targets
    for step in range(future_steps):
        for channel in range(num_channels):
            ax = axes[step + 1, channel]
            data = target_sample[step, channel].cpu().numpy()
            cmap = channel_cmaps[channel]
            im = ax.imshow(data, cmap=cmap)
            if channel_names:
                ax.set_title(f"Step {step + 1} - {channel_names[channel]}")
            else:
                ax.set_title(f"Step {step + 1} - Channel {channel}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def save_gifs_with_ground_truth(
    predictions,
    ground_truth,
    channels,
    cmaps,
    filename_prefix="comparison",
    interval=0.1,
    batch_idx=0,
):
    """
    Saves side-by-side GIFs of predictions and ground truth for each channel.

    This version uses a reliable (but less efficient) method of recreating the plot
    for each frame and uses the modern Matplotlib API for capturing frames.
    """
    print("--- Running Final GIF Generation ---")
    prediction_sequence = predictions[:, batch_idx].cpu()
    ground_truth_sequence = ground_truth[:, batch_idx].cpu()

    # Determine vmin/vmax based ONLY on the ground truth data
    gt_min_max = []
    for i in range(ground_truth_sequence.shape[1]):
        gt_data = ground_truth_sequence[:, i, :, :].numpy()
        vmin = gt_data.min()
        vmax = gt_data.max()
        gt_min_max.append((vmin, vmax))

    # Loop over each channel to create a separate GIF
    for i, channel_name in enumerate(channels):
        cmap = cmaps[i]
        vmin, vmax = gt_min_max[i]
        
        frames = [] # A list to hold the image data for each frame
        print(f"\n--- Processing channel: '{channel_name}' ---")

        # Loop through each timestep
        for t in range(prediction_sequence.shape[0]):
            # Create a new, clean figure for every single frame.
            fig, axes = plt.subplots(
                1, 2, figsize=(9, 4.5),
                gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.05}
            )
            
            gt_frame = ground_truth_sequence[t, i].numpy()
            pred_frame = prediction_sequence[t, i].numpy()

            # Plot Ground Truth
            axes[0].imshow(gt_frame, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0].set_title(f"Ground Truth - Timestep {t+1}")
            axes[0].axis("off")

            # Plot Prediction
            im_pred = axes[1].imshow(pred_frame, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1].set_title(f"Predicted - Timestep {t+1}")
            axes[1].axis("off")
            
            fig.colorbar(im_pred, ax=axes.ravel().tolist(), shrink=0.7)

            # Draw the canvas and capture it to an in-memory numpy array
            fig.canvas.draw()
            
            # --- THIS IS THE CORRECTED PART ---
            # Use buffer_rgba() which is the modern, correct method
            image_rgba = np.asarray(fig.canvas.buffer_rgba())
            # Append the RGB part of the image, slicing off the alpha channel
            frames.append(image_rgba[..., :3])
            # --- END OF CORRECTION ---

            # Close the figure to free memory.
            plt.close(fig)

            if (t + 1) % 5 == 0 or t == prediction_sequence.shape[0] - 1:
                print(f"  ...processed frame {t+1}/{prediction_sequence.shape[0]}")

        # Save the collected frames as a GIF
        gif_filename = f"{filename_prefix}_{channel_name}.gif"
        print(f"Saving {len(frames)} frames to {gif_filename}...")
        imageio.mimsave(gif_filename, frames, duration=int(interval * 1000), loop=0)
        print("Save complete.")
def save_comparison_snapshots(
    predictions,      # tensor of shape (T, B, C, H, W)
    ground_truth,     # same shape
    channels,         # list of C channel names
    cmaps,            # list of C matplotlib colormaps
    filename_prefix="snapshot",
    num_snapshots=4,
    batch_idx=0
):
    """
    For each channel, pick `num_snapshots` timesteps evenly spaced through T,
    plot GT on top row, prediction on bottom row,
    label only the rows ("Ground Truth" / "Prediction") centered above them,
    and label each column by its timestep.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # bring data to CPU numpy
    pred_seq = predictions[:, batch_idx].cpu().numpy()  # (T, C, H, W)
    gt_seq   = ground_truth[:, batch_idx].cpu().numpy()
    T, C, H, W = pred_seq.shape

    # compute snapshot indices
    snap_idxs = np.linspace(0, T - 1, num_snapshots, dtype=int)

    for i, ch_name in enumerate(channels):
        cmap = cmaps[i]
        # compute vmin/vmax across both GT & pred for this channel
        data_concat = np.concatenate([pred_seq[:, i], gt_seq[:, i]], axis=0)
        vmin, vmax = data_concat.min(), data_concat.max()

        # create figure
        fig, axes = plt.subplots(
            2, num_snapshots,
            figsize=(4 * num_snapshots, 6),
            constrained_layout=False
        )
        # adjust spacing: some horizontal, minimal vertical
        fig.subplots_adjust(left=0.05, right=0.92, top=0.88, bottom=0.08,
                            wspace=0.1, hspace=0.02)

        # plot each snapshot
        for j, t in enumerate(snap_idxs):
            ax_gt   = axes[0, j]
            ax_pred = axes[1, j]

            ax_gt.imshow(gt_seq[t, i],   cmap=cmap, vmin=vmin, vmax=vmax)
            ax_pred.imshow(pred_seq[t, i], cmap=cmap, vmin=vmin, vmax=vmax)

            # only column titles: timestep
            ax_gt.set_title(f"t={t+1}", fontsize=10, pad=4)
            ax_gt.axis("off")
            ax_pred.axis("off")

        # row titles centered above each row
        fig.text(0.5, 0.955,  "Ground Truth", ha='center', va='bottom', fontsize=14)
        fig.text(0.5, 0.495,  "Prediction",   ha='center', va='bottom', fontsize=14)

        # single colorbar on right
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

        # save and clean up
        out_fname = f"{filename_prefix}_{ch_name}_snapshots.png"
        fig.savefig(out_fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved snapshots for channel '{ch_name}' → {out_fname}")

