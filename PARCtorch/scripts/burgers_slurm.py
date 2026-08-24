#!/usr/bin/env python3
"""
Training Script for Burgers PARCTorch Model

This script performs the following steps:
1. Data Normalization
2. Dataset and DataLoader Preparation
3. Model Building
4. Training Loop with Logging and Checkpointing
5. Saving Loss History

Usage:
    python burgers.py --config config.yaml

Author: Your Name
Date: 2025-01-16
"""

import os
import sys
import argparse
import logging
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
# Add PARCTorch to system path
# ---------------------------
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# ---------------------------
# Import PARCTorch Modules
# ---------------------------
from data.normalization import compute_min_max
from data.dataset import (
    GenericPhysicsDataset,
    custom_collate_fn,
    InitialConditionDataset,
    initial_condition_collate_fn,
)
from utilities.viz import visualize_channels, save_gifs_with_ground_truth
from utilities.unet_deform import UNet
from utilities.load import load_model_weights
from PARCtorch.PARCv2 import PARCv2
from PARCtorch.differentiator.differentiator import Differentiator
from PARCtorch.differentiator.finitedifference import FiniteDifference
from PARCtorch.integrator.integrator import Integrator
from PARCtorch.integrator.heun import Heun

# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Burgers PARCTorch Model")
    parser.add_argument(
        "--train_dirs",
        nargs="+",
        required=True,
        help="List of training data directories",
    )
    parser.add_argument(
        "--test_dirs",
        nargs="+",
        required=True,
        help="List of testing data directories",
    )
    parser.add_argument(
        "--min_max_output",
        type=str,
        default="../data/b_min_max.json",
        help="Path to save min and max normalization values",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=250, help="Number of training epochs"
    )
    parser.add_argument(
        "--load_mod",
        type=str,
        default=None,
        help="Directory to load model path",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../weights/burgers",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--loss_dir",
        type=str,
        default="../loss/burgers",
        help="Directory to save loss history",
    )
    parser.add_argument(
        "--log_file", type=str, default="training_burgers.log", help="Log file name"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize a batch before training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    return parser.parse_args()

# ---------------------------
# Setup Logging
# ---------------------------
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

# ---------------------------
# Data Normalization
# ---------------------------
def perform_normalization(data_dirs, output_file, device):
    logging.info("Starting data normalization...")
    compute_min_max(data_dirs, output_file)
    logging.info(f"Min and max values saved to '{output_file}'.")

# ---------------------------
# Create DataLoader
# ---------------------------
def create_dataloader(train_dirs, min_max_path, batch_size, device):
    logging.info("Preparing DataLoader...")
    train_dataset = GenericPhysicsDataset(
        data_dirs=train_dirs, future_steps=5, min_max_path=min_max_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffled for training
        num_workers=4,  # Adjust based on your system
        pin_memory=True if device == "cuda" else False,
        collate_fn=custom_collate_fn,
    )

    logging.info(f"Total samples in training dataset: {len(train_dataset)}")
    return train_loader

# ---------------------------
# Visualize Data
# ---------------------------
def visualize_data(train_loader, device):
    logging.info("Visualizing a batch from the DataLoader...")
    for batch in train_loader:
        ic, t0, t1, target = batch
        channel_names = ["Reynolds (R)", "Velocity U", "Velocity V"]
        custom_cmaps = ["plasma", "inferno", "magma"]

        visualize_channels(
            ic,
            t0,
            t1,
            target,
            channel_names=channel_names,
            channel_cmaps=custom_cmaps,
        )
        break  # Visualize one batch for now
    logging.info("Data visualization completed.")

# ---------------------------
# Build Model
# ---------------------------
def build_model(device):
    logging.info("Building the Burgers PARCTorch model...")
    # Define model parameters
    n_fe_features = 64
    unet_burgers = UNet(
        block_dimensions=[64, 128, 256],
        input_channels=3,  # Assuming your input has 3 channels: Reynolds, u, v
        output_channels=n_fe_features,
        up_block_use_concat=[False, True],
        skip_connection_indices=[0],
        use_deform=False#True#False#True
    )

    # Comparable 5-layer deeper UNet
    #unet_burgers = UNet(
    #    block_dimensions=[64, 128, 256, 512, 1024],
    #    input_channels=3,  # Reynolds, u, v
    #    output_channels=n_fe_features,
    #    up_block_use_concat=[False, True, False, True],
    #    skip_connection_indices=[2, 0],  # Scaled from [0] -> keeping proportional depth
    #    use_deform=False
    #)

    right_diff = FiniteDifference(padding_mode="replicate").to(device)
    heun_int = Heun().to(device)

    diff_burgers = Differentiator(
        1,  # 1 state variable: Reynolds (R). Assuming u and v are velocity components
        n_fe_features,  # Number of features returned by the feature extraction network: 64
        [1, 2],  # Channel indices to calculate advection: u and v
        [1, 2],  # Channel indices to calculate diffusion: u and v
        unet_burgers,  # Feature extraction network: unet_burgers
        "constant",  # Padding mode: constant padding of zero
        right_diff,  # Finite difference method: replication of image gradients
        #False  # Additional parameter if required
    ).to(device)

    burgers_int = Integrator(
        True,  # Clip input data between 0 and 1
        [],  # No Poisson
        heun_int,  # Heun integration
        [None, None, None],  # Data-driven integrators: adjust as needed
        "constant",  # Padding mode: constant padding of zero
        right_diff,  # Finite difference method: replicate image gradients
    ).to(device)

    criterion = torch.nn.L1Loss().to(device)

    model = PARCv2(
        differentiator=diff_burgers,
        integrator=burgers_int,
        loss=criterion
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-5)

    logging.info("Model built successfully.")
    return model, optimizer, criterion

# ---------------------------
# Load Model Weights
# ---------------------------
def load_weights(model, weights_path, device):
    if weights_path is None:
        logging.info("No weights path provided. Training will start from scratch.")
        return model

    logging.info(f"Loading model weights from {weights_path}...")
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info("Model weights loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        raise
    return model

# ---------------------------
# Training Loop
# ---------------------------
def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs,
    save_dir,
    loss_dir,
    device,
):
    logging.info("Starting training...")

    # Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    epoch_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
            mininterval=30.0,
            dynamic_ncols=True,
            disable=False
        )

        for batch_idx, batch in progress_bar:
            ic, t0, t1, gt = batch

            # Move data to device
            ic = ic.to(device, non_blocking=True)
            t0 = t0.to(device, non_blocking=True)
            t1 = t1.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(ic, t0, t1)

            # Compute loss
            loss = criterion(predictions[:, :, 1:, :, :], gt[:, :, 1:, :, :])

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar with average batch loss
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        logging.info(
            f"Epoch [{epoch}/{num_epochs}], Average Loss: {epoch_loss:.7f}"
        )

        # Save model checkpoint every 10 epochs
        if epoch % 100 == 0 or epoch == num_epochs:
            model_save_path = os.path.join(
                save_dir, f"burgers_model_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model checkpoint saved at '{model_save_path}'")

    # Save loss history
    loss_save_path = os.path.join(loss_dir, "burgers_losses.pkl")
    with open(loss_save_path, "wb") as f:
        pickle.dump(epoch_losses, f)
    logging.info(f"Loss history saved at '{loss_save_path}'")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker="o")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    loss_plot_path = os.path.join(loss_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"Loss curve saved at '{loss_plot_path}'")

    logging.info("Training completed successfully.")

# ---------------------------
# Main Function
# ---------------------------
def main():
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.loss_dir, f"{timestamp}_{args.log_file}")
    os.makedirs(args.loss_dir, exist_ok=True)
    setup_logging(log_file)

    logging.info("Training script started.")

    device = (
        args.device
        if torch.cuda.is_available() and args.device == "cuda"
        else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Data Normalization
    perform_normalization(args.train_dirs, args.min_max_output, device)

    # Create DataLoader
    train_loader = create_dataloader(
        args.train_dirs, args.min_max_output, args.batch_size, device
    )

    # Optional Visualization
    if args.visualize:
        visualize_data(train_loader, device)

    # Build Model
    model, optimizer, criterion = build_model(device)

    # Load Model Weights
    model = load_weights(model, args.load_mod, device)

    # Start Training
    train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        args.num_epochs,
        args.save_dir,
        args.loss_dir,
        device,
    )

    logging.info("Training script finished.")

if __name__ == "__main__":
    main()
