#!/usr/bin/env python3
"""
Training Script for PARCTorch Model with Validation and Comprehensive Timing

This script performs the following steps:
1. Data Normalization
2. Dataset and DataLoader Preparation (Train + Validation)
3. Model Building
4. Training Loop with Validation and Best Model Checkpointing
5. Comprehensive Timing Analysis and Reporting
6. Saving Loss History and Training Metrics

Usage:
    python train_parc_with_validation.py --train_dirs <train_dir> --val_dirs <val_dir>

Author: Your Name
Date: 2024-10-24
"""

import os
import sys
import argparse
import logging
import pickle
import time
from datetime import datetime, timedelta

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
from data.dataset import GenericPhysicsDataset, custom_collate_fn
from utilities.viz import visualize_channels
from PARCtorch.PARCv2 import PARCv2
from PARCtorch.differentiator.differentiator import Differentiator
from PARCtorch.differentiator.finitedifference import FiniteDifference
from PARCtorch.integrator.integrator import Integrator
from PARCtorch.integrator.rk4 import RK4
from utilities.unet_deform import UNet
from utilities.load import load_model_weights
from dataset_model_configs import (
    get_dataset_config,
    get_model_config,
    get_integrator,
    print_dataset_config,
    print_model_config
)
from training_history_manager import (
    load_or_create_history,
    save_history_checkpoint,
    update_history_after_epoch,
    print_training_phase_info
)


# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train PARCTorch Model with Validation")
    parser.add_argument(
        "--train_dirs",
        nargs="+",
        required=True,
        help="List of training data directories",
    )
    parser.add_argument(
        "--val_dirs",
        nargs="+",
        required=True,
        help="List of validation data directories",
    )
    parser.add_argument(
        "--min_max_output",
        type=str,
        default="../data/ns_min_max.json",
        help="Path to save min and max normalization values",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--load_mod",
        type=str,
        default=None,
        help="Path to load model weights for resuming training",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="Starting epoch number (use when resuming training)",
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Resume training from checkpoint (loads loss history if available)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../weights",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--loss_dir",
        type=str,
        default="../loss",
        help="Directory to save loss history",
    )
    parser.add_argument(
        "--log_file", type=str, default="training.log", help="Log file name"
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
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=100,
        help="Save regular checkpoints every N epochs",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="large_no_deform",
        choices=["small_no_deform", "small_deform", "large_no_deform"],
        help="Model architecture configuration to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Alias for --model_config (for compatibility with history manager)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hmx", "burgers", "ns"],
        help="Dataset type (hmx, burgers, or ns)",
    )
    parser.add_argument(
        "--future_steps",
        type=int,
        default=1,
        help="Number of future timesteps to predict (tracked for phase management)",
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
def create_dataloader(data_dirs, min_max_path, batch_size, device, dataset_config, shuffle=True):
    logging.info(f"Preparing DataLoader for {data_dirs}...")
    dataset = GenericPhysicsDataset(
        data_dirs=data_dirs,
        future_steps=dataset_config['future_steps'],
        min_max_path=min_max_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=dataset_config['num_workers'],
        pin_memory=True if device == "cuda" else False,
        collate_fn=custom_collate_fn,
    )

    logging.info(f"Total samples in dataset: {len(dataset)}")
    return dataloader


# ---------------------------
# Visualize Data
# ---------------------------
def visualize_data(train_loader, device):
    logging.info("Visualizing a batch from the DataLoader...")
    for batch in train_loader:
        ic, t0, t1, target = batch
        channel_names = [
            "Temperature (T)",
            "Pressure (P)",
            "Microstructure (U)",
            "Velocity U",
            "Velocity V",
        ]
        custom_cmaps = ["jet", "seismic", "binary", "seismic", "seismic"]

        visualize_channels(
            ic,
            t0,
            t1,
            target,
            channel_names=channel_names,
            channel_cmaps=custom_cmaps,
        )
        break
    logging.info("Data visualization completed.")


# ---------------------------
# Build Model
# ---------------------------
def build_model(device, dataset_name, model_config_name="large_no_deform"):
    logging.info("Building the PARCTorch model...")
    
    # Get configurations
    dataset_config = get_dataset_config(dataset_name)
    model_config = get_model_config(dataset_name, model_config_name)
    
    print_dataset_config(dataset_name)
    print_model_config(dataset_name, model_config_name)
    
    # Build UNet with specified configuration
    unet = UNet(
        block_dimensions=model_config['block_dimensions'],
        input_channels=dataset_config['input_channels'],
        output_channels=model_config['n_fe_features'],
        padding_mode=model_config['padding_mode'],
        up_block_use_concat=model_config['up_block_use_concat'],
        skip_connection_indices=model_config['skip_connection_indices'],
        use_deform=model_config['use_deform']
    )

    right_diff = FiniteDifference(padding_mode="replicate").to(device)

    # Get appropriate integrator for dataset
    integrator_method = get_integrator(dataset_config['integrator_type'], device)

    diff = Differentiator(
        dataset_config['num_state_vars'],
        model_config['n_fe_features'],
        dataset_config['advection_indices'],
        dataset_config['diffusion_indices'],
        unet,
        "constant",
        right_diff,
    ).to(device)

    ddi_list = [None] * dataset_config['ddi_list_size']

    integrator = Integrator(
        True,
        dataset_config['poisson_config'],
        integrator_method,
        ddi_list,
        "constant",
        right_diff,
    ).to(device)

    criterion = torch.nn.L1Loss().to(device)

    model = PARCv2(
        differentiator=diff, integrator=integrator, loss=criterion
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-5)

    logging.info(f"Model built successfully using {dataset_config['name']} with {model_config['name']}")
    return model, optimizer, criterion, dataset_config


# ---------------------------
# Load Model Weights
# ---------------------------
def load_model_weights_func(model, weights_path, device):
    if weights_path is None:
        logging.info("No weights path provided. Training will start from scratch.")
        return model
    logging.info(f"Loading model weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle wrapped checkpoint format (with metadata)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logging.info(f"Loading from wrapped checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # Raw state dict
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        logging.info("Model weights loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        raise
    return model


# ---------------------------
# Validation Function
# ---------------------------
def validate_model(model, val_loader, criterion, device, dataset_config):
    """
    Perform validation and return average validation loss.
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            ic, t0, t1, gt = batch

            # Move data to device
            ic = ic.to(device, non_blocking=True)
            t0 = t0.to(device, non_blocking=True)
            t1 = t1.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # Forward pass
            predictions = model(ic, t0, t1)

            # Compute loss based on dataset configuration
            loss_start_ch = dataset_config['loss_start_channel']
            loss = criterion(
                predictions[:, :, loss_start_ch:, :, :],
                gt[:, :, loss_start_ch:, :, :]
            )
            running_loss += loss.item()

    avg_loss = running_loss / len(val_loader)
    return avg_loss


# ---------------------------
# Training Loop with Validation and Timing
# ---------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    save_dir,
    loss_dir,
    device,
    save_frequency=100,
    start_epoch=1,
    resume_training=False,
    model_config_name="large_no_deform",
    dataset_config=None,
    args=None,
):
    logging.info("Starting training with validation...")

    # Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    # Initialize or load training history
    best_model_path = os.path.join(save_dir, "best_model.pth")
    
    # Timing tracking
    epoch_times = []
    train_times = []
    val_times = []
    previous_total_time = 0.0
    
    # Load or create history using new system (handles future_steps changes!)
    loss_history_path = os.path.join(loss_dir, "loss_history.pkl")
    history, is_new_phase = load_or_create_history(loss_history_path, args)
    
    # Show phase info if we're starting a new phase (future_steps changed)
    if is_new_phase:
        logging.info("\n" + "=" * 90)
        logging.info("⚠️  NEW TRAINING PHASE DETECTED - FUTURE_STEPS CHANGED")
        logging.info("=" * 90)
        print_training_phase_info(history)
    
    # Extract convenience variables from history
    train_losses = history.get('train_losses', [])
    val_losses = history.get('val_losses', [])
    best_val_loss = history.get('best_val_loss', float('inf'))
    best_epoch = history.get('best_epoch', 0)
    
    # Load previous timing information if resuming
    if resume_training and 'timing_stats' in history:
        prev_timing = history['timing_stats']
        epoch_times = prev_timing.get('epoch_times', [])
        train_times = prev_timing.get('train_times', [])
        val_times = prev_timing.get('val_times', [])
        previous_total_time = prev_timing.get('total_training_time_seconds', 0.0)
        
        logging.info(f"Resuming training from epoch {start_epoch}...")
        logging.info(f"Loaded history: {len(train_losses)} previous epochs")
        logging.info(f"Previous best validation loss: {best_val_loss:.7f} at epoch {best_epoch}")
        logging.info(f"Previous total training time: {str(timedelta(seconds=int(previous_total_time)))}")
    
    total_start_time = time.time()
    actual_start_epoch = start_epoch if resume_training else 1
    actual_end_epoch = num_epochs if not resume_training else start_epoch + num_epochs - 1

    for epoch in range(actual_start_epoch, actual_end_epoch + 1):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        train_start_time = time.time()

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{actual_end_epoch} [Train]",
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

            # Compute loss based on dataset configuration
            loss_start_ch = dataset_config['loss_start_channel']
            loss = criterion(
                predictions[:, :, loss_start_ch:, :, :],
                gt[:, :, loss_start_ch:, :, :]
            )

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

        train_time = time.time() - train_start_time
        train_times.append(train_time)

        # Average training loss for the epoch
        avg_train_loss = running_train_loss / len(train_loader)

        # Validation phase
        val_start_time = time.time()
        avg_val_loss = validate_model(model, val_loader, criterion, device, dataset_config)
        val_time = time.time() - val_start_time
        
        epoch_time = time.time() - epoch_start_time
        
        # Update history using new system (handles everything automatically!)
        is_best = update_history_after_epoch(
            history, 
            epoch, 
            avg_train_loss, 
            avg_val_loss, 
            epoch_time
        )
        
        # Save history checkpoint
        save_history_checkpoint(history, loss_history_path)
        
        # Update local tracking variables (for plotting later)
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        best_val_loss = history['best_val_loss']
        best_epoch = history['best_epoch']
        epoch_times.append(epoch_time)
        train_times.append(train_time)
        val_times.append(val_time)

        logging.info(
            f"Epoch [{epoch}/{actual_end_epoch}] - "
            f"Train Loss: {avg_train_loss:.7f}, Val Loss: {avg_val_loss:.7f} - "
            f"Time: {epoch_time:.2f}s (Train: {train_time:.2f}s, Val: {val_time:.2f}s)"
            f"{' [BEST]' if is_best else ''}"
        )

        # Save best model if this is the best epoch
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'future_steps': dataset_config['future_steps'],
                'dataset': args.dataset,
                'model_name': model_config_name,
            }, best_model_path)
            logging.info(
                f"✓ New best model saved! Val Loss: {avg_val_loss:.7f}"
            )

        # Save regular checkpoint at specified frequency
        if epoch % save_frequency == 0 or epoch == actual_end_epoch:
            checkpoint_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Regular checkpoint saved at '{checkpoint_path}'")
        
        # Always save latest checkpoint for easy resumption
        latest_checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pth")
        torch.save(model.state_dict(), latest_checkpoint_path)

    current_session_time = time.time() - total_start_time
    total_time = previous_total_time + current_session_time
    total_epochs_completed = len(train_losses)
    
    # Calculate timing statistics
    timing_stats = {
        'total_training_time_seconds': total_time,
        'current_session_time_seconds': current_session_time,
        'previous_session_time_seconds': previous_total_time,
        'total_training_time_formatted': str(timedelta(seconds=int(total_time))),
        'current_session_time_formatted': str(timedelta(seconds=int(current_session_time))),
        'average_epoch_time': sum(epoch_times) / len(epoch_times) if epoch_times else 0,
        'average_train_time_per_epoch': sum(train_times) / len(train_times) if train_times else 0,
        'average_val_time_per_epoch': sum(val_times) / len(val_times) if val_times else 0,
        'min_epoch_time': min(epoch_times) if epoch_times else 0,
        'max_epoch_time': max(epoch_times) if epoch_times else 0,
        'epoch_times': epoch_times,
        'train_times': train_times,
        'val_times': val_times,
        'train_samples_per_second': len(train_loader.dataset) / (sum(train_times) / len(train_times)) if train_times else 0,
        'best_epoch': best_epoch,
        'time_to_best_model_seconds': sum(epoch_times[:best_epoch]) if best_epoch <= len(epoch_times) else sum(epoch_times),
        'time_to_best_model_formatted': str(timedelta(seconds=int(sum(epoch_times[:best_epoch]) if best_epoch <= len(epoch_times) else sum(epoch_times)))),
        'total_epochs_completed': total_epochs_completed,
        'resumed_from_epoch': actual_start_epoch if resume_training else None,
    }

    # Update history with final timing stats and metadata
    history['timing_stats'] = timing_stats
    history['batch_size'] = train_loader.batch_size
    history['num_train_samples'] = len(train_loader.dataset)
    history['num_val_samples'] = len(val_loader.dataset)
    history['num_train_batches'] = len(train_loader)
    history['num_val_batches'] = len(val_loader)
    history['resumed_training'] = resume_training
    history['model_config_name'] = model_config_name
    
    # Save final history checkpoint
    save_history_checkpoint(history, loss_history_path)
    logging.info(f"Final loss history saved at '{loss_history_path}'")

    # ==================================================================================
    # CORRECTED PLOTTING SECTION - Handles Multi-Phase Training
    # ==================================================================================
    
    # Plot comprehensive training metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    epoch_range = range(1, total_epochs_completed + 1)
    
    # ==================================================================================
    # Plot 1: Training and Validation Loss (ALL EPOCHS)
    # ==================================================================================
    ax1.plot(epoch_range, train_losses, marker="o", label="Train Loss", alpha=0.7, markersize=3)
    ax1.plot(epoch_range, val_losses, marker="s", label="Val Loss", alpha=0.7, markersize=3)
    ax1.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val Loss: {best_val_loss:.7f}', linewidth=2)
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch}', linewidth=2)
    if resume_training and actual_start_epoch > 1:
        ax1.axvline(x=actual_start_epoch, color='orange', linestyle=':', alpha=0.5, label=f'Resumed at Epoch {actual_start_epoch}', linewidth=2)
    ax1.set_title("Training and Validation Loss", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (L1)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ==================================================================================
    # Plot 2: Time per Epoch (CURRENT SESSION ONLY)
    # ==================================================================================
    if len(epoch_times) > 0:
        current_session_epoch_range = range(actual_start_epoch, actual_start_epoch + len(epoch_times))
        ax2.plot(current_session_epoch_range, epoch_times, marker="o", label="Total Epoch Time", alpha=0.7, markersize=3)
        ax2.plot(current_session_epoch_range, train_times, marker="s", label="Training Time", alpha=0.7, markersize=3)
        ax2.plot(current_session_epoch_range, val_times, marker="^", label="Validation Time", alpha=0.7, markersize=3)
        if timing_stats['average_epoch_time'] > 0:
            ax2.axhline(y=timing_stats['average_epoch_time'], color='purple', linestyle='--', 
                        label=f"Avg: {timing_stats['average_epoch_time']:.2f}s", linewidth=2)
        if resume_training and actual_start_epoch > 1:
            ax2.axvline(x=actual_start_epoch, color='orange', linestyle=':', alpha=0.5, linewidth=2)
        ax2.set_title("Time per Epoch (Current Session)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Time (seconds)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # ==================================================================================
    # Plot 3: Loss Improvement Over Time (ALL EPOCHS)
    # ==================================================================================
    if len(train_losses) > 0 and train_losses[0] > 0:
        train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
        val_improvement = [(val_losses[0] - loss) / val_losses[0] * 100 for loss in val_losses]
        ax3.plot(epoch_range, train_improvement, marker="o", label="Train Loss Improvement", alpha=0.7, markersize=3)
        ax3.plot(epoch_range, val_improvement, marker="s", label="Val Loss Improvement", alpha=0.7, markersize=3)
        ax3.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=2)
        if resume_training and actual_start_epoch > 1:
            ax3.axvline(x=actual_start_epoch, color='orange', linestyle=':', alpha=0.5, linewidth=2)
        ax3.set_title("Loss Improvement from Initial (%)", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Improvement (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # ==================================================================================
    # Plot 4: Cumulative Training Time (CURRENT SESSION ONLY)
    # ==================================================================================
    if len(epoch_times) > 0:
        cumulative_time = [sum(epoch_times[:i+1])/3600 for i in range(len(epoch_times))]  # in hours
        current_session_epoch_range = range(actual_start_epoch, actual_start_epoch + len(epoch_times))
        ax4.plot(current_session_epoch_range, cumulative_time, marker="o", color='navy', alpha=0.7, markersize=3)
        if timing_stats['average_epoch_time'] > 0:
            ax4.axhline(y=timing_stats['total_training_time_seconds']/3600, color='r', linestyle='--', 
                        label=f'Total time: {timing_stats["total_training_time_formatted"]}', linewidth=2)
        if resume_training and actual_start_epoch > 1:
            ax4.axvline(x=actual_start_epoch, color='orange', linestyle=':', alpha=0.5, 
                       label=f'Resumed at epoch {actual_start_epoch}', linewidth=2)
        ax4.set_title("Cumulative Training Time (Current Session)", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Cumulative Time (hours)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(loss_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Training metrics plot saved to '{plot_path}'")

    # Create comprehensive summary report for paper
    summary_report_path = os.path.join(loss_dir, "training_summary_report.txt")
    with open(summary_report_path, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("COMPREHENSIVE TRAINING SUMMARY REPORT\n")
        f.write("=" * 90 + "\n\n")
        
        if resume_training:
            f.write("TRAINING SESSION INFO:\n")
            f.write("-" * 90 + "\n")
            f.write(f"Training Mode: RESUMED from epoch {actual_start_epoch}\n")
            f.write(f"Current Session: Epochs {actual_start_epoch} to {epoch}\n")
            f.write(f"Current Session Time: {timing_stats['current_session_time_formatted']}\n")
            f.write(f"Previous Session(s) Time: {timing_stats.get('previous_session_time_formatted', 'N/A')}\n")
            f.write(f"Total Cumulative Time: {timing_stats['total_training_time_formatted']}\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 90 + "\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Architecture: {loss_history.get('model_config_name', 'N/A')}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n")
        f.write(f"Model Size (MB): {total_params * 4 / (1024**2):.2f}\n")  # Assuming float32
        f.write(f"Device: {device}\n")
        f.write(f"Optimizer: Adam (lr=1e-5)\n")
        f.write(f"Loss Function: L1Loss\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 90 + "\n")
        f.write(f"Training Samples: {len(train_loader.dataset):,}\n")
        f.write(f"Validation Samples: {len(val_loader.dataset):,}\n")
        f.write(f"Total Samples: {len(train_loader.dataset) + len(val_loader.dataset):,}\n")
        f.write(f"Train/Val Split: {len(train_loader.dataset)/(len(train_loader.dataset)+len(val_loader.dataset))*100:.1f}% / "
                f"{len(val_loader.dataset)/(len(train_loader.dataset)+len(val_loader.dataset))*100:.1f}%\n")
        f.write(f"Batch Size: {train_loader.batch_size}\n")
        f.write(f"Training Batches per Epoch: {len(train_loader)}\n")
        f.write(f"Validation Batches per Epoch: {len(val_loader)}\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total Epochs Completed: {total_epochs_completed}\n")
        f.write(f"Last Completed Epoch: {epoch}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.7f}\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.7f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.7f}\n")
        f.write(f"Initial Training Loss: {train_losses[0]:.7f}\n")
        f.write(f"Initial Validation Loss: {val_losses[0]:.7f}\n")
        f.write(f"Training Loss Reduction: {(1 - train_losses[-1]/train_losses[0])*100:.2f}%\n")
        f.write(f"Validation Loss Reduction: {(1 - val_losses[-1]/val_losses[0])*100:.2f}%\n\n")
        
        f.write("TIMING STATISTICS:\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total Training Time: {timing_stats['total_training_time_formatted']} "
                f"({timing_stats['total_training_time_seconds']:.2f} seconds)\n")
        f.write(f"Total Training Time (hours): {timing_stats['total_training_time_seconds']/3600:.2f} hours\n")
        f.write(f"Average Time per Epoch: {timing_stats['average_epoch_time']:.2f} seconds\n")
        f.write(f"  - Average Training Time per Epoch: {timing_stats['average_train_time_per_epoch']:.2f} seconds\n")
        f.write(f"  - Average Validation Time per Epoch: {timing_stats['average_val_time_per_epoch']:.2f} seconds\n")
        f.write(f"Fastest Epoch: {timing_stats['min_epoch_time']:.2f} seconds\n")
        f.write(f"Slowest Epoch: {timing_stats['max_epoch_time']:.2f} seconds\n")
        f.write(f"Training Throughput: {timing_stats['train_samples_per_second']:.2f} samples/second\n")
        f.write(f"Training Throughput: {timing_stats['train_samples_per_second']*3600:.0f} samples/hour\n\n")
        
        f.write("TIME TO BEST MODEL:\n")
        f.write("-" * 90 + "\n")
        f.write(f"Epochs to Best Model: {best_epoch}\n")
        f.write(f"Time to Best Model: {timing_stats['time_to_best_model_formatted']} "
                f"({timing_stats['time_to_best_model_seconds']:.2f} seconds)\n")
        f.write(f"Time to Best Model (hours): {timing_stats['time_to_best_model_seconds']/3600:.2f} hours\n")
        f.write(f"Percentage of Total Time: {timing_stats['time_to_best_model_seconds']/timing_stats['total_training_time_seconds']*100:.1f}%\n\n")
        
        f.write("COMPUTATIONAL EFFICIENCY:\n")
        f.write("-" * 90 + "\n")
        f.write(f"Time per Training Sample: {timing_stats['average_train_time_per_epoch']/len(train_loader.dataset)*1000:.2f} ms\n")
        f.write(f"Time per Validation Sample: {timing_stats['average_val_time_per_epoch']/len(val_loader.dataset)*1000:.2f} ms\n")
        f.write(f"Training/Validation Time Ratio: {timing_stats['average_train_time_per_epoch']/timing_stats['average_val_time_per_epoch']:.2f}:1\n")
        f.write(f"Estimated Time for 1000 Epochs: {timing_stats['average_epoch_time']*1000/3600:.2f} hours\n\n")
        
        f.write("=" * 90 + "\n")
        f.write("FOR ACADEMIC PAPER (Copy-Paste Ready):\n")
        f.write("=" * 90 + "\n\n")
        
        f.write("METHOD SECTION - Training Details:\n")
        f.write("-" * 90 + "\n")
        f.write(f"The model was trained for {total_epochs_completed} epochs using the Adam optimizer with a learning rate\n")
        f.write(f"of 1e-5 and L1 loss. The training dataset consisted of {len(train_loader.dataset):,} samples,\n")
        f.write(f"with {len(val_loader.dataset):,} samples reserved for validation. A batch size of {train_loader.batch_size} was used\n")
        f.write(f"throughout training. The model architecture contained {total_params:,} parameters.\n\n")
        
        f.write("RESULTS SECTION - Training Performance:\n")
        f.write("-" * 90 + "\n")
        f.write(f"Training converged after {total_epochs_completed} epochs, achieving a best validation loss of {best_val_loss:.4f}\n")
        f.write(f"at epoch {best_epoch}. The total training time was {timing_stats['total_training_time_formatted']}\n")
        f.write(f"({timing_stats['total_training_time_seconds']/3600:.2f} hours)")
        if resume_training:
            f.write(f", conducted across multiple sessions due to computational constraints")
        f.write(f".\n")
        f.write(f"The average time per epoch was {timing_stats['average_epoch_time']:.2f} seconds.\n")
        f.write(f"The model achieved a training throughput of {timing_stats['train_samples_per_second']:.2f} samples per second.\n")
        f.write(f"Training and validation losses decreased by {(1 - train_losses[-1]/train_losses[0])*100:.1f}% and\n")
        f.write(f"{(1 - val_losses[-1]/val_losses[0])*100:.1f}%, respectively, from their initial values.\n\n")
        
        f.write("TABLE FORMAT (LaTeX):\n")
        f.write("-" * 90 + "\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Training Statistics}\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Value} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Parameters & {total_params:,} \\\\\n")
        f.write(f"Training Samples & {len(train_loader.dataset):,} \\\\\n")
        f.write(f"Validation Samples & {len(val_loader.dataset):,} \\\\\n")
        f.write(f"Batch Size & {train_loader.batch_size} \\\\\n")
        f.write(f"Total Epochs & {total_epochs_completed} \\\\\n")
        f.write(f"Best Epoch & {best_epoch} \\\\\n")
        f.write(f"Best Val Loss & {best_val_loss:.4f} \\\\\n")
        f.write(f"Training Time & {timing_stats['total_training_time_seconds']/3600:.2f} hours \\\\\n")
        f.write(f"Time per Epoch & {timing_stats['average_epoch_time']:.2f} s \\\\\n")
        f.write(f"Throughput & {timing_stats['train_samples_per_second']:.2f} samples/s \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        f.write("=" * 90 + "\n")
    
    logging.info(f"Training summary report saved at '{summary_report_path}'")

    # Log final summary to console
    logging.info("\n" + "=" * 90)
    logging.info("TRAINING COMPLETED SUCCESSFULLY")
    logging.info("=" * 90)
    if resume_training:
        logging.info(f"Training resumed from epoch {actual_start_epoch}")
        logging.info(f"Current session time: {timing_stats['current_session_time_formatted']}")
    logging.info(f"Total Training Time: {timing_stats['total_training_time_formatted']} "
                 f"({timing_stats['total_training_time_seconds']/3600:.2f} hours)")
    logging.info(f"Total Epochs Completed: {total_epochs_completed}")
    logging.info(f"Best Validation Loss: {best_val_loss:.7f} (Epoch {best_epoch})")
    logging.info(f"Time to Best Model: {timing_stats['time_to_best_model_formatted']}")
    logging.info(f"Average Time per Epoch: {timing_stats['average_epoch_time']:.2f}s")
    logging.info(f"Training Throughput: {timing_stats['train_samples_per_second']:.2f} samples/sec")
    if len(train_losses) > 1 and train_losses[0] > 0:
        logging.info(f"Loss Reduction: Train {(1 - train_losses[-1]/train_losses[0])*100:.1f}%, "
                     f"Val {(1 - val_losses[-1]/val_losses[0])*100:.1f}%")
    logging.info(f"Best model saved at: {best_model_path}")
    logging.info(f"Latest checkpoint: {latest_checkpoint_path}")
    logging.info(f"Summary report saved at: {summary_report_path}")
    logging.info("=" * 90 + "\n")

    return timing_stats


# ---------------------------
# Main Function
# ---------------------------
def main():
    args = parse_args()
    
    # Handle model argument compatibility
    if args.model is None:
        args.model = args.model_config
    else:
        args.model_config = args.model

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.loss_dir, f"{timestamp}_{args.log_file}")
    os.makedirs(args.loss_dir, exist_ok=True)
    setup_logging(log_file)

    logging.info("=" * 90)
    logging.info("PARCTorch Training Script Started")
    logging.info("=" * 90)
    logging.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = (
        args.device
        if torch.cuda.is_available() and args.device == "cuda"
        else "cpu"
    )
    logging.info(f"Using device: {device}")
    
    if device == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Data Normalization (using training data)
    # Get dataset configuration for min_max file
    dataset_config_temp = get_dataset_config(args.dataset)
    if args.min_max_output == "../data/ns_min_max.json":  # Default value
        args.min_max_output = dataset_config_temp['min_max_file']
    
    perform_normalization(args.train_dirs, args.min_max_output, device)

    # Create DataLoaders with dataset configuration
    dataset_config = get_dataset_config(args.dataset)
    train_loader = create_dataloader(
        args.train_dirs, args.min_max_output, args.batch_size, device, dataset_config, shuffle=True
    )
    
    val_loader = create_dataloader(
        args.val_dirs, args.min_max_output, args.batch_size, device, dataset_config, shuffle=False
    )

    # Optional Visualization
    if args.visualize:
        visualize_data(train_loader, device)

    # Build Model with dataset and model configuration
    model, optimizer, criterion, dataset_config = build_model(device, args.dataset, args.model_config)
    
    # Load the model weights if provided
    model = load_model_weights_func(model, args.load_mod, device)

    # Start Training with Validation
    timing_stats = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.num_epochs,
        args.save_dir,
        args.loss_dir,
        device,
        args.save_frequency,
        args.start_epoch,
        args.resume_training,
        args.model_config,
        dataset_config,
        args,
    )

    logging.info("=" * 90)
    logging.info("Training Script Finished Successfully")
    logging.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 90)


if __name__ == "__main__":
    main()