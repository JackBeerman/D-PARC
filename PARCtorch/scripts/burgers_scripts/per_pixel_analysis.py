#!/usr/bin/env python3
"""
Final, vectorized analysis script to compute per-pixel metrics for ALL SAMPLES
in a Burgers dataset, focused on kernel and physical metrics.

This script iterates through every sample, computes a comprehensive set of metrics
using a centralized metrics module, and saves the results from all samples and all
timesteps into a single, unified CSV file for a complete correlational analysis.

This version has been optimized to handle large datasets by writing results to
disk incrementally, preventing memory overflow issues.

Date     : 2025-08-25
"""

# ------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------- #
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------- #
# Project Imports
# ------------------------------------------------------------------- #
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from PARCtorch.differentiator.differentiator import Differentiator
from PARCtorch.differentiator.finitedifference import FiniteDifference
from PARCtorch.integrator.heun import Heun
from PARCtorch.integrator.integrator import Integrator
from PARCtorch.PARCv2 import PARCv2
from data.dataset import GenericPhysicsDataset, custom_collate_fn
from utilities.unet_deform import UNet
# MODIFIED: Imported all the new methods from your finalized metrics module
from utilities.metrics import (
    denormalize,
    set_channel_mapping,
    calculate_kernel_metrics,
    calculate_strain_rate,
    calculate_gradient_magnitude,
    calculate_bilinear_metrics,
    get_deformable_offsets,
)

# ------------------------------------------------------------------- #
# Channel Mapping
# ------------------------------------------------------------------- #
ORIGINAL_INDICES = {"Reynolds R": 0, "Velocity U": 1, "Velocity V": 2}
CHANNEL_NAMES = list(ORIGINAL_INDICES.keys())

# ------------------------------------------------------------------- #
# Model Builder
# ------------------------------------------------------------------- #
def build_model(device: str) -> PARCv2:
    n_fe = 64
    unet = UNet(
        [64, 128, 256], input_channels=3, output_channels=n_fe,
        up_block_use_concat=[False, True], skip_connection_indices=[0], use_deform=True
    ).to(device)
    right_diff = FiniteDifference(padding_mode="replicate").to(device)
    heun = Heun().to(device)
    diff = Differentiator(1, n_fe, [1, 2], [1, 2], unet, "constant", right_diff).to(device)
    diff.unet = unet
    integ = Integrator(True, [], heun, [None, None, None], "constant", right_diff).to(device)
    loss = torch.nn.L1Loss().to(device)
    model = PARCv2(differentiator=diff, integrator=integ, loss=loss).to(device)
    return model

# =================================================================== #
# Main Entry
# =================================================================== #
def main():
    parser = argparse.ArgumentParser(description="Burgers' per-pixel analysis script.")
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--min_max_path", required=True)
    parser.add_argument("--output_dir", default="burgers_per_pixel_analysis")
    parser.add_argument("--layer_name", choices=["initial_doubleConv", "upBlock_1"], required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--future_steps", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    set_channel_mapping(ORIGINAL_INDICES)
    
    model = build_model(device)
    ckpt = torch.load(args.model_checkpoint, map_location=device, weights_only=True)
    key = 'model_state_dict' if 'model_state_dict' in ckpt else 'state_dict'
    model.load_state_dict(ckpt.get(key, ckpt), strict=False)
    model.eval()

    eval_set = GenericPhysicsDataset([args.eval_data], future_steps=args.future_steps, min_max_path=args.min_max_path)
    data_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    layers = {
        "initial_doubleConv": model.differentiator.unet.doubleConv.offset_conv,
        "upBlock_1": model.differentiator.unet.upBlocks[1].doubleConv.offset_conv
    }
    target_layer = layers[args.layer_name]
    
    with open(args.min_max_path) as f:
        norm_params = json.load(f)
        
    output_path = Path(args.output_dir) / f"burgers_{args.layer_name}_per_pixel_analysis.csv"
    is_first_write = True
    logging.info(f"Output will be saved incrementally to {output_path}")

    dx = 6.0 / 64.0
    dy = 6.0 / 64.0

    for batch_idx, (ic, t0, t1, gt) in enumerate(tqdm(data_loader, desc=f"Analyzing Layer {args.layer_name}")):
        ic, t0, t1 = ic.to(device), t0.to(device), t1.to(device)

        offsets_list: list[torch.Tensor] = []
        h = target_layer.register_forward_hook(lambda _, __, o: offsets_list.append(o.detach().cpu()))
        with torch.no_grad():
            preds = model(ic, t0, t1)
        h.remove()
        
        preds_cpu = preds.cpu()
        
        batch_df_list = []

        for ts in range(preds_cpu.shape[0]):
            offset_field_batch = offsets_list[ts * 2 + 1]
            
            for b in range(offset_field_batch.shape[0]):
                actual_sample_id = batch_idx * args.batch_size + b
                
                offset_field = offset_field_batch[b].view(-1, 2, *offset_field_batch.shape[2:]).numpy()
                H_feat, W_feat = offset_field.shape[2], offset_field.shape[3]
                
                # --- METRIC CALCULATIONS ---
                kernel_metrics = calculate_kernel_metrics(offset_field)
                
                pred_fields_denorm_torch = [denormalize(CHANNEL_NAMES[i], preds_cpu[ts, b, i], norm_params) for i in range(len(CHANNEL_NAMES))]
                pred_fields_denorm = np.array([p.numpy() for p in pred_fields_denorm_torch])
                H_full, W_full = pred_fields_denorm.shape[1], pred_fields_denorm.shape[2]
                
                u_pred = pred_fields_denorm[ORIGINAL_INDICES["Velocity U"]]
                v_pred = pred_fields_denorm[ORIGINAL_INDICES["Velocity V"]]
                
                strain_rate_magnitude = calculate_strain_rate(u_pred, v_pred, dx=dx, dy=dy)
                shock_gradient_magnitude = calculate_gradient_magnitude(u_pred, v_pred, dx=dx, dy=dy)
                
                physics_metrics = {
                    "strain_rate_magnitude": strain_rate_magnitude,
                    "shock_gradient_magnitude": shock_gradient_magnitude
                }
                
                # MODIFIED: Calculate bilinear and offset metrics
                bilinear_metrics = calculate_bilinear_metrics(
                    offset_field, H_full, W_full, H_full / H_feat, W_full / W_feat
                )
                offset_data = get_deformable_offsets(offset_field)
                
                # --- DATA ASSEMBLY ---
                yy, xx = np.mgrid[0:H_feat, 0:W_feat]
                yy_full, xx_full = (yy * H_full / H_feat).astype(int), (xx * W_full / W_feat).astype(int)
                
                data_for_df = {k: v.flatten() for k, v in kernel_metrics.items()}
                data_for_df.update({'sample_id': actual_sample_id, 'timestep': ts, 'pixel_y': yy.flatten(), 'pixel_x': xx.flatten()})
                
                for i, name in enumerate(CHANNEL_NAMES):
                    data_for_df[name] = pred_fields_denorm[i, yy_full, xx_full].flatten()
                
                for name, field in physics_metrics.items():
                    data_for_df[name] = field[yy_full, xx_full].flatten()
                    
                # MODIFIED: Add bilinear and offset metrics to the dataframe
                data_for_df['usage_map'] = bilinear_metrics['usage_map'][yy_full, xx_full].flatten()
                data_for_df['weight_map'] = bilinear_metrics['weight_map'][yy_full, xx_full].flatten()
                
                data_for_df.update(offset_data)
                
                batch_df_list.append(pd.DataFrame(data_for_df))
        
        if batch_df_list:
            batch_df = pd.concat(batch_df_list, ignore_index=True)
            if is_first_write:
                batch_df.to_csv(output_path, mode='w', header=True, index=False, float_format='%.6g')
                is_first_write = False
            else:
                batch_df.to_csv(output_path, mode='a', header=False, index=False, float_format='%.6g')

    if is_first_write:
        logging.warning("No data was generated to save.")
    else:
        logging.info(f"Successfully saved all data to {output_path}")

if __name__ == "__main__":
    main()