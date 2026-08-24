#!/usr/bin/env python3
"""
Final, vectorized analysis script to compute per-pixel metrics for ALL SAMPLES
in an HMX dataset and correlate them with physical data.

This script iterates through every sample, computes a comprehensive set of metrics
using a centralized metrics module, and saves the results from all samples and all
timesteps into a single, unified CSV file for a complete correlational analysis.

Date     : 2025-08-25
"""

# ------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------- #
import os, sys, json, argparse, logging
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

from data.dataset import GenericPhysicsDataset, custom_collate_fn
from utilities.unet_deform import UNet
from PARCtorch.PARCv2 import PARCv2
from PARCtorch.differentiator.differentiator import Differentiator
from PARCtorch.differentiator.finitedifference import FiniteDifference
from PARCtorch.integrator.integrator import Integrator
from PARCtorch.integrator.rk4 import RK4
# MODIFIED: Import the get_deformable_offsets function
from utilities.metrics import (
    denormalize,
    set_channel_mapping,
    calculate_kernel_metrics,
    calculate_strain_rate,
    calculate_bilinear_metrics,
    get_deformable_offsets,
)

# ------------------------------------------------------------------- #
# Channel Mapping
# ------------------------------------------------------------------- #
ORIGINAL_INDICES = {
    "Temperature (T)": 0, "Pressure (P)": 1, "Microstructure (M)": 2,
    "Velocity U (vx)": 3, "Velocity V (vy)": 4
}
CHANNEL_NAMES = list(ORIGINAL_INDICES.keys())

# ------------------------------------------------------------------- #
# Model Builder
# ------------------------------------------------------------------- #
def build_model(device: str) -> PARCv2:
    n_fe_features = 128
    unet_em = UNet(
        block_dimensions=[64, 128, 256, 512, 1024], input_channels=5,
        output_channels=n_fe_features, up_block_use_concat=[False, True, False, True],
        skip_connection_indices=[2, 0], use_deform=True,
    ).to(device)
    fd = FiniteDifference(padding_mode="replicate").to(device)
    rk4 = RK4().to(device)
    diff_em = Differentiator(3, n_fe_features, list(range(5)), [0], unet_em, "constant", fd).to(device)
    diff_em.unet_em = unet_em
    integrator = Integrator(True, [], rk4, [None]*5, "constant", fd).to(device)
    model = PARCv2(differentiator=diff_em, integrator=integrator, loss=torch.nn.L1Loss()).to(device)
    model.unet_em = unet_em
    return model

# ------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Vectorized script to compute metrics for all HMX samples.")
    parser.add_argument("--timesteps", type=int, nargs='+', required=True, help="A list of timesteps to analyze.")
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--min_max_path", required=True)
    parser.add_argument("--output_dir", default="hmx_per_pixel_analysis")
    parser.add_argument("--layer_name", choices=["initial_doubleConv", "upBlock_3"], required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_channel_mapping(ORIGINAL_INDICES)
    
    model = build_model(device)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device, weights_only=True), strict=False)
    model.eval()

    eval_set = GenericPhysicsDataset([args.eval_data], future_steps=14, min_max_path=args.min_max_path)
    
    layer_mapping = {
        "initial_doubleConv": model.differentiator.unet_em.doubleConv.offset_conv,
        "upBlock_3": model.differentiator.unet_em.upBlocks[3].doubleConv.offset_conv,
    }
    target_layer = layer_mapping[args.layer_name]

    with open(args.min_max_path) as f: norm_params = json.load(f)
    
    df_list = []
    output_path = os.path.join(args.output_dir, f"ALL_SAMPLES_{args.layer_name}_unified_data.csv")

    dx = 1.5 / 128.0
    dy = 3.0 / 256.0

    # ===== Main loop to iterate through ALL samples =====
    for sample_idx, sample in enumerate(tqdm(eval_set, desc="Processing Samples")):
        ic_norm, t0, t1, gt_norm = custom_collate_fn([sample])
        ic_norm, t0, t1, gt_norm = (x.to(device) for x in (ic_norm, t0, t1, gt_norm))

        offsets = []
        h = target_layer.register_forward_hook(lambda _,__,o: offsets.append(o.detach().cpu()))
        with torch.no_grad():
            preds_norm = model(ic_norm, t0, t1)
        h.remove()
        
        preds_norm_cpu = preds_norm.cpu()
        ic_norm_cpu = ic_norm.cpu()
        H_full, W_full, num_channels = preds_norm_cpu.shape[3], preds_norm_cpu.shape[4], len(CHANNEL_NAMES)
        
        ic_fields = np.array([denormalize(CHANNEL_NAMES[i], ic_norm_cpu[0, i], norm_params).numpy() for i in range(num_channels)])

        # --- Inner loop for timesteps ---
        for ts in args.timesteps:
            offset_idx_in_list = ts * 4 + 3
            if offset_idx_in_list >= len(offsets):
                logging.warning(f"Offset data for sample {sample_idx}, ts {ts} not found. Skipping."); continue

            offset_field = offsets[offset_idx_in_list][0].view(-1, 2, *offsets[offset_idx_in_list].shape[2:]).numpy()
            H_feat, W_feat = offset_field.shape[2], offset_field.shape[3]
            
            kernel_metric_fields = calculate_kernel_metrics(offset_field)
            
            physical_fields = np.array([denormalize(CHANNEL_NAMES[i], preds_norm_cpu[ts, 0, i], norm_params).numpy() for i in range(num_channels)])

            u_vel = physical_fields[ORIGINAL_INDICES["Velocity U (vx)"]]
            v_vel = physical_fields[ORIGINAL_INDICES["Velocity V (vy)"]]
            
            strain_rate_mag = calculate_strain_rate(u_vel, v_vel, dx=dx, dy=dy)
            temp_grad_y, temp_grad_x = np.gradient(physical_fields[ORIGINAL_INDICES["Temperature (T)"]], dy, dx)
            temp_grad_mag = np.sqrt(temp_grad_y**2 + temp_grad_x**2)
            pressure_grad_y, pressure_grad_x = np.gradient(physical_fields[ORIGINAL_INDICES["Pressure (P)"]], dy, dx)
            pressure_grad_mag = np.sqrt(pressure_grad_y**2 + pressure_grad_x**2)

            bilinear_metrics = calculate_bilinear_metrics(
                offset_field=offset_field,
                orig_H=H_full,
                orig_W=W_full,
                scale_y=H_full / H_feat,
                scale_x=W_full / W_feat
            )
            
            yy, xx = np.mgrid[0:H_feat, 0:W_feat]
            yy_full, xx_full = (yy * H_full / H_feat).astype(int), (xx * W_full / W_feat).astype(int)
            
            data_for_df = {k: v.flatten() for k, v in kernel_metric_fields.items()}
            data_for_df.update({'sample_id': sample_idx, 'timestep': ts, 'pixel_y': yy.flatten(), 'pixel_x': xx.flatten()})
            
            data_for_df['usage_map'] = bilinear_metrics['usage_map'][yy_full, xx_full].flatten()
            data_for_df['weight_map'] = bilinear_metrics['weight_map'][yy_full, xx_full].flatten()

            for i, name in enumerate(CHANNEL_NAMES):
                name_short = name.split(' ')[0]
                data_for_df[name] = physical_fields[i, yy_full, xx_full].flatten()
                data_for_df[f"{name_short}_initial"] = ic_fields[i, yy_full, xx_full].flatten()

            data_for_df["strain_rate_magnitude"] = strain_rate_mag[yy_full, xx_full].flatten()
            data_for_df["temp_gradient_magnitude"] = temp_grad_mag[yy_full, xx_full].flatten()
            data_for_df["pressure_gradient_magnitude"] = pressure_grad_mag[yy_full, xx_full].flatten()
            
            # MODIFIED: Use the method from metrics.py to get offset data
            offset_data = get_deformable_offsets(offset_field)
            data_for_df.update(offset_data)
            
            df_list.append(pd.DataFrame(data_for_df))

    if df_list:
        logging.info("Concatenating data from all samples and timesteps...")
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(output_path, index=False, float_format='%.6g')
        logging.info(f"Successfully saved final dataset for all samples to {output_path}")
    else:
        logging.warning("No data was generated to save.")

if __name__ == "__main__":
    main()