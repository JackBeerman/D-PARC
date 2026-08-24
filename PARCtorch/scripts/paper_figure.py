#!/usr/bin/env python3
"""
CNN PARCv2 Paper Figures — GT vs Prediction
=============================================
Publication-quality figures for CNN-based PARCv2 shock tube.

One figure per variable (density, x_momentum, total_energy).
Each figure shows 2 simulations stacked, GT vs prediction:

    Layout per figure:
        Columns: t_early | t_mid | t_late  (3 timesteps)
        Rows:    Sim1 GT | Sim1 PARCv2 | Sim2 GT | Sim2 PARCv2

    Per-row colorbars with shared GT-derived color range.

Usage:
    python paper_figure_cnn.py \\
        --test_dir "/standard/.../cnn_datasets/test" \\
        --model_path "/scratch/.../weights/large/shocktube_model_epoch_750.pth" \\
        --min_max_path "/path/to/shocktube_min_max.json" \\
        --output_dir /scratch/.../figures \\
        --sim_indices 0 5 \\
        --rollout_steps 40 --dpi 300
"""

import argparse
import sys
import os
import json
from pathlib import Path

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ============================================================================
# PARCtorch imports
# ============================================================================
# Ensure local modules are findable
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
try:
    from utilities.unet_deform import UNet
    from PARCtorch.PARCv2 import PARCv2
    from PARCtorch.differentiator.differentiator import Differentiator
    from PARCtorch.differentiator.finitedifference import FiniteDifference
    from PARCtorch.integrator.integrator import Integrator
    from PARCtorch.integrator.heun import Heun
    from data.includedt import GenericPhysicsDataset, custom_collate_fn
    print("✓ PARCtorch modules imported")
except ImportError as e:
    print(f"❌ PARCtorch import failed: {e}")
    print("   Run this script in the dparc conda environment.")
    sys.exit(1)


# ============================================================================
# CONSTANTS
# ============================================================================

VAR_NAMES = ['density', 'x_momentum', 'total_energy']

VAR_LABELS = {
    'density':      r'$\rho$',
    'x_momentum':   r'$\rho u$',
    'total_energy':  r'$E$',
}

# CNN channel indices for conserved variables
CNN_CONSERVED = [0, 1, 2]   # ρ, ρu, E in the 8-channel tensor
GT_CONSERVED = [0, 1, 2]    # ρ, ρu, E in the 5-channel GT

# Physical denormalization constants (from variable_statistics.json)
# These are the TRUE physical ranges — the CNN min_max.json only stores
# [0,1] ranges for normalized channels, not physical units.
DENORM = {
    'density':      {'min': 0.062139, 'max': 2.007674, 'unit': r'kg/m$^3$'},
    'x_momentum':   {'min': -2.600878, 'max': 235.423531, 'unit': r'kg/(m$^2 \cdot$s)'},
    'total_energy': {'min': 12399.199670, 'max': 424145.808785, 'unit': r'J/m$^3$'},
}


def denormalize_physical(arr, var_name):
    """Convert normalized [0,1] array to physical units using DENORM constants."""
    if var_name in DENORM:
        d = DENORM[var_name]
        return arr * (d['max'] - d['min']) + d['min']
    return arr


# ============================================================================
# MODEL
# ============================================================================

def build_cnn_parc_model(device):
    """Reconstruct the 8-channel CNN PARCv2 architecture (matches shocktube_new.py)."""
    n_fe_features = 64

    unet = UNet(
        block_dimensions=[64, 128, 256, 512, 1024],
        input_channels=8,
        output_channels=n_fe_features,
        padding_mode="reflect",
        up_block_use_concat=[False, True, False, True],
        skip_connection_indices=[2, 0],
        use_deform=False,
    )

    right_diff = FiniteDifference(padding_mode="replicate").to(device)
    heun_int = Heun().to(device)

    diff = Differentiator(
        n_state_var=6,
        n_fe_features=n_fe_features,
        list_adv_idx=[6, 7],
        list_dif_idx=[6, 7],
        feature_extraction=unet,
        padding_mode="constant",
        finite_difference_method=right_diff,
    ).to(device)

    ddi_list = [None] * 8
    integrator = Integrator(True, [], heun_int, ddi_list, "constant", right_diff).to(device)
    criterion = torch.nn.L1Loss().to(device)

    model = PARCv2(differentiator=diff, integrator=integrator, loss=criterion).to(device)
    return model


def load_cnn_parc(model_path, device):
    """Load trained CNN PARCv2 weights."""
    model = build_cnn_parc_model(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ CNN PARCv2 loaded: {n_params:,} parameters")
    return model


# ============================================================================
# DENORMALIZATION
# ============================================================================

def load_min_max(min_max_path):
    with open(min_max_path, 'r') as f:
        return json.load(f)


def denorm_channel(arr, ch_idx, norm):
    """Denormalize a channel from [0,1] to physical units."""
    c_min = norm['channel_min'][ch_idx]
    c_max = norm['channel_max'][ch_idx]
    return arr * (c_max - c_min) + c_min


def denorm_field(field_2d, var_idx, norm):
    """Denormalize a 2D field [H, W] for a conserved variable to physical units.

    var_idx: 0=density, 1=x_momentum, 2=total_energy
    These map to CNN channels 0, 1, 2 respectively.
    """
    ch_idx = CNN_CONSERVED[var_idx]
    return denorm_field_raw(field_2d, ch_idx, norm)


def denorm_field_raw(field_2d, ch_idx, norm):
    c_min = norm['channel_min'][ch_idx]
    c_max = norm['channel_max'][ch_idx]
    return field_2d * (c_max - c_min) + c_min


def get_unit_str(var_idx, norm):
    """Get a unit string for colorbar labels."""
    # Channel min/max give us the physical range
    ch_idx = CNN_CONSERVED[var_idx]
    c_min = norm['channel_min'][ch_idx]
    c_max = norm['channel_max'][ch_idx]
    # Infer units from variable name
    units = {
        'density':      r'kg/m$^3$',
        'x_momentum':   r'kg/(m$^2 \cdot$s)',
        'total_energy': r'J/m$^3$',
    }
    return units.get(VAR_NAMES[var_idx], '')


# ============================================================================
# ROLLOUT
# ============================================================================

@torch.no_grad()
def cnn_rollout(model, ic, t_len, device, total_timesteps=None):
    """Run PARCv2 rollout. Returns [Steps, 8, 64, 64].
    
    t_len: number of future steps to predict
    total_timesteps: total timesteps in the original npy file (used to compute
                     t1 values matching training). If None, uses t_len.
    """
    model.eval()
    t0 = torch.tensor(0.0).to(device)
    
    # Match training dataset: t1[i] = (i+1) / (total_timesteps + 1)
    # The training dataset used whole_t = timesteps + 1 where timesteps = npy.shape[0]
    if total_timesteps is not None:
        whole_t = total_timesteps + 1
    else:
        whole_t = t_len + 1
    
    t1 = torch.tensor([(i + 1) / whole_t for i in range(t_len)],
                       dtype=torch.float32).to(device)

    # [Steps, Batch=1, 8, H, W] → [Steps, 8, H, W]
    preds = model(ic, t0, t1).squeeze(1)
    return preds


# ============================================================================
# SIM PARAMS FROM IC
# ============================================================================

def extract_sim_label_from_filename(filename):
    """Parse pL and rhoL from filename like p_L_143750_rho_L_0.5625.npy"""
    import re
    pressure = None
    density = None

    m_p = re.search(r'p_L_(\d+\.?\d*)', filename)
    if m_p:
        pressure = float(m_p.group(1))

    m_rho = re.search(r'rho_L_(\d+\.?\d*)', filename)
    if m_rho:
        density = float(m_rho.group(1))

    parts = []
    if pressure is not None:
        parts.append(f'$p_L$ = {pressure:g} Pa')
    if density is not None:
        parts.append(f'$\\rho_L$ = {density:g} kg/m$^3$')
    return ',  '.join(parts) if parts else filename


def extract_sim_params_from_filename(filename):
    """Extract pressure and density from filename."""
    import re
    params = {}

    m_p = re.search(r'p_L_(\d+\.?\d*)', filename)
    if m_p:
        params['pressure'] = float(m_p.group(1))

    m_rho = re.search(r'rho_L_(\d+\.?\d*)', filename)
    if m_rho:
        params['density'] = float(m_rho.group(1))

    return params


def get_dataset_filenames(test_dir):
    """Get sorted list of .npy filenames (without metadata) from test dir."""
    from pathlib import Path
    npy_files = sorted(Path(test_dir).glob("*.npy"))
    # Filter out metadata files
    data_files = [f for f in npy_files if '_metadata' not in f.stem]
    return data_files


def extract_delta_t_from_metadata(test_dir, npy_filename):
    """Try to read delta_t from the companion metadata JSON."""
    meta_path = Path(test_dir) / f"{Path(npy_filename).stem}_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        # Common key names for dt
        for key in ['delta_t', 'dt', 'time_step', 'timestep']:
            if key in meta:
                return float(meta[key])
    return None


def extract_delta_t(ic, norm):
    """Fallback: extract raw delta_t from IC channel 5 (not denormalized)."""
    return ic[5, 0, 0].item()


# ============================================================================
# TIME FORMATTING
# ============================================================================

def format_physical_time(step, delta_t):
    if delta_t is None:
        return f't = {step}'
    phys_time = step * delta_t
    if phys_time < 1e-4:
        return f't = {phys_time:.2e} s'
    elif phys_time < 1.0:
        return f't = {phys_time:.4f} s'
    else:
        return f't = {phys_time:.2f} s'



def make_cnn_field_figure(sim_data_list, timesteps, var_idx, var_name,
                          output_path, norm, dpi=300, cmap='RdBu_r'):
    n_sims = len(sim_data_list)
    n_times = len(timesteps)
    n_rows = n_sims * 2

    gs_dim = sim_data_list[0]['gt'].shape[-1]

    var_label = VAR_LABELS.get(var_name, var_name)
    unit_str = DENORM.get(var_name, {}).get('unit', '')

    # Layout sizing
    cell_w, cell_h = 2.0, 1.9
    row_label_w = 1.5
    cbar_w = 0.45
    header_h = 0.9
    sim_gap_h = 0.5

    total_gaps = (n_sims - 1) * sim_gap_h
    fig_w = row_label_w + n_times * cell_w + cbar_w + 0.3
    fig_h = header_h + n_rows * cell_h + total_gaps + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    x0 = row_label_w / fig_w
    y_top = 1.0 - header_h / fig_h
    cw = cell_w / fig_w
    ch = cell_h / fig_h
    gap_norm = sim_gap_h / fig_h
    cbar_norm_w = 0.015
    cbar_gap = 0.008

    row_info = []
    for si, sd in enumerate(sim_data_list):
        sim_num = si + 1
        row_info.append((f'Sim {sim_num}: GT', sd['gt'], False, si))
        row_info.append((f'Sim {sim_num}: PARCv2', sd['pred'], True, si))

    for row_idx, (row_label, data_arr, is_pred, si) in enumerate(row_info):
        sim_pair_idx = row_idx // 2
        y_offset = sim_pair_idx * gap_norm

        # Compute per-row color range
        row_fields = []
        for t in timesteps:
            if t < data_arr.shape[0]:
                row_fields.append(denormalize_physical(data_arr[t, var_idx, :, :], var_name))
        row_vmin = min(f.min() for f in row_fields)
        row_vmax = max(f.max() for f in row_fields)
        row_pad = max((row_vmax - row_vmin) * 0.02, 1e-8)
        row_vmin -= row_pad
        row_vmax += row_pad
        row_norm = mcolors.Normalize(vmin=row_vmin, vmax=row_vmax)

        for ti, t in enumerate(timesteps):
            x = x0 + ti * cw
            y = y_top - (row_idx + 1) * ch - y_offset

            ax = fig.add_axes([x, y, cw * 0.93, ch * 0.90])

            if t < data_arr.shape[0]:
                field = denormalize_physical(data_arr[t, var_idx, :, :], var_name)
            else:
                field = np.full((gs_dim, gs_dim), np.nan)

            ax.imshow(field, cmap=cmap, aspect='equal',
                      vmin=row_vmin, vmax=row_vmax,
                      origin='lower', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if ti == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight='bold',
                              rotation=90, labelpad=8)

            if not is_pred:
                delta_t = sim_data_list[si].get('delta_t', None)
                if delta_t is not None:
                    time_str = f't = {t * delta_t:.3e} s'
                else:
                    time_str = f'Step {t}'
                ax.set_title(time_str, fontsize=9, fontweight='bold', pad=4)

        # Per-row colorbar using per-row norm
        cb_x = x0 + n_times * cw + cbar_gap
        cb_y = y_top - (row_idx + 1) * ch - y_offset + ch * 0.05
        cb_h_row = ch * 0.80

        cbar_ax = fig.add_axes([cb_x, cb_y, cbar_norm_w, cb_h_row])
        sm = cm.ScalarMappable(norm=row_norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(f'{var_label} ({unit_str})', fontsize=7, labelpad=3)

    for si, sd in enumerate(sim_data_list):
        gt_row_idx = si * 2
        y_subtitle = y_top - gt_row_idx * ch - si * gap_norm + 0.015
        fig.text(x0 + n_times * cw / 2, y_subtitle,
                 sd['label'], fontsize=10, ha='center', va='bottom',
                 fontweight='bold', color='#333333')

    fig.text(0.5, 0.98, f'{var_label} — PARCv2 (CNN)',
             fontsize=13, fontweight='bold', ha='center', va='top')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# ============================================================================
# ERROR FIGURE
# ============================================================================

def make_cnn_error_figure(sim_data_list, timesteps, var_idx, var_name,
                          output_path, norm, dpi=300):
    """
    Absolute error figure for one variable, 2 simulations.

    Layout:
        Columns: 3 timesteps
        Rows:    Sim1 PARCv2 Error | Sim2 PARCv2 Error
    """
    n_sims = len(sim_data_list)
    n_times = len(timesteps)
    n_rows = n_sims

    gs_dim = sim_data_list[0]['gt'].shape[-1]

    # Error range (99th percentile across all sims)
    emax = 0
    for sd in sim_data_list:
        for t in timesteps:
            if t < sd['gt'].shape[0] and t < sd['pred'].shape[0]:
                gt_field = denormalize_physical(sd['gt'][t, var_idx, :, :], var_name)
                pred_field = denormalize_physical(sd['pred'][t, var_idx, :, :], var_name)
                err = np.abs(pred_field - gt_field)
                emax = max(emax, float(np.percentile(err, 99)))
    emax = max(emax, 1e-8)
    color_norm = mcolors.Normalize(vmin=0, vmax=emax)

    var_label = VAR_LABELS.get(var_name, var_name)
    unit_str = DENORM.get(var_name, {}).get('unit', '')

    cell_w, cell_h = 2.0, 1.9
    row_label_w = 1.5
    header_h = 0.5
    fig_w = row_label_w + n_times * cell_w + 0.7
    fig_h = header_h + n_rows * cell_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    x0 = row_label_w / fig_w
    y_top = 1.0 - header_h / fig_h
    cw = cell_w / fig_w
    ch = cell_h / fig_h
    cbar_norm_w = 0.015
    cbar_gap = 0.008

    for row, sd in enumerate(sim_data_list):
        sim_label = sd.get('label', f'Sim {row+1}')
        label = f'Sim {row + 1}: Error'

        for ti, t in enumerate(timesteps):
            x = x0 + ti * cw
            y = y_top - (row + 1) * ch

            ax = fig.add_axes([x, y, cw * 0.93, ch * 0.90])

            if t < sd['gt'].shape[0] and t < sd['pred'].shape[0]:
                gt_field = denormalize_physical(sd['gt'][t, var_idx, :, :], var_name)
                pred_field = denormalize_physical(sd['pred'][t, var_idx, :, :], var_name)
                err = np.abs(pred_field - gt_field)
            else:
                err = np.full((gs_dim, gs_dim), np.nan)

            ax.imshow(err, cmap='hot_r', aspect='equal',
                      vmin=0, vmax=emax,
                      origin='lower', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if ti == 0:
                ax.set_ylabel(label, fontsize=9, fontweight='bold',
                              rotation=90, labelpad=8)
            if row == 0:
                ax.set_title(f'Step {t}',
                             fontsize=10, fontweight='bold', pad=6)

        # Per-row colorbar
        cb_x = x0 + n_times * cw + cbar_gap
        cb_y = y_top - (row + 1) * ch + ch * 0.05
        cb_h_row = ch * 0.80

        cbar_ax = fig.add_axes([cb_x, cb_y, cbar_norm_w, cb_h_row])
        sm = cm.ScalarMappable(norm=color_norm, cmap='hot_r')
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(f'|Δ{var_label}| ({unit_str})', fontsize=7, labelpad=3)

    fig.text(0.5, 0.97, f'Absolute Error: {var_label} — PARCv2 (CNN)',
             fontsize=11, fontweight='bold', ha='center', va='top')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CNN PARCv2 Paper Figures")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to CNN test data directory")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained CNN PARCv2 checkpoint")
    parser.add_argument("--min_max_path", type=str, required=True,
                        help="Path to shocktube_min_max.json")
    parser.add_argument("--output_dir", type=str, default="./cnn_figures")
    parser.add_argument("--sim_indices", type=int, nargs='+', default=[0, 5],
                        help="Indices of test samples to use (default: 0 5)")
    parser.add_argument("--rollout_steps", type=int, default=40)
    parser.add_argument("--timesteps", type=int, nargs='+', default=None,
                        help="Specific timesteps (default: 3 evenly spaced)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cmap", type=str, default='RdBu_r')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--error_fig", action='store_true',
                        help="Also generate absolute-error figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load normalization
    norm = load_min_max(args.min_max_path)
    print(f"Loaded min_max: {len(norm['channel_min'])} channels")

    # Load model
    print(f"\nLoading CNN PARCv2...")
    model = load_cnn_parc(args.model_path, device)

    # Get sorted test files (npy only, no metadata)
    data_files = get_dataset_filenames(args.test_dir)
    n_files = len(data_files)
    print(f"\nTest directory: {args.test_dir}")
    print(f"  Found {n_files} simulation files")

    # Validate sim indices (these index FILES, not dataset samples)
    sim_indices = [i for i in args.sim_indices if i < n_files]
    if len(sim_indices) < 2:
        print(f"⚠ Need at least 2 valid file indices. Available: 0-{n_files-1}")
        if n_files >= 2:
            sim_indices = [0, n_files // 2]
            print(f"  Using defaults: {sim_indices}")
        else:
            print("  Not enough test files!")
            return

    print(f"  Using file indices: {sim_indices}")

    # Load and rollout each simulation directly from npy files
    sim_data_list = []
    for idx in sim_indices:
        fpath = data_files[idx]
        fname = fpath.stem
        print(f"\n  Processing file {idx}: {fname}")

        # Load raw npy: [T, 5, 64, 64]
        raw = np.load(str(fpath))
        T_total, n_ch, H, W = raw.shape
        print(f"    Shape: {raw.shape}")

        # Normalize channels using min_max (same as GenericPhysicsDataset)
        data_tensor = torch.from_numpy(raw.copy()).float()
        for ch_i in range(min(n_ch, len(norm['channel_min']))):
            c_min = norm['channel_min'][ch_i]
            c_max = norm['channel_max'][ch_i]
            if c_max - c_min > 0:
                data_tensor[:, ch_i, :, :] = (data_tensor[:, ch_i, :, :] - c_min) / (c_max - c_min)
            else:
                data_tensor[:, ch_i, :, :] = data_tensor[:, ch_i, :, :] - c_min

        # Load metadata for global params (pL, rhoL, dt → channels 3,4,5 in 8-ch IC)
        meta_path = Path(args.test_dir) / f"{fname}_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            gp = meta[0]['global_params']  # [pL_norm, rhoL_norm, dt_norm]
        else:
            print(f"    ⚠ No metadata found, using zeros for global params")
            gp = [0.0, 0.0, 0.0]

        # Augment 5-ch → 8-ch: [ρ, ρu, E, pL, rhoL, dt, ux, uy]
        # Original 5 channels: [ρ, ρu, E, ux, uy]
        # Insert global params at channels 3,4,5; move ux,uy to 6,7
        T_all = data_tensor.shape[0]
        ic_8ch = torch.zeros(T_all, 8, H, W)
        ic_8ch[:, 0:3, :, :] = data_tensor[:, 0:3, :, :]  # ρ, ρu, E
        ic_8ch[:, 3, :, :] = gp[0]   # pL (broadcast)
        ic_8ch[:, 4, :, :] = gp[1]   # rhoL (broadcast)
        ic_8ch[:, 5, :, :] = gp[2]   # dt (broadcast)
        ic_8ch[:, 6:8, :, :] = data_tensor[:, 3:5, :, :]  # ux, uy

        # IC = first timestep (8-ch), GT = remaining timesteps (5-ch normalized)
        ic = ic_8ch[0]               # [8, 64, 64]
        gt = data_tensor[1:]         # [T-1, 5, 64, 64]
        t_len = min(args.rollout_steps, gt.shape[0])

        # Extract sim label from filename
        sim_label = extract_sim_label_from_filename(fname)

        # Extract delta_t from the metadata already loaded
        if meta_path.exists():
            delta_t = meta[0].get('delta_t', None)
            print(f"    raw delta_t from metadata: {delta_t}")
        else:
            delta_t = None
        if delta_t is not None:
            print(f"    {sim_label},  Δt = {delta_t:.2e} s")
        else:
            print(f"    {sim_label},  Δt = unknown")

        # Run rollout — pass total_timesteps so t1 matches training
        ic_batch = ic.unsqueeze(0).to(device)  # [1, 8, 64, 64]
        preds = cnn_rollout(model, ic_batch, t_len, device, total_timesteps=T_total)
        # preds: [Steps, 8, 64, 64]

        # Quick RRMSE on density
        pred_dens = preds[:, 0, :, :].cpu().numpy().flatten()
        gt_dens = gt[:t_len, 0, :, :].numpy().flatten()
        rrmse = (np.sqrt(np.mean((pred_dens - gt_dens)**2))
                 / max(np.sqrt(np.mean(gt_dens**2)), 1e-12))
        print(f"    Density RRMSE (norm): {rrmse:.4f}")

        sim_data_list.append({
            'gt':      gt[:t_len].numpy(),                            # [Steps, 5, 64, 64]
            'pred':    preds[:t_len, CNN_CONSERVED].cpu().numpy(),    # [Steps, 3, 64, 64]
            'label':   sim_label,
            'delta_t': delta_t,
            'index':   idx,
            'fname':   fname,
        })

    # Determine timesteps
    steps = min(sd['gt'].shape[0] for sd in sim_data_list)
    if args.timesteps:
        timesteps = [t for t in args.timesteps if t < steps]
    else:
        timesteps = [steps // 6, steps // 2, steps - 1]

    print(f"\n  Timestep indices: {timesteps} (of {steps} total)")
    for si, sd in enumerate(sim_data_list):
        dt = sd['delta_t']
        if dt is not None:
            phys = [f"{t * dt:.2e} s" for t in timesteps]
            print(f"    Sim {si+1} physical times: {phys}")
        else:
            print(f"    Sim {si+1}: delta_t unknown")

    # Generate figures — one per variable
    print(f"\nGenerating figures...")
    for vi, vn in enumerate(VAR_NAMES):
        for ext in ['png', 'pdf']:
            fpath = output_dir / f'cnn_{vn}.{ext}'
            make_cnn_field_figure(sim_data_list, timesteps, vi, vn,
                                  fpath, norm, dpi=args.dpi, cmap=args.cmap)

        if args.error_fig:
            for ext in ['png', 'pdf']:
                epath = output_dir / f'cnn_{vn}_error.{ext}'
                make_cnn_error_figure(sim_data_list, timesteps, vi, vn,
                                      epath, norm, dpi=args.dpi)

    # Print summary metrics
    print(f"\n{'=' * 60}")
    print("PARCv2 (CNN) QUICK METRICS (normalized space)")
    print(f"{'=' * 60}")
    for si, sd in enumerate(sim_data_list):
        print(f"\n  Sim {si+1} (index {sd['index']}):")
        for vi, vn in enumerate(VAR_NAMES):
            pred = sd['pred'][:, vi, :, :].flatten()
            gt = sd['gt'][:, vi, :, :].flatten()
            rrmse = (np.sqrt(np.mean((pred - gt)**2))
                     / max(np.sqrt(np.mean(gt**2)), 1e-12))
            print(f"    {vn:15s} RRMSE = {rrmse:.4f}")

    print(f"\n✓ Done! Figures in {output_dir}")


if __name__ == "__main__":
    main()