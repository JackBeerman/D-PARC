"""
Dataset and Model Configuration System

This file defines configurations for three different physics datasets:
1. HMX (Reactive Energetic Materials)
2. Burgers (Turbulence)
3. Navier-Stokes (Fluid Dynamics)

Each dataset has three model variants for comparison.
"""

from PARCtorch.integrator.rk4 import RK4
from PARCtorch.integrator.heun import Heun

# ==============================================================================
# DATASET CONFIGURATIONS
# ==============================================================================

DATASET_CONFIGS = {
    "hmx": {
        "name": "HMX (Reactive Energetic Materials)",
        "min_max_file": "../data/hmx_min_max.json",
        "future_steps": 2,
        "num_workers": 1,
        "input_channels": 5,  # T, P, mu, u, v
        "channel_names": [
            "Temperature (T)",
            "Pressure (P)", 
            "Microstructure (U)",
            "Velocity U",
            "Velocity V"
        ],
        "channel_cmaps": ["jet", "seismic", "binary", "seismic", "seismic"],
        "integrator_type": "rk4",
        "num_state_vars": 3,  # T, p, mu (velocities are separate)
        "advection_indices": [0, 1, 2, 3, 4],  # All channels
        "diffusion_indices": [0],  # Temperature only
        "velocity_indices": [3, 4],  # u, v
        "poisson_config": [],  # No Poisson
        "ddi_list_size": 5,
        "loss_start_channel": 0,  # Include all channels in loss
    },
    
    "burgers": {
        "name": "Burgers (Turbulence)",
        "min_max_file": "../data/b_min_max.json",
        "future_steps": 3,
        "num_workers": 4,
        "input_channels": 3,  # Reynolds, u, v
        "channel_names": ["Reynolds (R)", "Velocity U", "Velocity V"],
        "channel_cmaps": ["plasma", "inferno", "magma"],
        "integrator_type": "heun",
        "num_state_vars": 1,  # Reynolds only
        "advection_indices": [1, 2],  # u, v
        "diffusion_indices": [1, 2],  # u, v
        "velocity_indices": [1, 2],  # u, v
        "poisson_config": [],  # No Poisson
        "ddi_list_size": 3,
        "loss_start_channel": 1,  # Exclude Reynolds from loss (channel 0)
    },
    
    "ns": {
        "name": "Navier-Stokes (Fluid Dynamics)",
        "min_max_file": "../data/ns_min_max.json",
        "future_steps": 3,
        "num_workers": 1,
        "input_channels": 4,  # Reynolds, Pressure, u, v
        "channel_names": ["Reynolds (R)", "Pressure (P)", "Velocity U", "Velocity V"],
        "channel_cmaps": ["seismic", "seismic", "seismic", "seismic"],
        "integrator_type": "heun",
        "num_state_vars": 2,  # Pressure and Reynolds
        "advection_indices": [2, 3],  # u, v
        "diffusion_indices": [2, 3],  # u, v
        "velocity_indices": [2, 3],  # u, v
        "poisson_config": [(0, 2, 3, 1)],  # (re, u, v, pr)
        "ddi_list_size": 4,
        "loss_start_channel": 1,  # Exclude Reynolds from loss (channel 0)
    }
}

# ==============================================================================
# MODEL CONFIGURATIONS (per dataset)
# ==============================================================================

MODEL_CONFIGS = {
    "hmx": {
        "small_no_deform": {
            "name": "Small HMX UNet (No Deform)",
            "n_fe_features": 128,
            "block_dimensions": [64, 64*2, 64*4, 64*8, 64*16],  # 5 layers
            "padding_mode": "reflect",
            "up_block_use_concat": [False, True, False, True],
            "skip_connection_indices": [2, 0],
            "use_deform": False,
        },
        "small_deform": {
            "name": "Small HMX UNet (With Deform)",
            "n_fe_features": 128,
            "block_dimensions": [64, 64*2, 64*4, 64*8, 64*16],  # 5 layers
            "padding_mode": "reflect",
            "up_block_use_concat": [False, True, False, True],
            "skip_connection_indices": [2, 0],
            "use_deform": True,
        },
        "large_no_deform": {
            "name": "Large HMX UNet (No Deform)",
            "n_fe_features": 128,
            "block_dimensions": [64, 64*2, 64*4, 64*8, 64*16, 64*32, 64*64],  # 7 layers
            "padding_mode": "reflect",
            "up_block_use_concat": [False, True, False, True, False, False],
            "skip_connection_indices": [4, 2],
            "use_deform": False,
        }
    },
    
    "burgers": {
        "small_no_deform": {
            "name": "Small Burgers UNet (No Deform)",
            "n_fe_features": 64,
            "block_dimensions": [64, 128, 256],  # 3 layers
            "padding_mode": "reflect",
            "up_block_use_concat": [False, True],
            "skip_connection_indices": [0],
            "use_deform": False,
        },
        "small_deform": {
            "name": "Small Burgers UNet (With Deform)",
            "n_fe_features": 64,
            "block_dimensions": [64, 128, 256],  # 3 layers
            "padding_mode": "reflect",
            "up_block_use_concat": [False, True],
            "skip_connection_indices": [0],
            "use_deform": True,
        },
        "large_no_deform": {
            "name": "Large Burgers UNet (No Deform)",
            "n_fe_features": 64,
            "block_dimensions": [64, 128, 256, 512, 1024],  # 5 layers
            "padding_mode": "reflect",
            "up_block_use_concat": [False, True, False, True],
            "skip_connection_indices": [2, 0],
            "use_deform": False,
        }
    },
    
    "ns": {
        "small_no_deform": {
            "name": "Small NS UNet (No Deform)",
            "n_fe_features": 128,
            "block_dimensions": [64, 128, 256, 512, 1024],  # 5 layers
            "padding_mode": "reflect",
            "up_block_use_concat": [False, True, False, True],
            "skip_connection_indices": [2, 0],
            "use_deform": False,
        },
        "small_deform": {
            "name": "Small NS UNet (With Deform)",
            "n_fe_features": 128,
            "block_dimensions": [64, 128, 256, 512, 1024],  # 5 layers
            "padding_mode": "reflect",
            "up_block_use_concat": [False, True, False, True],
            "skip_connection_indices": [2, 0],
            "use_deform": True,
        },
        "large_no_deform": {
            "name": "Large NS UNet (No Deform)",
            "n_fe_features": 128,
            "block_dimensions": [64, 128, 256, 512, 1024, 2048, 4096],  # 7 layers
            "padding_mode": "reflect",
            "up_block_use_concat": [False, True, False, True, False, False],
            "skip_connection_indices": [4, 2],
            "use_deform": False,
        }
    }
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_dataset_config(dataset_name):
    """Get dataset configuration by name."""
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    return DATASET_CONFIGS[dataset_name]

def get_model_config(dataset_name, model_name):
    """Get model configuration for specific dataset and model variant."""
    if dataset_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    dataset_models = MODEL_CONFIGS[dataset_name]
    if model_name not in dataset_models:
        available = ", ".join(dataset_models.keys())
        raise ValueError(
            f"Unknown model for {dataset_name}: '{model_name}'. "
            f"Available models: {available}"
        )
    
    return dataset_models[model_name]

def get_integrator(integrator_type, device):
    """Get appropriate integrator for dataset."""
    if integrator_type == "rk4":
        return RK4().to(device)
    elif integrator_type == "heun":
        return Heun().to(device)
    else:
        raise ValueError(f"Unknown integrator type: {integrator_type}")

def print_dataset_config(dataset_name):
    """Print dataset configuration details."""
    config = get_dataset_config(dataset_name)
    print(f"\nDataset Configuration: {config['name']}")
    print("=" * 70)
    print(f"Input Channels: {config['input_channels']}")
    print(f"Channel Names: {config['channel_names']}")
    print(f"Future Steps: {config['future_steps']}")
    print(f"Integrator: {config['integrator_type']}")
    print(f"State Variables: {config['num_state_vars']}")
    print(f"Advection Indices: {config['advection_indices']}")
    print(f"Diffusion Indices: {config['diffusion_indices']}")
    print("=" * 70 + "\n")

def print_model_config(dataset_name, model_name):
    """Print model configuration details."""
    config = get_model_config(dataset_name, model_name)
    print(f"\nModel Configuration: {config['name']}")
    print("=" * 70)
    print(f"Feature Extraction Features: {config['n_fe_features']}")
    print(f"Block Dimensions: {config['block_dimensions']}")
    print(f"Number of Layers: {len(config['block_dimensions'])}")
    print(f"Deformable Convolutions: {config['use_deform']}")
    print(f"Skip Connections: {config['skip_connection_indices']}")
    print("=" * 70 + "\n")

def list_available_datasets():
    """List all available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 70)
    for name, config in DATASET_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Name: {config['name']}")
        print(f"  Channels: {config['input_channels']}")
        print(f"  Integrator: {config['integrator_type']}")
    print("=" * 70 + "\n")

def list_available_models(dataset_name):
    """List available model variants for a dataset."""
    if dataset_name not in MODEL_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    print(f"\nAvailable Models for {dataset_name}:")
    print("=" * 70)
    for model_name, config in MODEL_CONFIGS[dataset_name].items():
        print(f"\n{model_name}:")
        print(f"  Name: {config['name']}")
        print(f"  Layers: {len(config['block_dimensions'])}")
        print(f"  Deformable: {config['use_deform']}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    # Print all available configurations
    list_available_datasets()
    for dataset in DATASET_CONFIGS.keys():
        list_available_models(dataset)