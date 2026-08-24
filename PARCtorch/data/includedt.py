import os
import json
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from tqdm import tqdm

class GenericPhysicsDataset(Dataset):
    def __init__(self, data_dirs, future_steps=1, min_max_path=None, required_channels=None, validate=True):
        self.data_dirs = data_dirs
        self.future_steps = future_steps
        self.files = []

        # 1. Discover all .npy files
        for data_dir in data_dirs:
            dir_files = sorted([
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npy")
            ])
            self.files.extend(dir_files)

        # 2. Load Normalization (Min/Max for first 5 physical channels)
        with open(min_max_path, "r") as f:
            min_max = json.load(f)
            self.channel_min = min_max["channel_min"] # Should be length 5
            self.channel_max = min_max["channel_max"] # Should be length 5
        
        self.num_channels = len(self.channel_min)

        # 3. Precompute Sample Indices
        self.samples = []
        logging.info("Preparing dataset samples...")
        for file_idx, file in enumerate(tqdm(self.files, desc="Indexing files")):
            data_memmap = np.load(file, mmap_mode="r")
            timesteps = data_memmap.shape[0]
            
            # (timesteps - future_steps - 1) gives us enough room for start_t and target
            max_start_t = timesteps - self.future_steps - 1
            for start_t in range(0, max_start_t + 1):
                self.samples.append((file_idx, start_t))

        # 4. Standard Time Logic (As per your original code)
        # Using a representative file to define the t0 and t1 tensors
        sample_memmap = np.load(self.files[0], mmap_mode="r")
        timesteps = sample_memmap.shape[0]
        whole_t = timesteps + 1
        self.t1 = torch.tensor([(i + 1) / whole_t for i in range(self.future_steps)], dtype=torch.float32)
        self.t0 = torch.tensor(0.0, dtype=torch.float32)

        self._memmap_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start_t = self.samples[idx]
        file_path = self.files[file_idx]
        
        # Load .npy (0:rho, 1:rhou, 2:E, 3:ux, 4:uy)
        if file_path not in self._memmap_cache:
            self._memmap_cache[file_path] = np.load(file_path, mmap_mode="r")
        data_memmap = self._memmap_cache[file_path]
        
        # Pull required timesteps: (future_steps + 1, 5, 64, 64)
        data = data_memmap[start_t : start_t + 1 + self.future_steps, :, :, :].copy()
        data_tensor = torch.from_numpy(data).float()

        # --- SAFE PHYSICAL NORMALIZATION ---
        # Prevents 0/0 division when uy is constant zero
        for c in range(self.num_channels):
            min_v = self.channel_min[c]
            max_v = self.channel_max[c]
            range_v = max_v - min_v
            
            if range_v > 1e-9:
                data_tensor[:, c, :, :] = (data_tensor[:, c, :, :] - min_v) / range_v
            else:
                # Force pure zero for constant channels (like your uy)
                data_tensor[:, c, :, :] = 0.0

        # --- LOAD METADATA ---
        case_base = os.path.splitext(file_path)[0]
        metadata_path = f"{case_base}_metadata.json"
        
        global_params = [0.0, 0.0, 0.0] 
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta_list = json.load(f)
                global_params = meta_list[start_t].get("global_params", global_params)

        T_len, _, H, W = data_tensor.shape
        meta_maps = [torch.full((T_len, 1, H, W), v) for v in global_params]
        
        # --- REASSEMBLE CHANNELS FOR ADRDifferentiator([6, 7]) ---
        rho_rhou_e = data_tensor[:, :3, :, :] # Indices 0, 1, 2
        ux_uy      = data_tensor[:, 3:, :, :] # Indices 3, 4 (uy is now clean zero)
        
        # Concat: [Conserved(3)] + [Metadata(3)] + [Velocity(2)] = 8 Channels
        # Order: rho, rhou, E, pL, rhoL, dt, ux, uy
        full_tensor = torch.cat([rho_rhou_e] + meta_maps + [ux_uy], dim=1)

        # --- OUTPUT SLICING ---
        ic = full_tensor[0] # Step 0, all 8 channels
        
        # Target: Future steps, all 5 physical variables (rho, rhou, E, ux, uy)
        # We skip the metadata indices (3, 4, 5) for the target
        target_conserved = full_tensor[1:, :3, :, :] # rho, rhou, E
        target_velocity  = full_tensor[1:, 6:, :, :] # ux, uy
        target = torch.cat([target_conserved, target_velocity], dim=1) 

        return ic, self.t0, self.t1, target

    def __del__(self):
        for memmap in self._memmap_cache.values():
            del memmap
        self._memmap_cache.clear()

def custom_collate_fn(batch):
    """
    Args:
        batch: List of tuples (ic, t0, t1, target) from Dataset.__getitem__
        
    Returns:
        ic: (batch_size, 8, 64, 64) -> [rho, rhou, E, pL, rhoL, dt, ux, uy]
        t0: 0.0 (scalar tensor)
        t1: (future_steps,) -> The time points to predict
        target: (future_steps, batch_size, 5, 64, 64) -> [rho, rhou, E, ux, uy]
    """
    # Unzip the batch
    ics, t0s, t1s, targets = zip(*batch)

    # 1. Stack initial conditions
    # Result: (batch_size, 8, 64, 64)
    ic = torch.stack(ics, dim=0)

    # 2. Handle t0 (Starting time)
    # Since it's always 0.0, a single scalar tensor is standard
    t0 = torch.tensor(0.0, dtype=torch.float32)

    # 3. Handle t1 (Prediction time points)
    # Since t1 is a fixed property of the dataset, we just take the first one
    # Result: (future_steps,)
    t1 = t1s[0]

    # 4. Stack and Permute targets
    # Initial stack: (batch_size, future_steps, 5, 64, 64)
    # Target shape for PARC: (future_steps, batch_size, 5, 64, 64)
    target = torch.stack(targets, dim=0).permute(1, 0, 2, 3, 4)

    return ic, t0, t1, target