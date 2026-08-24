import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_data_format(
    data_dirs: list[str],
    future_steps: int = 1,
    min_max_path: str = None,
    required_channels: int = None,
    pre_normalized: bool = False, # New parameter
):
    """
    Validates the format of the data directories to ensure they contain properly formatted .npy files
    and corresponding min_max.json files (unless pre_normalized is True).

    Args:
        data_dirs (list of str): List of directories containing preprocessed `.npy` files.
        future_steps (int): Number of timesteps in the future the model will predict.
        min_max_path (str, optional): Path to the JSON file containing min and max values for each channel.
                                        If None, it will look for 'min_max.json' in the first data directory.
        required_channels (int, optional): Number of channels expected in the data.
        pre_normalized (bool): If True, skips the min_max.json check and normalization.

    Raises:
        ValueError: If any of the validation checks fail.
        FileNotFoundError: If a required min_max.json file is not found and pre_normalized is False.
    """
    logging.info("Starting data validation...")

    all_files = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            raise ValueError(
                f"Data directory '{data_dir}' does not exist or is not a directory."
            )
        dir_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npy")
            ]
        )
        if not dir_files:
            logging.warning(f"No .npy files found in directory '{data_dir}'.")
        all_files.extend(dir_files)

    if not all_files:
        raise ValueError(
            "No .npy files found in any of the specified directories."
        )

    channel_min = []
    channel_max = []
    num_channels = None

    if not pre_normalized:
        # Load min and max values if data is not pre-normalized
        if min_max_path is None:
            # Assume min_max.json is in the first data directory and applies to all
            if not data_dirs:
                raise ValueError("No data directories provided to infer min_max.json path.")
            
            default_min_max_path = os.path.join(data_dirs[0], "min_max.json")
            if not os.path.exists(default_min_max_path):
                raise FileNotFoundError(
                    f"Min and max values file not found at '{default_min_max_path}'. "
                    "Please ensure the file exists or provide 'min_max_path'."
                )
            logging.info(f"Using min_max.json from '{default_min_max_path}' for normalization.")
            with open(default_min_max_path, "r") as f:
                min_max = json.load(f)
        else:
            if not os.path.exists(min_max_path):
                raise FileNotFoundError(
                    f"Min and max values file not found at '{min_max_path}'. "
                    "Please ensure the file exists."
                )
            logging.info(f"Using min_max.json from '{min_max_path}' for normalization.")
            with open(min_max_path, "r") as f:
                min_max = json.load(f)

        if "channel_min" not in min_max or "channel_max" not in min_max:
            raise ValueError(
                f"'channel_min' or 'channel_max' not found in '{min_max_path or default_min_max_path}'."
            )
        channel_min = min_max["channel_min"]
        channel_max = min_max["channel_max"]

        num_channels = len(channel_min)
        if len(channel_max) != num_channels:
            raise ValueError(
                "Length of 'channel_min' and 'channel_max' must be the same."
            )

        if required_channels is not None:
            if num_channels != required_channels:
                raise ValueError(
                    f"Number of channels in min_max.json ({num_channels}) does not match "
                    f"the required_channels ({required_channels})."
                )
        logging.info(f"Number of channels validated: {num_channels}")
    else:
        logging.info("Skipping min_max.json validation as data is pre-normalized.")

    # Validate each .npy file
    logging.info("Validating .npy files...")
    for file in tqdm(all_files, desc="Validating files"):
        try:
            data = np.load(file, mmap_mode="r")
        except Exception as e:
            raise ValueError(f"Error loading file '{file}': {e}")

        if data.ndim != 4:
            raise ValueError(
                f"File '{file}' has {data.ndim} dimensions; expected 4 dimensions (timesteps, channels, height, width)."
            )

        timesteps, channels, height, width = data.shape
        if required_channels is not None and channels != required_channels:
            raise ValueError(
                f"File '{file}' has {channels} channels; expected {required_channels} channels."
            )
        elif required_channels is None and num_channels is not None and channels != num_channels:
            # If not pre_normalized, num_channels will be set from min_max.json
            raise ValueError(
                f"File '{file}' has {channels} channels; expected {num_channels} channels based on min_max.json."
            )

        if timesteps < future_steps + 1:
            raise ValueError(
                f"File '{file}' has {timesteps} timesteps; requires at least {future_steps + 1} timesteps for future_steps={future_steps}."
            )
        del data # Explicitly close the memmap

    logging.info("Data validation completed successfully.")


class GenericPhysicsDataset(Dataset):
    """
    A generic PyTorch Dataset for loading preprocessed physics data with sliding window sample generation
    and optional channel-wise normalization using precomputed min and max values.
    """

    def __init__(
        self,
        data_dirs: list[str],
        future_steps: int = 1,
        min_max_path: str = None,
        required_channels: int = None,
        validate: bool = True,
        pre_normalized: bool = False, # New parameter
    ):
        """
        Initializes the GenericPhysicsDataset.

        Args:
            data_dirs (list of str): List of directories containing preprocessed `.npy` files.
                                     Typically includes either train or test directories.
            future_steps (int): Number of timesteps in the future the model will predict.
            min_max_path (str, optional): Path to the JSON file containing min and max values for each channel.
                                          If None, it will look for 'min_max.json' in the first data directory
                                          provided in `data_dirs`.
            required_channels (int, optional): Number of channels expected in the data.
                                               If None, it will be inferred from the min_max.json file.
            validate (bool, optional): Whether to perform data validation upon initialization. Defaults to True.
            pre_normalized (bool): If True, assumes data is already normalized and skips min/max loading and normalization steps.
        """
        self.pre_normalized = pre_normalized

        if validate:
            validate_data_format(
                data_dirs, future_steps, min_max_path, required_channels, pre_normalized=self.pre_normalized
            )

        self.data_dirs = data_dirs
        self.future_steps = future_steps
        self.files = []

        # Aggregate all .npy files from the specified directories
        for data_dir in data_dirs:
            dir_files = sorted(
                [
                    os.path.join(data_dir, f)
                    for f in os.listdir(data_dir)
                    if f.endswith(".npy")
                ]
            )
            self.files.extend(dir_files)

        # Load min and max values only if data is not pre-normalized
        if not self.pre_normalized:
            if min_max_path is None:
                # Assume min_max.json is present in the first data directory and applies to all
                if not self.data_dirs:
                    raise ValueError("No data directories provided to infer min_max.json path.")
                
                default_min_max_path = os.path.join(self.data_dirs[0], "min_max.json")
                if not os.path.exists(default_min_max_path):
                    raise FileNotFoundError(
                        f"Min and max values file not found at '{default_min_max_path}'. "
                        "Please ensure the file exists or provide 'min_max_path'."
                    )
                with open(default_min_max_path, "r") as f:
                    min_max = json.load(f)
            else:
                if not os.path.exists(min_max_path):
                    raise FileNotFoundError(
                        f"Min and max values file not found at '{min_max_path}'. "
                        "Please ensure the file exists."
                    )
                with open(min_max_path, "r") as f:
                    min_max = json.load(f)

            if "channel_min" not in min_max or "channel_max" not in min_max:
                raise ValueError(
                    f"'channel_min' or 'channel_max' not found in '{min_max_path or default_min_max_path}'."
                )
            self.channel_min = min_max["channel_min"]
            self.channel_max = min_max["channel_max"]

            # Determine the number of channels
            self.num_channels = len(self.channel_min)
            if len(self.channel_max) != self.num_channels:
                raise ValueError("Length of 'channel_min' and 'channel_max' must be the same.")
        else:
            # If pre_normalized, infer num_channels from the first file directly if possible
            if self.files:
                sample_data = np.load(self.files[0], mmap_mode='r')
                self.num_channels = sample_data.shape[1] # (timesteps, channels, height, width)
                del sample_data
            else:
                raise ValueError("Cannot infer number of channels if no .npy files are found and pre_normalized is True.")
            logging.info(f"Data is pre-normalized. Inferred {self.num_channels} channels.")


        # Precompute the number of samples across all files
        self.samples = []
        logging.info("Preparing dataset samples...")
        for file_idx, file in enumerate(
            tqdm(self.files, desc="Listing samples")
        ):
            data_memmap = np.load(file, mmap_mode="r")
            timesteps = data_memmap.shape[0]  # Shape: (timesteps, channels, height, width)
            del data_memmap  # Close the memmap

            max_start_t = timesteps - self.future_steps - 1
            for start_t in range(0, max_start_t + 1):
                # Store (file_idx, start_t) for lazy loading later
                self.samples.append((file_idx, start_t))

        logging.info(f"Total samples in dataset: {len(self.samples)}")

        # Precompute t1 assuming all files have the same number of timesteps
        if len(self.files) > 0:
            sample_memmap = np.load(self.files[0], mmap_mode="r")
            timesteps = sample_memmap.shape[0]
            del sample_memmap
            whole_t = timesteps + 1
            self.t1 = torch.tensor(
                [(i + 1) / whole_t for i in range(self.future_steps)],
                dtype=torch.float32,
            )  # Shape: (future_steps,)
            self.t0 = torch.tensor(0.0, dtype=torch.float32)  # Scalar
        else:
            raise ValueError(
                "No valid .npy files found in the specified directories after validation."
            )

        # Initialize a cache for memory-mapped files to improve performance
        self._memmap_cache = {}

    def __len__(self) -> int:
        return len(self.samples)

    def normalize_channel(self, tensor: torch.Tensor, channel_idx: int) -> torch.Tensor:
        """
        Normalizes a specific channel of the tensor between 0 and 1.
        Only called if self.pre_normalized is False.

        Args:
            tensor (torch.Tensor): The tensor to normalize. Shape expected: (..., channels, height, width)
            channel_idx (int): The index of the channel to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        min_val = self.channel_min[channel_idx]
        max_val = self.channel_max[channel_idx]
        if max_val - min_val == 0:
            raise ValueError(
                f"Max and min values for channel {channel_idx} are the same. Cannot normalize due to division by zero."
            )
        
        # Apply normalization across the channel dimension
        tensor_slice = tensor.select(1, channel_idx) if tensor.ndim == 4 else tensor.select(0, channel_idx)
        tensor_slice.sub_(min_val).div_(max_val - min_val)
        return tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (ic, t0, t1, target, file_path)
                - ic: Initial condition tensor of shape (channels, height, width)
                - t0: Scalar tensor (0.0)
                - t1: Tensor of shape (future_steps,)
                - target: Tensor of shape (future_steps, channels, height, width)
                - file_path: The absolute path to the .npy file this sample came from.
        """
        # Retrieve the (file_idx, start_t) tuple for this sample
        file_idx, start_t = self.samples[idx]
        file_path = self.files[file_idx]

        # Check if the memmap for this file is already cached
        if file_path not in self._memmap_cache:
            try:
                # Memory-map the file and store in cache
                data_memmap = np.load(file_path, mmap_mode="r")
                self._memmap_cache[file_path] = data_memmap
            except Exception as e:
                raise ValueError(f"Error loading file '{file_path}': {e}")

        data_memmap = self._memmap_cache[file_path]

        # Convert to PyTorch tensor
        try:
            # Access the required timesteps: start_t to start_t + 1 + future_steps
            required_timesteps = slice(
                start_t, start_t + 1 + self.future_steps
            )
            data = data_memmap[
                required_timesteps, :, :, :
            ]  # Shape: (future_steps + 1, channels, height, width)
        except Exception as e:
            raise ValueError(
                f"Error accessing timesteps {start_t} to {start_t + self.future_steps + 1} in file '{file_path}': {e}"
            )

        # Convert to PyTorch tensor. Using .copy() to ensure data is not read-only for normalization.
        data_tensor = torch.from_numpy(data.copy()).float()  # Shape: (future_steps + 1, channels, height, width)

        # Normalize each channel between 0 and 1 using precomputed min and max, only if not pre_normalized
        if not self.pre_normalized:
            for channel_idx in range(self.num_channels):
                data_tensor = self.normalize_channel(data_tensor, channel_idx)

        # Extract input timestep
        ic = data_tensor[0]  # Shape: (channels, height, width)

        # Prepare the target sequence (ground truth)
        target = data_tensor[
            1:
        ]  # Shape: (future_steps, channels, height, width)

        return ic, self.t0, self.t1, target, file_path

    def __del__(self):
        # Close all memmap files when the dataset is deleted
        for memmap in self._memmap_cache.values():
            if isinstance(memmap, np.memmap):
                memmap._mmap.close() # Explicitly close the memory map
        self._memmap_cache.clear()


def custom_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Custom collate function to rearrange the target tensor and include file paths.

    Args:
        batch: A list of tuples (ic, t0, t1, target, file_path) from GenericPhysicsDataset.__getitem__

    Returns:
        Batched tensors and fixed time indicators, along with a list of file paths:
            - ic: (batch_size, channels, height, width)
            - t0: 0.0 (scalar tensor)
            - t1: (future_steps,) tensor
            - target: (future_steps, batch_size, channels, height, width)
            - file_paths: list of strings (batch_size)
    """
    ics, t0s, t1s, targets, file_paths = zip(*batch)

    # Stack the initial conditions into a tensor
    ic = torch.stack(ics, dim=0)  # Shape: (batch_size, channels, height, width)

    # Since t0 is always 0.0, return a single scalar tensor
    t0 = t0s[0] # Take the first t0, as they are all the same scalar

    # Since t1 is consistent across all samples, take the first one
    t1 = t1s[0]  # Shape: (future_steps,)

    # Stack targets into a tensor and permute to match desired shape
    target = torch.stack(targets, dim=0).permute(
        1, 0, 2, 3, 4
    )  # Shape: (future_steps, batch_size, channels, height, width)

    return ic, t0, t1, target, list(file_paths)


class InitialConditionDataset(Dataset):
    """
    A PyTorch Dataset for loading only the initial condition (first time step) from preprocessed physics data.
    Supports optional skipping of min/max normalization if data is pre-normalized.
    """

    def __init__(
        self,
        data_dirs: list[str],
        future_steps: int = 1,
        min_max_path: str = None,
        required_channels: int = None,
        validate: bool = True,
        pre_normalized: bool = False, # New parameter
    ):
        """
        Initializes the InitialConditionDataset.

        Args:
            data_dirs (list of str): List of directories containing preprocessed `.npy` files.
            future_steps (int): Number of timesteps in the future the model will predict.
            min_max_path (str, optional): Path to the JSON file containing min and max values for each channel.
                                          If None, it will look for 'min_max.json' in the first data directory
                                          provided in `data_dirs`.
            required_channels (int, optional): Number of channels expected in the data.
                                               If None, it will be inferred from the min_max.json file.
            validate (bool, optional): Whether to perform data validation upon initialization. Defaults to True.
            pre_normalized (bool): If True, assumes data is already normalized and skips min/max loading and normalization steps.
        """
        self.pre_normalized = pre_normalized

        if validate:
            validate_data_format(
                data_dirs, future_steps, min_max_path, required_channels, pre_normalized=self.pre_normalized
            )

        self.data_dirs = data_dirs
        self.future_steps = future_steps
        self.files = []

        # Aggregate all .npy files from the specified directories
        for data_dir in data_dirs:
            dir_files = sorted(
                [
                    os.path.join(data_dir, f)
                    for f in os.listdir(data_dir)
                    if f.endswith(".npy")
                ]
            )
            if not dir_files:
                logging.warning(
                    f"No .npy files found in directory '{data_dir}'."
                )
            self.files.extend(dir_files)

        if not self.files:
            raise ValueError(
                "No .npy files found in any of the specified directories."
            )

        # Load min and max values only if data is not pre-normalized
        if not self.pre_normalized:
            if min_max_path is None:
                # Assume min_max.json is present in the first data directory and applies to all
                if not self.data_dirs:
                    raise ValueError("No data directories provided to infer min_max.json path.")
                
                default_min_max_path = os.path.join(self.data_dirs[0], "min_max.json")
                if not os.path.exists(default_min_max_path):
                    raise FileNotFoundError(
                        f"Min and max values file not found at '{default_min_max_path}'. "
                        "Please ensure the file exists or provide 'min_max_path'."
                    )
                with open(default_min_max_path, "r") as f: # Corrected variable name here
                    min_max = json.load(f)
            else:
                if not os.path.exists(min_max_path):
                    raise FileNotFoundError(
                        f"Min and max values file not found at '{min_max_path}'. "
                        "Please ensure the file exists."
                    )
                with open(min_max_path, "r") as f:
                    min_max = json.load(f)

            if "channel_min" not in min_max or "channel_max" not in min_max:
                raise ValueError(
                    f"'channel_min' or 'channel_max' not found in '{min_max_path or default_min_max_path}'."
                )
            self.channel_min = min_max["channel_min"]
            self.channel_max = min_max["channel_max"]

            num_channels = len(self.channel_min)
            if len(self.channel_max) != num_channels:
                raise ValueError(
                    "Length of 'channel_min' and 'channel_max' must be the same."
                )

            if required_channels is not None:
                if num_channels != required_channels:
                    raise ValueError(
                        f"Number of channels in min_max.json ({num_channels}) does not match "
                        f"the required_channels ({required_channels})."
                    )
            self.num_channels = num_channels
            logging.info(f"Number of channels validated: {self.num_channels}")
        else:
            # If pre_normalized, infer num_channels from the first file directly if possible
            if self.files:
                sample_data = np.load(self.files[0], mmap_mode='r')
                self.num_channels = sample_data.shape[1] # (timesteps, channels, height, width)
                del sample_data
            else:
                raise ValueError("Cannot infer number of channels if no .npy files are found and pre_normalized is True.")
            logging.info(f"Data is pre-normalized. Inferred {self.num_channels} channels.")


        # Determine the total number of timesteps from a sample file
        if len(self.files) > 0:
            sample_memmap = np.load(self.files[0], mmap_mode="r")
            timesteps = sample_memmap.shape[0]
            del sample_memmap
            whole_t = timesteps + 1
            self.t1 = torch.tensor(
                [(i + 1) / whole_t for i in range(self.future_steps)],
                dtype=torch.float32,
            )  # Shape: (future_steps,)
            self.t0 = torch.tensor(0.0, dtype=torch.float32)  # Scalar
        else:
            raise ValueError(
                "No valid .npy files found in the specified directories."
            )

        # Initialize a cache for memory-mapped files to improve performance
        self._memmap_cache = {}

    def __len__(self) -> int:
        return len(self.files)

    def normalize_channel(self, tensor: torch.Tensor, channel_idx: int) -> torch.Tensor:
        """
        Normalizes a specific channel of the tensor between 0 and 1.
        Only called if self.pre_normalized is False.

        Args:
            tensor (torch.Tensor): The tensor to normalize. Shape expected: (channels, height, width)
            channel_idx (int): The index of the channel to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        min_val = self.channel_min[channel_idx]
        max_val = self.channel_max[channel_idx]
        if max_val - min_val == 0:
            raise ValueError(
                f"Max and min values for channel {channel_idx} are the same. Cannot normalize due to division by zero."
            )
        tensor.select(0, channel_idx).sub_(min_val).div_(max_val - min_val)
        return tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, str]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (ic, t0, t1, target, file_path)
                - ic: Initial condition tensor of shape (channels, height, width)
                - t0: Scalar tensor (0.0)
                - t1: Tensor of shape (future_steps,)
                - target: None (since target is not used)
                - file_path: The absolute path to the .npy file this sample came from.
        """
        file_path = self.files[idx]

        # Check if the memmap for this file is already cached
        if file_path not in self._memmap_cache:
            try:
                # Memory-map the file and store in cache
                data_memmap = np.load(file_path, mmap_mode="r")
                self._memmap_cache[file_path] = data_memmap
            except Exception as e:
                raise ValueError(f"Error loading file '{file_path}': {e}")

        data_memmap = self._memmap_cache[file_path]

        # Convert to PyTorch tensor
        try:
            # Access the first timestep
            data = data_memmap[0, :, :, :]  # Shape: (channels, height, width)
        except Exception as e:
            raise ValueError(
                f"Error accessing the first timestep in file '{file_path}': {e}"
            )

        # Convert to PyTorch tensor. Using .copy() to ensure data is not read-only for normalization.
        data_tensor = torch.from_numpy(data.copy()).float()  # Shape: (channels, height, width)

        # Normalize each channel between 0 and 1 using precomputed min and max, only if not pre_normalized
        if not self.pre_normalized:
            for channel_idx in range(self.num_channels):
                data_tensor = self.normalize_channel(data_tensor, channel_idx)

        ic = data_tensor  # Shape: (channels, height, width)

        return ic, self.t0, self.t1, None, file_path

    def __del__(self):
        # Close all memmap files when the dataset is deleted
        for memmap in self._memmap_cache.values():
            if isinstance(memmap, np.memmap):
                memmap._mmap.close() # Explicitly close the memory map
        self._memmap_cache.clear()


def initial_condition_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, list[str]]:
    """
    Custom collate function for InitialConditionDataset.

    Args:
        batch: A list of tuples (ic, t0, t1, target, file_path) from InitialConditionDataset.__getitem__

    Returns:
        tuple: Batched tensors and fixed time indicators, along with a list of file paths:
            - ic: (batch_size, channels, height, width)
            - t0: Scalar tensor (0.0)
            - t1: (future_steps,) tensor
            - target: None (since target is not used)
            - file_paths: list of strings (batch_size)
    """
    ics, t0s, t1s, targets, file_paths = zip(*batch)

    # Stack the initial conditions into a tensor
    ic = torch.stack(ics, dim=0)  # Shape: (batch_size, channels, height, width)

    # Since t0 is always 0.0, return a single scalar tensor
    t0 = t0s[0] # Take the first t0, as they are all the same scalar

    # Since t1 is consistent across all samples, take the first one
    t1 = t1s[0]  # Shape: (future_steps,)

    # Targets are None, so we can return None or handle accordingly
    target = None

    return ic, t0, t1, target, list(file_paths)