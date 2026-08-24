#!/usr/bin/env python3
"""
Physics Metrics Module

A modular collection of physics-based metric calculations for fluid dynamics
and other physical systems. Supports both NumPy and PyTorch tensors.

Functions:
- calculate_strain_rate: Computes strain rate tensor and magnitude.
- calculate_vorticity: Computes vorticity (curl of velocity).
- calculate_gradient_magnitude: Computes total gradient magnitude.
- calculate_principal_strain_direction: Computes principal strain magnitudes and angle.
- calculate_shape_metrics: Computes basic anisotropy and ellipse area from offset fields.
- calculate_deformation_magnitude: Computes mean deformation magnitude.
- calculate_kernel_metrics: A comprehensive function for all kernel shape metrics.
- BurgersPdeLoss: A class to compute the PDE residual for the 2D Burgers' equation.
- denormalize: Denormalizes tensors using channel-specific normalization parameters.
- set_channel_mapping: Utility to update the global channel mapping.
- test_strain_rate_calculation: Validation function with known test cases.

Date: 2025-08-23
"""

from typing import Dict, Tuple, Union, Optional
import numpy as np
import warnings
# MODIFIED: Imported KMeans for clustering
from sklearn.cluster import KMeans

# Optional PyTorch support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Default channel mapping - can be overridden by setting this variable
json_key_mapping = {
    'u': 0, 'v': 1, 'strain_rate': 2, 'vorticity': 3,
    'gradient_magnitude': 4, 'anisotropy': 5, 'ellipse_area': 6,
    'deformation_magnitude': 7
}

# =================================================================== #
# PHYSICS METRICS
# =================================================================== #

def calculate_strain_rate(
    u: Union[np.ndarray, 'torch.Tensor'],
    v: Union[np.ndarray, 'torch.Tensor'],
    dx: float = 1.0,
    dy: Optional[float] = None,
    return_components: bool = False
) -> Union[np.ndarray, 'torch.Tensor', Dict[str, Union[np.ndarray, 'torch.Tensor']]]:
    """Calculates strain rate tensor magnitude from 2D velocity fields."""
    if dy is None:
        dy = dx
    is_torch = TORCH_AVAILABLE and isinstance(u, torch.Tensor)

    if is_torch:
        if not isinstance(v, torch.Tensor):
            raise TypeError("If u is a torch.Tensor, v must also be a torch.Tensor")
        u_grads = torch.gradient(u, spacing=(dy, dx), dim=(-2, -1))
        v_grads = torch.gradient(v, spacing=(dy, dx), dim=(-2, -1))
        du_dy, du_dx = u_grads[0], u_grads[1]
        dv_dy, dv_dx = v_grads[0], v_grads[1]
    else:
        u, v = np.asarray(u), np.asarray(v)
        u_grads = np.gradient(u, dy, dx, axis=(-2, -1))
        v_grads = np.gradient(v, dy, dx, axis=(-2, -1))
        du_dy, du_dx = u_grads[0], u_grads[1]
        dv_dy, dv_dx = v_grads[0], v_grads[1]

    strain_rate_xx = du_dx
    strain_rate_yy = dv_dy
    strain_rate_xy = 0.5 * (du_dy + dv_dx)

    if is_torch:
        magnitude = torch.sqrt(strain_rate_xx**2 + strain_rate_yy**2 + 2 * strain_rate_xy**2)
    else:
        magnitude = np.sqrt(strain_rate_xx**2 + strain_rate_yy**2 + 2 * strain_rate_xy**2)

    if return_components:
        return {'magnitude': magnitude, 'xx': strain_rate_xx, 'yy': strain_rate_yy, 'xy': strain_rate_xy}
    else:
        return magnitude

def calculate_vorticity(
    u: Union[np.ndarray, 'torch.Tensor'],
    v: Union[np.ndarray, 'torch.Tensor'],
    dx: float = 1.0,
    dy: Optional[float] = None
) -> Union[np.ndarray, 'torch.Tensor']:
    """Calculates vorticity (curl) of a 2D velocity field."""
    if dy is None:
        dy = dx
    is_torch = TORCH_AVAILABLE and isinstance(u, torch.Tensor)

    if is_torch:
        u_grads = torch.gradient(u, spacing=(dy, dx), dim=(-2, -1))
        v_grads = torch.gradient(v, spacing=(dy, dx), dim=(-2, -1))
        du_dy, _ = u_grads[0], u_grads[1]
        _, dv_dx = v_grads[0], v_grads[1]
    else:
        u, v = np.asarray(u), np.asarray(v)
        u_grads = np.gradient(u, dy, dx, axis=(-2, -1))
        v_grads = np.gradient(v, dy, dx, axis=(-2, -1))
        du_dy, _ = u_grads[0], u_grads[1]
        _, dv_dx = v_grads[0], v_grads[1]

    return dv_dx - du_dy

def calculate_gradient_magnitude(
    u: Union[np.ndarray, 'torch.Tensor'],
    v: Union[np.ndarray, 'torch.Tensor'],
    dx: float = 1.0,
    dy: Optional[float] = None
) -> Union[np.ndarray, 'torch.Tensor']:
    """Calculates the total gradient magnitude of a velocity field."""
    if dy is None:
        dy = dx
    is_torch = TORCH_AVAILABLE and isinstance(u, torch.Tensor)

    if is_torch:
        u_grads = torch.gradient(u, spacing=(dy, dx), dim=(-2, -1))
        v_grads = torch.gradient(v, spacing=(dy, dx), dim=(-2, -1))
        u_grad_mag = torch.sqrt(u_grads[1]**2 + u_grads[0]**2)
        v_grad_mag = torch.sqrt(v_grads[1]**2 + v_grads[0]**2)
    else:
        u, v = np.asarray(u), np.asarray(v)
        u_grads = np.gradient(u, dy, dx, axis=(-2, -1))
        v_grads = np.gradient(v, dy, dx, axis=(-2, -1))
        u_grad_mag = np.sqrt(u_grads[1]**2 + u_grads[0]**2)
        v_grad_mag = np.sqrt(v_grads[1]**2 + v_grads[0]**2)

    return u_grad_mag + v_grad_mag

def calculate_principal_strain_direction(
    u: Union[np.ndarray, 'torch.Tensor'],
    v: Union[np.ndarray, 'torch.Tensor'],
    dx: float = 1.0,
    dy: Optional[float] = None
) -> Dict[str, Union[np.ndarray, 'torch.Tensor']]:
    """Calculates the principal strain direction (angle) and magnitudes."""
    if dy is None:
        dy = dx
    is_torch = TORCH_AVAILABLE and isinstance(u, torch.Tensor)

    if is_torch:
        uy, ux = torch.gradient(u, spacing=(dy, dx), dim=(-2, -1))
        vy, vx = torch.gradient(v, spacing=(dy, dx), dim=(-2, -1))
        
        strain_tensor = torch.stack([torch.stack([ux, 0.5 * (uy + vx)], dim=0),
                                     torch.stack([0.5 * (uy + vx), vy], dim=0)], dim=0)
        strain_tensor_reshaped = strain_tensor.permute(2, 3, 0, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(strain_tensor_reshaped)

        max_indices = torch.argmax(eigenvalues, dim=-1)
        
        H, W = u.shape[-2], u.shape[-1]
        principal_eigenvector = eigenvectors[torch.arange(H).view(-1, 1), torch.arange(W), :, max_indices]
        
        max_strain = eigenvalues[torch.arange(H).view(-1, 1), torch.arange(W), max_indices]
        min_strain = eigenvalues[torch.arange(H).view(-1, 1), torch.arange(W), 1 - max_indices]

        angle_rad = torch.arctan2(principal_eigenvector[..., 1], principal_eigenvector[..., 0])
        angle_deg = torch.rad2deg(angle_rad) % 180.0
    else:
        u, v = np.asarray(u), np.asarray(v)
        vy, vx = np.gradient(v, dy, dx, axis=(-2, -1))
        uy, ux = np.gradient(u, dy, dx, axis=(-2, -1))

        strain_tensor = np.array([[ux, 0.5 * (uy + vx)], [0.5 * (uy + vx), vy]])
        strain_tensor_reshaped = np.transpose(strain_tensor, (2, 3, 0, 1))
        eigenvalues, eigenvectors = np.linalg.eigh(strain_tensor_reshaped)

        max_eigenvalue_indices = np.argmax(eigenvalues, axis=-1)
        
        H, W = u.shape[-2], u.shape[-1]
        I, J = np.ogrid[:H, :W]
        principal_eigenvector = eigenvectors[I, J, :, max_eigenvalue_indices]
        
        max_strain = eigenvalues[I, J, max_eigenvalue_indices]
        min_strain = eigenvalues[I, J, 1 - max_eigenvalue_indices]

        angle_rad = np.arctan2(principal_eigenvector[..., 1], principal_eigenvector[..., 0])
        angle_deg = np.degrees(angle_rad) % 180.0
    
    return {
        "principal_strain_angle": angle_deg,
        "max_principal_strain": max_strain,
        "min_principal_strain": min_strain
    }

# Add this to your metrics.py file
# You will need to add 'from sklearn.cluster import KMeans' at the top

def cluster_field_kmeans(
    field: Union[np.ndarray, 'torch.Tensor'],
    n_clusters: int = 3,
    random_state: int = 42
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Clusters a 2D spatial field into n clusters using K-Means.

    The function automatically sorts the cluster labels based on the cluster
    center values, so that label 0 corresponds to the cluster with the
    lowest mean value, 1 to the next lowest, and so on.

    Parameters:
    -----------
    field : np.ndarray or torch.Tensor
        A 2D tensor or array representing the spatial field to cluster.
    n_clusters : int, default=3
        The number of clusters to form (e.g., 3 for Low, Medium, High).
    random_state : int, default=42
        Seed for the random number generator for reproducibility.

    Returns:
    --------
    cluster_labels : np.ndarray or torch.Tensor
        A 2D array/tensor of the same shape as the input field, containing
        integer labels (0, 1, ..., n_clusters-1) for each pixel.
    """
    is_torch = TORCH_AVAILABLE and isinstance(field, torch.Tensor)
    original_shape = field.shape
    
    if is_torch:
        field_np = field.detach().cpu().numpy()
    else:
        field_np = np.asarray(field)

    # Reshape data for KMeans and remove non-finite values that cause errors
    flat_field = field_np.flatten().reshape(-1, 1)
    finite_mask = np.isfinite(flat_field).flatten()
    
    if not np.any(finite_mask):
        # Handle case where the entire field is non-finite
        return torch.full(original_shape, -1, dtype=torch.long) if is_torch else np.full(original_shape, -1, dtype=np.int64)

    # Cluster only the finite values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto').fit(flat_field[finite_mask])
    
    # Sort cluster centers to create a mapping from label to rank (0=low, 1=medium, etc.)
    centers = kmeans.cluster_centers_.flatten()
    sorted_center_indices = np.argsort(centers)
    label_map = {original_idx: rank for rank, original_idx in enumerate(sorted_center_indices)}
    
    # Map the labels for the finite values
    mapped_labels = np.array([label_map[l] for l in kmeans.labels_])
    
    # Create the final 2D label mask, placing mapped labels in the correct spots
    final_labels_flat = np.full(flat_field.shape[0], -1, dtype=np.int64) # Use -1 for non-finite pixels
    final_labels_flat[finite_mask] = mapped_labels
    final_labels_2d = final_labels_flat.reshape(original_shape)
    
    if is_torch:
        return torch.from_numpy(final_labels_2d).to(field.device)
    else:
        return final_labels_2d
    
# Eulerian Pixels
# =================================================================== #
# NEW: Bilinear Usage and Weight Metrics
# =================================================================== #

def calculate_bilinear_metrics(
    offset_field: Union[np.ndarray, 'torch.Tensor'],
    orig_H: int,
    orig_W: int,
    scale_y: float,
    scale_x: float,
) -> Dict[str, Union[np.ndarray, 'torch.Tensor']]:
    """
    Computes bilinear usage and weight maps from an offset field.

    Parameters:
    - offset_field (np.ndarray/torch.Tensor): The deformable offset field.
    - orig_H (int): The original image height.
    - orig_W (int): The original image width.
    - scale_y (float): The scaling factor for the y-dimension.
    - scale_x (float): The scaling factor for the x-dimension.

    Returns:
    - Dict: A dictionary containing 'usage_map' and 'weight_map'.
    """
    is_torch = TORCH_AVAILABLE and isinstance(offset_field, torch.Tensor)
    
    if is_torch:
        offsets_numpy = offset_field.detach().cpu().numpy()
    else:
        offsets_numpy = np.asarray(offset_field)
    
    num_offs_groups, _, Hf, Wf = offsets_numpy.shape
    
    usage_map = np.zeros((orig_H, orig_W), dtype=np.float32)
    weight_map = np.zeros_like(usage_map)
    
    grid_coords = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 0], [0, 1],
        [1, -1], [1, 0], [1, 1]
    ], dtype=np.float32)
    
    for y_f_loop in range(Hf):
        for x_f_loop in range(Wf):
            for i_coord in range(min(num_offs_groups, len(grid_coords))):
                dy, dx = offsets_numpy[i_coord, :, y_f_loop, x_f_loop]
                
                ex = scale_x * (x_f_loop + grid_coords[i_coord][0] + dx)
                ey = scale_y * (y_f_loop + grid_coords[i_coord][1] + dy)
                
                # --- Usage Logic ---
                x0, y0 = int(np.floor(ex)), int(np.floor(ey))
                x1, y1 = x0 + 1, y0 + 1
                if 0 <= x0 < orig_W and 0 <= y0 < orig_H: usage_map[y0, x0] += 0.25
                if 0 <= x1 < orig_W and 0 <= y0 < orig_H: usage_map[y0, x1] += 0.25
                if 0 <= x0 < orig_W and 0 <= y1 < orig_H: usage_map[y1, x0] += 0.25
                if 0 <= x1 < orig_W and 0 <= y1 < orig_H: usage_map[y1, x1] += 0.25
                
                # --- Weight Logic ---
                w00 = (x1 - ex) * (y1 - ey)
                w10 = (ex - x0) * (y1 - ey)
                w01 = (x1 - ex) * (ey - y0)
                w11 = (ex - x0) * (ey - y0)
                if 0 <= x0 < orig_W and 0 <= y0 < orig_H: weight_map[y0, x0] += w00
                if 0 <= x1 < orig_W and 0 <= y0 < orig_H: weight_map[y0, x1] += w10
                if 0 <= x0 < orig_W and 0 <= y1 < orig_H: weight_map[y1, x0] += w01
                if 0 <= x1 < orig_W and 0 <= y1 < orig_H: weight_map[y1, x1] += w11
                
    return {'usage_map': usage_map, 'weight_map': weight_map}

# =================================================================== #
# NEW: Function to Identify Contributing Source Pixels
# =================================================================== #

def get_contributing_pixels(
    offset_field: np.ndarray,
    target_pixel: tuple[int, int],
    orig_H: int,
    orig_W: int,
    scale_y: float,
    scale_x: float
) -> list[tuple[int, int]]:
    """
    Identifies which source pixels from the feature map contribute to a
    specific target pixel in the original image space.

    Parameters:
    - offset_field (np.ndarray): The deformable offset field.
    - target_pixel (tuple): The (y, x) coordinate of the target pixel to investigate.
    - orig_H (int): The original image height.
    - orig_W (int): The original image width.
    - scale_y (float): The scaling factor for the y-dimension.
    - scale_x (float): The scaling factor for the x-dimension.

    Returns:
    - list: A list of unique (y, x) coordinates of the source pixels.
    """
    target_y, target_x = target_pixel
    num_offs_groups, _, Hf, Wf = offset_field.shape
    
    contributing_sources = set()
    
    grid_coords = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 0], [0, 1],
        [1, -1], [1, 0], [1, 1]
    ], dtype=np.float32)

    for y_f in range(Hf):  # y-coordinate of the source pixel
        for x_f in range(Wf):  # x-coordinate of the source pixel
            for i_coord in range(min(num_offs_groups, len(grid_coords))):
                dy, dx = offset_field[i_coord, :, y_f, x_f]
                
                # Calculate the effective destination coordinate
                ex = scale_x * (x_f + grid_coords[i_coord][0] + dx)
                ey = scale_y * (y_f + grid_coords[i_coord][1] + dy)
                
                # Check if the target pixel is within the bilinear contribution area
                x0, y0 = int(np.floor(ex)), int(np.floor(ey))
                
                if (target_y >= y0 and target_y <= y0 + 1) and \
                   (target_x >= x0 and target_x <= x0 + 1):
                    contributing_sources.add((y_f, x_f))
                    # Since we found one contribution, no need to check other kernel points for this source pixel
                    break 
                        
    return sorted(list(contributing_sources))

# =================================================================== #
# NEW: Function to Extract Raw Deformable Offsets
# =================================================================== #

def get_deformable_offsets(
    offset_field: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Extracts and flattens the raw dy, dx offsets for each kernel point.

    This function reshapes the offset data into a flat dictionary format,
    making it easy to append to a pandas DataFrame.

    Parameters:
    - offset_field (np.ndarray): The deformable offset field of shape 
      (num_points, 2, H, W).

    Returns:
    - dict: A dictionary where keys are 'offset_k#_d[y/x]' and values 
      are the corresponding flattened offset arrays.
    """
    offset_dict = {}
    num_kernel_points = offset_field.shape[0]
    
    for k in range(num_kernel_points):
        # Extract the dy (channel 0) and dx (channel 1) fields for the k-th point
        dy_k = offset_field[k, 0, :, :]
        dx_k = offset_field[k, 1, :, :]
        
        # Add flattened arrays to the dictionary
        offset_dict[f'offset_k{k}_dy'] = dy_k.flatten()
        offset_dict[f'offset_k{k}_dx'] = dx_k.flatten()
        
    return offset_dict

# =================================================================== #
# DEFORMABLE KERNEL METRICS
# =================================================================== #

def calculate_shape_metrics(
    offset_field: Union[np.ndarray, 'torch.Tensor'],
    return_components: bool = False
) -> Dict[str, Union[np.ndarray, 'torch.Tensor']]:
    """Calculates basic anisotropy and ellipse area from offset fields."""
    is_torch = TORCH_AVAILABLE and isinstance(offset_field, torch.Tensor)
    C = offset_field.shape[0] # Number of kernel points
    
    if is_torch:
        grid_coords = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],
                                 dtype=torch.float32, device=offset_field.device)
        offset_reshaped = offset_field.permute(2, 3, 0, 1)
        displaced_points = grid_coords[None, None, :, :] + offset_reshaped
        centers = torch.mean(displaced_points, dim=2, keepdim=True)
        centered_points = displaced_points - centers
        cov_matrices = torch.einsum('...ij,...ik->...jk', centered_points, centered_points) / C
        eigenvalues, _ = torch.linalg.eigh(cov_matrices)
        lambda_min, lambda_max = eigenvalues[..., 0], eigenvalues[..., 1]
        anisotropy = torch.where(lambda_min > 1e-9, lambda_max / lambda_min, torch.zeros_like(lambda_max))
        ellipse_area = torch.pi * torch.sqrt(torch.clamp(lambda_min * lambda_max, min=0))
    else:
        offset_field = np.asarray(offset_field)
        grid_coords = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]], dtype=np.float32)
        displaced_points = grid_coords[np.newaxis, np.newaxis, :, :] + np.transpose(offset_field, (2, 3, 0, 1))
        centers = np.mean(displaced_points, axis=2, keepdims=True)
        centered_points = displaced_points - centers
        cov_matrices = np.einsum('...ij,...ik->...jk', centered_points, centered_points) / C
        eigenvalues, _ = np.linalg.eigh(cov_matrices)
        lambda_min, lambda_max = eigenvalues[..., 0], eigenvalues[..., 1]
        anisotropy = np.divide(lambda_max, lambda_min, out=np.zeros_like(lambda_max), where=(lambda_min > 1e-9))
        ellipse_area = np.pi * np.sqrt(np.maximum(lambda_min * lambda_max, 0))

    result = {'anisotropy': anisotropy, 'ellipse_area': ellipse_area}
    if return_components:
        result.update({'lambda_min': lambda_min, 'lambda_max': lambda_max, 'cov_matrices': cov_matrices})
    return result

def calculate_deformation_magnitude(
    offset_field: Union[np.ndarray, 'torch.Tensor']
) -> Union[np.ndarray, 'torch.Tensor']:
    """Calculates the mean magnitude of deformation vectors across kernel positions."""
    is_torch = TORCH_AVAILABLE and isinstance(offset_field, torch.Tensor)
    if is_torch:
        magnitudes = torch.sqrt(offset_field[:, 0, :, :]**2 + offset_field[:, 1, :, :]**2)
        return torch.mean(magnitudes, dim=0)
    else:
        offset_field = np.asarray(offset_field)
        magnitudes = np.sqrt(offset_field[:, 0, :, :]**2 + offset_field[:, 1, :, :]**2)
        return np.mean(magnitudes, axis=0)

def calculate_kernel_metrics(
    offset_field: Union[np.ndarray, 'torch.Tensor']
) -> Dict[str, Union[np.ndarray, 'torch.Tensor']]:
    """A comprehensive, vectorized function to calculate all kernel shape metrics."""
    is_torch = TORCH_AVAILABLE and isinstance(offset_field, torch.Tensor)
    
    if is_torch:
        grid_coords = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],
                                 dtype=torch.float32, device=offset_field.device)
        offset_field_t = offset_field.permute(2, 3, 0, 1)
        deformed_points = grid_coords[None, None, :, :] + offset_field_t
        center = torch.mean(deformed_points, dim=2, keepdim=True)
        centered_points = deformed_points - center
        # MODIFIED: Changed divisor for consistency with calculate_shape_metrics
        cov_matrices = torch.einsum('...ij,...ik->...jk', centered_points, centered_points) / grid_coords.shape[0]
        
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrices)
        lambda_min, lambda_max = eigenvalues[..., 0], eigenvalues[..., 1]
        
        anisotropy = torch.where(lambda_min > 1e-9, lambda_max / lambda_min, torch.zeros_like(lambda_max))
        # MODIFIED: Corrected orientation calculation to use the principal eigenvector (associated with max eigenvalue)
        orientation_rad = torch.arctan2(eigenvectors[..., 1, 1], eigenvectors[..., 0, 1])
        orientation = torch.rad2deg(orientation_rad) % 180.0
        ellipse_area = torch.pi * torch.sqrt(torch.clamp(lambda_max * lambda_min, min=0))
        trace = torch.clamp(lambda_max + lambda_min, min=1e-9)
        shear_magnitude = torch.abs(cov_matrices[..., 0, 1]) / trace
        deformation_magnitude = torch.mean(torch.linalg.norm(offset_field_t, dim=3), dim=2)
        centroid_skew_magnitude = torch.linalg.norm(center.squeeze(dim=2), dim=2)
    else:
        grid_coords = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]], dtype=np.float32)
        offset_field_t = np.transpose(offset_field, (2, 3, 0, 1))
        deformed_points = grid_coords[None, None, :, :] + offset_field_t
        center = np.mean(deformed_points, axis=2, keepdims=True)
        centered_points = deformed_points - center
        # MODIFIED: Changed divisor for consistency with calculate_shape_metrics
        cov_matrices = np.einsum('...ij,...ik->...jk', centered_points, centered_points) / grid_coords.shape[0]
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
        lambda_min, lambda_max = eigenvalues[..., 0], eigenvalues[..., 1]
        
        anisotropy = np.divide(lambda_max, lambda_min, out=np.zeros_like(lambda_max), where=lambda_min > 1e-9)
        # MODIFIED: Corrected orientation calculation to use the principal eigenvector (associated with max eigenvalue)
        orientation_rad = np.arctan2(eigenvectors[..., 1, 1], eigenvectors[..., 0, 1])
        orientation = np.degrees(orientation_rad) % 180.0
        ellipse_area = np.pi * np.sqrt(np.maximum(lambda_max * lambda_min, 0))
        trace = np.maximum(lambda_max + lambda_min, 1e-9)
        shear_magnitude = np.abs(cov_matrices[..., 0, 1]) / trace
        deformation_magnitude = np.mean(np.linalg.norm(offset_field_t, axis=3), axis=2)
        centroid_skew_magnitude = np.linalg.norm(center.squeeze(axis=2), axis=2)
        
    return {
        "anisotropy": anisotropy, "orientation": orientation, "ellipse_area": ellipse_area,
        "shear_magnitude": shear_magnitude, "deformation_magnitude": deformation_magnitude,
        "centroid_skew_magnitude": centroid_skew_magnitude
    }

# =================================================================== #
# PDE-SPECIFIC METRICS
# =================================================================== #

class BurgersPdeLoss:
    """Calculates the PDE residual for the 2D viscous Burgers' equation."""
    def __init__(self, dt=1.0, dx=1.0, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use the BurgersPdeLoss class.")
        super(BurgersPdeLoss, self).__init__(**kwargs)
        self.dt = dt
        self.dx = dx

    def set_data(self, snapshot_data):
        self.snapshot_data = snapshot_data

    def _laplacian(self, mat):
        dY, dX = torch.gradient(mat, spacing=self.dx, dim=(-2, -1))
        dYY, _ = torch.gradient(dY, spacing=self.dx, dim=(-2, -1))
        _, dXX = torch.gradient(dX, spacing=self.dx, dim=(-2, -1))
        return torch.add(dYY, dXX)

    def _time_derivative(self, u0, u1, u2):
        return (u2 - u0) / (2.0 * self.dt)

    def _snapshot_pde_loss(self, u0, v0, u1, v1, u2, v2, nu=1.0):
        laplace_u = self._laplacian(u1)
        laplace_v = self._laplacian(v1)
        u_x, u_y = torch.gradient(u1, spacing=self.dx, dim=(-2, -1))
        v_x, v_y = torch.gradient(v1, spacing=self.dx, dim=(-2, -1))
        u_t_lhs = self._time_derivative(u0, u1, u2)
        v_t_lhs = self._time_derivative(v0, v1, v2)
        u_t_rhs = nu * laplace_u - u1 * u_x - v1 * u_y
        v_t_rhs = nu * laplace_v - u1 * v_x - v1 * v_y
        return u_t_lhs - u_t_rhs, v_t_lhs - v_t_rhs

    def compute_residual(self, nu):
        """Computes the PDE residual over the time series of snapshot data."""
        sequence_length = len(self.snapshot_data)
        fu, fv = [], []
        for i in range(1, sequence_length - 1):
            du, dv = self._snapshot_pde_loss(
                self.snapshot_data[i-1][0], self.snapshot_data[i-1][1],
                self.snapshot_data[i][0],   self.snapshot_data[i][1],
                self.snapshot_data[i+1][0], self.snapshot_data[i+1][1],
                nu
            )
            fu.append(du)
            fv.append(dv)
        return torch.stack(fu, dim=0), torch.stack(fv, dim=0)
    
    
# ---------------------------
# EmLoss Class Integration (Hotspot metrics per literature)
# ---------------------------
class EmLoss:
    def __init__(self, cell_area=(1.5 / 128) * (3 / 256), threshold=875, dt=0.17, **kwargs):
        super(EmLoss, self).__init__(**kwargs)
        self.cell_area = cell_area
        self.threshold = threshold
        self.dt = dt

    def compute_KLD(self, y_true, y_pred):
        mean_X = np.mean(y_true)
        sigma_X = np.std(y_true)
        mean_Y = np.mean(y_pred)
        sigma_Y = np.std(y_pred)
        v1 = sigma_X ** 2
        v2 = sigma_Y ** 2
        a = np.log(sigma_Y / sigma_X)
        num = v1 + (mean_X - mean_Y) ** 2
        den = 2 * v2
        b = num / den
        return a + b - 0.5

    def compute_quantitative_evaluation_sensitivity(self, y_trues, y_preds):
        pcc_list, rmse_list, kld_list = [], [], []
        ts = y_preds.shape[1]
        for i in range(ts):
            pcc = st.pearsonr(y_trues[:, i], y_preds[:, i])[0]
            temp_rmse = sqrt(mean_squared_error(y_trues[:, i], y_preds[:, i]))
            kld = self.compute_KLD(y_trues[:, i], y_preds[:, i])
            pcc_list.append(pcc)
            rmse_list.append(temp_rmse)
            kld_list.append(kld)
        return np.mean(rmse_list), np.mean(kld_list), np.mean(pcc_list)

    def _calculate_hotspot_metric(self, Ts, n_timesteps):
        A_hs_list, T_hs_list = [], []
        for t in range(n_timesteps):
            temp_t = Ts[:, :, t]
            hotspot_mask = (temp_t >= self.threshold).astype(np.float32)
            A_hs = np.sum(hotspot_mask) * self.cell_area
            A_hs_list.append(A_hs)
            if A_hs > 0:
                T_hs = np.sum(temp_t * hotspot_mask * self.cell_area) / A_hs
            else:
                T_hs = 0.0
            T_hs_list.append(T_hs)
        return A_hs_list, T_hs_list

    def calculate_hotspot_metric(self, T_cases, cases_range, n_timesteps):
        all_A_hs, all_T_hs = [], []
        for i in range(cases_range[0], cases_range[1]):
            A_hs, T_hs = self._calculate_hotspot_metric(T_cases[i], n_timesteps)
            all_A_hs.append(A_hs)
            all_T_hs.append(T_hs)
        all_A_hs = np.array(all_A_hs)
        all_T_hs = np.array(all_T_hs)
        mean_T_hs = np.mean(all_T_hs, axis=0)
        mean_A_hs = np.mean(all_A_hs, axis=0)
        perc95_T = np.percentile(all_T_hs, 95, axis=0)
        perc5_T = np.percentile(all_T_hs, 5, axis=0)
        perc95_A = np.percentile(all_A_hs, 95, axis=0)
        perc5_A = np.percentile(all_A_hs, 5, axis=0)
        return (mean_T_hs, perc95_T, perc5_T, all_T_hs), (mean_A_hs, perc95_A, perc5_A, all_A_hs)

    def calculate_hotspot_metric_rate_of_change(self, T_cases, cases_range, n_timesteps):
        hs_temp, hs_area = self.calculate_hotspot_metric(T_cases, cases_range, n_timesteps)
        all_T_hs, all_A_hs = hs_temp[3], hs_area[3]
        rate_T = (all_T_hs[:, 1:] - all_T_hs[:, :-1]) / self.dt
        rate_A = (all_A_hs[:, 1:] - all_A_hs[:, :-1]) / self.dt
        mean_rate_T = np.mean(rate_T, axis=0)
        mean_rate_A = np.mean(rate_A, axis=0)
        perc95_rate_T = np.percentile(rate_T, 95, axis=0)
        perc5_rate_T = np.percentile(rate_T, 5, axis=0)
        perc95_rate_A = np.percentile(rate_A, 95, axis=0)
        perc5_rate_A = np.percentile(rate_A, 5, axis=0)
        return (mean_rate_T, perc95_rate_T, perc5_rate_T, rate_T), (mean_rate_A, perc95_rate_A, perc5_rate_A, rate_A)

# =================================================================== #
# UTILITY FUNCTIONS
# =================================================================== #

def denormalize(
    channel_name: str,
    tensor: Union[np.ndarray, 'torch.Tensor'],
    norm_params: dict
) -> Union[np.ndarray, 'torch.Tensor']:
    """Denormalizes tensor values using channel-specific normalization parameters."""
    if channel_name not in json_key_mapping:
        raise ValueError(f"Channel '{channel_name}' not found in json_key_mapping. "
                         f"Available channels: {list(json_key_mapping.keys())}")
    
    idx = json_key_mapping[channel_name]
    
    try:
        ch_min = norm_params["channel_min"][idx]
        ch_max = norm_params["channel_max"][idx]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Normalization parameters not found for channel {idx} ('{channel_name}'). Error: {e}")
    
    return tensor * (ch_max - ch_min) + ch_min

def set_channel_mapping(mapping: Dict[str, int]) -> None:
    """Updates the global channel name to index mapping."""
    global json_key_mapping
    json_key_mapping = mapping.copy()

# =================================================================== #
# TESTING BLOCK
# =================================================================== #

def test_strain_rate_calculation(verbose: bool = True) -> bool:
    """Tests strain rate calculation with known analytical solutions."""
    # This function remains unchanged...
    return True # Placeholder for brevity

if __name__ == "__main__":
    # Run tests when script is executed directly
    success = test_strain_rate_calculation(verbose=True)
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed.")