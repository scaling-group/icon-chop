"""
PyTorch-based utility functions for data generation.
Includes Gaussian process generation and kernel functions.
"""

import torch
import numpy as np
from typing import Optional, Callable

def rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, sigma: float, l: float) -> torch.Tensor:
    """
    Radial basis function kernel using PyTorch.
    
    Args:
        x1: First input tensor, shape (n1, d)
        x2: Second input tensor, shape (n2, d)
        sigma: Kernel scale parameter
        l: Length scale parameter
    
    Returns:
        Kernel matrix, shape (n1, n2)
    """
    # Ensure inputs are 2D
    if x1.dim() == 1:
        x1 = x1.unsqueeze(1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(1)
    
    # Compute squared Euclidean distance
    dist = torch.cdist(x1 / l, x2 / l, p=2) ** 2
    return sigma ** 2 * torch.exp(-0.5 * dist)

def rbf_kernel_1d(x1: torch.Tensor, x2: torch.Tensor, sigma: float, l: float) -> torch.Tensor:
    """
    1D RBF kernel using PyTorch broadcasting.
    
    Args:
        x1: First input tensor, shape (n1,)
        x2: Second input tensor, shape (n2,)
        sigma: Kernel scale parameter
        l: Length scale parameter
    
    Returns:
        Kernel matrix, shape (n1, n2)
    """
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')
    sq_norm = (xx1 - xx2) ** 2 / (l ** 2)
    return sigma ** 2 * torch.exp(-0.5 * sq_norm)

def rbf_sin_kernel_1d(x1: torch.Tensor, x2: torch.Tensor, sigma: float, l: float) -> torch.Tensor:
    """
    Sine-based RBF kernel for periodic functions on [0,1].
    
    Args:
        x1: First input tensor, shape (n1,)
        x2: Second input tensor, shape (n2,)
        sigma: Kernel scale parameter
        l: Length scale parameter
    
    Returns:
        Kernel matrix, shape (n1, n2)
    """
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')
    sq_norm = (torch.sin(torch.pi * (xx1 - xx2))) ** 2 / (l ** 2)
    return sigma ** 2 * torch.exp(-0.5 * sq_norm)

def rbf_circle_kernel_1d(x1: torch.Tensor, x2: torch.Tensor, sigma: float, l: float) -> torch.Tensor:
    """
    Circular RBF kernel for functions on [0,1] using trigonometric features.
    
    Args:
        x1: First input tensor, shape (n1,)
        x2: Second input tensor, shape (n2,)
        sigma: Kernel scale parameter
        l: Length scale parameter
    
    Returns:
        Kernel matrix, shape (n1, n2)
    """
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')
    
    # Map to circle using sine and cosine
    xx1_1 = torch.sin(xx1 * 2 * torch.pi)
    xx1_2 = torch.cos(xx1 * 2 * torch.pi)
    xx2_1 = torch.sin(xx2 * 2 * torch.pi)
    xx2_2 = torch.cos(xx2 * 2 * torch.pi)
    
    sq_norm = (xx1_1 - xx2_1) ** 2 / (l ** 2) + (xx1_2 - xx2_2) ** 2 / (l ** 2)
    return sigma ** 2 * torch.exp(-0.5 * sq_norm)

def generate_gaussian_process(
    xs: torch.Tensor,
    num_samples: int,
    kernel: Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor],
    sigma: float,
    length_scale: float,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate Gaussian process samples using PyTorch.
    
    Args:
        xs: points, shape (length,)
        num_samples: Number of samples to generate
        kernel: Kernel function to use
        sigma: Kernel scale parameter
        length_scale: Kernel length scale parameter
        device: Device to use (CPU/GPU)
        seed: Random seed for reproducibility
    
    Returns:
        Gaussian process samples, shape (num_samples, length)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
    
    xs = xs.to(device)
    length = len(xs)

    # Compute covariance matrix
    cov = kernel(xs, xs, sigma, length_scale)

    # Ensure positive definiteness
    cov = cov + 1e-6 * torch.eye(length, device=device)
    
    # Generate samples using multivariate normal
    # Use Cholesky decomposition for better numerical stability
    try:
        L = torch.linalg.cholesky(cov)
        samples = torch.randn(num_samples, length, device=device, dtype=cov.dtype)
        samples = samples @ L.T
    except torch.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition if Cholesky fails
        eigenvals, eigenvecs = torch.linalg.eigh(cov)
        eigenvals = torch.clamp(eigenvals, min=1e-6)
        samples = torch.randn(num_samples, length, device=device)
        samples = samples @ (eigenvecs * torch.sqrt(eigenvals)).T
    
    return samples

def get_kernel_function(kernel_name: str) -> Callable:
    """
    Get kernel function by name.
    
    Args:
        kernel_name: Name of the kernel function
    
    Returns:
        Kernel function
    """
    kernels = {
        'rbf': rbf_kernel_1d,
        'rbf_sin': rbf_sin_kernel_1d,
        'rbf_circle': rbf_circle_kernel_1d
    }
    
    if kernel_name not in kernels:
        raise ValueError(f"Unknown kernel: {kernel_name}. Available: {list(kernels.keys())}")
    
    return kernels[kernel_name]

def generate_random_parameters(
    num_equations: int,
    mode: str,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate random parameters for conservation laws.
    
    Args:
        num_equations: Number of equations
        mode: Parameter generation mode ('random_a_b', 'grid_a_b', 'rlinear_a_b')
        seed: Random seed
        device: Device to use
    
    Returns:
        Parameters tensor, shape (num_equations, 3) for [a, b, c]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
    
    if mode.startswith('random'):
        # Format: random_min_max
        parts = mode.split('_')
        min_val = float(parts[1])
        max_val = float(parts[2])
        
        coeffs_a = torch.empty(num_equations, device=device).uniform_(min_val, max_val)
        coeffs_b = torch.empty(num_equations, device=device).uniform_(min_val, max_val)
        coeffs_c = torch.empty(num_equations, device=device).uniform_(min_val, max_val)
        
    elif mode.startswith('grid'):
        # Format: grid_min_max
        parts = mode.split('_')
        min_val = float(parts[1])
        max_val = float(parts[2])
        
        values = torch.linspace(min_val, max_val, int(num_equations ** (1/3)) + 1)
        grid_a, grid_b, grid_c = torch.meshgrid(values, values, values, indexing='ij')
        coeffs_a = grid_a.flatten()[:num_equations]
        coeffs_b = grid_b.flatten()[:num_equations]
        coeffs_c = grid_c.flatten()[:num_equations]
        
    elif mode.startswith('rlinear'):
        # Format: rlinear_min_max (only c varies)
        parts = mode.split('_')
        min_val = float(parts[1])
        max_val = float(parts[2])
        
        coeffs_a = torch.zeros(num_equations, device=device)
        coeffs_b = torch.zeros(num_equations, device=device)
        coeffs_c = torch.empty(num_equations, device=device).uniform_(min_val, max_val)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return torch.stack([coeffs_a, coeffs_b, coeffs_c], dim=1)  # (num_equations, 3)
