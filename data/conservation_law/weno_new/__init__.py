"""
PyTorch implementation of Weighted Essentially Non-Oscillatory (WENO) schemes
for solving hyperbolic conservation laws.

This package provides a complete PyTorch implementation of WENO schemes
using einops for efficient tensor operations.

Modules:
    weno_3_coeff: Coefficients and beta calculations for 3rd order WENO
    weno_roll: Boundary handling and stencil construction
    weno_scheme: Core WENO algorithm and Euler equations
The standalone data generator calls ``weno_scheme.weno_step`` directly.
"""

from . import weno_3_coeff
from . import weno_roll
from . import weno_scheme

__all__ = [
    'weno_3_coeff',
    'weno_roll', 
    'weno_scheme',
]

__version__ = "1.0.0"
