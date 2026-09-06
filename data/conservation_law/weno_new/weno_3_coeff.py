import torch
from einops import rearrange, repeat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
k = 3
c_rj = torch.tensor([
    [11/6, -7/6, 1/3], 
    [1/3, 5/6, -1/6], 
    [-1/6, 5/6, 1/3], 
    [1/3, -7/6, 11/6], 
], device=device, dtype=torch.float32)

d_r = torch.tensor([3/10, 3/5, 1/10], device=device, dtype=torch.float32)
d_r_t = torch.tensor([1/10, 3/5, 3/10], device=device, dtype=torch.float32)

roll_list = (2, 1, 0, -1, -2)

def get_beta(u_roll):
    """
    u_roll: [..., 5], [2,1,0,-1,-2]
    """
    # Extract the required indices using einops for clarity
    us_0 = u_roll[..., 2]  # u_i
    us_1 = u_roll[..., 3]  # u_{i-1}
    us_2 = u_roll[..., 4]  # u_{i-2}
    us_3 = u_roll[..., 0]  # u_{i+2}
    us_4 = u_roll[..., 1]  # u_{i+1}
    
    # Reorder as [0, -1, -2, 2, 1] -> [u_i, u_{i-1}, u_{i-2}, u_{i+2}, u_{i+1}]
    us = [us_0, us_1, us_2, us_3, us_4]
    
    beta_0 = 13/12 * (us[0] - 2 * us[1] + us[2])**2 + 1/4 * (3 * us[0] - 4 * us[1] + us[2])**2
    beta_1 = 13/12 * (us[-1] - 2 * us[0] + us[1])**2 + 1/4 * (us[-1] - us[1])**2
    beta_2 = 13/12 * (us[-2] - 2 * us[-1] + us[0])**2 + 1/4 * (us[-2] - 4 * us[-1] + 3 * us[0])**2
    return torch.stack([beta_0, beta_1, beta_2], dim=-1)  # [..., 3]

epsilon = 1e-6
