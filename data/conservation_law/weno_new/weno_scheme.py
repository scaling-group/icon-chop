import torch
import numpy as np
from einops import rearrange, repeat
from . import weno_3_coeff as coeff

def get_euler_eigen_vector(u, c, H, gamma):
    """
    each of size [...]
    right eigen vectors of Euler equations, R
    [3, 3, ...]
         _                    _ 
        |                      |
        |   1      1       1   |
        |                      |
    R = |  u-c     u      u+c  |
        |                      |
        |  H-uc   u^2/2   H+uc |
        |_                    _|
    """
    # Create eigenvectors using einops for broadcasting
    v1 = torch.stack([torch.ones_like(u), torch.ones_like(u), torch.ones_like(u)], dim=0)  # [3, ...]
    v2 = torch.stack([u - c, u, u + c], dim=0)  # [3, ...]
    v3 = torch.stack([H - u * c, 0.5 * u**2, H + u * c], dim=0)  # [3, ...]
    R = torch.stack([v1, v2, v3], dim=0)  # [3, 3, ...]
    
    # left eigen vectors of Euler equations, R^{-1}
    # [3, 3, ...]
    #                          _                                       _ 
    #                         |                                         |
    #                         |  uc/(gamma-1)+u^2/2  -c/(gamma-1)-u   1 |
    #                         |                                         |
    # R^{-1}=(gamma-1)/(2c^2)*|  2(H-u^2)             2u             -2 |
    #                         |                                         |
    #                         | -uc/(gamma-1)+u^2/2   c/(gamma-1)-u   1 |
    #                         |_                                       _|
    l1 = torch.stack([u * c / (gamma-1) + 0.5 * u**2,  -c / (gamma-1) - u,  torch.ones_like(u)], dim=0)  # [3, ...]
    l2 = torch.stack([2 * (H - u**2),                  2 * u,               -2 * torch.ones_like(u)], dim=0)  # [3, ...]
    l3 = torch.stack([-u * c / (gamma-1) + 0.5 * u**2, c / (gamma-1) - u,   torch.ones_like(u)], dim=0)  # [3, ...]
    R_inv = (gamma-1)/(2*c**2) * torch.stack([l1, l2, l3], dim=0)  # [3, 3, ...]
    
    return R, R_inv  # [3, 3, ...]

def get_euler_properties(U, gamma):
    """
    U: [..., 3], with or without batch
    """
    rho, m, E = U[..., 0], U[..., 1], U[..., 2]
    u = m / rho  # [...]
    p = (gamma - 1) * (E - 0.5 * m * u)
    c = torch.sqrt(gamma * p / rho)
    H = (E + p) / rho
    eigenvalues = torch.stack([u - c, u, u + c], dim=0)  # [3, ...]
    return rho, u, p, c, H, eigenvalues

def get_scalar_alpha(u, grad_fn, gs, redundancy):
    umin, umax = torch.min(u), torch.max(u)
    u_range = torch.linspace(umin, umax, gs, device=u.device)
    grad_fn_u = grad_fn(u_range)
    alpha = torch.max(torch.abs(grad_fn_u))
    return alpha * (1 + redundancy)

def get_scalar_alpha_batch(u_batch, grad_fn, gs, redundancy):
    return torch.stack([
        get_scalar_alpha(u, grad_fn, gs, redundancy) for u in u_batch
    ])

def get_tv(u, left_bound, right_bound, boundary_type):
    if boundary_type == "dirichlet":
        left = torch.cat([left_bound[None, :], u], dim=0)
        right = torch.cat([u, right_bound[None, :]], dim=0)
        return torch.sum(torch.abs(left - right))
    elif boundary_type == "periodic":
        return torch.sum(torch.abs(u - torch.roll(u, 1, dims=0)))
    else:
        raise NotImplementedError

def flux_Lax_Friedrichs(f, alpha, a, b):
    h = 0.5 * (f(a) + f(b) - alpha[:, None, None] * (b - a))
    return h

def get_v_r_rhalf(u_roll, r):
    """
    u_roll: [usize + 2, udim, 5]
    v^{(r)}_{i+1/2}, LHS of 2.51
    """
    # Use einsum for tensor contraction
    coeff_slice = coeff.c_rj[r+1, :]
    v_r = torch.einsum('bjki,i->bjk', u_roll[:, :, :, coeff.k-1-r:2*coeff.k-1-r], coeff_slice)
    return v_r

def get_v_r_lhalf(u_roll, r):
    """
    v^{(r)}_{i-1/2}
    """
    coeff_slice = coeff.c_rj[r, :]
    v_r = torch.einsum('bjki,i->bjk', u_roll[:, :, :, coeff.k-1-r:2*coeff.k-1-r], coeff_slice)
    return v_r

def get_w_d(u_roll):
    """
    u_roll: [usize + 2, udim, 5]
    """
    usize, udim, _ = u_roll.shape
    w_r = repeat(coeff.d_r, 'i -> usize udim i', usize=usize, udim=udim)
    w_r_t = repeat(coeff.d_r_t, 'i -> usize udim i', usize=usize, udim=udim)
    return w_r, w_r_t

def get_w_classic(u_roll):
    """
    u_roll: [usize + 2, udim, 5]
    """
    beta = coeff.get_beta(u_roll)  # (usize + 2, udim, 3)
    alpha_r = coeff.d_r[None, None, None, :] / (coeff.epsilon + beta)**2  # (usize + 2, udim, 3)
    w_r = alpha_r / torch.sum(alpha_r, dim=-1, keepdim=True)  # (usize + 2, udim, 3)
    
    alpha_r_t = coeff.d_r_t[None, None, None, :] / (coeff.epsilon + beta)**2  # (usize + 2, udim, 3)
    w_r_t = alpha_r_t / torch.sum(alpha_r_t, dim=-1, keepdim=True)  # (usize + 2, udim, 3)
    return w_r, w_r_t

def get_v_candidates(u_roll):
    """
    get the candidates for weighted summation
    return [usize + 2, udim, 3], two ghost points
    """
    v_minus_rhalf_c = torch.stack([get_v_r_rhalf(u_roll, r) for r in range(coeff.k)], dim=-1)  # candidates for v^-_{i+1/2}
    v_plus_lhalf_c = torch.stack([get_v_r_lhalf(u_roll, r) for r in range(coeff.k)], dim=-1)  # candidates for v^+_{i-1/2}
    
    return v_minus_rhalf_c, v_plus_lhalf_c

def get_v(u_roll, w_r, w_r_t):
    """
    w_r, w_r_t: [usize + 2, udim, 3]
    output: [usize + 2, udim]
    """
    v_minus_rhalf_c, v_plus_lhalf_c = get_v_candidates(u_roll)  # [usize + 2, udim, 3]
    v_minus_rhalf_all = torch.einsum("bijk,bijk->bij", v_minus_rhalf_c, w_r)  # v^-_{i+1/2}
    v_plus_lhalf_all = torch.einsum("bijk,bijk->bij", v_plus_lhalf_c, w_r_t)  # v^+_{i-1/2}
    return v_minus_rhalf_all, v_plus_lhalf_all  # [usize + 2, udim]

def get_flux(f, alpha, u_roll, w_r, w_r_t):
    v_minus_rhalf_all, v_plus_lhalf_all = get_v(u_roll, w_r, w_r_t)  # [usize + 2, udim]
    
    v_minus_lhalf = v_minus_rhalf_all[:, :-2, :]
    v_minus_rhalf = v_minus_rhalf_all[:, 1:-1, :]
    v_plus_lhalf = v_plus_lhalf_all[:, 1:-1, :]
    v_plus_rhalf = v_plus_lhalf_all[:, 2:, :]
    
    f_rhalf = flux_Lax_Friedrichs(f, alpha, v_minus_rhalf, v_plus_rhalf)  # [usize, udim]
    f_lhalf = flux_Lax_Friedrichs(f, alpha, v_minus_lhalf, v_plus_lhalf)
    return f_rhalf, f_lhalf

def get_rhs(f, alpha, u_roll, dx, w_r, w_r_t):
    f_rhalf, f_lhalf = get_flux(f, alpha, u_roll, w_r, w_r_t)  # [usize, udim]
    rhs = - (f_rhalf - f_lhalf) / dx  # [usize, udim]
    return rhs

def weno_reconstruct(u, get_w, get_u_roll, left_bound, right_bound):
    this_u_roll = get_u_roll(u, left_bound, right_bound, coeff.roll_list)  # [usize + 2, udim, 5]
    w_r, w_r_t = get_w(this_u_roll)  # [usize + 2, udim, 3]
    v_minus_rhalf_all, v_plus_lhalf_all = get_v(this_u_roll, w_r, w_r_t)  # [usize, udim]
    return v_minus_rhalf_all, v_plus_lhalf_all

def weno_step(dt, dx, u, get_w, get_u_roll, f, alpha, scheme, left_bound, right_bound):
    def get_current_rhs(this_u):
        this_u_roll = get_u_roll(this_u, left_bound, right_bound, coeff.roll_list)  # [usize + 2, udim, 5]
        w_r, w_r_t = get_w(this_u_roll)  # [usize + 2, udim, 3]
        rhs = get_rhs(f, alpha, this_u_roll, dx, w_r, w_r_t)  # [usize, udim]
        return rhs
    
    if scheme == "euler":
        rhs = get_current_rhs(u)
        new_u = u + dt * rhs
        return new_u
    elif scheme == "rk3":
        k1 = get_current_rhs(u)
        k2 = get_current_rhs(u + 0.5 * dt * k1)
        k3 = get_current_rhs(u - dt * k1 + 2 * dt * k2)
        new_u = u + dt/6 * (k1 + 4 * k2 + k3)
        return new_u
    elif scheme == "rk4":
        k1 = get_current_rhs(u)
        k2 = get_current_rhs(u + 0.5 * dt * k1)
        k3 = get_current_rhs(u + 0.5 * dt * k2)
        k4 = get_current_rhs(u + dt * k3)
        new_u = u + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_u
    else:
        raise NotImplementedError

def get_euler_eigen_on_interface(U_l, U_r, gamma, mode="roe"):
    # ul, ur: [usize, udim]
    if mode == "simple":
        mean = 0.5 * (U_l + U_r)
        rho, u, p, c, H, eigenvalues = get_euler_properties(mean, gamma)
        R, R_inv = get_euler_eigen_vector(u, c, H, gamma)
        return eigenvalues, R, R_inv
    
    elif mode == "roe":
        rho_l, u_l, p_l, c_l, H_l, eigenvalues_l = get_euler_properties(U_l, gamma)
        rho_r, u_r, p_r, c_r, H_r, eigenvalues_r = get_euler_properties(U_r, gamma)
        
        # Roe average
        sqrt_rho_l = torch.sqrt(rho_l)
        sqrt_rho_r = torch.sqrt(rho_r)
        
        u_mean = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) / (sqrt_rho_l + sqrt_rho_r)
        H_mean = (sqrt_rho_l * H_l + sqrt_rho_r * H_r) / (sqrt_rho_l + sqrt_rho_r)
        c_mean = torch.sqrt((gamma - 1) * (H_mean - 0.5 * u_mean**2))
        eigenvalues = torch.stack([u_mean - c_mean, u_mean, u_mean + c_mean], dim=0)  # [3, usize]
        R, R_inv = get_euler_eigen_vector(u_mean, c_mean, H_mean, gamma)
        return eigenvalues, R, R_inv
    else:
        raise NotImplementedError

def weno_step_euler(dt, dx, u, get_w, get_u_roll, gamma, f, alpha, scheme, left_bound, right_bound):
    average_mode = 'roe'
    
    def get_current_rhs(this_u):
        this_u_roll = get_u_roll(this_u, left_bound, right_bound, coeff.roll_list)  # [usize + 2, udim, 5]
        
        # Right eigenvalues and vectors
        right_ev, right_R, right_R_inv = get_euler_eigen_on_interface(
            this_u_roll[:, :, 2], this_u_roll[:, :, 3], gamma, average_mode)  # [3, 3, usize + 2]
        
        # Transform to characteristic variables
        char_roll = torch.einsum("ijk,kjn->kin", right_R_inv, this_u_roll)  # [usize + 2, udim, 5]
        w_r, w_r_t = get_w(char_roll)  # [usize + 2, udim, 3]
        char_minus_rhalf_all, _ = get_v(char_roll, w_r, w_r_t)  # [usize + 2, udim]
        v_minus_rhalf_all = torch.einsum("ijk,kj->ki", right_R, char_minus_rhalf_all)  # [usize + 2, udim]
        
        # Left eigenvalues and vectors
        left_ev, left_R, left_R_inv = get_euler_eigen_on_interface(
            this_u_roll[:, :, 1], this_u_roll[:, :, 2], gamma, average_mode)  # [3, 3, usize + 2]
        
        char_roll = torch.einsum("ijk,kjn->kin", left_R_inv, this_u_roll)  # [usize + 2, udim, 5]
        w_r, w_r_t = get_w(char_roll)  # [usize + 2, udim, 3]
        _, char_plus_lhalf_all = get_v(char_roll, w_r, w_r_t)  # [usize + 2, udim]
        v_plus_lhalf_all = torch.einsum("ijk,kj->ki", left_R, char_plus_lhalf_all)  # [usize + 2, udim]
        
        # Extract interface values
        v_minus_lhalf = v_minus_rhalf_all[:-2, :]  # [usize, udim]
        v_minus_rhalf = v_minus_rhalf_all[1:-1, :]
        v_plus_lhalf = v_plus_lhalf_all[1:-1, :]
        v_plus_rhalf = v_plus_lhalf_all[2:, :]
        
        char_minus_lhalf = char_minus_rhalf_all[:-2, :]  # [usize, udim]
        char_minus_rhalf = char_minus_rhalf_all[1:-1, :]
        char_plus_lhalf = char_plus_lhalf_all[1:-1, :]
        char_plus_rhalf = char_plus_lhalf_all[2:, :]
        
        # Compute flux differences
        ev_abs_max = torch.max(torch.abs(right_ev[:, :-1]), dim=-1)[0]  # [3]
        
        du_rhalf = torch.einsum("ijk,kj->ki", right_R[:, :, 1:-1], ev_abs_max * (char_plus_rhalf - char_minus_rhalf))
        f_rhalf = 0.5 * (f(v_minus_rhalf) + f(v_plus_rhalf) - du_rhalf)  # [usize, udim]
        
        du_lhalf = torch.einsum("ijk,kj->ki", left_R[:, :, 1:-1], ev_abs_max * (char_plus_lhalf - char_minus_lhalf))
        f_lhalf = 0.5 * (f(v_minus_lhalf) + f(v_plus_lhalf) - du_lhalf)
        
        rhs = - (f_rhalf - f_lhalf) / dx  # [usize, udim]
        return rhs
    
    if scheme == "euler":
        rhs = get_current_rhs(u)
        new_u = u + dt * rhs
        return new_u
    elif scheme == "rk3":
        k1 = get_current_rhs(u)
        k2 = get_current_rhs(u + 0.5 * dt * k1)
        k3 = get_current_rhs(u - dt * k1 + 2 * dt * k2)
        new_u = u + dt/6 * (k1 + 4 * k2 + k3)
        return new_u
    elif scheme == "rk4":
        k1 = get_current_rhs(u)
        k2 = get_current_rhs(u + 0.5 * dt * k1)
        k3 = get_current_rhs(u + 0.5 * dt * k2)
        k4 = get_current_rhs(u + dt * k3)
        new_u = u + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_u
    else:
        raise NotImplementedError
