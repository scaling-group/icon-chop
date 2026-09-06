"""
PyTorch-based Mean Field Control solver using Hamilton-Jacobi (HJ) approach.
Ported from the JAX implementation in the ICON reference codebase.

Solves the MFC problem:
  inf_{rho, m} ∫∫ (|m|^2 / rho) dx dt + ∫ g(x) rho(T,x) dx
  s.t.  ∂_t rho + ∇·m = eps * Δ rho,   rho(0,x) = rho_0(x)

via the viscous Hamilton-Jacobi formulation (Hopf-Lax / log-sum-exp).
"""

import torch
import math
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────
#  Utility: periodic interpolation (PyTorch has no built-in)
# ─────────────────────────────────────────────────────────────
def interp_periodic(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor,
                    period: float) -> torch.Tensor:
    """1-D periodic interpolation, matching jnp.interp(..., period=P).

    Args:
        x:  query points  [n]
        xp: knot points   [m]  (assumed sorted, equally spaced over one period)
        fp: values at xp  [m]
        period: period length
    Returns:
        interpolated values [n]
    """
    # Bring x into [xp[0], xp[0]+period)
    x_mod = xp[0] + (x - xp[0]) % period
    # torch.searchsorted gives index of first xp >= x_mod
    idx = torch.searchsorted(xp, x_mod) - 1
    idx = idx.clamp(0, len(xp) - 1)
    idx_next = (idx + 1) % len(xp)
    dx = (xp[1] - xp[0])
    t = (x_mod - xp[idx]) / dx
    return fp[idx] * (1 - t) + fp[idx_next] * t


def interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Simple 1-D linear interpolation (non-periodic).

    Args:
        x:  query points [n]
        xp: knot points  [m]  (sorted)
        fp: values at xp [m]
    Returns:
        interpolated values [n]
    """
    idx = torch.searchsorted(xp, x) - 1
    idx = idx.clamp(0, len(xp) - 2)
    dx = xp[idx + 1] - xp[idx]
    t = (x - xp[idx]) / dx
    return fp[idx] * (1.0 - t) + fp[idx + 1] * t


# ─────────────────────────────────────────────────────────────
#  Viscous HJ solver  (Hopf-Lax via log-sum-exp)
# ─────────────────────────────────────────────────────────────
def viscous_HJ_solver_1d_Riemann(
    g_u: torch.Tensor, u_grids: torch.Tensor,
    x: torch.Tensor, t: torch.Tensor,
    diffusion_eps: float,
    if_u_grids_full_domain: bool = True,
) -> torch.Tensor:
    """Solve viscous Hamilton-Jacobi equation via Hopf-Lax formula.

    phi(x,t) = -eps * log ∫ exp(-1/eps * (g(u) + (x-u)^2/(2t))) / sqrt(2π t eps) du

    Args:
        g_u: [nu] or [n_pts, nu]  – initial condition g at grid points
        u_grids: [nu] or [n_pts, nu]  – grid points of u
        x: [n_pts]
        t: [n_pts]
        diffusion_eps: scalar
        if_u_grids_full_domain: whether u_grids cover the full domain
    Returns:
        phi: [n_pts]
    """
    quad = (x[:, None] - u_grids) ** 2 / 2 / t[:, None]       # [n_pts, nu]
    g_plus_quad = g_u + quad                                     # [n_pts, nu]
    fn_val = -g_plus_quad / diffusion_eps                        # [n_pts, nu]
    fn_max = fn_val.max(dim=1, keepdim=True).values              # [n_pts, 1]

    phi = -diffusion_eps * (
        torch.log(torch.exp(fn_val - fn_max).sum(dim=1)) + fn_max[:, 0]
    )  # [n_pts]

    if if_u_grids_full_domain:
        phi = phi + diffusion_eps * torch.log(
            torch.exp(-quad / diffusion_eps).sum(dim=1)
        )
    else:
        du = u_grids[1] - u_grids[0]
        phi = (phi
               + diffusion_eps * torch.log(2 * math.pi * t * diffusion_eps) / 2
               - diffusion_eps * torch.log(du))
    return phi


def viscous_HJ_solver_1d_Riemann_periodic(
    g_u: torch.Tensor, u_grids: torch.Tensor,
    x: torch.Tensor, t: torch.Tensor,
    diffusion_eps: float,
    half_unroll_num: int = 6,
) -> torch.Tensor:
    """Viscous HJ solver for periodic domain by unrolling g_u.

    Args:
        g_u: [nu]
        u_grids: [nu]
        x: [n_pts]
        t: [n_pts]
        diffusion_eps: scalar
        half_unroll_num: number of periods to unroll on each side
    Returns:
        phi: [n_pts]
    """
    du = u_grids[1] - u_grids[0]
    nu = u_grids.shape[0]
    u_period = du * nu
    unroll_num = 2 * half_unroll_num + 1

    g_u_unroll = g_u.repeat(unroll_num)                          # [nu * unroll_num]
    shifts = torch.arange(-half_unroll_num, half_unroll_num + 1,
                           device=u_grids.device, dtype=u_grids.dtype)
    u_grids_unroll = torch.cat([u_grids + i * u_period for i in shifts])

    phi = viscous_HJ_solver_1d_Riemann(
        g_u_unroll, u_grids_unroll, x, t, diffusion_eps
    )
    return phi


# ─────────────────────────────────────────────────────────────
#  Proximal map & gradient
# ─────────────────────────────────────────────────────────────
def viscous_prox_1d_Riemann(
    g_u: torch.Tensor, u_grids: torch.Tensor,
    x: torch.Tensor, t: torch.Tensor,
    diffusion_eps: float,
) -> torch.Tensor:
    """Proximal map: u_PM = E[u exp(...)]/E[exp(...)].

    Returns:
        u_pm: [n_pts]
    """
    g_plus_quad = g_u + (x[:, None] - u_grids) ** 2 / 2 / t[:, None]
    fn_val = -g_plus_quad / diffusion_eps
    fn_max = fn_val.max(dim=1, keepdim=True).values
    weights = torch.exp(fn_val - fn_max)
    u_pm = (weights * u_grids).sum(dim=1) / weights.sum(dim=1)
    return u_pm


def viscous_HJ_gradx_1d_Riemann(
    g_u: torch.Tensor, u_grids: torch.Tensor,
    x: torch.Tensor, t: torch.Tensor,
    diffusion_eps: float,
) -> torch.Tensor:
    """grad_x phi(x,t) = (x - u_pm) / t."""
    u_pm = viscous_prox_1d_Riemann(g_u, u_grids, x, t, diffusion_eps)
    return (x - u_pm) / t


# ─────────────────────────────────────────────────────────────
#  Extension helpers
# ─────────────────────────────────────────────────────────────
def extend_fn_periodic(fn_val: torch.Tensor, repeat_num: int = 1) -> torch.Tensor:
    """Tile fn_val repeat_num times (periodic extension)."""
    return fn_val.repeat(repeat_num)


def extend_fn_padding(fn_val: torch.Tensor, pad_left: int, pad_right: int,
                       padding_val: float = 0.0) -> torch.Tensor:
    """Pad fn_val with a constant on both sides."""
    return torch.nn.functional.pad(fn_val, (pad_left, pad_right),
                                    mode='constant', value=padding_val)


# ─────────────────────────────────────────────────────────────
#  Finite-difference helpers
# ─────────────────────────────────────────────────────────────
def df_dt(f: torch.Tensor, dt: float) -> torch.Tensor:
    """Time derivative via forward differences.  f: [nx, nt]  →  [nx, nt-1]"""
    return (f[:, 1:] - f[:, :-1]) / dt


def df_dx(f: torch.Tensor, dx: float, periodic: bool = False) -> torch.Tensor:
    """Spatial derivative via central differences.  f: [nx, nt]  →  [nx, nt]"""
    if periodic:
        return (torch.roll(f, -1, 0) - torch.roll(f, 1, 0)) / (2 * dx)
    else:
        ret = (f[2:, :] - f[:-2, :]) / (2 * dx)
        return torch.cat([ret[0:1, :], ret, ret[-1:, :]], dim=0)


def d2f_dx2(f: torch.Tensor, dx: float, periodic: bool = False) -> torch.Tensor:
    """Laplacian via central differences.  f: [nx, nt]  →  [nx, nt]"""
    if periodic:
        return (torch.roll(f, -1, 0) - 2 * f + torch.roll(f, 1, 0)) / (dx ** 2)
    else:
        ret = (f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / (dx ** 2)
        return torch.nn.functional.pad(ret, (0, 0, 1, 1), mode='constant', value=0.0)


# ─────────────────────────────────────────────────────────────
#  MFC solvers
# ─────────────────────────────────────────────────────────────
def solve_mfc_unbdd(
    g_u: torch.Tensor, u_grids: torch.Tensor,
    rho0_y: torch.Tensor, y_grids: torch.Tensor,
    x: torch.Tensor, t: torch.Tensor,
    terminal_time: float, diffusion_eps: float,
    if_u_grids_full_domain: bool = True,
    if_y_grids_full_domain: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve MFC on unbounded domain ℝ.

    Args:
        g_u: [nu]  terminal cost g at grid points
        u_grids: [nu]
        rho0_y: [ny]  initial density at grid points
        y_grids: [ny]
        x: [n_pts]  query x
        t: [n_pts]  query t
        terminal_time: T
        diffusion_eps: ε
    Returns:
        (rho, phi):  each [n_pts]
    """
    # Backward HJ from terminal to y_grids
    T_arr = terminal_time + torch.zeros_like(y_grids)
    phi_bkwd0 = viscous_HJ_solver_1d_Riemann(
        g_u, u_grids, y_grids, T_arr, diffusion_eps, if_u_grids_full_domain
    )

    # Forward HJ
    phi_fwd0 = -diffusion_eps * torch.log(rho0_y) - phi_bkwd0
    phi_fwdt = viscous_HJ_solver_1d_Riemann(
        phi_fwd0, y_grids, x, t, diffusion_eps, if_y_grids_full_domain
    )

    # Backward HJ at query points
    phi_bkwdt = viscous_HJ_solver_1d_Riemann(
        g_u, u_grids, x, terminal_time - t, diffusion_eps, if_u_grids_full_domain
    )
    g_interp = interp1d(x, u_grids, g_u)
    phi_bkwdt = torch.where(t < terminal_time - 1e-6, phi_bkwdt, g_interp)

    rho = torch.exp(-(phi_fwdt + phi_bkwdt) / diffusion_eps)
    rho0_interp = interp1d(x, y_grids, rho0_y)
    rho = torch.where(t > 1e-6, rho, rho0_interp)
    return rho, phi_bkwdt


def get_half_unroll_nums(
    u_grids: torch.Tensor, y_grids: torch.Tensor,
    t: torch.Tensor, terminal_time: float,
    diffusion_eps: float,
) -> Tuple[int, int, int]:
    """Compute safe half-unroll numbers for periodic MFC solver."""
    minimum = 2
    u_span = u_grids[-1] - u_grids[0]
    y_span = y_grids[-1] - y_grids[0]

    h1 = math.sqrt(20 * 2 * diffusion_eps * terminal_time) // u_span + 1
    h2 = math.sqrt(20 * 2 * diffusion_eps * float(t.max())) // y_span + 1
    h3 = math.sqrt(20 * 2 * diffusion_eps * abs(terminal_time - float(t.min()))) // u_span + 1

    return (max(int(h1), minimum), max(int(h2), minimum), max(int(h3), minimum))


def solve_mfc_periodic(
    g_u: torch.Tensor, u_grids: torch.Tensor,
    rho0_y: torch.Tensor, y_grids: torch.Tensor,
    x: torch.Tensor, t: torch.Tensor,
    terminal_time: float, diffusion_eps: float,
    half_unroll_nums: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve MFC on periodic domain [0,1].

    Args:
        g_u: [nu]
        u_grids: [nu]
        rho0_y: [ny]
        y_grids: [ny]
        x: [n_pts]
        t: [n_pts]
        terminal_time: T
        diffusion_eps: ε
        half_unroll_nums: (h1, h2, h3)
    Returns:
        (rho, phi):  each [n_pts]
    """
    period_g = (u_grids[1] - u_grids[0]) * len(u_grids)
    period_rho = (y_grids[1] - y_grids[0]) * len(y_grids)
    h1, h2, h3 = half_unroll_nums

    # Backward HJ at y_grids
    T_arr = terminal_time + torch.zeros_like(y_grids)
    phi_bkwd0 = viscous_HJ_solver_1d_Riemann_periodic(
        g_u, u_grids, y_grids, T_arr, diffusion_eps, h1
    )

    # Forward HJ
    phi_fwd0 = -diffusion_eps * torch.log(rho0_y) - phi_bkwd0
    phi_fwdt = viscous_HJ_solver_1d_Riemann_periodic(
        phi_fwd0, y_grids, x, t, diffusion_eps, h2
    )

    # Backward HJ at query points
    phi_bkwdt = viscous_HJ_solver_1d_Riemann_periodic(
        g_u, u_grids, x, terminal_time - t, diffusion_eps, h3
    )
    g_interp = interp_periodic(x, u_grids, g_u, float(period_g))
    phi_bkwdt = torch.where(t < terminal_time - 1e-6, phi_bkwdt, g_interp)

    rho = torch.exp(-(phi_fwdt + phi_bkwdt) / diffusion_eps)
    rho0_interp = interp_periodic(x, y_grids, rho0_y, float(period_rho))
    rho = torch.where(t > 1e-6, rho, rho0_interp)
    return rho, phi_bkwdt


def solve_mfc_periodic_time_reversal(
    g_u: torch.Tensor, u_grids: torch.Tensor,
    rho1_y: torch.Tensor, y_grids: torch.Tensor,
    x: torch.Tensor, t: torch.Tensor,
    terminal_time: float, diffusion_eps: float,
    half_unroll_nums: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve MFC on periodic domain with time reversal."""
    return solve_mfc_periodic(
        g_u, u_grids, rho1_y, y_grids, x, terminal_time - t,
        terminal_time, diffusion_eps, half_unroll_nums
    )


# ─────────────────────────────────────────────────────────────
#  Batch wrappers
# ─────────────────────────────────────────────────────────────
def solve_mfc_periodic_batch(
    gs: torch.Tensor, u_grids: torch.Tensor,
    rho_0: torch.Tensor, y_grids: torch.Tensor,
    x: torch.Tensor, t: torch.Tensor,
    terminal_time: float, diffusion_eps: float,
    half_unroll_nums: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch version of solve_mfc_periodic.

    Args:
        gs: [batch, nu]
        rho_0: [batch, ny]
        (other args same as solve_mfc_periodic)
    Returns:
        (rho_all, phi_all):  each [batch, n_pts]
    """
    batch = gs.shape[0]
    rho_list, phi_list = [], []
    for i in range(batch):
        rho_i, phi_i = solve_mfc_periodic(
            gs[i], u_grids, rho_0[i], y_grids,
            x, t, terminal_time, diffusion_eps, half_unroll_nums
        )
        rho_list.append(rho_i)
        phi_list.append(phi_i)
    return torch.stack(rho_list), torch.stack(phi_list)


def solve_mfc_periodic_time_reversal_batch(
    gs: torch.Tensor, u_grids: torch.Tensor,
    rho_1: torch.Tensor, y_grids: torch.Tensor,
    x: torch.Tensor, t: torch.Tensor,
    terminal_time: float, diffusion_eps: float,
    half_unroll_nums: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch version of solve_mfc_periodic_time_reversal."""
    batch = gs.shape[0]
    rho_list, phi_list = [], []
    for i in range(batch):
        rho_i, phi_i = solve_mfc_periodic_time_reversal(
            gs[i], u_grids, rho_1[i], y_grids,
            x, t, terminal_time, diffusion_eps, half_unroll_nums
        )
        rho_list.append(rho_i)
        phi_list.append(phi_i)
    return torch.stack(rho_list), torch.stack(phi_list)


# ─────────────────────────────────────────────────────────────
#  Residual and objective computation
# ─────────────────────────────────────────────────────────────
def compute_mfc_residual_obj_periodic(
    gs: torch.Tensor, rho_0: torch.Tensor, us: torch.Tensor,
    xs: torch.Tensor, ts: torch.Tensor,
    terminal_t: float, diffusion_eps: float,
    dx_res: float, dt_res: float,
    half_unroll_nums: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute continuity and HJ residuals + objective value.

    Args:
        gs: [nu]
        rho_0: [nu]
        us: [nu]
        xs, ts: [n_pts]
        terminal_t, diffusion_eps, dx_res, dt_res: scalars
        half_unroll_nums: tuple
    Returns:
        (cont_res, HJ_res, obj_val)
    """
    period = (us[1] - us[0]) * len(us)
    xs_left = torch.where(xs - dx_res >= us[0], xs - dx_res, xs - dx_res + period)
    xs_right = torch.where(xs + dx_res <= us[0] + period, xs + dx_res, xs + dx_res - period)

    rho_mid, phi_mid = solve_mfc_periodic(gs, us, rho_0, us, xs, ts,
                                           terminal_t, diffusion_eps, half_unroll_nums)
    rho_left, phi_left = solve_mfc_periodic(gs, us, rho_0, us, xs_left, ts,
                                             terminal_t, diffusion_eps, half_unroll_nums)
    rho_right, phi_right = solve_mfc_periodic(gs, us, rho_0, us, xs_right, ts,
                                               terminal_t, diffusion_eps, half_unroll_nums)
    rho_up, phi_up = solve_mfc_periodic(gs, us, rho_0, us, xs, ts + dt_res,
                                         terminal_t, diffusion_eps, half_unroll_nums)

    dphidx = (phi_right - phi_left) / (2 * dx_res)
    dphidt = (phi_up - phi_mid) / dt_res
    d2phidx2 = (phi_right - 2 * phi_mid + phi_left) / (dx_res ** 2)
    HJ_res = dphidt - 0.5 * dphidx ** 2 + 0.5 * diffusion_eps * d2phidx2

    vrho_left = -(phi_mid - phi_left) / dx_res * rho_left
    vrho_mid = -(phi_right - phi_mid) / dx_res * rho_mid
    dvrhodx = (vrho_mid - vrho_left) / dx_res
    drhodt = (rho_up - rho_mid) / dt_res
    d2rhodx2 = (rho_right - 2 * rho_mid + rho_left) / (dx_res ** 2)
    cont_res = drhodt + dvrhodx - d2rhodx2 / 2 * diffusion_eps

    obj_val = (rho_mid * dphidx ** 2).sum() / 2
    return cont_res, HJ_res, obj_val


def compute_mfc_residual_obj_periodic_batch(
    gs: torch.Tensor, rho_0: torch.Tensor, us: torch.Tensor,
    xs: torch.Tensor, ts: torch.Tensor,
    terminal_t: float, diffusion_eps: float,
    dx_res: float, dt_res: float,
    half_unroll_nums: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch version: gs [batch, nu], rho_0 [batch, nu]."""
    batch = gs.shape[0]
    cont_list, hj_list, obj_list = [], [], []
    for i in range(batch):
        c, h, o = compute_mfc_residual_obj_periodic(
            gs[i], rho_0[i], us, xs, ts,
            terminal_t, diffusion_eps, dx_res, dt_res, half_unroll_nums
        )
        cont_list.append(c)
        hj_list.append(h)
        obj_list.append(o)
    return torch.stack(cont_list), torch.stack(hj_list), torch.stack(obj_list)

