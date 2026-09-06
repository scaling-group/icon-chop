"""
MFC training data generation using HJ solver (PyTorch).
Saves to HDF5 format, compatible with WenoDataset.

H5 structure (matching WenoDataset):
  /eid_{i}_sid_{j}/
    equation:  string
    cond_k:    [num, cond_len, cond_k_dim]
    cond_v:    [num, cond_len, 1]
    qoi_k:     [num, qoi_len, qoi_k_dim]
    qoi_v:     [num, qoi_len, 1]

Each group = one operator (fixed g or fixed rho_0).
Inside each group, `num` samples share the same operator.
WenoDataset randomly picks demo_num demos + 1 quest from these num samples.

Reference settings (from datagen.sh):
  length=100, dx=0.01, dt=0.02, nu_nx_ratio=1
  run_cost=20.0, terminal_t=1.0, diffusion_eps=0.04
  Train: eqns=1000, quests=1, num=100
  Test:  eqns=100,  quests=5, num=100

Usage:
    cd mean_filed
    # Small validation
    python generate_mfc.py --mode both --eqns 5 --num 10 --name test_mfc
    # Full training (matches ICON reference)
    python generate_mfc.py --mode gparam --eqns 1000 --num 100 --name train --seed 8
    
    # OOD Operator Test
    python generate_mfc.py --mode both --name test_ood_operator --ood_target operator --eqns 100 --num 10
    python generate_mfc.py --mode rhoparam --eqns 1000 --num 100 --name train --seed 9
"""

import os
import sys
import argparse
import time
import h5py
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_utils
import mfc_hj

torch.set_default_dtype(torch.float64)

# ═══════════════════════════════════════════════════════
#  Fixed physical parameters (from ICON reference)
# ═══════════════════════════════════════════════════════
NX = 100
DT_MFC = 0.02
TERMINAL_T = 1.0
NT = int(TERMINAL_T / DT_MFC)  # = 50
DIFFUSION_EPS = 0.04
RUN_COST = 20.0
NU_NX_RATIO = 1
GP_K_SIGMA = 1.0
GP_K_L = 1.0


def build_grids(device):
    """Build (t, x) query grids on the specified device."""
    nu = NU_NX_RATIO * NX
    xs = torch.arange(NX, dtype=torch.float64, device=device) / NX
    us = torch.arange(nu, dtype=torch.float64, device=device) / nu
    ts_1d = torch.linspace(0, TERMINAL_T, NT + 1, device=device)

    # Flatten: order is (t0,x0),...,(t0,xN-1),(t1,x0),...
    xs_flat = xs.unsqueeze(0).expand(NT + 1, NX).reshape(-1)
    ts_flat = ts_1d.unsqueeze(1).expand(NT + 1, NX).reshape(-1)

    # (t, x) coordinate grid
    txs_grid = torch.stack([ts_flat, xs_flat], dim=-1)  # [(NT+1)*NX, 2]

    half_unroll_nums = mfc_hj.get_half_unroll_nums(us, us, ts_flat, TERMINAL_T, DIFFUSION_EPS)
    return xs, us, ts_1d, xs_flat, ts_flat, txs_grid, half_unroll_nums


def generate_gparam(args):
    """gparam: g(x) defines operator, rho_0(x) varies.

    Output files (one h5 per problem type):
      forward11: cond=rho(t=0),       qoi=rho(t=T)
      forward12: cond=rho(t=0),       qoi=rho(t in [T/2,T])
      forward22: cond=rho(t in [0,T/2]), qoi=rho(t in [T/2,T])
    """
    device = args.device
    xs, us, ts_1d, xs_flat, ts_flat, txs_grid, half_unroll_nums = build_grids(device)
    nu = len(us)
    kernel_fn = data_utils.rbf_circle_kernel_1d
    first_half_t = (NT + 1) // 2

    # txs_grid expanded: [num, (NT+1)*NX, 2]
    txs_batch = txs_grid.unsqueeze(0).expand(args.num, -1, -1).cpu().numpy().astype(np.float32)

    # OOD Operator logic for gparam
    l_scale = getattr(args, "ood_l_scale", 0.5)
    l_g = l_scale if args.ood_target == "operator" else GP_K_L
    l_rho = l_scale if args.ood_target == "condition" else GP_K_L

    # Sample all g(x) at once
    torch.manual_seed(args.seed)
    all_gs = data_utils.generate_gaussian_process(
        us.cpu(), args.eqns, kernel_fn, GP_K_SIGMA, l_g,
        seed=args.seed, device=torch.device('cpu')
    ).to(device)
    all_gs = all_gs - all_gs.mean(dim=-1, keepdim=True)

    os.makedirs(args.dir, exist_ok=True)
    file_split = args.file_split
    ptypes = args.ptype.split(",") if getattr(args, "ptype", "") else ["forward11", "forward12", "forward22"]

    rng_counter = 0
    total_time = 0.0
    h5_files = {}
    file_idx = 0

    for eq_idx in range(args.eqns):
        # Open new files at split boundaries
        if eq_idx % file_split == 0:
            # Close previous files
            for ptype, f in h5_files.items():
                f.close()
            file_idx = eq_idx // file_split + 1
            h5_files = {}
            for ptype in ptypes:
                fname = os.path.join(args.dir, f"{args.name}_mfc_gparam_{ptype}_{file_idx}.h5")
                h5_files[ptype] = h5py.File(fname, 'w')

        g = all_gs[eq_idx]
        g_scaled = g / RUN_COST

        for q_idx in range(args.quests):
            rng_counter += 1
            rho_seed = args.seed * 10000 + rng_counter

            # Sample num initial densities
            init_rho = data_utils.generate_gaussian_process(
                us.cpu(), args.num, kernel_fn, GP_K_SIGMA, l_rho,
                seed=rho_seed, device=torch.device('cpu')
            ).to(device)
            init_rho = torch.nn.functional.softplus(init_rho)
            init_rho = init_rho / init_rho.mean(dim=-1, keepdim=True)

            g_batch = g_scaled.unsqueeze(0).expand(args.num, -1)

            t0 = time.time()
            rhos, _ = mfc_hj.solve_mfc_periodic_batch(
                g_batch, us, init_rho, us,
                xs_flat, ts_flat, TERMINAL_T, DIFFUSION_EPS,
                half_unroll_nums
            )  # [num, (NT+1)*NX]
            total_time += time.time() - t0

            rhos_np = rhos.unsqueeze(-1).cpu().numpy().astype(np.float32)  # [num, (NT+1)*NX, 1]

            # Build group name
            group_name = f"eid_{eq_idx}_sid_{q_idx}"
            param_str = "_".join(f"{g[k*nu//10].item():.4f}" for k in range(10))
            equation_base = f"mfc_gparam_hj_{param_str}"

            # Write each problem type
            for ptype in ptypes:
                if ptype == "forward11":
                    cond_k = txs_batch[:, :NX, :]
                    cond_v = rhos_np[:, :NX, :]
                    qoi_k = txs_batch[:, -NX:, :]
                    qoi_v = rhos_np[:, -NX:, :]
                elif ptype == "forward12":
                    cond_k = txs_batch[:, :NX, :]
                    cond_v = rhos_np[:, :NX, :]
                    qoi_k = txs_batch[:, first_half_t*NX:, :]
                    qoi_v = rhos_np[:, first_half_t*NX:, :]
                elif ptype == "forward22":
                    cond_k = txs_batch[:, :first_half_t*NX, :]
                    cond_v = rhos_np[:, :first_half_t*NX, :]
                    qoi_k = txs_batch[:, first_half_t*NX:, :]
                    qoi_v = rhos_np[:, first_half_t*NX:, :]

                grp = h5_files[ptype].create_group(group_name)
                grp.create_dataset('equation', data=np.bytes_(f"{equation_base}_{ptype}"))
                grp.create_dataset('cond_k', data=cond_k, compression='gzip', compression_opts=1)
                grp.create_dataset('cond_v', data=cond_v, compression='gzip', compression_opts=1)
                grp.create_dataset('qoi_k', data=qoi_k, compression='gzip', compression_opts=1)
                grp.create_dataset('qoi_v', data=qoi_v, compression='gzip', compression_opts=1)
                h5_files[ptype].flush()

        if (eq_idx + 1) % max(1, args.eqns // 10) == 0 or eq_idx == 0:
            print(f"  gparam [{eq_idx+1}/{args.eqns}] done, "
                  f"time={total_time:.1f}s", flush=True)

    for ptype, f in h5_files.items():
        f.close()
    print(f"  Saved {file_idx} file(s) per problem type, "
          f"{file_split} groups/file, to {args.dir}")


def generate_rhoparam(args):
    """rhoparam: rho_0(x) defines operator, g(x) varies.

    Output files:
      forward11: cond=g(x),  qoi=rho(t=T)
      forward12: cond=g(x),  qoi=rho(t in [T/2,T])
    """
    device = args.device
    xs, us, ts_1d, xs_flat, ts_flat, txs_grid, half_unroll_nums = build_grids(device)
    nu = len(us)
    kernel_fn = data_utils.rbf_circle_kernel_1d
    first_half_t = (NT + 1) // 2

    txs_batch = txs_grid.unsqueeze(0).expand(args.num, -1, -1).cpu().numpy().astype(np.float32)
    xs_key = xs.cpu().reshape(-1, 1).unsqueeze(0).expand(args.num, -1, -1).numpy().astype(np.float32)
    # [num, NX, 1]

    # OOD Operator logic for rhoparam
    l_scale = getattr(args, "ood_l_scale", 0.5)
    l_rho = l_scale if args.ood_target == "operator" else GP_K_L
    l_g = l_scale if args.ood_target == "condition" else GP_K_L

    torch.manual_seed(args.seed)
    all_init_rhos = data_utils.generate_gaussian_process(
        us.cpu(), args.eqns, kernel_fn, GP_K_SIGMA, l_rho,
        seed=args.seed, device=torch.device('cpu')
    ).to(device)
    all_init_rhos = torch.nn.functional.softplus(all_init_rhos)
    all_init_rhos = all_init_rhos / all_init_rhos.mean(dim=-1, keepdim=True)

    os.makedirs(args.dir, exist_ok=True)
    file_split = args.file_split
    ptypes = args.ptype.split(",") if getattr(args, "ptype", "") else ["forward11", "forward12"]

    rng_counter = 0
    total_time = 0.0
    h5_files = {}
    file_idx = 0

    for eq_idx in range(args.eqns):
        # Open new files at split boundaries
        if eq_idx % file_split == 0:
            for ptype, f in h5_files.items():
                f.close()
            file_idx = eq_idx // file_split + 1
            h5_files = {}
            for ptype in ptypes:
                fname = os.path.join(args.dir, f"{args.name}_mfc_rhoparam_{ptype}_{file_idx}.h5")
                h5_files[ptype] = h5py.File(fname, 'w')

        init_rho = all_init_rhos[eq_idx]

        for q_idx in range(args.quests):
            rng_counter += 1
            g_seed = args.seed * 10000 + rng_counter

            g_batch = data_utils.generate_gaussian_process(
                us.cpu(), args.num, kernel_fn, GP_K_SIGMA, l_g,
                seed=g_seed, device=torch.device('cpu')
            ).to(device)
            g_batch = g_batch - g_batch.mean(dim=-1, keepdim=True)

            init_rho_batch = init_rho.unsqueeze(0).expand(args.num, -1)

            t0 = time.time()
            rhos, _ = mfc_hj.solve_mfc_periodic_batch(
                g_batch / RUN_COST, us, init_rho_batch, us,
                xs_flat, ts_flat, TERMINAL_T, DIFFUSION_EPS,
                half_unroll_nums
            )
            total_time += time.time() - t0

            rhos_np = rhos.unsqueeze(-1).cpu().numpy().astype(np.float32)
            g_on_xs = g_batch[:, ::NU_NX_RATIO].unsqueeze(-1).cpu().numpy().astype(np.float32)
            # [num, NX, 1]

            group_name = f"eid_{eq_idx}_sid_{q_idx}"
            param_str = "_".join(f"{init_rho[k*nu//10].item():.4f}" for k in range(10))
            equation_base = f"mfc_rhoparam_hj_{param_str}"

            for ptype in ptypes:
                cond_k = xs_key        # g coordinates [num, NX, 1]
                cond_v = g_on_xs       # g values [num, NX, 1]
                if ptype == "forward11":
                    qoi_k = txs_batch[:, -NX:, :]
                    qoi_v = rhos_np[:, -NX:, :]
                elif ptype == "forward12":
                    qoi_k = txs_batch[:, first_half_t*NX:, :]
                    qoi_v = rhos_np[:, first_half_t*NX:, :]

                grp = h5_files[ptype].create_group(group_name)
                grp.create_dataset('equation', data=np.bytes_(f"{equation_base}_{ptype}"))
                grp.create_dataset('cond_k', data=cond_k, compression='gzip', compression_opts=1)
                grp.create_dataset('cond_v', data=cond_v, compression='gzip', compression_opts=1)
                grp.create_dataset('qoi_k', data=qoi_k, compression='gzip', compression_opts=1)
                grp.create_dataset('qoi_v', data=qoi_v, compression='gzip', compression_opts=1)
                h5_files[ptype].flush()

        if (eq_idx + 1) % max(1, args.eqns // 10) == 0 or eq_idx == 0:
            print(f"  rhoparam [{eq_idx+1}/{args.eqns}] done, "
                  f"time={total_time:.1f}s", flush=True)

    for ptype, f in h5_files.items():
        f.close()
    print(f"  Saved {file_idx} file(s) per problem type, "
          f"{file_split} groups/file, to {args.dir}")


def intended_output_paths(args):
    """List all HDF5 chunks and reject unsupported problem-type requests."""
    mode_ptypes = {
        "gparam": ["forward11", "forward12", "forward22"],
        "rhoparam": ["forward11", "forward12"],
    }
    modes = [args.mode] if args.mode != "both" else ["gparam", "rhoparam"]
    chunk_count = (args.eqns + args.file_split - 1) // args.file_split
    paths = []
    for mode in modes:
        ptypes = args.ptype.split(",") if args.ptype else mode_ptypes[mode]
        unsupported = sorted(set(ptypes) - set(mode_ptypes[mode]))
        if unsupported:
            raise ValueError(f"Unsupported {mode} problem type(s): {unsupported}")
        for ptype in ptypes:
            for file_idx in range(1, chunk_count + 1):
                paths.append(os.path.join(
                    args.dir,
                    f"{args.name}_mfc_{mode}_{ptype}_{file_idx}.h5",
                ))
    return paths


def smoke_check(device):
    xs, us, _, xs_flat, ts_flat, _, half_unroll_nums = build_grids(device)
    del xs
    g = torch.zeros((1, NX), dtype=torch.float64, device=device)
    rho0 = torch.ones((1, NX), dtype=torch.float64, device=device)
    rho, phi = mfc_hj.solve_mfc_periodic_batch(
        g, us, rho0, us, xs_flat, ts_flat,
        TERMINAL_T, DIFFUSION_EPS, half_unroll_nums
    )
    if tuple(rho.shape) != (1, (NT + 1) * NX):
        raise RuntimeError(f"Unexpected smoke-test shape: {tuple(rho.shape)}")
    if not torch.isfinite(rho).all() or not torch.isfinite(phi).all():
        raise RuntimeError("MFC smoke check produced non-finite values")
    print("MFC generator self-check passed")


def main():
    parser = argparse.ArgumentParser(description="MFC data generation (HDF5)")
    parser.add_argument("--mode", type=str, default="gparam",
                        choices=["gparam", "rhoparam", "both"])
    parser.add_argument("--eqns", type=int, default=1000)
    parser.add_argument("--quests", type=int, default=1)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--file_split", type=int, default=100,
                        help="Number of equations per h5 file")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "generated"))
    parser.add_argument("--ood_target", type=str, default="",
                        choices=["", "operator", "condition"],
                        help="Target for OOD generation. Modifies GP length-scale.")
    parser.add_argument("--ood_l_scale", type=float, default=0.5,
                        help="Length scale used for OOD target.")
    parser.add_argument("--ptype", type=str, default="",
                        help="Comma separated problem types to generate (e.g., forward22)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--check-only", action="store_true",
                        help="Run a no-output solver self-check and exit")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.device = torch.device(args.device)

    if args.check_only:
        smoke_check(args.device)
        return

    if args.eqns <= 0 or args.quests <= 0 or args.num <= 0 or args.file_split <= 0:
        raise ValueError("eqns, quests, num and file_split must be positive")
    collisions = [path for path in intended_output_paths(args) if os.path.exists(path)]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing output(s):\n" + "\n".join(collisions)
        )

    print("=" * 60)
    print(f"MFC Data Generation (HDF5) — mode={args.mode}, device={args.device}")
    print(f"  eqns={args.eqns}, quests={args.quests}, num={args.num}")
    print(f"  nx={NX}, nt={NT}, T={TERMINAL_T}, eps={DIFFUSION_EPS}")
    n_groups = args.eqns * args.quests
    print(f"  Groups (operators): {n_groups}, samples/group: {args.num}")
    print("=" * 60)

    t_start = time.time()

    if args.mode in ("gparam", "both"):
        print("\n[gparam] Generating...")
        generate_gparam(args)

    if args.mode in ("rhoparam", "both"):
        print("\n[rhoparam] Generating...")
        if args.mode == "both":
            args.seed += 1
        generate_rhoparam(args)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
