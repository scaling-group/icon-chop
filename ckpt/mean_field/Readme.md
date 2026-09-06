# MFC Dataset Structure (Mean Field Control)

This document describes the mathematical foundation of the MFC dataset, the randomized definitions used by each generation mode, the input/output (`cond`/`qoi`) combinations, and the corresponding data structures and dimensions.

## 0. Mathematical Formulation

Under the mean-field-control framework, the dataset is based on the following stochastic optimal-control problem for a Fokker--Planck density:

$$ \inf_{\rho, m} \int_0^1 \int_0^1 \frac{c m^2(t,x)}{2 \rho(t,x)} \, dx dt + \int_0^1 g(x)\rho(1,x) \, dx $$

subject to the continuous physical dynamics

$$ \partial_t \rho(t, x) + \nabla_x m(t,x) = \mu \Delta_x \rho(t,x). $$

The setup is as follows:

- **Domain:** $t \in [0,1]$ and $x \in [0,1]$.
- **Spatial boundary:** periodic boundary conditions on the spatial domain.
- **Fixed constants:** quadratic running-cost coefficient $c = 20$ and diffusion coefficient $\mu = 0.02$. The principal sources of variation are the terminal potential operator $g(x)$ and the initial state $\rho(0,x)$.

### Data Generation with Gaussian Processes

To produce diverse continuous functional inputs, both operators are sampled independently in one spatial dimension with Gaussian processes (GPs). To respect the periodic boundary condition, the implementation uses a circular RBF kernel (`rbf_circle_kernel_1d`) with $\sigma = 1.0$ and $l = 1.0$.

- **Sampling the terminal cost $g(x)$:**
  - After drawing a one-dimensional sequence from the GP, its arithmetic mean is subtracted (`g(x) - mean(g)`). This enforces a zero-mean driving field with both positive and negative oscillations.
- **Sampling the initial density $\rho_0(x) = \rho(0,x)$:**
  - Because the density must remain nonnegative, the GP sample is first passed through **Softplus** (`softplus(rho)`), which smoothly maps negative values to positive values.
  - The result is then normalized (`rho / mean(rho)`) to preserve the spatial integral property over the domain, giving unit total mass.

## 1. HDF5 Group Structure

All datasets use HDF5. Each file contains multiple groups named `/eid_{i}_sid_{j}/`.

- **`eid_{i}` (Equation ID):** identifies an operator. In `gparam`, it denotes one unique cost/forcing function $g(x)$; in `rhoparam`, it denotes one unique initial state $\rho_0(x)$. With the default full-scale training configuration (`--eqns 1000`), there are **1,000** operators, corresponding to $i \in [0, 999]$ and 1,000 groups.
- **`sid_{j}` (Seed ID):** identifies a distinct subsample batch for the operator. The number of `sid` entries (`quests`) is usually 1.
- **`num` (samples per operator):** within one operator/group and its fixed environment, `num` distinct physical trajectories are generated. The default is `--num 100`, so **each operator** contains **100** trajectories. The complete dataset therefore contains $1000 \text{ (operators)} \times 100 \text{ (samples per operator)} = \mathbf{100,000}$ physical trajectories.

Each group contains the following five dataset nodes:

- `equation` [String]: a string describing the equation, for example `mfc_gparam_hj_..._forward11`.
- `cond_k` [Float Array]: coordinates of the conditioning input, such as time $t$ and position $x$.
- `cond_v` [Float Array]: physical values of the conditioning input.
- `qoi_k` [Float Array]: coordinates of the quantity-of-interest output.
- `qoi_v` [Float Array]: physical values of the quantity-of-interest output.

---

## 2. Dimensions

With the defaults `NX = 100`, `DT = 0.02`, and `TERMINAL_T = 1.0`, the system evolves for $NT=50$ steps and has 51 temporal snapshots in total:

- Each time slice is a one-dimensional curve with **100** spatial grid points.
- Either half of the temporal interval contains 26 snapshots. For example, $T=0.5$ through $T=1.0$ includes steps 25 through 50, giving a flattened full-field sequence length of $26 \times 100 = \mathbf{2600}$.
- The last dimension of `cond_k` and `qoi_k` is the coordinate dimension: **2** when both $(t,x)$ are present, and **1** when only $x$ is present.

---

## 3. `gparam` Mode: Fixed $g$, Varying $\rho_0$

One $g(x)$ defines an operator/group. The group contains 100 trajectories $\rho(x,t)$ generated from different initial densities $\rho_0(x)$.

### `forward11`

- **Meaning:** given the initial density, predict the terminal density.
- **`cond` (input):** $\rho(x,t=0)$, the initial density.
  - `cond_k` shape: `[100, 100, 2]`, or `[num, 100 spatial points, (t,x) coordinates]`.
  - `cond_v` shape: `[100, 100, 1]`, or `[num, 100 spatial points, physical value]`.
- **`qoi` (output):** $\rho(x,t=T)$, the terminal density.
  - `qoi_k` shape: `[100, 100, 2]`.
  - `qoi_v` shape: `[100, 100, 1]`.

### `forward12`

- **Meaning:** given the initial density, predict the spatiotemporal evolution over the second half of the interval.
- **`cond` (input):** $\rho(x,t=0)$, the initial density.
  - `cond_k` shape: `[100, 100, 2]`.
  - `cond_v` shape: `[100, 100, 1]`.
- **`qoi` (output):** $\rho(x,t \in [T/2,T])$, the full density field over the second half of the interval.
  - `qoi_k` shape: `[100, 2600, 2]`.
  - `qoi_v` shape: `[100, 2600, 1]`.

### `forward22`

- **Meaning:** given the spatiotemporal evolution over the first half of the interval, predict the evolution over the second half.
- **`cond` (input):** $\rho(x,t \in [0,T/2])$, the full density field over the first half of the interval.
  - `cond_k` shape: `[100, 2600, 2]`.
  - `cond_v` shape: `[100, 2600, 1]`.
- **`qoi` (output):** $\rho(x,t \in [T/2,T])$, the full density field over the second half of the interval.
  - `qoi_k` shape: `[100, 2600, 2]`.
  - `qoi_v` shape: `[100, 2600, 1]`.

---

## 4. `rhoparam` Mode: Fixed $\rho_0$, Varying $g$

One initial state $\rho_0(x)$ defines an operator/group. The group contains 100 trajectories $\rho(x,t)$ driven by different forcing functions $g(x)$.

### `forward11`

- **Meaning:** given the cost/forcing function, predict the terminal density.
- **`cond` (input):** $g(x)$, the spatially distributed terminal cost function.
  - `cond_k` shape: `[100, 100, 1]`; only the spatial coordinate $x$ is provided, without $t$.
  - `cond_v` shape: `[100, 100, 1]`; values of the cost function $g$.
- **`qoi` (output):** $\rho(x,t=T)$, the terminal density.
  - `qoi_k` shape: `[100, 100, 2]`, containing $(t,x)$ coordinates.
  - `qoi_v` shape: `[100, 100, 1]`.

### `forward12`

- **Meaning:** given the cost/forcing function, predict the spatiotemporal evolution over the second half of the interval.
- **`cond` (input):** $g(x)$, the spatially distributed terminal cost function.
  - `cond_k` shape: `[100, 100, 1]`.
  - `cond_v` shape: `[100, 100, 1]`.
- **`qoi` (output):** $\rho(x,t \in [T/2,T])$, the full density field over the second half of the interval.
  - `qoi_k` shape: `[100, 2600, 2]`.
  - `qoi_v` shape: `[100, 2600, 1]`.

---
