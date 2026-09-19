# cpTC: Tensor-Core-Accelerated Sparse Tensor CP Decomposition

cpTC accelerates large-scale sparse tensor CANDECOMP/PARAFAC (CP)
decomposition on NVIDIA GPUs by splitting the tensor into dense blocks
(processed by Tensor Cores via WMMA TF32) and residual values (processed
by FP64 CUDA cores), with a fill-rate-adaptive path-selection rule.

Beyond the original order-3 path, the repository also contains a generic
N-order CP-ALS path (`src/generic.cu`, orders 3-5) that applies the same
hybrid idea at the granularity of 16^3 *dense units* over the three inner
modes, with per-unit density selection and tensor reordering.

## Supported GPUs

Works on any sm_89 (Ada) GPU: RTX 4060/4060 Ti/4070/4080/4090, RTX
4000 Ada/5000 Ada/6000 Ada. Runtime device adaptation:

- The fill-rate threshold (0.11 @ 24 SM) is auto-scaled by SM count
  (e.g., ~0.08 on RTX 4090's 128 SMs).
- The auto-balance rates in partitioning are scaled by SM count.
- Build with `-gencode arch=compute_89,code=sm_89` (already set).

## Features

- Sparse MTTKRP kernel: row-sorted views, register-cached factor rows,
  atomic-storm elimination (174.5ms -> 67ms per mode on NELL-2).
- Dense-block tensor-core kernel: WMMA TF32, full 16^3 tiles
  materialized in shared memory.
- Exact truncated-SVD pseudoinverse solve (cuSOLVER gesvdj) with lambda
  absorption and O(R^2) gram-based fit.
- Fill-rate-adaptive path selection:
  `avg_fill = dense_nnz / (num_dense_tiles * 16^3)`; tensor-core path if
  `avg_fill >= 0.11`, pure CUDA-core path otherwise.
- Dual-stream serial execution with a documented grid-level
  serialization study (8 micro-benchmarks + grid-size sweep).
- Generic N-order (3-5) CP-ALS driver: row-sorted segmented FP64 MTTKRP
  with rank-dependent element packing, cached per-mode Grams composed by
  Hadamard product, in-kernel Cholesky solve, and fused column norms plus
  normalisation.
- Dense-unit tensor-core path for higher orders: one warp per 16^3 unit of
  the three inner modes (WMMA TF32, two K-steps of m16n16k8), FP64
  outer-mode weight and FP64 output accumulation, one `atomicAdd` per
  (row, rank) per unit; units are picked by a per-unit density cutoff and
  every remaining element stays on the FP64 kernel.
- Element reordering (CADR / HITS / sampled) applied before view
  construction, to raise the fill of the dense units.
- Factor export (`GEN_DUMP`) plus `GEN_SELFTEST` unit test for the
  dense-unit kernel.

## Results (RTX 4060 Laptop, CUDA 12.6, R=32 unless noted)

| Workload | Per-iteration | fit |
|---|---|---|
| NELL-2 (12092x9184x28818, 76.9M nnz), R=128 | 0.81 s | 0.4371 |
| dense_lr2 (256^3, 15.1M nnz), R=32 | 18.8 ms | 0.6832 |
| 11 synthetic tensors (512^3) | faster than BLCO on 9/10 (up to 1.5x), comparable on s1 | --- |

## Build

```bash
./scripts/build_opt.sh          # builds Release_opt/cptc_opt
```

Dependencies: CUDA 12.6 (nvcc), MAGMA (LU fallback), cuBLAS/cuSPARSE/cuSOLVER.

## Usage

```bash
./Release_opt/cptc_opt <tns_file> [max_iters] [R] [dense_threshold]
# examples
./Release_opt/cptc_opt nell-2.tns 20 32 0.03
./Release_opt/cptc_opt dense_lr2.tns 20 32 0.03
```

Order-3 tensors take this path. Tensors of any other order are routed
automatically to the N-order path below (`ALS_GENERIC_FORCE=1` routes
order-3 tensors through it as well, for cross-validation).

Environment switches:

| Variable | Effect |
|---|---|
| `ALS_FORCE_WMMA=1` | force tensor-core dual-stream path |
| `ALS_FORCE_CUDA=1` | force pure CUDA-core path |
| `ALS_FILL_MIN=<x>` | override fill-rate threshold (default 0.11) |
| `ALS_PROFILE=1` | per-phase timing breakdown |
| `ALS_DEBUG=1` | verbose diagnostics |
| `ALS_SPARSE_WMMA=1` | sparse WMMA variant (accuracy OK, slower) |
| `ALS_EXPORT_FACTORS=<dir>` | export factor matrices |

## Higher-order (N-order) path

`src/generic.cu` implements CP-ALS for tensors of order 3-5 (any N in
principle; the current driver accepts `R <= 32`). For an order-N tensor
each mode update runs two kernels whose contributions are accumulated into
the same `FtV`:

- **FP64 residual kernel.** Per-mode views are built by sorting the
  nonzeros of each row into fixed-length segments (`GEN_SEG = 1024`) and
  packing `1`, `2` or `4` elements per thread depending on the rank
  (`R<=8`, `R<=16`, else). Warp shuffles and a shared-memory reduction
  combine the packed slots, and a row whose elements form a single
  segment *stores* its result while all other rows use `atomicAdd`.
- **Dense-unit tensor-core kernel.** A *unit* is a 16^3 tile of the three
  inner modes with the remaining modes fixed to one coordinate tuple. A
  unit contributes to an inner target mode exactly as the order-3 dense
  block does, scaled by the per-rank scalar `prod_{m outer} F_m[c_m, r]`,
  so one warp per unit runs two WMMA TF32 `m16n16k8` steps per 16-rank
  block, multiplies by the outer-mode weight in FP64 and accumulates in
  FP64 registers, issuing one `atomicAdd` per (row, rank) per unit. On the
  dense-covered inner modes the single-segment store of the FP64 kernel is
  forced to `atomicAdd` as well, so the two kernels write disjoint
  contributions and may complete in any order.

Units are selected by a per-unit density cutoff; everything not covered
by a unit falls through to the FP64 kernel, which skips those elements.

```bash
# order-4/5 tensor: only <tns> <iters> <R>; the rest is environment
./Release_opt/cptc_opt uber.tns 20 16
```

| Variable | Effect |
|---|---|
| `GEN_DENSE=0` | disable the dense/tensor-core kernel (pure FP64) |
| `GEN_DENSE_THRESHOLD=<x>` | per-unit density cutoff (default 0.03) |
| `GEN_DENSE_INNER=<i,j,k>` | inner mode set for the units (default `0,1,2`) |
| `GEN_REORDER=cadr\|hits\|auto\|none` | reordering policy (default `hits` for order>=4, `cadr` for order 3; `auto` skips reordering when the probed fill is below 0.02 and otherwise keeps HITS only if it raises the fill) |
| `GEN_REORDER_T=<n>` | number of HITS refinement rounds (default 3) |
| `GEN_REORDER_SAMPLE=<n>` | sampling budget for the HITS reordering (default 8e6) |
| `GEN_SOLVER=svd` | truncated-SVD pseudoinverse instead of Cholesky |
| `GEN_SEED=<n>` | factor initialisation seed |
| `GEN_DUMP=<prefix>` | write `<prefix>.factors.txt` (column-normalised factors plus `lambda`) |
| `GEN_PROFILE=1` | per-phase timing breakdown |
| `GEN_SELFTEST=1` | run the dense-unit kernel unit test and exit |

Non-default `GEN_DENSE_INNER` sets are currently refused by the dense
kernel (they have not been validated); the FP64 path is used instead.

## Data

- **NELL-2**: available from the FROSTT repository
  (http://frostt.io/tensors/nell-2/) — 12092x9184x28818, 76,879,419
  nonzeros. Not included here due to size (1.5 GB).
- **dense_lr2**: generate with `scripts/gen_lowrank_struct.py 256 0.8 dense_lr2.tns`.
- **synthetic s1-s11**: generate with `scripts/gen_synth.py` using the
  parameter table in `docs/`.
- **order-4/5 tensors** for the N-order path: Uber, Enron, LBNL and LANL2
  from the FROSTT repository; MovieLens-10M/25M from GroupLens, with the
  ratings aggregated into an order-4/5 count tensor. Sorted `.tns` files
  with coordinate-major layout are expected (`scripts/gen_norder.py`
  builds synthetic ones).

## License

See LICENSE.
