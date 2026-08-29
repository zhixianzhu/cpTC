# cpTC: Tensor-Core-Accelerated Sparse Tensor CP Decomposition

cpTC accelerates large-scale sparse tensor CANDECOMP/PARAFAC (CP)
decomposition on NVIDIA GPUs by splitting the tensor into dense blocks
(processed by Tensor Cores via WMMA TF32) and residual values (processed
by FP64 CUDA cores), with a fill-rate-adaptive path-selection rule.

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

## Results (RTX 4060 Laptop, CUDA 12.6, R=32 unless noted)

| Workload | Per-iteration | fit |
|---|---|---|
| NELL-2 (12092x9184x28818, 76.9M nnz), R=128 | 0.81 s | 0.4371 |
| dense_lr2 (256^3, 15.1M nnz), R=32 | 18.8 ms | 0.6832 |
| 11 synthetic tensors (512^3) | 1.04x-2.0x faster than BLCO | --- |

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

## Data

- **NELL-2**: available from the FROSTT repository
  (http://frostt.io/tensors/nell-2/) — 12092x9184x28818, 76,879,419
  nonzeros. Not included here due to size (1.5 GB).
- **dense_lr2**: generate with `scripts/gen_lowrank_struct.py 256 0.8 dense_lr2.tns`.
- **synthetic s1-s11**: generate with `scripts/gen_synth.py` using the
  parameter table in `docs/`.

## License

See LICENSE.
