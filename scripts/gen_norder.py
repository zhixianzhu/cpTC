#!/usr/bin/env python3
"""Generate a low-rank structured N-order sparse tensor (.tns).

Ground truth: N factor matrices of smooth sin/cos form, true rank R0,
weights lambda decaying; values on a uniform random sample of the
index space equal the low-rank reconstruction.  Good for validating
that an N-order CP solver recovers high fit (and near-exact on
noiseless samples).

Usage:
  gen_norder.py <N> <R0> <density> <out.tns> [seed] [dims...]
  dims default: N moderate sizes; pass explicit per-mode dims after seed.
"""
import sys
import numpy as np

N = int(sys.argv[1])
R0 = int(sys.argv[2])
density = float(sys.argv[3])
out = sys.argv[4]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
extra = sys.argv[6:]

if extra:
    dims = [int(x) for x in extra]
    assert len(dims) == N, "explicit dims must match N"
else:
    base = 160 if N <= 4 else 90
    dims = [max(20, base - 8 * i) for i in range(N)]

lam = np.array([1.0 / (1.0 + 0.7 * r) for r in range(R0)])

factors = []
for m in range(N):
    d = dims[m]
    t = np.arange(d) / d
    A = np.column_stack([
        (np.sin(2 * np.pi * (r + 1) * t * (1 + 0.3 * m) + r + m) +
         np.cos(2 * np.pi * (r + 1) * t * 0.5 + 2 * r + m)) * 0.5 + 1.2
        for r in range(R0)])
    factors.append(A)

# sample unique positions
rng = np.random.default_rng(seed)
total = int(np.prod(dims))
nnz_target = max(1000, int(total * density))
nnz_target = min(nnz_target, total)

flat_idx = rng.choice(total, size=nnz_target, replace=False)
coords = np.unravel_index(flat_idx, dims)  # tuple of N arrays

# low-rank value
X = lam[None, :]  # (1,R0)
for m in range(N):
    # build in chunks to bound memory: accumulate per-sample only
    pass

# per-sample value: sum_r lam[r] prod_m A_m[coord_m, r]
coords_c = [c.astype(np.int64) for c in coords]
vals = np.zeros(nnz_target)
for r in range(R0):
    p = np.ones(nnz_target) * lam[r]
    for m in range(N):
        p *= factors[m][coords_c[m], r]
    vals += p

# write tns (1-based)
with open(out, "w") as f:
    for e in range(nnz_target):
        line = " ".join(str(int(coords[m][e]) + 1) for m in range(N))
        f.write(f"{line} {vals[e]:.9e}\n")

print(f"wrote {nnz_target} nnz  order={N}  dims={dims}  R0={R0}  "
      f"density={density:.4f}  seed={seed}")
