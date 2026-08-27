#!/usr/bin/env python3
"""Structured low-rank dense synthetic tensor generator (tns format, 1-based).

Recipe used in the paper (fit > 0.6):
- dims = I^3, R_true = 2 (small true rank -> significant low-rank structure)
- factors use smooth functions (sin/cos combinations), values ~[0,3]
- lambda = [1, 1/sqrt(2)] (mild spectral decay)
- uniformly sample density fraction of positions (unique), value = low-rank reconstruction
- sparse CP fit computed on observed positions; measured fit ~0.68 (R=32, 20 iters)

Usage: gen_lowrank_struct.py <I> <density> <out> [seed]
"""
import numpy as np, sys

I = int(sys.argv[1])
density = float(sys.argv[2])
out = sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

R_true = 2
t = np.arange(I) / I
A = np.column_stack([np.sin(2*np.pi*(r+1)*t + r) for r in range(R_true)]) * 2 + 1
B = np.column_stack([np.cos(2*np.pi*(r+1)*t*0.7 + r*2) for r in range(R_true)]) * 2 + 1
C = np.column_stack([np.sin(2*np.pi*(r+1)*t*1.3 + r*3) for r in range(R_true)]) * 2 + 1
lam = np.array([1.0, 1.0/np.sqrt(2)])

X = np.einsum('ir,jr,kr,r->ijk', A, B, C, lam)

rng = np.random.default_rng(seed)
mask = rng.random(X.shape) < density
nnz = int(mask.sum())

lines = []
for i in range(I):
    for j in range(I):
        for k in range(I):
            if mask[i, j, k]:
                lines.append(f"{i+1} {j+1} {k+1} {X[i,j,k]:.6f}")

with open(out, 'w') as f:
    f.write("\n".join(lines) + "\n")

print(f"generated {nnz} nnz  dims={I}^3  density={density:.2f}  R_true={R_true}  seed={seed}")
