#!/usr/bin/env python3
"""Parameterized synthetic 3rd-order tensor generator (tns format, 1-based, values 1..100).

Dual-fill-rate block model:
  - tensor is partitioned into 16^3 tiles
  - block_ratio of tiles are "dense blocks": filled at dense_fill
  - remaining tiles are "sparse blocks": filled at sparse_fill (low-density residual)

Usage:
  gen_synth.py <I> <J> <K> <block_ratio> <dense_fill> <sparse_fill> <out> [seed]
"""
import random, sys

I, J, K = (int(x) for x in sys.argv[1:4])
block_ratio = float(sys.argv[4])
dense_fill = float(sys.argv[5])
sparse_fill = float(sys.argv[6])
out = sys.argv[7]
seed = int(sys.argv[8]) if len(sys.argv) > 8 else 42

random.seed(seed)
T = 16

def tile_range(dim, b):
    return range(b * T, min((b + 1) * T, dim))

lines = []
count = 0
nb_i = (I + T - 1) // T
nb_j = (J + T - 1) // T
nb_k = (K + T - 1) // T

for bi in range(nb_i):
    for bj in range(nb_j):
        for bk in range(nb_k):
            is_dense = random.random() < block_ratio
            fill = dense_fill if is_dense else sparse_fill
            for ii in tile_range(I, bi):
                for jj in tile_range(J, bj):
                    for kk in tile_range(K, bk):
                        if random.random() < fill:
                            lines.append(f"{ii+1} {jj+1} {kk+1} {random.uniform(1,100):.1f}")
                            count += 1

with open(out, 'w') as f:
    f.write("\n".join(lines) + "\n")

print(f"generated {count} nnz  dims={I}x{J}x{K}  block_ratio={block_ratio}  "
      f"dense_fill={dense_fill}  sparse_fill={sparse_fill}", flush=True)
