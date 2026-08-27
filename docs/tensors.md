# Synthetic tensor parameter table (s1-s11)

All tensors are 512x512x512 with sparse residual fill 0.003, generated
with `scripts/gen_synth.py` and seed 42:

| ID | block_ratio | dense_fill | nnz | avg_fill |
|----|-------------|------------|-----|----------|
| s1  | 0.15 | 0.50 | 10,488,021 | 0.090 |
| s2  | 0.05 | 0.50 |  3,750,209 | 0.058 |
| s3  | 0.50 | 0.50 | 33,739,408 | 0.252 |
| s4  | 0.15 | 0.10 |  2,338,744 | 0.033 |
| s5  | 0.15 | 1.00 | 20,486,466 | 0.327 |
| s6  | 0.30 | 0.30 | 12,374,737 | 0.093 |
| s7  | 0.05 | 0.05 |    729,147 | 0.000 |
| s8  | 0.30 | 0.50 | 20,267,063 | 0.153 |
| s9  | 0.50 | 0.30 | 20,419,030 | 0.152 |
| s10 | 1.00 | 0.05 |  6,710,258 | 0.050 |
| s11 | 0.05 | 1.00 |  6,989,640 | 0.134 |

Example:

```bash
python3 scripts/gen_synth.py 512 512 512 0.15 0.5 0.003 s1.tns 42
```

## dense_lr2 (dense low-rank, fit > 0.6)

256^3, true rank 2, smooth sin/cos structured factors, sampling density
0.8, 13,421,118 nonzeros, seed 42:

```bash
python3 scripts/gen_lowrank_struct.py 256 0.8 dense_lr2.tns 42
```

Measured fit = 0.6832 (R=32, 20 iters), tensor-core path 18.8 ms/iter
vs BLCO 41.0 ms/iter.
