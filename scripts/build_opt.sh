#!/bin/bash
# ============================================================
# cpTC - one-shot build script
#
# Usage:
#     ./build_opt.sh              # build Release_opt/cptc_opt
#     ./build_opt.sh run          # build and run (full 20 iters)
#
# Dependencies:
#     /usr/local/cuda-12.6  (nvcc 12.6)
#     /home/zzx/magma        (MAGMA library, used for LU fallback)
# ============================================================
set -e
cd "$(dirname "$0")"

NVCC=/usr/local/cuda-12.6/bin/nvcc
INC="-I include -I /home/zzx/magma/build/include -I /home/zzx/magma/include"
LIBS="-L/home/zzx/magma/build/lib -lmagma -lcublas -lcusparse -lcusolver"

mkdir -p Release_opt

echo "==> Compiling (-O3, sm_89) ..."
$NVCC -O3 -gencode arch=compute_89,code=sm_89 -ccbin gcc \
    $INC -o Release_opt/cptc_opt \
    src/als.cu src/cadr.cu src/common.cu src/compute_fit.cu \
    src/dense.cu src/main.cu src/partition.cu src/solver.cu \
    src/sparse.cu src/tensor.cu \
    --cudart=static $LIBS

echo "==> Done: Release_opt/cptc_opt"
echo "    Usage:"
echo "    ./Release_opt/cptc_opt nell-2.tns 20 32"

if [ "$1" = "run" ]; then
    ./Release_opt/cptc_opt nell-2.tns 20 32
fi
