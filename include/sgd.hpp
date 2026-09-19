#pragma once

#include "common.hpp"
#include "tensor.hpp"
#include <cuda_runtime.h>

// CP-SGD（保持 CP 分解 + 随机梯度下降，参照论文 cuFastTuckerPlusTC
// 的 warp 采样路线）。稀疏残差由 SGD 更新；稠密 tile 由调用方用
// 现有 WMMA 路径贡献。
void execute_sgd_decomposition(
    const HybridCOOTensor& hybrid,
    double* d_A,
    double* d_B,
    double* d_C,
    double* d_lambda_out,
    int R,
    int max_iters,
    double gamma,
    int batch_size,
    unsigned long long seed);
