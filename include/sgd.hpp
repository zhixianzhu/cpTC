#pragma once

#include "common.hpp"
#include "tensor.hpp"
#include <cuda_runtime.h>

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
