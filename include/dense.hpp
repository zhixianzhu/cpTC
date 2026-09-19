#pragma once

#include "common.hpp"
#include "tensor.hpp"
#include <cuda_runtime.h>

void compute_dense_mttkrp_async(
    const HybridCOOTensor& hybrid,
    int mode,
    double* d_A,
    double* d_B,
    double* d_C,
    double* d_FtV,
    double* d_FtF,
    int R,
    cudaStream_t stream);

void compute_dense_mttkrp(
    const HybridCOOTensor& hybrid,
    int mode,
    double* d_A,
    double* d_B,
    double* d_C,
    double* d_FtV,
    double* d_FtF,
    int R);
// 合并内核（稠密 tile + 稀疏段单网格，R==32）
void compute_mttkrp_merged_async(
    const HybridCOOTensor& hybrid,
    int mode,
    double* d_A,
    double* d_B,
    double* d_C,
    double* d_FtV,
    double* d_FtF,
    int R,
    cudaStream_t stream);
