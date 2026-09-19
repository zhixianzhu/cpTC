#ifndef COMPUTE_FIT_HPP
#define COMPUTE_FIT_HPP

#include "common.hpp"
#include "tensor.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>

// ============================================================
// Kernel
//
// pred = sum_r lambda[r]
//              * A[i,r]
//              * B[j,r]
//              * C[k,r]
//
// 如果 lambda == nullptr，则认为 lambda[r] = 1
// ============================================================
__global__
void compute_residual_kernel(
    const double* val,
    const int* m0,
    const int* m1,
    const int* m2,
    size_t nnz,

    const double* A,
    const double* B,
    const double* C,

    const double* lambda,
    bool has_lambda,

    int R,

    double* d_res_norm_sq);

// ============================================================
// Compute fit
//
// fit = 1 - ||X - X_hat||_F / ||X||_F
//
// X_hat =
//     sum_r lambda[r]
//              * A[:,r]
//              * B[:,r]
//              * C[:,r]
//
// lambda == nullptr 时：lambda[r] = 1
//
// 注意：基于 HybridCOOTensor 的 MTTKRP 视图计算
//（COO 设备数组在视图构建后会被释放）。
// ============================================================
double compute_fit(
    const HybridCOOTensor& h1,
    const HybridCOOTensor& h2,
    const double* d_A,
    const double* d_B,
    const double* d_C,
    const double* d_lambda,
    int R);

// ============================================================
// Fit report
// ============================================================
void print_fit_report(
    const HybridCOOTensor& h1,
    const HybridCOOTensor& h2,
    const double* d_A,
    const double* d_B,
    const double* d_C,
    const double* d_lambda,
    int R,
    double rmse,
    const char* tag = "");

#endif // COMPUTE_FIT_HPP