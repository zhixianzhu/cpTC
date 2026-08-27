#ifndef COMPUTE_FIT_HPP
#define COMPUTE_FIT_HPP

#include "common.hpp"
#include "tensor.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>

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

double compute_fit(
    const HybridCOOTensor& h1,
    const HybridCOOTensor& h2,
    const double* d_A,
    const double* d_B,
    const double* d_C,
    const double* d_lambda,
    int R);

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

#endif