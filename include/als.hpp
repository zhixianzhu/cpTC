#ifndef ALS_HPP
#define ALS_HPP

#include "tensor.hpp"
#include "partition.hpp"

void execute_als_decomposition(
    const COOTensor& tensor,
    HybridCOOTensor& hybrid,
    const HybridCOOTensor& hybrid_dense,
    double* d_A,
    double* d_B,
    double* d_C,
    double* d_lambda_out,
    int R,
    int max_iters,
    double lambda_reg);

#endif