#ifndef COMMON_HPP
#define COMMON_HPP

#include "tensor.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstddef>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void debug_cuda_device_state();
void init_factors_random(double** d_factors, const size_t* dims, int R);
void clamp_factor_thrust(double* d_factor, size_t size, double min_val, double max_val);
double calculate_rmse(const HybridCOOTensor& h1, const HybridCOOTensor& h2,
                      double* d_A, double* d_B, double* d_C,
                      const double* d_lambda, int R);
void compute_residual_and_norm_sq(
    const HybridCOOTensor& h1,
    const HybridCOOTensor& h2,
    const double* d_A,
    const double* d_B,
    const double* d_C,
    const double* d_lambda,
    bool has_lambda,
    int R,
    double& h_res_sq,
    double& h_x_sq);

#endif