#include "als.hpp"
#include "dense.hpp"
#include "sparse.hpp"
#include "solver.hpp"
#include "common.hpp"
#include "compute_fit.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <vector>

static const bool g_als_profile = (std::getenv("ALS_PROFILE") != nullptr);

__global__ void check_nan_inf_kernel(const double* data, size_t size, int* d_has_bad);
__global__ void compute_column_norms_kernel(const double* factor, int rows, int R, double* d_norms);
__global__ void elementwise_mult_kernel(const double* A, const double* B, double* C, size_t size);
__global__ void compute_lambda_kernel(const double* na, const double* nb, const double* nc,
                                      double* lambda, int R);
__global__ void divide_factor_kernel(double* factor, int rows, int R, const double* d_norms);
__global__ void normalize_with_scale_kernel(double* factor, int rows, int R,
                                            const double* norms);

#if ENABLE_NAN_CHECK
__global__ void check_nan_inf_kernel(const double* data, size_t size, int* d_has_bad)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < size) {
        double v = data[idx];
        if (isnan(v) || isinf(v)) atomicExch(d_has_bad, 1);
    }
}

static void check_data_integrity(const double* d_data, size_t size,
                                 const char* step_name, int* d_has_bad)
{
    int h = 0;
    CHECK_CUDA(cudaMemcpy(d_has_bad, &h, sizeof(int), cudaMemcpyHostToDevice));
    int blocks = static_cast<int>((size + 255) / 256);
    check_nan_inf_kernel<<<blocks, 256>>>(d_data, size, d_has_bad);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h, d_has_bad, sizeof(int), cudaMemcpyDeviceToHost));

    if (h) {
        std::cerr << "[ALS FATAL] NaN/Inf detected at: "
                  << step_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#else
#define check_data_integrity(data, size, name, flag)
#endif

__global__ void compute_column_norms_kernel(const double* factor, int rows, int R,
                                            double* d_norms)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;

    double s = 0.0;
    for (int i = 0; i < rows; ++i) {
        double v = factor[static_cast<size_t>(i) * R + r];
        s += v * v;
    }

    double n = sqrt(s);
    if (!isfinite(n) || n < 1e-14) n = 1e-14;
    d_norms[r] = n;
}

__global__ void divide_factor_kernel(double* factor, int rows, int R,
                                     const double* d_norms)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(rows) * R;
    if (idx >= total) return;

    int r = static_cast<int>(idx % R);
    factor[idx] /= d_norms[r];
}

__global__ void elementwise_mult_kernel(const double* A, const double* B,
                                        double* C, size_t size)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] * B[idx];
}

__global__ void compute_lambda_kernel(const double* na, const double* nb,
                                      const double* nc, double* lambda, int R)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < R) lambda[r] = na[r] * nb[r] * nc[r];
}

static void compute_gram_matrix(cublasHandle_t handle, const double* d_factor,
                                size_t rows, int R, double* d_gram)
{
    if (rows == 0 || R <= 0) return;

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasStatus_t stat = cublasDgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        R, R, static_cast<int>(rows),
        &alpha, d_factor, R,
        d_factor, R,
        &beta, d_gram, R);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ALS] cuBLAS Gram calculation failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void normalize_factor(double* d_factor, size_t rows, int R,
                             double* d_norms, int grid_r)
{
    compute_column_norms_kernel<<<grid_r, 256>>>(
        d_factor, static_cast<int>(rows), R, d_norms);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    size_t total = rows * static_cast<size_t>(R);
    int blocks = static_cast<int>((total + 255) / 256);

    divide_factor_kernel<<<blocks, 256>>>(
        d_factor, static_cast<int>(rows), R, d_norms);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void column_dot_kernel(
    const double* __restrict__ U,
    const double* __restrict__ V,
    int rows,
    int R,
    double* __restrict__ out)
{
    const int r = blockIdx.x;

    if (r >= R) return;

    double acc = 0.0;

    for (int i = threadIdx.x; i < rows; i += blockDim.x)
    {
        const size_t p = static_cast<size_t>(i) * R + r;
        acc += U[p] * V[p];
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ double s_part[8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if (lane == 0) s_part[warp] = acc;

    __syncthreads();

    if (warp == 0)
    {
        double total = (lane < 8) ? s_part[lane] : 0.0;

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            total += __shfl_down_sync(0xffffffffu, total, off);
        }

        if (lane == 0) out[r] = total;
    }
}

__global__ void vecsum_kernel(
    const double* __restrict__ x,
    size_t n,
    double* __restrict__ out)
{
    double acc = 0.0;

    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
    {
        acc += x[i];
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ double s_part[8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if (lane == 0) s_part[warp] = acc;

    __syncthreads();

    if (warp == 0)
    {
        double total = (lane < 8) ? s_part[lane] : 0.0;

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            total += __shfl_down_sync(0xffffffffu, total, off);
        }

        if (lane == 0) *out = total;
    }
}

__global__ void fill_value_kernel(double* x, size_t n, double val)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = val;
}

__global__ void column_norms_kernel2(
    const double* __restrict__ factor,
    int rows,
    int R,
    double* __restrict__ d_norms)
{
    const int r = blockIdx.x;

    if (r >= R) return;

    double acc = 0.0;

    for (int i = threadIdx.x; i < rows; i += blockDim.x)
    {
        const double v = factor[static_cast<size_t>(i) * R + r];
        acc += v * v;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ double s_part[8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if (lane == 0) s_part[warp] = acc;

    __syncthreads();

    if (warp == 0)
    {
        double total = (lane < 8) ? s_part[lane] : 0.0;

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            total += __shfl_down_sync(0xffffffffu, total, off);
        }

        if (lane == 0) d_norms[r] = sqrt(total);
    }
}

__global__ void normalize_absorb_lambda_kernel(
    double* __restrict__ factor,
    int rows,
    int R,
    const double* __restrict__ d_norms,
    double* __restrict__ lambda)
{
    const size_t total = static_cast<size_t>(rows) * R;

    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const int r = static_cast<int>(idx % R);
        const double n = d_norms[r];

        if (n < 1e-14)
        {
            factor[idx] = 0.0;
        }
        else
        {
            factor[idx] /= n;
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int r = 0; r < R; ++r)
        {
            const double n = d_norms[r];
            lambda[r] = (n < 1e-14) ? 0.0 : lambda[r] * n;
        }
    }
}

__global__ void factor_divide_lambda_kernel(
    double* __restrict__ factor,
    int rows,
    int R,
    const double* __restrict__ lambda)
{
    const size_t total = static_cast<size_t>(rows) * R;

    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const int r = static_cast<int>(idx % R);
        factor[idx] /= lambda[r];
    }
}

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
    double lambda_reg)
{
    (void)lambda_reg;

    if (R <= 0 || max_iters <= 0 ||
        !d_A || !d_B || !d_C || !d_lambda_out) {
        std::cerr << "[ALS ERROR] Invalid parameters." << std::endl;
        return;
    }

    const size_t I = tensor.dims[0];
    const size_t J = tensor.dims[1];
    const size_t K = tensor.dims[2];
    const size_t RR = static_cast<size_t>(R) * R;
    const size_t max_dim = std::max({I, J, K});

    std::cout << "[ALS] Starting CP-ALS: rank=" << R
              << ", max_iters=" << max_iters
              << " (SVD pseudoinverse solve, per-iter normalization)"
              << std::endl;

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ALS] cublasCreate failed." << std::endl;
        return;
    }

    cudaStream_t dense_stream = nullptr;
    cudaStream_t sparse_stream = nullptr;
    cudaEvent_t zero_event = nullptr;

    CHECK_CUDA(cudaStreamCreateWithFlags(&dense_stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&sparse_stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreateWithFlags(&zero_event, cudaEventDisableTiming));

    cudaEvent_t factor_event = nullptr;
    CHECK_CUDA(cudaEventCreateWithFlags(&factor_event, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(factor_event));

    double *d_GtA = nullptr, *d_GtB = nullptr, *d_GtC = nullptr, *d_FtF = nullptr;
    double *d_FtV = nullptr;

    CHECK_CUDA(cudaMalloc(&d_GtA, RR * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_GtB, RR * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_GtC, RR * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_FtF, RR * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_FtV, max_dim * static_cast<size_t>(R) * sizeof(double)));

    CHECK_CUDA(cudaMemset(d_lambda_out, 0, R * sizeof(double)));
    fill_value_kernel<<<(R + 255) / 256, 256>>>(
        d_lambda_out, R, 1.0);
    CHECK_CUDA(cudaGetLastError());

    SVDWorkspace svd_ws;
    init_svd_workspace(svd_ws, R);

    double *d_work_r = nullptr, *d_model = nullptr, *d_scalar = nullptr;

    CHECK_CUDA(cudaMalloc(&d_work_r, R * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_model, RR * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_scalar, sizeof(double)));

    double h_X_norm_sq = 0.0, h_res_ignored = 0.0;

    compute_residual_and_norm_sq(
        hybrid, hybrid_dense,
        d_A, d_B, d_C,
        nullptr,
        false,
        R,
        h_res_ignored,
        h_X_norm_sq);

    const double X_norm = std::sqrt(h_X_norm_sq);

#if ENABLE_NAN_CHECK
    int* d_nan_flag = nullptr;
    CHECK_CUDA(cudaMalloc(&d_nan_flag, sizeof(int)));
#endif

    const int grid_rr = static_cast<int>((RR + 255) / 256);

    cudaEvent_t total_start, total_stop, iter_start, iter_stop;
    CHECK_CUDA(cudaEventCreate(&total_start));
    CHECK_CUDA(cudaEventCreate(&total_stop));
    CHECK_CUDA(cudaEventCreate(&iter_start));
    CHECK_CUDA(cudaEventCreate(&iter_stop));

    cudaEvent_t t_gram, t_mtt, t_solve, t_fit;
    CHECK_CUDA(cudaEventCreate(&t_gram));
    CHECK_CUDA(cudaEventCreate(&t_mtt));
    CHECK_CUDA(cudaEventCreate(&t_solve));
    CHECK_CUDA(cudaEventCreate(&t_fit));

    CHECK_CUDA(cudaEventRecord(total_start));

    auto update_mode =
        [&](int mode, double* d_F, size_t dim, double* d_G) -> bool
    {

        const bool ok =
            solve_als_system_svd(
                svd_ws,
                handle,
                d_FtF,
                d_FtV,
                d_F,
                dim,
                R);

        if (!ok) return false;

        factor_divide_lambda_kernel<<<
            static_cast<int>((dim * static_cast<size_t>(R) + 255) / 256), 256>>>(
            d_F, static_cast<int>(dim), R, d_lambda_out);
        CHECK_CUDA(cudaGetLastError());

        column_norms_kernel2<<<R, 256>>>(d_F, static_cast<int>(dim), R, d_work_r);
        CHECK_CUDA(cudaGetLastError());

        normalize_absorb_lambda_kernel<<<
            static_cast<int>((dim * static_cast<size_t>(R) + 255) / 256), 256>>>(
            d_F, static_cast<int>(dim), R, d_work_r, d_lambda_out);
        CHECK_CUDA(cudaGetLastError());

        compute_gram_matrix(handle, d_F, dim, R, d_G);

        (void)mode;
        return true;
    };

    for (int iter = 0; iter < max_iters; ++iter) {
        CHECK_CUDA(cudaEventRecord(iter_start));

        compute_gram_matrix(handle, d_B, J, R, d_GtB);
        compute_gram_matrix(handle, d_C, K, R, d_GtC);
        elementwise_mult_kernel<<<grid_rr, 256>>>(d_GtB, d_GtC, d_FtF, RR);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaStreamWaitEvent(dense_stream, factor_event, 0));
        CHECK_CUDA(cudaStreamWaitEvent(sparse_stream, factor_event, 0));

        CHECK_CUDA(cudaMemsetAsync(
            d_FtV, 0, I * static_cast<size_t>(R) * sizeof(double), dense_stream));
        CHECK_CUDA(cudaEventRecord(zero_event, dense_stream));
        CHECK_CUDA(cudaStreamWaitEvent(sparse_stream, zero_event, 0));

        compute_dense_mttkrp_async(hybrid, 0, d_A, d_B, d_C,
                                   d_FtV, d_FtF, R, dense_stream);

        compute_sparse_mttkrp_async(hybrid, 0, d_A, d_B, d_C,
                                    d_FtV, d_FtF, R, sparse_stream);

        if (hybrid_dense.mttkrp_ready) {
            compute_sparse_mttkrp_async(hybrid_dense, 0, d_A, d_B, d_C,
                                        d_FtV, d_FtF, R, sparse_stream);
        }

        CHECK_CUDA(cudaStreamSynchronize(dense_stream));
        CHECK_CUDA(cudaStreamSynchronize(sparse_stream));

        if (!update_mode(0, d_A, I, d_GtA)) break;
        CHECK_CUDA(cudaEventRecord(factor_event));

        compute_gram_matrix(handle, d_A, I, R, d_GtA);
        compute_gram_matrix(handle, d_C, K, R, d_GtC);
        elementwise_mult_kernel<<<grid_rr, 256>>>(d_GtA, d_GtC, d_FtF, RR);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaStreamWaitEvent(dense_stream, factor_event, 0));
        CHECK_CUDA(cudaStreamWaitEvent(sparse_stream, factor_event, 0));

        CHECK_CUDA(cudaMemsetAsync(
            d_FtV, 0, J * static_cast<size_t>(R) * sizeof(double), dense_stream));
        CHECK_CUDA(cudaEventRecord(zero_event, dense_stream));
        CHECK_CUDA(cudaStreamWaitEvent(sparse_stream, zero_event, 0));

        compute_dense_mttkrp_async(hybrid, 1, d_A, d_B, d_C,
                                   d_FtV, d_FtF, R, dense_stream);

        compute_sparse_mttkrp_async(hybrid, 1, d_A, d_B, d_C,
                                    d_FtV, d_FtF, R, sparse_stream);

        if (hybrid_dense.mttkrp_ready) {
            compute_sparse_mttkrp_async(hybrid_dense, 1, d_A, d_B, d_C,
                                        d_FtV, d_FtF, R, sparse_stream);
        }

        CHECK_CUDA(cudaStreamSynchronize(dense_stream));
        CHECK_CUDA(cudaStreamSynchronize(sparse_stream));

        if (!update_mode(1, d_B, J, d_GtB)) break;
        CHECK_CUDA(cudaEventRecord(factor_event));

        compute_gram_matrix(handle, d_A, I, R, d_GtA);
        compute_gram_matrix(handle, d_B, J, R, d_GtB);
        elementwise_mult_kernel<<<grid_rr, 256>>>(d_GtA, d_GtB, d_FtF, RR);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(t_gram));

        CHECK_CUDA(cudaStreamWaitEvent(dense_stream, factor_event, 0));
        CHECK_CUDA(cudaStreamWaitEvent(sparse_stream, factor_event, 0));

        CHECK_CUDA(cudaMemsetAsync(
            d_FtV, 0, K * static_cast<size_t>(R) * sizeof(double), dense_stream));
        CHECK_CUDA(cudaEventRecord(zero_event, dense_stream));
        CHECK_CUDA(cudaStreamWaitEvent(sparse_stream, zero_event, 0));

        auto t0 = std::chrono::high_resolution_clock::now();

        compute_dense_mttkrp_async(hybrid, 2, d_A, d_B, d_C,
                                   d_FtV, d_FtF, R, dense_stream);

        compute_sparse_mttkrp_async(hybrid, 2, d_A, d_B, d_C,
                                    d_FtV, d_FtF, R, sparse_stream);

        if (hybrid_dense.mttkrp_ready) {
            compute_sparse_mttkrp_async(hybrid_dense, 2, d_A, d_B, d_C,
                                        d_FtV, d_FtF, R, sparse_stream);
        }

        CHECK_CUDA(cudaStreamSynchronize(dense_stream));
        CHECK_CUDA(cudaStreamSynchronize(sparse_stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        if (std::getenv("ALS_DEBUG")) {
            std::cout << "[DBG] modeC mttkrp wall = "
                      << std::chrono::duration<double, std::milli>(t1 - t0).count()
                      << " ms" << std::endl;
        }
        CHECK_CUDA(cudaEventRecord(t_mtt));

        if (!update_mode(2, d_C, K, d_GtC)) break;
        CHECK_CUDA(cudaEventRecord(factor_event));

        CHECK_CUDA(cudaEventRecord(t_solve));

#if ENABLE_NAN_CHECK
        check_data_integrity(d_A, I * static_cast<size_t>(R), "Mode A factor", d_nan_flag);
        check_data_integrity(d_B, J * static_cast<size_t>(R), "Mode B factor", d_nan_flag);
        check_data_integrity(d_C, K * static_cast<size_t>(R), "Mode C factor", d_nan_flag);
#endif

        column_dot_kernel<<<R, 256>>>(d_C, d_FtV, static_cast<int>(K), R, d_work_r);
        CHECK_CUDA(cudaGetLastError());

        elementwise_mult_kernel<<<(R + 255) / 256, 256>>>(
            d_lambda_out, d_work_r, d_work_r, R);
        vecsum_kernel<<<1, 256>>>(d_work_r, R, d_scalar);
        CHECK_CUDA(cudaGetLastError());

        double h_iprod = 0.0;
        CHECK_CUDA(cudaMemcpy(&h_iprod, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));

        {

            fill_value_kernel<<<(RR + 255) / 256, 256>>>(
                d_model, RR, 0.0);
            CHECK_CUDA(cudaGetLastError());

            const double one = 1.0;
            cublasStatus_t st =
                cublasDger(
                    handle, R, R,
                    &one, d_lambda_out, 1, d_lambda_out, 1,
                    d_model, R);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "[ALS] cublasDger failed." << std::endl;
            }
        }

        elementwise_mult_kernel<<<grid_rr, 256>>>(d_model, d_GtA, d_model, RR);
        elementwise_mult_kernel<<<grid_rr, 256>>>(d_model, d_GtB, d_model, RR);
        elementwise_mult_kernel<<<grid_rr, 256>>>(d_model, d_GtC, d_model, RR);
        CHECK_CUDA(cudaGetLastError());

        vecsum_kernel<<<1, 256>>>(d_model, RR, d_scalar);
        CHECK_CUDA(cudaGetLastError());

        double h_model_sq = 0.0;
        CHECK_CUDA(cudaMemcpy(&h_model_sq, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));

        const double residual_sq =
            std::max(h_X_norm_sq - 2.0 * h_iprod + h_model_sq, 0.0);

        const double fit =
            (X_norm > 0.0)
                ? 1.0 - std::sqrt(residual_sq) / X_norm
                : 0.0;

        CHECK_CUDA(cudaEventRecord(t_fit));

        CHECK_CUDA(cudaEventRecord(factor_event));

        CHECK_CUDA(cudaEventRecord(iter_stop));
        CHECK_CUDA(cudaEventSynchronize(iter_stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, iter_start, iter_stop));

        std::cout << "[ALS] iter " << iter + 1 << "/" << max_iters
                  << " time=" << ms << " ms fit=" << fit << std::endl;

        if (std::getenv("ALS_DEBUG")) {
            double hL[4]; double hA[4];
            cudaMemcpy(hL, d_lambda_out, 4 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hA, d_A, 4 * sizeof(double), cudaMemcpyDeviceToHost);
            double hX2 = h_X_norm_sq, hIp = h_iprod, hM2 = h_model_sq;
            double hW[4]; double hF[4];
            cudaMemcpy(hW, d_work_r, 4 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hF, d_FtV, 4 * sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << "[DBG] lambda[0..3]=" << hL[0] << " " << hL[1] << " " << hL[2] << " " << hL[3]
                      << " A[0..3]=" << hA[0] << " " << hA[1] << " " << hA[2] << " " << hA[3]
                      << " X2=" << hX2 << " iprod=" << hIp << " model2=" << hM2
                      << " work_r[0..3]=" << hW[0] << " " << hW[1] << " " << hW[2] << " " << hW[3]
                      << " FtV[0..3]=" << hF[0] << " " << hF[1] << " " << hF[2] << " " << hF[3] << std::endl;
            double rmse_exact = calculate_rmse(hybrid, hybrid_dense, d_A, d_B, d_C, d_lambda_out, R);
            std::cout << "[DBG] exact RMSE=" << rmse_exact << std::endl;
        }

        if (g_als_profile) {
            float ms_gram = 0.0f, ms_mtt = 0.0f, ms_solve = 0.0f, ms_fit = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms_gram, iter_start, t_gram));
            CHECK_CUDA(cudaEventElapsedTime(&ms_mtt, t_gram, t_mtt));
            CHECK_CUDA(cudaEventElapsedTime(&ms_solve, t_mtt, t_solve));
            CHECK_CUDA(cudaEventElapsedTime(&ms_fit, t_solve, t_fit));
            std::cout << "[ALS][profile] iter " << iter + 1
                      << " gram=" << ms_gram
                      << " mttkrp=" << ms_mtt
                      << " solve=" << ms_solve
                      << " fit=" << ms_fit
                      << " (iter=" << ms << ")" << std::endl;
        }
    }

    double final_rmse = calculate_rmse(
        hybrid, hybrid_dense, d_A, d_B, d_C, d_lambda_out, R);

    print_fit_report(
        hybrid, hybrid_dense, d_A, d_B, d_C, d_lambda_out, R, final_rmse, "final");

    std::cout << "[ALS] Final RMSE=" << final_rmse << std::endl;

    std::vector<double> h_lambda(R);
    CHECK_CUDA(cudaMemcpy(
        h_lambda.data(), d_lambda_out,
        R * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "[ALS] Lambda:";
    for (double x : h_lambda) std::cout << " " << x;
    std::cout << std::endl;

    CHECK_CUDA(cudaEventRecord(total_stop));
    CHECK_CUDA(cudaEventSynchronize(total_stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, total_start, total_stop));
    std::cout << "[ALS] Total time=" << total_ms << " ms" << std::endl;

    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    cudaEventDestroy(iter_start);
    cudaEventDestroy(iter_stop);
    cudaEventDestroy(t_gram);
    cudaEventDestroy(t_mtt);
    cudaEventDestroy(t_solve);
    cudaEventDestroy(t_fit);
    CHECK_CUDA(cudaEventDestroy(zero_event));
    CHECK_CUDA(cudaEventDestroy(factor_event));

    CHECK_CUDA(cudaStreamDestroy(dense_stream));
    CHECK_CUDA(cudaStreamDestroy(sparse_stream));

    cublasDestroy(handle);

    cudaFree(d_GtA);
    cudaFree(d_GtB);
    cudaFree(d_GtC);
    cudaFree(d_FtF);
    cudaFree(d_FtV);
    cudaFree(d_work_r);
    cudaFree(d_model);
    cudaFree(d_scalar);

    free_svd_workspace(svd_ws);

#if ENABLE_NAN_CHECK
    cudaFree(d_nan_flag);
#endif
}
