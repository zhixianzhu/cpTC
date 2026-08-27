#include "common.hpp"
#include "tensor.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <chrono>

__device__ __forceinline__
void decode_packed_sgd(
    uint64_t pc,
    uint32_t& i,
    uint32_t& j,
    uint32_t& k)
{
    i = static_cast<uint32_t>((pc >> 30) & 0x7FFFu);
    j = static_cast<uint32_t>((pc >> 15) & 0x7FFFu);
    k = static_cast<uint32_t>(pc & 0x7FFFu);
}

__global__ void cpsgd_update_kernel(
    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_vals,
    const int* __restrict__ d_sample_idx,
    int batch_size,

    double* __restrict__ d_A,
    double* __restrict__ d_B,
    double* __restrict__ d_C,
    double* __restrict__ d_lambda,

    int R,
    double gamma)
{
    const int warp_id =
        (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    const int lane =
        threadIdx.x & 31;

    if (warp_id >= batch_size)
        return;

    const int elem =
        d_sample_idx[warp_id];

    const uint64_t pc =
        __ldg(&d_coords[elem]);

    const double v =
        __ldg(&d_vals[elem]);

    uint32_t gi, gj, gk;
    decode_packed_sgd(pc, gi, gj, gk);

#ifdef SGD_DEBUG
    if (warp_id < 4 && lane == 0) {
        printf("[SGD-DBG] elem=%d pc=%llx i=%u j=%u k=%u v=%.3e\n",
               elem, (unsigned long long)pc, gi, gj, gk, v);
    }
#endif

    const int r = lane;

    const double a = __ldg(&d_A[static_cast<size_t>(gi) * R + r]);
    const double b = __ldg(&d_B[static_cast<size_t>(gj) * R + r]);
    const double c = __ldg(&d_C[static_cast<size_t>(gk) * R + r]);
    const double lam = __ldg(&d_lambda[r]);

    double xhat = lam * a * b * c;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        xhat += __shfl_down_sync(0xffffffffu, xhat, off);
    }

    xhat = __shfl_sync(0xffffffffu, xhat, 0);

    const double e = v - xhat;

#ifdef SGD_DEBUG
    if (warp_id < 2 && lane == 0) {
        printf("[SGD-DBG] elem=%d i=%u j=%u k=%u v=%.3e xhat=%.3e e=%.3e\n",
               elem, gi, gj, gk, v, xhat, e);
    }
#endif

    const double g_a   = gamma * e * b * c;
    const double g_b   = gamma * e * a * c;
    const double g_c   = gamma * e * a * b;

    atomicAdd(&d_A[static_cast<size_t>(gi) * R + r], g_a);
    atomicAdd(&d_B[static_cast<size_t>(gj) * R + r], g_b);
    atomicAdd(&d_C[static_cast<size_t>(gk) * R + r], g_c);
}

__global__ void cpsgd_loss_kernel(
    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_vals,
    const int* __restrict__ d_sample_idx,
    int batch_size,

    const double* __restrict__ d_A,
    const double* __restrict__ d_B,
    const double* __restrict__ d_C,
    const double* __restrict__ d_lambda,

    double* __restrict__ d_loss_sum,
    int R)
{
    const int warp_id =
        (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    const int lane =
        threadIdx.x & 31;

    if (warp_id >= batch_size)
        return;

    const int elem =
        d_sample_idx[warp_id];

    const uint64_t pc =
        __ldg(&d_coords[elem]);

    const double v =
        __ldg(&d_vals[elem]);

    uint32_t gi, gj, gk;
    decode_packed_sgd(pc, gi, gj, gk);

    const int r = lane;

    const double a = __ldg(&d_A[static_cast<size_t>(gi) * R + r]);
    const double b = __ldg(&d_B[static_cast<size_t>(gj) * R + r]);
    const double c = __ldg(&d_C[static_cast<size_t>(gk) * R + r]);
    const double lam = __ldg(&d_lambda[r]);

    double xhat = lam * a * b * c;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        xhat += __shfl_down_sync(0xffffffffu, xhat, off);
    }

    xhat = __shfl_sync(0xffffffffu, xhat, 0);

    const double e = v - xhat;

    if (lane == 0)
    {
        atomicAdd(d_loss_sum, e * e);
    }
}

__global__ void cpsgd_sample_kernel(
    int* __restrict__ d_sample_idx,
    int batch_size,
    int nnz,
    unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= batch_size)
        return;

    unsigned long long x = seed + static_cast<unsigned long long>(i) * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;

    d_sample_idx[i] = static_cast<int>(x % static_cast<unsigned long long>(nnz));
}

__global__ void sgd_column_norms_kernel(
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
        acc += __shfl_down_sync(0xffffffffu, acc, off);

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
            total += __shfl_down_sync(0xffffffffu, total, off);
        if (lane == 0) d_norms[r] = sqrt(total);
    }
}

__global__ void sgd_normalize_absorb_kernel(
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
            factor[idx] = 0.0;
        else
            factor[idx] /= n;
    }

    if (threadIdx.x < R)
    {
        const int r = threadIdx.x;
        const double n = d_norms[r];

        lambda[r] = (n < 1e-14) ? 0.0 : n;
    }
}

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
    unsigned long long seed)
{
    if (hybrid.sparse_nnz == 0 || !hybrid.mttkrp_ready)
    {
        std::cerr << "[SGD] sparse views not ready." << std::endl;
        return;
    }

    const uint64_t* d_coords = hybrid.d_sp2_coords[0];
    const double* d_vals = hybrid.d_sp2_val[0];

    int* d_sample_idx = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_sample_idx,
        static_cast<size_t>(batch_size) * sizeof(int)));

    double* d_loss = nullptr;

    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(double)));

    const int threads = 128;
    const int blocks =
        (batch_size + threads / 32 - 1) / (threads / 32);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::cout << "[SGD] CP-SGD start: R=" << R
              << " batch=" << batch_size
              << " gamma=" << gamma
              << " iters=" << max_iters
              << " sparse_nnz=" << hybrid.sparse_nnz
              << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < max_iters; ++it)
    {

        cpsgd_sample_kernel<<<(batch_size + 255) / 256, 256, 0, stream>>>(
            d_sample_idx,
            batch_size,
            static_cast<int>(hybrid.sparse_nnz),
            seed + static_cast<unsigned long long>(it));

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));

        if (it == 0) {
            int h_idx[4];
            cudaMemcpy(h_idx, d_sample_idx, 4 * sizeof(int), cudaMemcpyDeviceToHost);
            printf("[SGD-DBG] sample[0..3] = %d %d %d %d\n", h_idx[0], h_idx[1], h_idx[2], h_idx[3]);
            double hA[4], hL[4];
            cudaMemcpy(hA, d_A, 4*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hL, d_lambda_out, 4*sizeof(double), cudaMemcpyDeviceToHost);
            printf("[SGD-DBG] before: A[0..3]=%.3e %.3e %.3e %.3e  lam[0..3]=%.3e %.3e %.3e %.3e\n",
                   hA[0],hA[1],hA[2],hA[3], hL[0],hL[1],hL[2],hL[3]);
        }

        cpsgd_update_kernel<<<blocks, threads, 0, stream>>>(
            d_coords,
            d_vals,
            d_sample_idx,
            batch_size,
            d_A, d_B, d_C, d_lambda_out,
            R,
            gamma);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));

        if (it == 0) {
            double hL[8];
            cudaMemcpy(hL, d_lambda_out, 8*sizeof(double), cudaMemcpyDeviceToHost);
            printf("[SGD-DBG] after update lam[0..7]=%.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e\n",
                   hL[0],hL[1],hL[2],hL[3],hL[4],hL[5],hL[6],hL[7]);
        }

        if ((it % 5) == 0)
        {
            const int dims[3] = {
                static_cast<int>(hybrid.dims[0]),
                static_cast<int>(hybrid.dims[1]),
                static_cast<int>(hybrid.dims[2])};
            double* factors[3] = {d_A, d_B, d_C};

            for (int n = 0; n < 3; ++n)
            {
                double* d_n = nullptr;
                CHECK_CUDA(cudaMalloc(&d_n, R * sizeof(double)));
                sgd_column_norms_kernel<<<R, 256, 0, stream>>>(
                    factors[n], dims[n], R, d_n);
                sgd_normalize_absorb_kernel<<<
                    static_cast<int>((static_cast<size_t>(dims[n]) * R + 255) / 256),
                    256, 0, stream>>>(
                        factors[n], dims[n], R, d_n, d_lambda_out);
                CHECK_CUDA(cudaGetLastError());
                cudaFree(d_n);
            }
        }

        if ((it % 10) == 0 || it == max_iters - 1)
        {
            if (std::getenv("ALS_SGD_DBG") && it == 0) {
                double hA[4], hL[4];
                cudaMemcpy(hA, d_A, 4*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(hL, d_lambda_out, 4*sizeof(double), cudaMemcpyDeviceToHost);
                printf("[SGD-DBG] it=%d A0=%.3e A1=%.3e lam0=%.3e lam1=%.3e\n",
                       it, hA[0], hA[1], hL[0], hL[1]);
            }
        }
        if ((it % 10) == 0 || it == max_iters - 1)
        {
            CHECK_CUDA(cudaMemsetAsync(d_loss, 0, sizeof(double), stream));

            cpsgd_loss_kernel<<<blocks, threads, 0, stream>>>(
                d_coords,
                d_vals,
                d_sample_idx,
                batch_size,
                d_A, d_B, d_C, d_lambda_out,
                d_loss,
                R);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(stream));

            double h_loss = 0.0;
            CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost));

            std::cout << "[SGD] iter " << it
                      << " batch-rmse=" << std::sqrt(h_loss / batch_size)
                      << std::endl;
        }
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "[SGD] done in "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms" << std::endl;

    cudaFree(d_sample_idx);
    cudaFree(d_loss);
    cudaStreamDestroy(stream);
}
