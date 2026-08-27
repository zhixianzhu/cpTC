#include "common.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

void debug_cuda_device_state()
{
int deviceCount = 0;

cudaError_t err = cudaGetDeviceCount(&deviceCount);

if (err != cudaSuccess)
{
    std::cerr
        << "[CUDA Error] cudaGetDeviceCount failed: "
        << cudaGetErrorString(err)
        << std::endl;

    std::exit(EXIT_FAILURE);
}

if (deviceCount == 0)
{
    std::cerr
        << "[Error] No CUDA devices found!"
        << std::endl;

    std::exit(EXIT_FAILURE);
}

cudaDeviceProp prop{};

CHECK_CUDA(
    cudaGetDeviceProperties(&prop, 0)
);

std::cout
    << "[CUDA Device] "
    << prop.name
    << " (Compute "
    << prop.major
    << "."
    << prop.minor
    << ")"
    << std::endl;

}

void init_factors_random(
double** d_factors,
const size_t* dims,
int R)
{
if (d_factors == nullptr ||
dims == nullptr ||
R <= 0)
{
std::cerr
<< "[Error] Invalid arguments in init_factors_random"
<< std::endl;

    std::exit(EXIT_FAILURE);
}

std::mt19937 gen(42);

std::uniform_real_distribution<double> dis(0.0, 1.0);
for (int mode = 0; mode < 3; ++mode)
{
    size_t num_elements =
        dims[mode] *
        static_cast<size_t>(R);

    std::vector<double>
        h_factor(num_elements);

    for (size_t i = 0;
         i < num_elements;
         ++i)
    {
        h_factor[i] = dis(gen);
    }

    CHECK_CUDA(
        cudaMalloc(
            reinterpret_cast<void**>(&d_factors[mode]),
            num_elements *
            sizeof(double)
        )
    );

    CHECK_CUDA(
        cudaMemcpy(
            d_factors[mode],
            h_factor.data(),
            num_elements *
            sizeof(double),
            cudaMemcpyHostToDevice
        )
    );
}

}

struct ClampOp
{
double min_v;
double max_v;

__host__ __device__
ClampOp(
    double mn,
    double mx)
    : min_v(mn),
      max_v(mx)
{
}

__host__ __device__
double operator()(double x) const
{
    return fmin(
        fmax(x, min_v),
        max_v
    );
}

};

void clamp_factor_thrust(
double* d_factor,
size_t size,
double min_val,
double max_val)
{
if (d_factor == nullptr ||
size == 0)
{
return;
}

thrust::device_ptr<double>
    dev_ptr(d_factor);

thrust::transform(
    dev_ptr,
    dev_ptr + size,
    dev_ptr,
    ClampOp(
        min_val,
        max_val
    )
);

CHECK_CUDA(
    cudaGetLastError()
);

}

__device__ __forceinline__
void block_reduce_add(double my, double* d_out)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        my += __shfl_down_sync(0xffffffffu, my, off);
    }

    __shared__ double s_err[8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if (lane == 0)
    {
        s_err[warp] = my;
    }

    __syncthreads();

    if (warp == 0)
    {
        double total =
            (lane < 8) ? s_err[lane] : 0.0;

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            total += __shfl_down_sync(0xffffffffu, total, off);
        }

        if (lane == 0)
        {
            atomicAdd(d_out, total);
        }
    }
}

__global__
void residual_sparse_view_kernel(
    const uint64_t* __restrict__ coords,
    const double* __restrict__ vals,
    size_t nnz,

    const double* A,
    const double* B,
    const double* C,

    const double* lambda,
    bool has_lambda,

    int R,

    double* d_error)
{
    constexpr int ELEMS = 4;

    const size_t base =
        (
            static_cast<size_t>(blockIdx.x) *
                static_cast<size_t>(blockDim.x) +
            static_cast<size_t>(threadIdx.x)
        ) *
        ELEMS;

    double my = 0.0;

    #pragma unroll
    for (int t = 0; t < ELEMS; ++t)
    {
        const size_t idx = base + static_cast<size_t>(t);

        if (idx >= nnz)
            break;

        const uint64_t pc = __ldcs(&coords[idx]);
        const double v = __ldcs(&vals[idx]);

        const uint32_t i =
            static_cast<uint32_t>((pc >> 30) & 0x7FFFu);
        const uint32_t j =
            static_cast<uint32_t>((pc >> 15) & 0x7FFFu);
        const uint32_t k =
            static_cast<uint32_t>(pc & 0x7FFFu);

        const double* pa = A + static_cast<size_t>(i) * R;
        const double* pb = B + static_cast<size_t>(j) * R;
        const double* pc3 = C + static_cast<size_t>(k) * R;

        double p0 = 0.0;
        double p1 = 0.0;

        int r = 0;

        for (; r + 1 < R; r += 2)
        {
            double x0 = pa[r] * pb[r] * pc3[r];
            double x1 = pa[r + 1] * pb[r + 1] * pc3[r + 1];

            if (has_lambda)
            {
                x0 *= lambda[r];
                x1 *= lambda[r + 1];
            }

            p0 += x0;
            p1 += x1;
        }

        for (; r < R; ++r)
        {
            double x = pa[r] * pb[r] * pc3[r];

            if (has_lambda)
                x *= lambda[r];

            p0 += x;
        }

        const double pred = p0 + p1;

        const double diff = v - pred;

        my += diff * diff;
    }

    block_reduce_add(my, d_error);
}

__global__
void residual_dense_tiles_kernel(
    const DenseTile16* __restrict__ d_tiles,
    int num_tiles,

    const double* A,
    const double* B,
    const double* C,

    const double* lambda,
    bool has_lambda,

    int R,

    double* d_error)
{
    const int tile_id =
        static_cast<int>(blockIdx.x);

    if (tile_id >= num_tiles)
        return;

    const DenseTile16 t =
        d_tiles[tile_id];

    const int b0 = t.block_idx[0] * 16;
    const int b1 = t.block_idx[1] * 16;
    const int b2 = t.block_idx[2] * 16;

    double my = 0.0;

    for (int e = threadIdx.x;
         e < t.nnz;
         e += blockDim.x)
    {
        const int lid =
            t.d_coords[e];

        const int li = lid >> 8;
        const int lj = (lid >> 4) & 15;
        const int lk = lid & 15;

        const double v =
            t.d_values[e];

        const int i = b0 + li;
        const int j = b1 + lj;
        const int k = b2 + lk;

        const double* pa = A + static_cast<size_t>(i) * R;
        const double* pb = B + static_cast<size_t>(j) * R;
        const double* pc3 = C + static_cast<size_t>(k) * R;

        double p0 = 0.0;
        double p1 = 0.0;

        int r = 0;

        for (; r + 1 < R; r += 2)
        {
            double x0 = pa[r] * pb[r] * pc3[r];
            double x1 = pa[r + 1] * pb[r + 1] * pc3[r + 1];

            if (has_lambda)
            {
                x0 *= lambda[r];
                x1 *= lambda[r + 1];
            }

            p0 += x0;
            p1 += x1;
        }

        for (; r < R; ++r)
        {
            double x = pa[r] * pb[r] * pc3[r];

            if (has_lambda)
                x *= lambda[r];

            p0 += x;
        }

        const double pred = p0 + p1;

        const double diff = v - pred;

        my += diff * diff;
    }

    block_reduce_add(my, d_error);
}

__global__
void sumsq_sparse_view_kernel(
    const double* __restrict__ vals,
    size_t nnz,
    double* d_out)
{
    constexpr int ELEMS = 4;

    const size_t base =
        (
            static_cast<size_t>(blockIdx.x) *
                static_cast<size_t>(blockDim.x) +
            static_cast<size_t>(threadIdx.x)
        ) *
        ELEMS;

    double my = 0.0;

    #pragma unroll
    for (int t = 0; t < ELEMS; ++t)
    {
        const size_t idx = base + static_cast<size_t>(t);

        if (idx >= nnz)
            break;

        const double v = vals[idx];

        my += v * v;
    }

    block_reduce_add(my, d_out);
}

__global__
void sumsq_dense_tiles_kernel(
    const DenseTile16* __restrict__ d_tiles,
    int num_tiles,
    double* d_out)
{
    const int tile_id =
        static_cast<int>(blockIdx.x);

    if (tile_id >= num_tiles)
        return;

    const DenseTile16 t =
        d_tiles[tile_id];

    double my = 0.0;

    for (int e = threadIdx.x;
         e < t.nnz;
         e += blockDim.x)
    {
        const double v =
            t.d_values[e];

        my += v * v;
    }

    block_reduce_add(my, d_out);
}

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
    double& h_x_sq)
{
    const size_t nnz_total =
        h1.sparse_nnz +
        h2.sparse_nnz +
        h1.dense_coords_capacity +
        h2.dense_coords_capacity;

    h_res_sq = 0.0;
    h_x_sq = 0.0;

    if (nnz_total == 0)
        return;

    const int threads = 256;

    double* d_res = nullptr;
    double* d_x = nullptr;

    CHECK_CUDA(cudaMalloc(&d_res, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, sizeof(double)));
    CHECK_CUDA(cudaMemset(d_res, 0, sizeof(double)));
    CHECK_CUDA(cudaMemset(d_x, 0, sizeof(double)));

    const HybridCOOTensor* hybrids[2] = {&h1, &h2};

    for (int hh = 0; hh < 2; ++hh)
    {
        const HybridCOOTensor& h = *hybrids[hh];

        if (h.sparse_nnz == 0 || !h.mttkrp_ready)
            continue;

        const size_t nnz = h.sparse_nnz;
        const int blocks =
            static_cast<int>((nnz + threads * 4 - 1) / (threads * 4));

        residual_sparse_view_kernel<<<blocks, threads>>>(
            h.d_sp2_coords[0],
            h.d_sp2_val[0],
            nnz,

            d_A, d_B, d_C,

            d_lambda,
            has_lambda,

            R,

            d_res);

        sumsq_sparse_view_kernel<<<blocks, threads>>>(
            h.d_sp2_val[0],
            nnz,
            d_x);

        CHECK_CUDA(cudaGetLastError());
    }

    if (h1.num_dense_tiles > 0 && h1.d_dense_coords_pool)
    {
        DenseTile16* d_tiles = nullptr;

        const size_t tile_bytes =
            static_cast<size_t>(h1.num_dense_tiles) *
            sizeof(DenseTile16);

        CHECK_CUDA(cudaMalloc(&d_tiles, tile_bytes));

        CHECK_CUDA(cudaMemcpy(
            d_tiles,
            h1.dense_tiles.data(),
            tile_bytes,
            cudaMemcpyHostToDevice));

        residual_dense_tiles_kernel<<<
            h1.num_dense_tiles,
            threads>>>(
                d_tiles,
                h1.num_dense_tiles,

                d_A, d_B, d_C,

                d_lambda,
                has_lambda,

                R,

                d_res);

        sumsq_dense_tiles_kernel<<<
            h1.num_dense_tiles,
            threads>>>(
                d_tiles,
                h1.num_dense_tiles,
                d_x);

        CHECK_CUDA(cudaGetLastError());

        cudaFree(d_tiles);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        &h_res_sq,
        d_res,
        sizeof(double),
        cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(
        &h_x_sq,
        d_x,
        sizeof(double),
        cudaMemcpyDeviceToHost));

    cudaFree(d_res);
    cudaFree(d_x);
}

double calculate_rmse(
    const HybridCOOTensor& h1,
    const HybridCOOTensor& h2,
    double* d_A,
    double* d_B,
    double* d_C,
    const double* d_lambda,
    int R)
{
    const size_t nnz_total =
        h1.sparse_nnz +
        h2.sparse_nnz +
        h1.dense_coords_capacity +
        h2.dense_coords_capacity;

    if (nnz_total == 0)
    {
        return 0.0;
    }

    if (d_A == nullptr ||
        d_B == nullptr ||
        d_C == nullptr ||
        R <= 0)
    {
        std::cerr
            << "[RMSE] Invalid factor pointer or rank"
            << std::endl;
        return -1.0;
    }

    double h_res_sq = 0.0;
    double h_x_sq = 0.0;

    compute_residual_and_norm_sq(
        h1, h2,
        d_A, d_B, d_C,
        d_lambda,
        (d_lambda != nullptr),
        R,
        h_res_sq,
        h_x_sq);

    if (!std::isfinite(h_res_sq))
    {
        std::cerr
            << "[RMSE] Non-finite squared error detected!"
            << std::endl;
        return std::numeric_limits<double>::infinity();
    }

    if (h_res_sq < 0.0)
    {
        h_res_sq = 0.0;
    }

    return std::sqrt(
        h_res_sq /
        static_cast<double>(nnz_total)
    );
}
