#include "solver.hpp"
#include "common.hpp"

#include <cuda_runtime.h>
#include <magma_v2.h>
#include <magma_lapack.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

static const bool g_verbose =
    (std::getenv("ALS_VERBOSE") != nullptr);

__global__ void add_regularization_kernel(double* A, int R, double lambda)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;

    size_t p = static_cast<size_t>(r) * R + r;
    double d = A[p];

    if (!isfinite(d) || d < 0.0) d = 0.0;
    A[p] = d + lambda;
}

__global__ void symmetrize_kernel(double* A, int R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;

    for (int j = i + 1; j < R; ++j) {
        size_t a = static_cast<size_t>(i) * R + j;
        size_t b = static_cast<size_t>(j) * R + i;
        double v = 0.5 * (A[a] + A[b]);
        A[a] = v;
        A[b] = v;
    }
}

__global__ void symmetric_scale_kernel(
    double* A, const double* invD, int R)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t n = static_cast<size_t>(R) * R;
    if (idx >= n) return;

    int row = static_cast<int>(idx / R);
    int col = static_cast<int>(idx % R);
    A[idx] *= invD[row] * invD[col];
}

__global__ void scale_rhs_kernel(
    double* B, const double* invD, size_t rhs_count, int R)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t n = rhs_count * static_cast<size_t>(R);
    if (idx >= n) return;

    int r = static_cast<int>(idx % R);
    B[idx] *= invD[r];
}

__global__ void recover_solution_kernel(
    double* X, const double* invD, size_t rhs_count, int R)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t n = rhs_count * static_cast<size_t>(R);
    if (idx >= n) return;

    int r = static_cast<int>(idx % R);
    X[idx] *= invD[r];
}

__global__ void cholesky_factor_kernel(
    double* __restrict__ d_A,
    int R,
    int* __restrict__ d_info)
{
    constexpr int MAXR = 64;

    __shared__ double sA[MAXR][MAXR + 1];

    const int tid = threadIdx.x;

    for (int i = tid; i < R * R; i += blockDim.x)
    {
        int r = i / R;
        int c = i % R;
        if (c <= r)
            sA[r][c] = d_A[static_cast<size_t>(r) * R + c];
        else
            sA[r][c] = 0.0;
    }

    __syncthreads();

    int fail = 0;

    for (int j = 0; j < R; ++j)
    {

        if (tid == 0)
        {
            double s = sA[j][j];

            for (int k = 0; k < j; ++k)
            {
                s -= sA[j][k] * sA[j][k];
            }

            if (s <= 0.0 || !isfinite(s))
            {
                fail = j + 1;
                s = 1.0;
            }

            sA[j][j] = sqrt(s);
        }

        __syncthreads();

        if (fail == 0)
        {

            for (int i = j + 1 + tid; i < R; i += blockDim.x)
            {
                double s = sA[i][j];

                for (int k = 0; k < j; ++k)
                {
                    s -= sA[i][k] * sA[j][k];
                }

                sA[i][j] = s / sA[j][j];
            }
        }

        __syncthreads();
    }

    if (tid == 0)
    {

        for (int r = 0; r < R; ++r)
        {
            for (int c = 0; c <= r; ++c)
            {
                d_A[static_cast<size_t>(r) * R + c] = sA[r][c];
            }
        }

        *d_info = fail;
    }
}

__global__ void cholesky_solve32_kernel(
    const double* __restrict__ d_L,
    double* __restrict__ d_B,
    size_t rhs_count,
    int R)
{
    constexpr int MAXR = 32;
    constexpr int WARPS = 8;

    __shared__ double sL[MAXR][MAXR + 1];

    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    const size_t col =
        static_cast<size_t>(blockIdx.x) * WARPS +
        warp;

    const bool active =
        col < rhs_count;

    for (int i = t; i < R * R; i += blockDim.x)
    {
        int r = i / R;
        int c = i % R;
        sL[r][c] = (c <= r)
            ? d_L[static_cast<size_t>(r) * R + c]
            : 0.0;
    }

    double bv = 0.0;

    if (active && lane < R)
    {
        bv = d_B[static_cast<size_t>(lane) + col * R];
    }

    __syncthreads();

    if (active)
    {

        double yv = 0.0;

        for (int i = 0; i < R; ++i)
        {

            double partial =
                (lane < i) ? sL[i][lane] * yv : 0.0;

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
            {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }

            if (lane == i)
            {
                yv = (bv - partial) / sL[i][i];
            }

            __syncwarp();
        }

        double xv = 0.0;

        for (int i = R - 1; i >= 0; --i)
        {
            double partial =
                (lane > i && lane < R) ? sL[lane][i] * xv : 0.0;

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
            {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }

            if (lane == i)
            {
                xv = (yv - partial) / sL[i][i];
            }

            __syncwarp();
        }

        if (lane < R)
        {
            d_B[static_cast<size_t>(lane) + col * R] = xv;
        }
    }
}

__global__ void cholesky_solve64_kernel(
    const double* __restrict__ d_L,
    double* __restrict__ d_B,
    size_t rhs_count,
    int R)
{
    constexpr int MAXR = 64;

    __shared__ double sL[MAXR][MAXR + 1];
    __shared__ double sy[MAXR];
    __shared__ double sx[MAXR];
    __shared__ double sb[MAXR];
    __shared__ double s_part[2];

    const size_t col =
        static_cast<size_t>(blockIdx.x);

    if (col >= rhs_count)
        return;

    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    for (int i = t; i < R * R; i += blockDim.x)
    {
        int r = i / R;
        int c = i % R;
        sL[r][c] = (c <= r)
            ? d_L[static_cast<size_t>(r) * R + c]
            : 0.0;
    }

    if (t < R)
    {
        sb[t] = d_B[static_cast<size_t>(t) + col * R];
    }

    __syncthreads();

    for (int i = 0; i < R; ++i)
    {
        double acc = 0.0;

        if (t < i)
        {
            acc = sL[i][t] * sy[t];
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }

        if (warp == 0 && lane == 0)
        {
            s_part[0] = acc;
        }

        if (warp == 1 && lane == 0)
        {
            s_part[1] = acc;
        }

        __syncthreads();

        if (t == i)
        {
            const double total =
                (R <= 32)
                    ? s_part[0]
                    : s_part[0] + s_part[1];

            sy[i] = (sb[i] - total) / sL[i][i];
        }

        __syncthreads();
    }

    for (int i = R - 1; i >= 0; --i)
    {
        double acc = 0.0;

        if (t > i && t < R)
        {
            acc = sL[t][i] * sx[t];
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }

        if (warp == 0 && lane == 0)
        {
            s_part[0] = acc;
        }

        if (warp == 1 && lane == 0)
        {
            s_part[1] = acc;
        }

        __syncthreads();

        if (t == i)
        {
            const double total =
                (R <= 32)
                    ? s_part[0]
                    : s_part[0] + s_part[1];

            sx[i] = (sy[i] - total) / sL[i][i];
        }

        __syncthreads();
    }

    if (t < R)
    {
        d_B[static_cast<size_t>(t) + col * R] = sx[t];
    }
}

static void copy_matrix_to_host(
    const double* d_A,
    std::vector<double>& mat,
    int R)
{
    mat.resize(static_cast<size_t>(R) * R);

    CHECK_CUDA(cudaMemcpy(
        mat.data(),
        d_A,
        static_cast<size_t>(R) * R * sizeof(double),
        cudaMemcpyDeviceToHost));
}

static double compute_adaptive_lambda(
    const double* d_FtF,
    int R,
    double lambda_rel,
    double& mean_diag)
{
    std::vector<double> mat;
    copy_matrix_to_host(d_FtF, mat, R);

    std::vector<double> diag(R);

    for (int r = 0; r < R; ++r)
    {
        diag[r] =
            mat[static_cast<size_t>(r) * R + r];
    }

    double trace = 0.0;

    for (double d : diag) {
        if (!std::isfinite(d) || d < 0.0) {
            mean_diag = 1.0;
            return 1e-12;
        }
        trace += d;
    }

    mean_diag = trace / static_cast<double>(R);

    if (!std::isfinite(mean_diag) || mean_diag <= 0.0)
        mean_diag = 1.0;

    double lambda = lambda_rel * mean_diag;

    if (!std::isfinite(lambda) || lambda < 0.0)
        lambda = 1e-12;

    lambda = std::max(lambda, 1e-12);

    const double lambda_max = std::max(1e-12, 1e-2 * mean_diag);
    lambda = std::min(lambda, lambda_max);

    return lambda;
}

static bool solve_cholesky_custom(
    double* d_A,
    double* d_B,
    int R,
    size_t dim_len)
{
    int* d_info = nullptr;
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));

    cholesky_factor_kernel<<<1, 64>>>(d_A, R, d_info);
    CHECK_CUDA(cudaGetLastError());

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_info != 0)
    {
        cudaFree(d_info);
        return false;
    }

    if (R <= 32)
    {

        const unsigned int blocks =
            static_cast<unsigned int>((dim_len + 7) / 8);

        cholesky_solve32_kernel<<<blocks, 256>>>(d_A, d_B, dim_len, R);
        CHECK_CUDA(cudaGetLastError());
    }
    else
    {
        const unsigned int blocks =
            static_cast<unsigned int>(dim_len);

        cholesky_solve64_kernel<<<blocks, 64>>>(d_A, d_B, dim_len, R);
        CHECK_CUDA(cudaGetLastError());
    }

    cudaFree(d_info);
    return true;
}

static bool solve_cholesky_magma(
    double* d_A,
    double* d_B,
    int R,
    size_t dim_len,
    magma_int_t& info)
{
    magma_dpotrf_gpu(
        MagmaLower,
        R,
        d_A,
        R,
        &info);

    if (info != 0)
        return false;

    magma_dpotrs_gpu(
        MagmaLower,
        R,
        static_cast<magma_int_t>(dim_len),
        d_A,
        R,
        d_B,
        R,
        &info);

    return info == 0;
}

void solve_als_system_magma(
    double* d_FtF,
    double* d_FtV,
    double* d_Factor,
    size_t dim_len,
    int R,
    double lambda_rel)
{
    if (!d_FtF || !d_FtV || !d_Factor || R <= 0 || dim_len == 0) {
        std::cerr << "[Solver] Invalid arguments." << std::endl;
        return;
    }

    if (R > 512) {
        std::cerr << "[Solver] R > 512 unsupported." << std::endl;
        return;
    }

    if (!std::isfinite(lambda_rel) || lambda_rel < 0.0) {
        std::cerr << "[Solver] Invalid lambda_rel." << std::endl;
        return;
    }

    const bool use_custom = (R <= 64);

    const size_t matrix_elements = static_cast<size_t>(R) * R;
    const size_t rhs_elements = dim_len * static_cast<size_t>(R);

    double mean_diag = 0.0;
    const double lambda_base =
        compute_adaptive_lambda(
            d_FtF, R, lambda_rel, mean_diag);

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_invD = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, matrix_elements * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, rhs_elements * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_invD, R * sizeof(double)));

    bool solved = false;
    double lambda_used = lambda_base;

    constexpr int MAX_RETRY = 6;

    for (int retry = 0; retry < MAX_RETRY; ++retry) {
        double current_lambda =
            lambda_base * std::pow(10.0, static_cast<double>(retry));

        const double retry_max =
            std::max(1e-12, 5e-2 * mean_diag);

        current_lambda = std::min(current_lambda, retry_max);
        current_lambda = std::max(current_lambda, 1e-12);

        CHECK_CUDA(cudaMemcpy(
            d_A, d_FtF,
            matrix_elements * sizeof(double),
            cudaMemcpyDeviceToDevice));

        CHECK_CUDA(cudaMemcpy(
            d_B, d_FtV,
            rhs_elements * sizeof(double),
            cudaMemcpyDeviceToDevice));

        symmetrize_kernel<<<(R + 255) / 256, 256>>>(d_A, R);
        CHECK_CUDA(cudaGetLastError());

        add_regularization_kernel<<<(R + 255) / 256, 256>>>(
            d_A, R, current_lambda);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<double> mat;
        copy_matrix_to_host(d_A, mat, R);

        double min_diag = std::numeric_limits<double>::max();
        double max_diag = 0.0;

        bool valid = true;

        for (int r = 0; r < R; ++r) {
            const double d =
                mat[static_cast<size_t>(r) * R + r];

            if (!std::isfinite(d) || d <= 0.0) {
                valid = false;
                break;
            }

            min_diag = std::min(min_diag, d);
            max_diag = std::max(max_diag, d);
        }

        if (!valid) {
            if (g_verbose)
                std::cerr << "[Solver] Invalid diagonal; retrying." << std::endl;
            continue;
        }

        double diag_ratio = max_diag / min_diag;

        const bool use_scaling = diag_ratio > 100.0;

        std::vector<double> h_invD(R);

        for (int r = 0; r < R; ++r)
            h_invD[r] = 1.0 / std::sqrt(mat[static_cast<size_t>(r) * R + r]);

        CHECK_CUDA(cudaMemcpy(
            d_invD, h_invD.data(),
            R * sizeof(double),
            cudaMemcpyHostToDevice));

        if (use_scaling) {
            const int matrix_blocks =
                static_cast<int>((matrix_elements + 255) / 256);
            const int rhs_blocks =
                static_cast<int>((rhs_elements + 255) / 256);

            symmetric_scale_kernel<<<matrix_blocks, 256>>>(
                d_A, d_invD, R);

            scale_rhs_kernel<<<rhs_blocks, 256>>>(
                d_B, d_invD, dim_len, R);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        bool solve_ok = false;

        if (use_custom)
        {
            solve_ok =
                solve_cholesky_custom(d_A, d_B, R, dim_len);
        }
        else
        {
            magma_int_t minfo = 0;

            solve_ok =
                solve_cholesky_magma(d_A, d_B, R, dim_len, minfo);
        }

        if (solve_ok) {
            if (use_scaling) {
                const int blocks =
                    static_cast<int>((rhs_elements + 255) / 256);

                recover_solution_kernel<<<blocks, 256>>>(
                    d_B, d_invD, dim_len, R);

                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            solved = true;
            lambda_used = current_lambda;

            if (g_verbose)
                std::cout << "[Solver] Cholesky succeeded. lambda="
                          << lambda_used << std::endl;
            break;
        }

        if (g_verbose)
            std::cerr << "[Solver] Cholesky failed; retrying." << std::endl;
    }

    if (!solved) {
        std::cerr << "[Solver] Cholesky retries exhausted; "
                     "trying MAGMA LU fallback." << std::endl;

        const double lambda_lu =
            std::max(
                lambda_base,
                std::min(
                    1e-1 * std::max(mean_diag, 1e-12),
                    lambda_base * 1e5));

        CHECK_CUDA(cudaMemcpy(
            d_A, d_FtF,
            matrix_elements * sizeof(double),
            cudaMemcpyDeviceToDevice));

        CHECK_CUDA(cudaMemcpy(
            d_B, d_FtV,
            rhs_elements * sizeof(double),
            cudaMemcpyDeviceToDevice));

        symmetrize_kernel<<<(R + 255) / 256, 256>>>(d_A, R);
        add_regularization_kernel<<<(R + 255) / 256, 256>>>(
            d_A, R, lambda_lu);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        magma_int_t* d_ipiv = nullptr;
        CHECK_CUDA(cudaMalloc(
            &d_ipiv,
            R * sizeof(magma_int_t)));

        magma_int_t info = 0;

        magma_dgetrf_gpu(
            R, R, d_A, R, d_ipiv, &info);

        if (info == 0) {
            magma_dgetrs_gpu(
                MagmaNoTrans,
                R,
                static_cast<magma_int_t>(dim_len),
                d_A,
                R,
                d_ipiv,
                d_B,
                R,
                &info);
        }

        if (info == 0) {
            solved = true;
            lambda_used = lambda_lu;

            std::cout << "[Solver] LU fallback succeeded. lambda="
                      << lambda_used << std::endl;
        } else {
            std::cerr << "[Solver] LU fallback failed, info="
                      << info << std::endl;
        }

        CHECK_CUDA(cudaFree(d_ipiv));
    }

    if (solved) {
        CHECK_CUDA(cudaMemcpy(
            d_Factor,
            d_B,
            rhs_elements * sizeof(double),
            cudaMemcpyDeviceToDevice));
    } else {
        std::cerr << "[Solver] ERROR: ALS system could not be solved."
                  << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_invD));
}

__global__ void reciprocal_with_tol_kernel(
    double* __restrict__ s,
    int n,
    double tol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        double v = s[idx];
        s[idx] = (fabs(v) > tol) ? 1.0 / v : 0.0;
    }
}

__global__ void scale_columns_kernel(
    double* __restrict__ U,
    const double* __restrict__ S,
    int n)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(n) * n;
    if (idx >= total) return;

    int col = static_cast<int>(idx / n);
    U[idx] *= S[col];
}

void init_svd_workspace(
    SVDWorkspace& ws,
    int R)
{
    if (R <= 0) return;

    ws.R = R;

    if (ws.cusolver == nullptr)
    {
        cusolverStatus_t st = cusolverDnCreate(&ws.cusolver);
        if (st != CUSOLVER_STATUS_SUCCESS)
        {
            std::cerr << "[Solver] cusolverDnCreate failed: "
                      << st << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    if (ws.gesvd_info == nullptr)
    {
        cusolverDnCreateGesvdjInfo(&ws.gesvd_info);
        cusolverDnXgesvdjSetMaxSweeps(ws.gesvd_info, 15);
        cusolverDnXgesvdjSetTolerance(ws.gesvd_info, 1e-14);
    }

    const size_t n2 = static_cast<size_t>(R) * R;

    if (!ws.d_U)  CHECK_CUDA(cudaMalloc(&ws.d_U,  n2 * sizeof(double)));
    if (!ws.d_V)  CHECK_CUDA(cudaMalloc(&ws.d_V,  n2 * sizeof(double)));
    if (!ws.d_Ainv) CHECK_CUDA(cudaMalloc(&ws.d_Ainv, n2 * sizeof(double)));
    if (!ws.d_S)  CHECK_CUDA(cudaMalloc(&ws.d_S,  R * sizeof(double)));
    if (!ws.d_info) CHECK_CUDA(cudaMalloc(&ws.d_info, sizeof(int)));

    int lwork_int = 0;

    cusolverStatus_t st =
        cusolverDnDgesvdj_bufferSize(
            ws.cusolver,
            CUSOLVER_EIG_MODE_VECTOR,
            0,
            R, R,
            ws.d_Ainv, R,
            ws.d_S,
            ws.d_U, R,
            ws.d_V, R,
            &lwork_int,
            ws.gesvd_info);

    if (st != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "[Solver] gesvdj_bufferSize failed: "
                  << st << std::endl;
        std::exit(EXIT_FAILURE);
    }

    ws.work_len = static_cast<size_t>(lwork_int) + 2 * n2 + R;

    if (!ws.d_work)
        CHECK_CUDA(cudaMalloc(&ws.d_work, ws.work_len * sizeof(double)));
}

void free_svd_workspace(
    SVDWorkspace& ws)
{
    if (ws.d_U)      { cudaFree(ws.d_U);      ws.d_U = nullptr; }
    if (ws.d_V)      { cudaFree(ws.d_V);      ws.d_V = nullptr; }
    if (ws.d_Ainv)   { cudaFree(ws.d_Ainv);   ws.d_Ainv = nullptr; }
    if (ws.d_S)      { cudaFree(ws.d_S);      ws.d_S = nullptr; }
    if (ws.d_work)   { cudaFree(ws.d_work);   ws.d_work = nullptr; }
    if (ws.d_info)   { cudaFree(ws.d_info);   ws.d_info = nullptr; }

    if (ws.gesvd_info)
    {
        cusolverDnDestroyGesvdjInfo(ws.gesvd_info);
        ws.gesvd_info = nullptr;
    }

    if (ws.cusolver)
    {
        cusolverDnDestroy(ws.cusolver);
        ws.cusolver = nullptr;
    }
}

bool solve_als_system_svd(
    const SVDWorkspace& ws,
    cublasHandle_t cublas,
    double* d_FtF,
    double* d_FtV,
    double* d_Factor,
    size_t dim_len,
    int R)
{
    if (!ws.cusolver || !ws.gesvd_info || R <= 0 || R != ws.R)
    {
        std::cerr << "[Solver] SVD workspace not initialized."
                  << std::endl;
        return false;
    }

    const size_t n2 = static_cast<size_t>(R) * R;

    CHECK_CUDA(cudaMemcpy(
        ws.d_Ainv, d_FtF,
        n2 * sizeof(double),
        cudaMemcpyDeviceToDevice));

    int* d_info_host_tmp = nullptr;
    (void)d_info_host_tmp;

    cusolverStatus_t st =
        cusolverDnDgesvdj(
            ws.cusolver,
            CUSOLVER_EIG_MODE_VECTOR,
            0,
            R, R,
            ws.d_Ainv, R,
            ws.d_S,
            ws.d_U, R,
            ws.d_V, R,
            ws.d_work, static_cast<int>(ws.work_len),
            ws.d_info,
            ws.gesvd_info);

    if (st != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "[Solver] gesvdj failed: " << st << std::endl;
        return false;
    }

    double s_max = 0.0;

    CHECK_CUDA(cudaMemcpy(
        &s_max, ws.d_S, sizeof(double),
        cudaMemcpyDeviceToHost));

    const double tol =
        static_cast<double>(R) *
        (std::nextafter(s_max, s_max + 1.0) - s_max);

    reciprocal_with_tol_kernel<<<(R + 255) / 256, 256>>>(
        ws.d_S, R, tol);
    CHECK_CUDA(cudaGetLastError());

    scale_columns_kernel<<<(n2 + 255) / 256, 256>>>(
        ws.d_U, ws.d_S, R);
    CHECK_CUDA(cudaGetLastError());

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasStatus_t cst =
        cublasDgemm(
            cublas,
            CUBLAS_OP_N, CUBLAS_OP_T,
            R, R, R,
            &alpha,
            ws.d_V, R,
            ws.d_U, R,
            &beta,
            ws.d_Ainv, R);

    if (cst != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "[Solver] Ainv gemm failed." << std::endl;
        return false;
    }

    cst =
        cublasDgemm(
            cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            R, static_cast<int>(dim_len), R,
            &alpha,
            ws.d_Ainv, R,
            d_FtV, R,
            &beta,
            d_Factor, R);

    if (cst != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "[Solver] factor gemm failed." << std::endl;
        return false;
    }

    return true;
}
