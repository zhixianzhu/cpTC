#include "sparse.hpp"
#include "common.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <mma.h>

using namespace nvcuda;

__device__ __forceinline__
void decode_packed_coords(
    uint64_t pc,
    int mode,
    uint32_t& target,
    uint32_t& o0,
    uint32_t& o1)
{
    const uint32_t i = static_cast<uint32_t>((pc >> 30) & 0x7FFFu);
    const uint32_t j = static_cast<uint32_t>((pc >> 15) & 0x7FFFu);
    const uint32_t k = static_cast<uint32_t>(pc & 0x7FFFu);

    if (mode == 0)
    {
        target = i;
        o0 = j;
        o1 = k;
    }
    else if (mode == 1)
    {
        target = j;
        o0 = i;
        o1 = k;
    }
    else
    {
        target = k;
        o0 = i;
        o1 = j;
    }
}

#ifndef MTTKRP_LB
#define MTTKRP_LB 5
#endif

template <int R, int RB>
__global__ void __launch_bounds__(MTTKRP_BLOCK, MTTKRP_LB)
mttkrp_rowsorted_kernel(
    int mode,

    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_sval,

    const uint32_t* __restrict__ d_blk_row,
    const uint32_t* __restrict__ d_blk_start,
    const uint32_t* __restrict__ d_blk_cnt,

    const double* __restrict__ F1,
    const double* __restrict__ F2,

    double* __restrict__ d_FtV)
{
    const uint32_t row = d_blk_row[blockIdx.x];
    const uint32_t start = d_blk_start[blockIdx.x];
    const uint32_t cnt = d_blk_cnt[blockIdx.x];

    for (int rb = 0; rb < R; rb += RB)
    {
        double acc[RB];

        #pragma unroll
        for (int r = 0; r < RB; ++r)
        {
            acc[r] = 0.0;
        }

        const uint32_t tbase =
            threadIdx.x *
            static_cast<uint32_t>(MTTKRP_ITEMS);

        uint32_t prev_o1 =
            0xFFFFFFFFu;

        double f2reg[RB];

        #pragma unroll
        for (int t = 0; t < MTTKRP_ITEMS; ++t)
        {
            const uint32_t ee =
                tbase + static_cast<uint32_t>(t);

            if (ee >= cnt)
                break;

            const uint32_t pos =
                start + ee;

            const uint64_t pc =
                __ldcv(&d_coords[pos]);

            const double v =
                __ldcv(&d_sval[pos]);

            uint32_t target;
            uint32_t o0;
            uint32_t o1;

            decode_packed_coords(
                pc,
                mode,
                target,
                o0,
                o1);

            (void)target;

            if (o1 != prev_o1)
            {
                const double* f2 =
                    F2 +
                    static_cast<size_t>(o1) * R +
                    rb;

                #pragma unroll
                for (int r = 0; r < RB; ++r)
                {
                    f2reg[r] = f2[r];
                }

                prev_o1 = o1;
            }

            const double* f1 =
                F1 +
                static_cast<size_t>(o0) * R +
                rb;

            #pragma unroll
            for (int r = 0; r < RB; ++r)
            {
                acc[r] += v * f1[r] * f2reg[r];
            }
        }

        constexpr int WARPS = MTTKRP_BLOCK / 32;

        __shared__ double s_red[WARPS][RB];

        const int lane =
            threadIdx.x & 31;

        const int warp =
            threadIdx.x >> 5;

        #pragma unroll
        for (int r = 0; r < RB; ++r)
        {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
            {
                acc[r] +=
                    __shfl_down_sync(
                        0xffffffffu,
                        acc[r],
                        off);
            }
        }

        if (lane == 0)
        {
            #pragma unroll
            for (int r = 0; r < RB; ++r)
            {
                s_red[warp][r] = acc[r];
            }
        }

        __syncthreads();

        if (warp == 0)
        {
            #pragma unroll
            for (int r = 0; r < RB; ++r)
            {
                double sum =
                    (lane < WARPS)
                        ? s_red[lane][r]
                        : 0.0;

                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                {
                    sum +=
                        __shfl_down_sync(
                            0xffffffffu,
                            sum,
                            off);
                }

                if (lane == 0 && sum != 0.0)
                {
                    atomicAdd(
                        &d_FtV[
                            static_cast<size_t>(row) * R +
                            rb + r],
                        sum);
                }
            }
        }

        __syncthreads();
    }
}

__global__ void mttkrp_rowsorted_generic_kernel(
    int mode,

    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_sval,

    const uint32_t* __restrict__ d_blk_row,
    const uint32_t* __restrict__ d_blk_start,
    const uint32_t* __restrict__ d_blk_cnt,

    const double* __restrict__ F1,
    const double* __restrict__ F2,

    double* __restrict__ d_FtV,

    int R)
{
    const uint32_t row = d_blk_row[blockIdx.x];
    const uint32_t start = d_blk_start[blockIdx.x];
    const uint32_t cnt = d_blk_cnt[blockIdx.x];

    for (uint32_t e = threadIdx.x;
         e < cnt;
         e += blockDim.x)
    {
        const uint32_t pos =
            start + e;

        const uint64_t pc =
            d_coords[pos];

        const double v =
            d_sval[pos];

        uint32_t target;
        uint32_t o0;
        uint32_t o1;

        decode_packed_coords(
            pc,
            mode,
            target,
            o0,
            o1);

        (void)target;

        const double* f1 =
            F1 + static_cast<size_t>(o0) * R;

        const double* f2 =
            F2 + static_cast<size_t>(o1) * R;

        for (int r = 0; r < R; ++r)
        {
            atomicAdd(
                &d_FtV[
                    static_cast<size_t>(row) * R +
                    r],
                v * f1[r] * f2[r]);
        }
    }
}

__global__ void mttkrp_range_generic_kernel(
    int mode,

    size_t start,
    size_t count,

    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_sval,

    const double* __restrict__ F1,
    const double* __restrict__ F2,

    double* __restrict__ d_FtV,

    int R)
{
    size_t i =
        static_cast<size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    if (i >= count)
        return;

    const size_t idx =
        start + i;

    const uint64_t pc =
        d_coords[idx];

    const double v =
        d_sval[idx];

    uint32_t target;
    uint32_t o0;
    uint32_t o1;

    decode_packed_coords(
        pc,
        mode,
        target,
        o0,
        o1);

    const double* f1 =
        F1 + static_cast<size_t>(o0) * R;

    const double* f2 =
        F2 + static_cast<size_t>(o1) * R;

    for (int r = 0; r < R; ++r)
    {
        atomicAdd(
            &d_FtV[
                static_cast<size_t>(target) * R +
                r],
            v * f1[r] * f2[r]);
    }
}

__global__ void mttkrp_rowsorted_wmma32_kernel(
    int mode,

    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_sval,

    const uint32_t* __restrict__ d_blk_row,
    const uint32_t* __restrict__ d_blk_start,
    const uint32_t* __restrict__ d_blk_cnt,

    const double* __restrict__ F1,
    const double* __restrict__ F2,

    double* __restrict__ d_FtV)
{
    constexpr int R = 32;
    constexpr int NWARP = 4;

    const int warp =
        threadIdx.x >> 5;

    const int lane =
        threadIdx.x & 31;

    const int seg_id =
        blockIdx.x * NWARP + warp;

    if (seg_id >= (int)gridDim.x * 4)
        return;

    const uint32_t row =
        d_blk_row[seg_id];

    const uint32_t start =
        d_blk_start[seg_id];

    const uint32_t cnt =
        d_blk_cnt[seg_id];

    const uint32_t end =
        start + cnt;

    __shared__ float s_a[NWARP][2][16][8];
    __shared__ float s_b[NWARP][2][16][8];
    __shared__ double s_f1[NWARP][8][32];
    __shared__ double s_f2[NWARP][8][32];
    __shared__ uint32_t s_o[NWARP][2][8];

    float (*s_a_w)[16][8] = s_a[warp];
    float (*s_b_w)[16][8] = s_b[warp];
    double (*s_f1_w)[32] = s_f1[warp];
    double (*s_f2_w)[32] = s_f2[warp];
    uint32_t (*s_o_w)[8] = s_o[warp];

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[2];

    wmma::fill_fragment(c_frag[0], 0.0f);
    wmma::fill_fragment(c_frag[1], 0.0f);

    for (uint32_t e0 = start;
         e0 < end;
         e0 += 8)
    {

        if (lane < 8 && (e0 + lane) < end)
        {
            const uint64_t pc =
                __ldcv(&d_coords[e0 + lane]);

            uint32_t target;
            decode_packed_coords(
                pc,
                mode,
                target,
                s_o_w[0][lane],
                s_o_w[1][lane]);
        }
        else if (lane < 8)
        {
            s_o_w[0][lane] = 0;
            s_o_w[1][lane] = 0;
        }

        __syncwarp();

        for (int i = lane; i < 8 * 32; i += 32)
        {
            const int k = i / 32;
            const int r = i % 32;

            const uint32_t o0 = s_o_w[0][k];
            const uint32_t o1 = s_o_w[1][k];

            s_f1_w[k][r] =
                F1[static_cast<size_t>(o0) * R + r];

            s_f2_w[k][r] =
                F2[static_cast<size_t>(o1) * R + r];
        }

        __syncwarp();

        const double v =
            (lane < 8 && (e0 + lane) < end)
                ? __ldcv(&d_sval[e0 + lane])
                : 0.0;

        for (int i = lane; i < 8 * 32; i += 32)
        {
            const int k = i / 32;
            const int r = i % 32;
            const int rb = r / 16;
            const int rl = r % 16;

            const double vk =
                (k < 8 && (e0 + k) < end)
                    ? (k == (lane < 8 ? lane : -1) ? v : s_f1_w[k][0] * 0.0 + (k == (lane < 8 ? lane : -1) ? 0.0 : 0.0))
                    : 0.0;

            const double vk2 =
                (e0 + k) < end
                    ? __ldcv(&d_sval[e0 + k])
                    : 0.0;

            s_a_w[rb][rl][k] =
                static_cast<float>(vk2 * s_f1_w[k][r]);

            s_b_w[rb][rl][k] =
                static_cast<float>(s_f2_w[k][r]);
        }

        __syncwarp();

        for (int rb = 0; rb < 2; ++rb)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                           wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                           wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, &s_a_w[rb][0][0], 8);
            wmma::load_matrix_sync(b_frag, &s_b_w[rb][0][0], 8);

            wmma::mma_sync(c_frag[rb], a_frag, b_frag, c_frag[rb]);
        }
    }

    __shared__ float s_out[NWARP][2][16][16];

    wmma::store_matrix_sync(&s_out[warp][0][0][0], c_frag[0], 16, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_out[warp][1][0][0], c_frag[1], 16, wmma::mem_row_major);

    __syncwarp();

    const int r = lane;
    const int rb = r / 16;
    const int rl = r % 16;

    const double diag_val =
        static_cast<double>(s_out[warp][rb][rl][rl]);

    if (diag_val != 0.0)
    {
        atomicAdd(
            &d_FtV[
                static_cast<size_t>(row) * R +
                r],
            diag_val);
    }
}

template <int R>
__global__ void mttkrp_rowsorted_coalesced_kernel(
    int mode,

    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_sval,

    const uint32_t* __restrict__ d_blk_row,
    const uint32_t* __restrict__ d_blk_start,
    const uint32_t* __restrict__ d_blk_cnt,

    const double* __restrict__ F1,
    const double* __restrict__ F2,

    double* __restrict__ d_FtV)
{
    constexpr int THREADS = 128;
    constexpr int GROUPS = THREADS / R;

    const uint32_t row = d_blk_row[blockIdx.x];
    const uint32_t start = d_blk_start[blockIdx.x];
    const uint32_t cnt = d_blk_cnt[blockIdx.x];
    const uint32_t end = start + cnt;

    const int r = threadIdx.x % R;
    const int g = threadIdx.x / R;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    double acc = 0.0;

    for (uint32_t e = start + static_cast<uint32_t>(g);
         e < end;
         e += static_cast<uint32_t>(GROUPS))
    {

        const uint64_t pc = __ldcv(&d_coords[e]);
        const double v = __ldcv(&d_sval[e]);

        uint32_t target;
        uint32_t o0;
        uint32_t o1;

        decode_packed_coords(
            pc,
            mode,
            target,
            o0,
            o1);

        (void)target;

        acc +=
            v *
            F1[static_cast<size_t>(o0) * R + r] *
            F2[static_cast<size_t>(o1) * R + r];
    }

    constexpr int WARPS = THREADS / 32;

    __shared__ double s_red[WARPS][R];

    for (int i = threadIdx.x; i < WARPS * R; i += THREADS)
    {
        (reinterpret_cast<double*>(s_red))[i] = 0.0;
    }

    __syncthreads();

    s_red[warp][r] = acc;

    __syncthreads();

    const int my_rank = warp * 32 + lane;

    if (my_rank < R)
    {
        double total = 0.0;

        #pragma unroll
        for (int w = 0; w < WARPS; ++w)
        {
            total += s_red[w][my_rank];
        }

        if (total != 0.0)
        {
            atomicAdd(
                &d_FtV[
                    static_cast<size_t>(row) * R +
                    my_rank],
                total);
        }
    }
}

static void launch_mttkrp_rowsorted(
    const HybridCOOTensor& hybrid,

    int mode,

    const double* d_A,
    const double* d_B,
    const double* d_C,

    double* d_FtV,

    int R,

    cudaStream_t stream)
{
    const size_t blocks =
        hybrid.mttkrp_blocks[mode];

    if (blocks == 0)
        return;

    const double* F1 = nullptr;
    const double* F2 = nullptr;

    if (mode == 0)
    {
        F1 = d_B;
        F2 = d_C;
    }
    else if (mode == 1)
    {
        F1 = d_A;
        F2 = d_C;
    }
    else
    {
        F1 = d_A;
        F2 = d_B;
    }

    const uint64_t* coords =
        hybrid.d_sp2_coords[mode];

    const double* sval =
        hybrid.d_sp2_val[mode];

    const uint32_t* bro =
        hybrid.d_blk_row[mode];

    const uint32_t* bst =
        hybrid.d_blk_start[mode];

    const uint32_t* bcn =
        hybrid.d_blk_cnt[mode];

    const unsigned int gx =
        static_cast<unsigned int>(blocks);

    switch (R)
    {
        case 8:
            mttkrp_rowsorted_coalesced_kernel<8><<<gx, 128, 0, stream>>>(
                mode, coords, sval, bro, bst, bcn, F1, F2, d_FtV);
            break;

        case 16:
            mttkrp_rowsorted_coalesced_kernel<16><<<gx, 128, 0, stream>>>(
                mode, coords, sval, bro, bst, bcn, F1, F2, d_FtV);
            break;

        case 32:

            mttkrp_rowsorted_coalesced_kernel<32><<<gx, 128, 0, stream>>>(
                mode, coords, sval, bro, bst, bcn, F1, F2, d_FtV);
            break;

        case 64:
            mttkrp_rowsorted_coalesced_kernel<64><<<gx, 128, 0, stream>>>(
                mode, coords, sval, bro, bst, bcn, F1, F2, d_FtV);
            break;

        case 128:
            mttkrp_rowsorted_coalesced_kernel<128><<<gx, 128, 0, stream>>>(
                mode, coords, sval, bro, bst, bcn, F1, F2, d_FtV);
            break;

        default:
            mttkrp_rowsorted_generic_kernel<<<gx, MTTKRP_BLOCK, 0, stream>>>(
                mode, coords, sval, bro, bst, bcn, F1, F2, d_FtV, R);
            break;
    }

    CHECK_CUDA(cudaGetLastError());
}

void compute_sparse_mttkrp_async(
    const HybridCOOTensor& hybrid,

    int target_mode,

    const double* d_A,
    const double* d_B,
    const double* d_C,

    double* d_FtV,
    double* d_FtF,

    int R,

    cudaStream_t stream)
{
    (void)d_FtF;

    if (!hybrid.mttkrp_ready ||
        hybrid.sparse_nnz == 0)
    {
        return;
    }

    launch_mttkrp_rowsorted(
        hybrid,
        target_mode,
        d_A,
        d_B,
        d_C,
        d_FtV,
        R,
        stream);
}

void compute_sparse_mttkrp_async(
    const HybridCOOTensor& hybrid,

    int target_mode,

    const double* d_A,
    const double* d_B,
    const double* d_C,

    double* d_FtV,
    double* d_FtF,

    int R,

    cudaStream_t stream,

    size_t start_idx,
    size_t count)
{
    (void)d_FtF;

    if (!hybrid.mttkrp_ready ||
        hybrid.sparse_nnz == 0)
    {
        return;
    }

    if (count == 0)
        return;

    if (start_idx >= hybrid.sparse_nnz)
        return;

    count =
        std::min(
            count,
            hybrid.sparse_nnz -
                start_idx);

    if (start_idx == 0 &&
        count == hybrid.sparse_nnz)
    {
        launch_mttkrp_rowsorted(
            hybrid,
            target_mode,
            d_A,
            d_B,
            d_C,
            d_FtV,
            R,
            stream);

        return;
    }

    const double* F1 = nullptr;
    const double* F2 = nullptr;

    if (target_mode == 0)
    {
        F1 = d_B;
        F2 = d_C;
    }
    else if (target_mode == 1)
    {
        F1 = d_A;
        F2 = d_C;
    }
    else
    {
        F1 = d_A;
        F2 = d_B;
    }

    const unsigned int threads = 256;

    const unsigned int blocks =
        static_cast<unsigned int>(
            (count + threads - 1) /
            threads);

    mttkrp_range_generic_kernel<<<
        blocks,
        threads,
        0,
        stream>>>(
            target_mode,

            start_idx,
            count,

            hybrid.d_sp2_coords[target_mode],
            hybrid.d_sp2_val[target_mode],

            F1,
            F2,

            d_FtV,

            R);

    CHECK_CUDA(cudaGetLastError());
}

void compute_sparse_mttkrp(
    const HybridCOOTensor& hybrid,

    int target_mode,

    const double* d_A,
    const double* d_B,
    const double* d_C,

    double* d_FtV,
    double* d_FtF,

    int R)
{
    cudaStream_t stream;

    CHECK_CUDA(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking));

    compute_sparse_mttkrp_async(
        hybrid,

        target_mode,

        d_A,
        d_B,
        d_C,

        d_FtV,
        d_FtF,

        R,

        stream);

    CHECK_CUDA(
        cudaStreamSynchronize(
            stream));

    CHECK_CUDA(
        cudaStreamDestroy(
            stream));
}
