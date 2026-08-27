#include "dense.hpp"
#include "common.hpp"

#include <iostream>
#include <mma.h>
#include <cfloat>

using namespace nvcuda;

#define TILE_DIM 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

__device__ __forceinline__
void decode_dense_local_id(
    int local_id,
    int& local_i,
    int& local_j,
    int& local_k)
{
    local_k =
        local_id % TILE_DIM;

    local_j =
        (local_id / TILE_DIM) % TILE_DIM;

    local_i =
        local_id / (TILE_DIM * TILE_DIM);
}

__device__ __forceinline__
void dense_tile_mttkrp_wmma(
    const DenseTile16 tile,

    int lane_id,

    float* s_mem_float,

    int mode,

    const double* __restrict__ d_A,
    const double* __restrict__ d_B,
    const double* __restrict__ d_C,

    double* __restrict__ d_FtV,

    int R,

    int dimI,
    int dimJ,
    int dimK)
{

    int b0 =
        tile.block_idx[0] *
        TILE_DIM;

    int b1 =
        tile.block_idx[1] *
        TILE_DIM;

    int b2 =
        tile.block_idx[2] *
        TILE_DIM;

    const int SHARED_PER_WARP =
        4096 +
        256 +
        256 +
        256;

    (void)SHARED_PER_WARP;

    float* s_tile_dense =
        s_mem_float;

    float* s_slice_k =
        s_tile_dense +
        4096;

    float* s_factor =
        s_slice_k +
        256;

    float* s_gemm_out =
        s_factor +
        256;

    for (int i = lane_id;
         i < 4096;
         i += 32)
    {
        s_tile_dense[i] = 0.0f;
    }

    __syncwarp();

    for (int i = lane_id;
         i < tile.nnz;
         i += 32)
    {

        int local_id =
            tile.d_coords[i];

        int local_i;
        int local_j;
        int local_k;

        decode_dense_local_id(
            local_id,
            local_i,
            local_j,
            local_k);

        if (local_i < 0 ||
            local_i >= TILE_DIM ||
            local_j < 0 ||
            local_j >= TILE_DIM ||
            local_k < 0 ||
            local_k >= TILE_DIM)
        {
            continue;
        }

        double raw_val =
            tile.d_values[i];

        if (raw_val > FLT_MAX)
            raw_val = FLT_MAX;

        if (raw_val < -FLT_MAX)
            raw_val = -FLT_MAX;

        s_tile_dense[
            local_i * 256 +
            local_j * 16 +
            local_k] =
            static_cast<float>(
                raw_val);
    }

    __syncwarp();

    wmma::fragment<
        wmma::matrix_a,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        wmma::precision::tf32,
        wmma::row_major>
        a_frag_0,
        a_frag_1;

    wmma::fragment<
        wmma::matrix_b,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        wmma::precision::tf32,
        wmma::row_major>
        b_frag_0,
        b_frag_1;

    wmma::fragment<
        wmma::accumulator,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        float>
        c_frag;

    for (int r_base = 0;
         r_base < R;
         r_base += 16)
    {

        if (mode == 0)
        {
            double total_accum[8];

            #pragma unroll
            for (int iter = 0;
                 iter < 8;
                 ++iter)
            {
                total_accum[iter] = 0.0;
            }

            for (int i = lane_id;
                 i < 256;
                 i += 32)
            {
                int r_idx =
                    i / 16;

                int c_idx =
                    i % 16;

                int global_r =
                    r_base +
                    c_idx;

                int global_b_row =
                    b1 +
                    r_idx;

                if (global_r < R &&
                    global_b_row < dimJ)
                {
                    s_factor[i] =
                        static_cast<float>(
                            d_B[
                                global_b_row * R +
                                global_r]);
                }
                else
                {
                    s_factor[i] =
                        0.0f;
                }
            }

            __syncwarp();

            wmma::load_matrix_sync(
                b_frag_0,
                s_factor,
                16);

            wmma::load_matrix_sync(
                b_frag_1,
                s_factor + 128,
                16);

            for (int k = 0;
                 k < 16;
                 ++k)
            {
                if ((b2 + k) >= dimK)
                    continue;

                for (int i = lane_id;
                     i < 256;
                     i += 32)
                {
                    int r_idx =
                        i / 16;

                    int c_idx =
                        i % 16;

                    s_slice_k[i] =
                        s_tile_dense[
                            r_idx * 256 +
                            c_idx * 16 +
                            k];
                }

                __syncwarp();

                wmma::fill_fragment(
                    c_frag,
                    0.0f);

                wmma::load_matrix_sync(
                    a_frag_0,
                    s_slice_k,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_0,
                    b_frag_0,
                    c_frag);

                wmma::load_matrix_sync(
                    a_frag_1,
                    s_slice_k + 8,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_1,
                    b_frag_1,
                    c_frag);

                wmma::store_matrix_sync(
                    s_gemm_out,
                    c_frag,
                    16,
                    wmma::mem_row_major);

                __syncwarp();

                for (int iter = 0;
                     iter < 8;
                     ++iter)
                {
                    int idx =
                        iter * 32 +
                        lane_id;

                    int row =
                        idx / 16;

                    int col =
                        idx % 16;

                    int global_r =
                        r_base +
                        col;

                    int global_a_row =
                        b0 +
                        row;

                    if (global_r < R &&
                        global_a_row < dimI)
                    {
                        double c_val =
                            d_C[
                                (b2 + k) * R +
                                global_r];

                        total_accum[iter] +=
                            static_cast<double>(
                                s_gemm_out[idx]) *
                            c_val;
                    }
                }

                __syncwarp();
            }

            for (int iter = 0;
                 iter < 8;
                 ++iter)
            {
                int idx =
                    iter * 32 +
                    lane_id;

                int row =
                    idx / 16;

                int col =
                    idx % 16;

                int global_r =
                    r_base +
                    col;

                int global_a_row =
                    b0 +
                    row;

                if (global_r < R &&
                    global_a_row < dimI &&
                    total_accum[iter] != 0.0)
                {
                    atomicAdd(
                        &d_FtV[
                            global_a_row * R +
                            global_r],
                        total_accum[iter]);
                }
            }
        }

        else if (mode == 1)
        {
            double total_accum[8];

            #pragma unroll
            for (int iter = 0;
                 iter < 8;
                 ++iter)
            {
                total_accum[iter] = 0.0;
            }

            for (int i = lane_id;
                 i < 256;
                 i += 32)
            {
                int r_idx =
                    i / 16;

                int c_idx =
                    i % 16;

                int global_r =
                    r_base +
                    c_idx;

                int global_a_row =
                    b0 +
                    r_idx;

                if (global_r < R &&
                    global_a_row < dimI)
                {
                    s_factor[i] =
                        static_cast<float>(
                            d_A[
                                global_a_row * R +
                                global_r]);
                }
                else
                {
                    s_factor[i] =
                        0.0f;
                }
            }

            __syncwarp();

            wmma::load_matrix_sync(
                b_frag_0,
                s_factor,
                16);

            wmma::load_matrix_sync(
                b_frag_1,
                s_factor + 128,
                16);

            for (int k = 0;
                 k < 16;
                 ++k)
            {
                if ((b2 + k) >= dimK)
                    continue;

                for (int i = lane_id;
                     i < 256;
                     i += 32)
                {
                    int m1 =
                        i / 16;

                    int m0 =
                        i % 16;

                    s_slice_k[i] =
                        s_tile_dense[
                            m0 * 256 +
                            m1 * 16 +
                            k];
                }

                __syncwarp();

                wmma::fill_fragment(
                    c_frag,
                    0.0f);

                wmma::load_matrix_sync(
                    a_frag_0,
                    s_slice_k,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_0,
                    b_frag_0,
                    c_frag);

                wmma::load_matrix_sync(
                    a_frag_1,
                    s_slice_k + 8,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_1,
                    b_frag_1,
                    c_frag);

                wmma::store_matrix_sync(
                    s_gemm_out,
                    c_frag,
                    16,
                    wmma::mem_row_major);

                __syncwarp();

                for (int iter = 0;
                     iter < 8;
                     ++iter)
                {
                    int idx =
                        iter * 32 +
                        lane_id;

                    int row =
                        idx / 16;

                    int col =
                        idx % 16;

                    int global_r =
                        r_base +
                        col;

                    int global_b_row =
                        b1 +
                        row;

                    if (global_r < R &&
                        global_b_row < dimJ)
                    {
                        double c_val =
                            d_C[
                                (b2 + k) * R +
                                global_r];

                        total_accum[iter] +=
                            static_cast<double>(
                                s_gemm_out[idx]) *
                            c_val;
                    }
                }

                __syncwarp();
            }

            for (int iter = 0;
                 iter < 8;
                 ++iter)
            {
                int idx =
                    iter * 32 +
                    lane_id;

                int row =
                    idx / 16;

                int col =
                    idx % 16;

                int global_r =
                    r_base +
                    col;

                int global_b_row =
                    b1 +
                    row;

                if (global_r < R &&
                    global_b_row < dimJ &&
                    total_accum[iter] != 0.0)
                {
                    atomicAdd(
                        &d_FtV[
                            global_b_row * R +
                            global_r],
                        total_accum[iter]);
                }
            }
        }

        else if (mode == 2)
        {
            double total_accum[8];

            #pragma unroll
            for (int iter = 0;
                 iter < 8;
                 ++iter)
            {
                total_accum[iter] = 0.0;
            }

            for (int i = lane_id;
                 i < 256;
                 i += 32)
            {
                int r_idx =
                    i / 16;

                int c_idx =
                    i % 16;

                int global_r =
                    r_base +
                    c_idx;

                int global_a_row =
                    b0 +
                    r_idx;

                if (global_r < R &&
                    global_a_row < dimI)
                {
                    s_factor[i] =
                        static_cast<float>(
                            d_A[
                                global_a_row * R +
                                global_r]);
                }
                else
                {
                    s_factor[i] =
                        0.0f;
                }
            }

            __syncwarp();

            wmma::load_matrix_sync(
                b_frag_0,
                s_factor,
                16);

            wmma::load_matrix_sync(
                b_frag_1,
                s_factor + 128,
                16);

            for (int j = 0;
                 j < 16;
                 ++j)
            {
                if ((b1 + j) >= dimJ)
                    continue;

                for (int i = lane_id;
                     i < 256;
                     i += 32)
                {
                    int m2 =
                        i / 16;

                    int m0 =
                        i % 16;

                    s_slice_k[i] =
                        s_tile_dense[
                            m0 * 256 +
                            j * 16 +
                            m2];
                }

                __syncwarp();

                wmma::fill_fragment(
                    c_frag,
                    0.0f);

                wmma::load_matrix_sync(
                    a_frag_0,
                    s_slice_k,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_0,
                    b_frag_0,
                    c_frag);

                wmma::load_matrix_sync(
                    a_frag_1,
                    s_slice_k + 8,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_1,
                    b_frag_1,
                    c_frag);

                wmma::store_matrix_sync(
                    s_gemm_out,
                    c_frag,
                    16,
                    wmma::mem_row_major);

                __syncwarp();

                for (int iter = 0;
                     iter < 8;
                     ++iter)
                {
                    int idx =
                        iter * 32 +
                        lane_id;

                    int row =
                        idx / 16;

                    int col =
                        idx % 16;

                    int global_r =
                        r_base +
                        col;

                    int global_c_row =
                        b2 +
                        row;

                    if (global_r < R &&
                        global_c_row < dimK)
                    {
                        double b_val =
                            d_B[
                                (b1 + j) * R +
                                global_r];

                        total_accum[iter] +=
                            static_cast<double>(
                                s_gemm_out[idx]) *
                            b_val;
                    }
                }

                __syncwarp();
            }

            for (int iter = 0;
                 iter < 8;
                 ++iter)
            {
                int idx =
                    iter * 32 +
                    lane_id;

                int row =
                    idx / 16;

                int col =
                    idx % 16;

                int global_r =
                    r_base +
                    col;

                int global_c_row =
                    b2 +
                    row;

                if (global_r < R &&
                    global_c_row < dimK &&
                    total_accum[iter] != 0.0)
                {
                    atomicAdd(
                        &d_FtV[
                            global_c_row * R +
                            global_r],
                        total_accum[iter]);
                }
            }
        }
    }
}

__device__ __forceinline__
void dense_tile_mttkrp_wmma_compact(
    const DenseTile16 tile,

    int tile_index,

    int lane_id,

    float* s_mem_float,

    const int* __restrict__ d_k_offsets,

    int mode,

    const double* __restrict__ d_A,
    const double* __restrict__ d_B,
    const double* __restrict__ d_C,

    double* __restrict__ d_FtV,

    int R,

    int dimI,
    int dimJ,
    int dimK)
{
    const int SHARED_PER_WARP =
        256 +
        256 +
        256;

    float* s_slice_k =
        s_mem_float;

    float* s_factor =
        s_slice_k +
        256;

    float* s_gemm_out =
        s_factor +
        256;

    const int b0 =
        tile.block_idx[0] * TILE_DIM;

    const int b1 =
        tile.block_idx[1] * TILE_DIM;

    const int b2 =
        tile.block_idx[2] * TILE_DIM;

    const int* k_off =
        d_k_offsets
            ? d_k_offsets + tile_index * 16
            : nullptr;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> a_frag_0;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> a_frag_1;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> b_frag_0;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> b_frag_1;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    for (int r_base = 0;
         r_base < R;
         r_base += WMMA_N)
    {

        const double* factor_mat =
            (mode == 0) ? d_B : d_A;

        const int dim_o =
            (mode == 0) ? dimJ : dimI;

        const int origin_o =
            (mode == 0) ? b1 : b0;

        for (int i = lane_id;
             i < 256;
             i += 32)
        {
            const int o_row = i / 16;
            const int r_col = i % 16;

            const int gr = r_base + r_col;

            const int global_o =
                origin_o + o_row;

            if (global_o < dim_o && gr < R)
            {
                s_factor[i] =
                    static_cast<float>(
                        factor_mat[
                            static_cast<size_t>(global_o) * R +
                            gr]);
            }
            else
            {
                s_factor[i] = 0.0f;
            }
        }

        __syncwarp();

        if (mode == 2)
        {

            {
                for (int i = lane_id;
                     i < 256;
                     i += 32)
                {
                    const int o_row = i / 16;
                    const int r_col = i % 16;

                    const int gr = r_base + r_col;

                    const int global_j = b1 + o_row;

                    if (global_j < dimJ && gr < R)
                    {
                        s_factor[i] =
                            static_cast<float>(
                                d_B[
                                    static_cast<size_t>(global_j) * R +
                                    gr]);
                    }
                    else
                    {
                        s_factor[i] = 0.0f;
                    }
                }

                __syncwarp();
            }

            for (int k = 0;
                 k < TILE_DIM;
                 ++k)
            {
                if ((b2 + k) >= dimK)
                    continue;

                for (int i = lane_id;
                     i < 256;
                     i += 32)
                {
                    s_slice_k[i] = 0.0f;
                }

                __syncwarp();

                if (k_off != nullptr)
                {
                    const int k0 = k_off[k];
                    const int k1 =
                        (k == 15) ? tile.nnz : k_off[k + 1];

                    for (int e = k0 + lane_id;
                         e < k1;
                         e += 32)
                    {
                        const int lid = tile.d_coords[e];

                        double raw_val = tile.d_values[e];

                        if (raw_val > FLT_MAX) raw_val = FLT_MAX;
                        if (raw_val < -FLT_MAX) raw_val = -FLT_MAX;

                        s_slice_k[
                            (lid >> 8) * 16 +
                            ((lid >> 4) & 15)] =
                            static_cast<float>(raw_val);
                    }
                }
                else
                {
                    for (int e = lane_id;
                         e < tile.nnz;
                         e += 32)
                    {
                        const int lid = tile.d_coords[e];

                        if ((lid & 15) == k)
                        {
                            double raw_val = tile.d_values[e];

                            if (raw_val > FLT_MAX) raw_val = FLT_MAX;
                            if (raw_val < -FLT_MAX) raw_val = -FLT_MAX;

                            s_slice_k[
                                (lid >> 8) * 16 +
                                ((lid >> 4) & 15)] =
                                static_cast<float>(raw_val);
                        }
                    }
                }

                __syncwarp();

                wmma::fill_fragment(c_frag, 0.0f);

                wmma::load_matrix_sync(a_frag_0, s_slice_k, 16);
                wmma::load_matrix_sync(b_frag_0, s_factor, 16);
                wmma::mma_sync(c_frag, a_frag_0, b_frag_0, c_frag);

                wmma::load_matrix_sync(a_frag_1, s_slice_k + 8, 16);
                wmma::load_matrix_sync(b_frag_1, s_factor + 128, 16);
                wmma::mma_sync(c_frag, a_frag_1, b_frag_1, c_frag);

                wmma::store_matrix_sync(
                    s_gemm_out, c_frag, 16, wmma::mem_row_major);

                __syncwarp();

                for (int i = lane_id;
                     i < 256;
                     i += 32)
                {
                    const int row = i / 16;
                    const int col = i % 16;

                    const int gr = r_base + col;

                    const int global_k = b2 + k;

                    const int global_i = b0 + row;

                    if (gr < R && global_k < dimK && global_i < dimI)
                    {
                        const double a_val =
                            d_A[
                                static_cast<size_t>(global_i) * R +
                                gr];

                        const double contrib =
                            static_cast<double>(s_gemm_out[i]) * a_val;

                        if (contrib != 0.0)
                        {
                            atomicAdd(
                                &d_FtV[
                                    static_cast<size_t>(global_k) * R +
                                    gr],
                                contrib);
                        }
                    }
                }

                __syncwarp();
            }

            continue;
        }

        for (int k = 0;
             k < TILE_DIM;
             ++k)
        {
            if ((b2 + k) >= dimK)
                continue;

            for (int i = lane_id;
                 i < 256;
                 i += 32)
            {
                s_slice_k[i] = 0.0f;
            }

            __syncwarp();

            if (mode == 0)
            {

                if (k_off != nullptr)
                {
                    const int k0 = k_off[k];
                    const int k1 =
                        (k == 15) ? tile.nnz : k_off[k + 1];

                    for (int e = k0 + lane_id;
                         e < k1;
                         e += 32)
                    {
                        const int lid = tile.d_coords[e];

                        double raw_val = tile.d_values[e];

                        if (raw_val > FLT_MAX) raw_val = FLT_MAX;
                        if (raw_val < -FLT_MAX) raw_val = -FLT_MAX;

                        s_slice_k[
                            (lid >> 8) * 16 +
                            ((lid >> 4) & 15)] =
                            static_cast<float>(raw_val);
                    }
                }
                else
                {
                    for (int e = lane_id;
                         e < tile.nnz;
                         e += 32)
                    {
                        const int lid = tile.d_coords[e];

                        if ((lid & 15) == k)
                        {
                            double raw_val = tile.d_values[e];

                            if (raw_val > FLT_MAX) raw_val = FLT_MAX;
                            if (raw_val < -FLT_MAX) raw_val = -FLT_MAX;

                            s_slice_k[
                                (lid >> 8) * 16 +
                                ((lid >> 4) & 15)] =
                                static_cast<float>(raw_val);
                        }
                    }
                }
            }
            else if (mode == 1)
            {

                if (k_off != nullptr)
                {
                    const int k0 = k_off[k];
                    const int k1 =
                        (k == 15) ? tile.nnz : k_off[k + 1];

                    for (int e = k0 + lane_id;
                         e < k1;
                         e += 32)
                    {
                        const int lid = tile.d_coords[e];

                        double raw_val = tile.d_values[e];

                        if (raw_val > FLT_MAX) raw_val = FLT_MAX;
                        if (raw_val < -FLT_MAX) raw_val = -FLT_MAX;

                        s_slice_k[
                            ((lid >> 4) & 15) * 16 +
                            (lid >> 8)] =
                            static_cast<float>(raw_val);
                    }
                }
                else
                {
                    for (int e = lane_id;
                         e < tile.nnz;
                         e += 32)
                    {
                        const int lid = tile.d_coords[e];

                        if ((lid & 15) == k)
                        {
                            double raw_val = tile.d_values[e];

                            if (raw_val > FLT_MAX) raw_val = FLT_MAX;
                            if (raw_val < -FLT_MAX) raw_val = -FLT_MAX;

                            s_slice_k[
                                ((lid >> 4) & 15) * 16 +
                                (lid >> 8)] =
                                static_cast<float>(raw_val);
                        }
                    }
                }
            }
            else
            {

            }

            __syncwarp();

            wmma::fill_fragment(c_frag, 0.0f);

            wmma::load_matrix_sync(a_frag_0, s_slice_k, 16);
            wmma::load_matrix_sync(b_frag_0, s_factor, 16);
            wmma::mma_sync(c_frag, a_frag_0, b_frag_0, c_frag);

            wmma::load_matrix_sync(a_frag_1, s_slice_k + 8, 16);
            wmma::load_matrix_sync(b_frag_1, s_factor + 128, 16);
            wmma::mma_sync(c_frag, a_frag_1, b_frag_1, c_frag);

            wmma::store_matrix_sync(
                s_gemm_out, c_frag, 16, wmma::mem_row_major);

            __syncwarp();

            const double* c_mat =
                (mode == 0) ? d_C :
                (mode == 1) ? d_C : d_B;

            const int dim_other =
                (mode == 2) ? dimJ : dimK;

            for (int i = lane_id;
                 i < 256;
                 i += 32)
            {
                const int row = i / 16;
                const int col = i % 16;

                const int gr = r_base + col;

                const int global_row =
                    (mode == 0) ? b0 + row :
                    (mode == 1) ? b1 + row : b2 + row;

                const int dim_target =
                    (mode == 0) ? dimI :
                    (mode == 1) ? dimJ : dimK;

                if (gr < R && global_row < dim_target)
                {
                    const double c_val =
                        c_mat[
                            (mode == 0)
                                ? (static_cast<size_t>(b2 + k) * R + gr)
                                : (static_cast<size_t>(b2 + k) * R + gr)];

                    (void)dim_other;

                    const double contrib =
                        static_cast<double>(s_gemm_out[i]) * c_val;

                    if (contrib != 0.0)
                    {
                        atomicAdd(
                            &d_FtV[
                                static_cast<size_t>(global_row) * R +
                                gr],
                            contrib);
                    }
                }
            }

            __syncwarp();
        }
    }
}

__global__
void dense_mttkrp_wmma_kernel(
    const DenseTile16* __restrict__ tiles,

    int num_tiles,

    int mode,

    const double* __restrict__ d_A,
    const double* __restrict__ d_B,
    const double* __restrict__ d_C,

    double* __restrict__ d_FtV,

    int R,

    int dimI,
    int dimJ,
    int dimK)
{
    const int global_warp_id =
        blockIdx.x;

    if (global_warp_id >= num_tiles)
        return;

    extern __shared__ float s_mem_float[];

    DenseTile16 tile =
        tiles[global_warp_id];

    dense_tile_mttkrp_wmma(
        tile,

        threadIdx.x % 32,

        s_mem_float,

        mode,

        d_A,
        d_B,
        d_C,

        d_FtV,

        R,

        dimI,
        dimJ,
        dimK);
}

__device__ __forceinline__
void decode_packed_coords_merged(
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

__device__ __forceinline__
void sparse_segment_mttkrp_warp(
    int mode,

    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_sval,

    const uint32_t* __restrict__ d_blk_row,
    const uint32_t* __restrict__ d_blk_start,
    const uint32_t* __restrict__ d_blk_cnt,

    const double* __restrict__ F1,
    const double* __restrict__ F2,

    double* __restrict__ d_FtV,

    int segment_id,

    int lane_id)
{
    constexpr int R = 32;

    const uint32_t row =
        d_blk_row[segment_id];

    const uint32_t start =
        d_blk_start[segment_id];

    const uint32_t cnt =
        d_blk_cnt[segment_id];

    const uint32_t end =
        start + cnt;

    const int r =
        lane_id;

    double acc = 0.0;

    for (uint32_t e = start;
         e < end;
         ++e)
    {
        const uint64_t pc =
            __ldcv(&d_coords[e]);

        const double v =
            __ldcv(&d_sval[e]);

        uint32_t target;
        uint32_t o0;
        uint32_t o1;

        decode_packed_coords_merged(
            pc,
            mode,
            target,
            o0,
            o1);

        (void)target;

        acc +=
            v *
            F1[
                static_cast<size_t>(o0) * R +
                r] *
            F2[
                static_cast<size_t>(o1) * R +
                r];
    }

    if (acc != 0.0)
    {
        atomicAdd(
            &d_FtV[
                static_cast<size_t>(row) * R +
                r],
            acc);
    }
}

__device__ __forceinline__
void sparse_segment_mttkrp_block(
    int mode,

    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_sval,

    const uint32_t* __restrict__ d_blk_row,
    const uint32_t* __restrict__ d_blk_start,
    const uint32_t* __restrict__ d_blk_cnt,

    const double* __restrict__ F1,
    const double* __restrict__ F2,

    double* __restrict__ d_FtV,

    int segment_id,

    int thread_id)
{
    constexpr int R = 32;
    constexpr int THREADS = 128;
    constexpr int GROUPS = THREADS / R;

    const uint32_t row =
        d_blk_row[segment_id];

    const uint32_t start =
        d_blk_start[segment_id];

    const uint32_t cnt =
        d_blk_cnt[segment_id];

    const uint32_t end =
        start + cnt;

    const int r =
        thread_id % R;

    const int g =
        thread_id / R;

    const int lane =
        thread_id & 31;

    const int warp =
        thread_id >> 5;

    double acc = 0.0;

    for (uint32_t e = start + static_cast<uint32_t>(g);
         e < end;
         e += static_cast<uint32_t>(GROUPS))
    {
        const uint64_t pc =
            __ldcv(&d_coords[e]);

        const double v =
            __ldcv(&d_sval[e]);

        uint32_t target;
        uint32_t o0;
        uint32_t o1;

        decode_packed_coords_merged(
            pc,
            mode,
            target,
            o0,
            o1);

        (void)target;

        acc +=
            v *
            F1[
                static_cast<size_t>(o0) * R +
                r] *
            F2[
                static_cast<size_t>(o1) * R +
                r];
    }

    constexpr int WARPS = THREADS / 32;

    __shared__ double s_red[WARPS][R];

    for (int i = thread_id;
         i < WARPS * R;
         i += THREADS)
    {
        (reinterpret_cast<double*>(s_red))[i] = 0.0;
    }

    __syncthreads();

    s_red[warp][r] = acc;

    __syncthreads();

    const int my_rank =
        warp * 32 + lane;

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

__global__
void mttkrp_merged_kernel(
    const DenseTile16* __restrict__ tiles,

    int num_dense_tiles,

    int mode,

    const int* __restrict__ d_k_offsets,

    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_sval,

    const uint32_t* __restrict__ d_blk_row,
    const uint32_t* __restrict__ d_blk_start,
    const uint32_t* __restrict__ d_blk_cnt,

    int num_sparse_segments,

    int dense_stride,

    const double* __restrict__ d_A,
    const double* __restrict__ d_B,
    const double* __restrict__ d_C,

    double* __restrict__ d_FtV,

    int dimI,
    int dimJ,
    int dimK)
{
    const int lane_id =
        threadIdx.x & 31;

    extern __shared__ float s_mem_float[];

    const bool is_dense =
        (num_dense_tiles > 0) &&
        ((blockIdx.x % dense_stride) == 0);

    const int dense_slot =
        blockIdx.x / dense_stride;

    const int sparse_slot =
        blockIdx.x -
        blockIdx.x / dense_stride -
        1;

    const int work_id =
        is_dense ? dense_slot : sparse_slot;

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

    if (is_dense)
    {

        if (work_id >= num_dense_tiles)
            return;

        DenseTile16 tile =
            tiles[work_id];

        dense_tile_mttkrp_wmma_compact(
            tile,

            work_id,

            lane_id,

            s_mem_float,

            d_k_offsets,

            mode,

            d_A,
            d_B,
            d_C,

            d_FtV,

            32,

            dimI,
            dimJ,
            dimK);
    }
    else
    {

        if (work_id >= num_sparse_segments)
            return;

        sparse_segment_mttkrp_warp(
            mode,

            d_coords,
            d_sval,

            d_blk_row,
            d_blk_start,
            d_blk_cnt,

            F1,
            F2,

            d_FtV,

            work_id,

            lane_id);
    }
}

void compute_dense_mttkrp_async(
    const HybridCOOTensor& hybrid,

    int mode,

    double* d_A,
    double* d_B,
    double* d_C,

    double* d_FtV,
    double* d_FtF,

    int R,

    cudaStream_t stream)
{
    (void)d_FtF;

    if (std::getenv("ALS_DBG_KGROUP")) {
        std::cout << "[DBG] kgroup: tiles=" << hybrid.num_dense_tiles
                  << " k_offsets=" << (hybrid.d_dense_k_offsets ? "SET" : "NULL")
                  << std::endl;
    }

    if (hybrid.num_dense_tiles == 0)
        return;

    DenseTile16* d_tiles =
        nullptr;

    const size_t tile_bytes =
        static_cast<size_t>(
            hybrid.num_dense_tiles) *
        sizeof(DenseTile16);

    CHECK_CUDA(
        cudaMallocAsync(
            &d_tiles,
            tile_bytes,
            stream));

    CHECK_CUDA(
        cudaMemcpyAsync(
            d_tiles,

            hybrid.dense_tiles.data(),

            tile_bytes,

            cudaMemcpyHostToDevice,

            stream));

    constexpr int warps_per_block =
        1;

    constexpr int threads_per_block =
        warps_per_block * 32;

    const int blocks =
        (
            hybrid.num_dense_tiles +
            warps_per_block -
            1
        ) /
        warps_per_block;

    const size_t shared_mem_size =
        static_cast<size_t>(
            warps_per_block) *

        (
            4096 +
            256 +
            256 +
            256
        ) *

        sizeof(float);

    CHECK_CUDA(
        cudaFuncSetAttribute(
            dense_mttkrp_wmma_kernel,

            cudaFuncAttributeMaxDynamicSharedMemorySize,

            static_cast<int>(
                shared_mem_size)));

    dense_mttkrp_wmma_kernel<<<
        blocks,
        threads_per_block,
        shared_mem_size,
        stream>>>(
            d_tiles,

            hybrid.num_dense_tiles,

            mode,

            d_A,
            d_B,
            d_C,

            d_FtV,

            R,

            static_cast<int>(
                hybrid.dims[0]),

            static_cast<int>(
                hybrid.dims[1]),

            static_cast<int>(
                hybrid.dims[2]));

    CHECK_CUDA(
        cudaGetLastError());

    CHECK_CUDA(
        cudaFreeAsync(
            d_tiles,
            stream));
}

void compute_mttkrp_merged_async(
    const HybridCOOTensor& hybrid,

    int mode,

    double* d_A,
    double* d_B,
    double* d_C,

    double* d_FtV,
    double* d_FtF,

    int R,

    cudaStream_t stream)
{
    (void)d_FtF;
    (void)R;

    static const bool no_dense =
        std::getenv("ALS_MERGED_NO_DENSE") != nullptr;

    static const bool no_sparse =
        std::getenv("ALS_MERGED_NO_SPARSE") != nullptr;

    if (hybrid.num_dense_tiles == 0 &&
        hybrid.mttkrp_blocks[mode] == 0)
    {
        return;
    }

    DenseTile16* d_tiles =
        nullptr;

    const size_t tile_bytes =
        static_cast<size_t>(
            hybrid.num_dense_tiles) *
        sizeof(DenseTile16);

    if (hybrid.num_dense_tiles > 0)
    {
        CHECK_CUDA(
            cudaMallocAsync(
                &d_tiles,
                tile_bytes,
                stream));

        CHECK_CUDA(
            cudaMemcpyAsync(
                d_tiles,

                hybrid.dense_tiles.data(),

                tile_bytes,

                cudaMemcpyHostToDevice,

                stream));
    }

    const int num_dense_blocks =
        no_dense
            ? 0
            : hybrid.num_dense_tiles;

    const int num_sparse_segments =
        no_sparse
            ? 0
            : static_cast<int>(
                hybrid.mttkrp_blocks[mode]);

    const int max_work =
        num_dense_blocks > num_sparse_segments
            ? num_dense_blocks
            : num_sparse_segments;

    static const int stride_override =
        std::getenv("ALS_MERGED_STRIDE")
            ? std::atoi(std::getenv("ALS_MERGED_STRIDE"))
            : 0;

    const int dense_stride =
        stride_override > 0 ? stride_override :
        (num_sparse_segments == 0) ? 1 :
        (num_dense_blocks > 0)
            ? (num_dense_blocks + num_sparse_segments +
               num_dense_blocks - 1) / num_dense_blocks
            : 2;

    const int grid =
        (no_dense || no_sparse)
            ? max_work
            : (num_dense_blocks * dense_stride > num_sparse_segments
                ? num_dense_blocks * dense_stride
                : num_sparse_segments + num_dense_blocks);

    if (grid == 0)
    {
        return;
    }

    constexpr int shared_per_block =
        (
            256 +
            256 +
            256
        );

    const size_t shared_mem_size =
        static_cast<size_t>(
            shared_per_block) *
        sizeof(float);

    CHECK_CUDA(
        cudaFuncSetAttribute(
            mttkrp_merged_kernel,

            cudaFuncAttributeMaxDynamicSharedMemorySize,

            static_cast<int>(
                shared_mem_size)));

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

    mttkrp_merged_kernel<<<
        grid,
        32,
        shared_mem_size,
        stream>>>(
            d_tiles,

            hybrid.num_dense_tiles,

            mode,

            hybrid.d_dense_k_offsets,

            coords,
            sval,

            bro,
            bst,
            bcn,

            num_sparse_segments,

            dense_stride,

            d_A,
            d_B,
            d_C,

            d_FtV,

            static_cast<int>(
                hybrid.dims[0]),

            static_cast<int>(
                hybrid.dims[1]),

            static_cast<int>(
                hybrid.dims[2]));

    CHECK_CUDA(
        cudaGetLastError());

    if (d_tiles != nullptr)
    {
        CHECK_CUDA(
            cudaFreeAsync(
                d_tiles,
                stream));
    }
}

void compute_dense_mttkrp(
    const HybridCOOTensor& hybrid,

    int mode,

    double* d_A,
    double* d_B,
    double* d_C,

    double* d_FtV,
    double* d_FtF,

    int R)
{
    cudaStream_t stream =
        nullptr;

    CHECK_CUDA(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking));

    compute_dense_mttkrp_async(
        hybrid,

        mode,

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
