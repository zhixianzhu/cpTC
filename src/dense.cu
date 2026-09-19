#include "dense.hpp"
#include "common.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <mma.h>
#include <cfloat>

using namespace nvcuda;

// ============================================================
// Tile configuration
// ============================================================

#define TILE_DIM 16

// CUDA WMMA TF32
// M = 16
// N = 16
// K = 8

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8


// ============================================================
// Device helper
//
// 新数据格式：
//
// DenseTile16
//
//     block_idx[0] = block_i
//     block_idx[1] = block_j
//     block_idx[2] = block_k
//
//     d_coords[i] = local_id
//
// local_id:
//
//     local_id =
//         (local_i * 16 + local_j) * 16
//         + local_k
//
// 因此：
//
//     local_i = local_id / 256
//     local_j = (local_id / 16) % 16
//     local_k = local_id % 16
//
// ============================================================

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


// ============================================================
// Dense Tile MTTKRP Kernel
//
// Tensor Core / WMMA / TF32
//
// 每个 warp 处理一个 Dense Tile。
//
// ============================================================

// ============================================================
// Dense Tile MTTKRP - single warp implementation
//
// 可由独立内核（1 warp/block）或合并内核（稠密+稀疏单网格）
// 调用。s_mem 指向本 warp 的共享内存区域（19KB 完整 tile）。
// ============================================================

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

    // ========================================================
    // Tile global origin
    //
    // block_idx 保存的是块号，不是全局坐标。
    //
    // 例如：
    //
    // block_idx = (2,1,3)
    //
    // TILE_DIM = 16
    //
    // tile origin =
    //
    //     (32,16,48)
    // ========================================================

    int b0 =
        tile.block_idx[0] *
        TILE_DIM;

    int b1 =
        tile.block_idx[1] *
        TILE_DIM;

    int b2 =
        tile.block_idx[2] *
        TILE_DIM;

    // ========================================================
    // Shared memory layout (19KB per warp, full tile)
    // ========================================================

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


    // ========================================================
    // Clear dense tile
    // ========================================================

    for (int i = lane_id;
         i < 4096;
         i += 32)
    {
        s_tile_dense[i] = 0.0f;
    }

    __syncwarp();


    // ========================================================
    // Load sparse entries into dense tile
    //
    // 新格式：
    //
    //     d_coords[i] = local_id
    //
    // 不再使用：
    //
    //     packed_coord >> 8
    //     packed_coord >> 4
    //
    // ========================================================

    for (int i = lane_id;
         i < tile.nnz;
         i += 32)
    {
        // ----------------------------------------------------
        // local_id
        // ----------------------------------------------------

        int local_id =
            tile.d_coords[i];

        // ----------------------------------------------------
        // Decode local coordinate
        // ----------------------------------------------------

        int local_i;
        int local_j;
        int local_k;

        decode_dense_local_id(
            local_id,
            local_i,
            local_j,
            local_k);

        // ----------------------------------------------------
        // Defensive protection
        // ----------------------------------------------------

        if (local_i < 0 ||
            local_i >= TILE_DIM ||
            local_j < 0 ||
            local_j >= TILE_DIM ||
            local_k < 0 ||
            local_k >= TILE_DIM)
        {
            continue;
        }

        // ----------------------------------------------------
        // Value
        // ----------------------------------------------------

        double raw_val =
            tile.d_values[i];

        // ----------------------------------------------------
        // TF32 path uses float.
        //
        // Protect conversion from extreme double values.
        // ----------------------------------------------------

        if (raw_val > FLT_MAX)
            raw_val = FLT_MAX;

        if (raw_val < -FLT_MAX)
            raw_val = -FLT_MAX;

        // ----------------------------------------------------
        // Dense tile layout:
        //
        //     [i][j][k]
        //
        // linear:
        //
        //     i * 256
        //   + j * 16
        //   + k
        // ----------------------------------------------------

        s_tile_dense[
            local_i * 256 +
            local_j * 16 +
            local_k] =
            static_cast<float>(
                raw_val);
    }

    __syncwarp();


    // ========================================================
    // WMMA fragments
    //
    // TF32 input
    // FP32 accumulator
    // ========================================================

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


    // ========================================================
    // Rank blocking
    //
    // 每次处理 16 个 rank。
    // ========================================================

    for (int r_base = 0;
         r_base < R;
         r_base += 16)
    {

        // ====================================================
        // Mode 0
        //
        // 更新 A
        //
        // A(i,r) =
        //
        //     sum_jk
        //
        //         X(i,j,k)
        //         * B(j,r)
        //         * C(k,r)
        //
        // ====================================================

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


            // ------------------------------------------------
            // B tile 一次性载入共享内存（k 不变，避免 16 次重载）
            // ------------------------------------------------

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

            // ------------------------------------------------
            // Iterate k inside tile
            // ------------------------------------------------

            for (int k = 0;
                 k < 16;
                 ++k)
            {
                if ((b2 + k) >= dimK)
                    continue;


                // ============================================
                // Construct X(:,:,k)
                //
                // s_slice_k:
                //
                //     rows = i
                //     cols = j
                //
                // ============================================

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


                // ============================================
                // Tensor Core GEMM
                //
                // 16x16x16
                //
                // WMMA K = 8
                //
                // split:
                //
                //     K = 0..7
                //     K = 8..15
                // ============================================

                wmma::fill_fragment(
                    c_frag,
                    0.0f);


                // ------------------------------------------------
                // K = 0..7
                // ------------------------------------------------

                wmma::load_matrix_sync(
                    a_frag_0,
                    s_slice_k,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_0,
                    b_frag_0,
                    c_frag);


                // ------------------------------------------------
                // K = 8..15
                // ------------------------------------------------

                wmma::load_matrix_sync(
                    a_frag_1,
                    s_slice_k + 8,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_1,
                    b_frag_1,
                    c_frag);


                // ============================================
                // Store GEMM output
                // ============================================

                wmma::store_matrix_sync(
                    s_gemm_out,
                    c_frag,
                    16,
                    wmma::mem_row_major);

                __syncwarp();


                // ============================================
                // Multiply C
                // ============================================

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


            // ================================================
            // Atomic update A
            // ================================================

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


        // ====================================================
        // Mode 1
        //
        // 更新 B
        //
        // B(j,r) =
        //
        //     sum_ik
        //
        //         X(i,j,k)
        //         * A(i,r)
        //         * C(k,r)
        //
        // ====================================================

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


            // A tile 一次性载入共享内存（k 不变）
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


                // ============================================
                // X(:, :, k)
                //
                // rows = j
                // cols = i
                // ============================================

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


                // ============================================
                // WMMA
                // ============================================

                wmma::fill_fragment(
                    c_frag,
                    0.0f);


                // K = 0..7

                wmma::load_matrix_sync(
                    a_frag_0,
                    s_slice_k,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_0,
                    b_frag_0,
                    c_frag);


                // K = 8..15

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


                // ============================================
                // Multiply C
                // ============================================

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


            // ================================================
            // Atomic update B
            // ================================================

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


        // ====================================================
        // Mode 2
        //
        // 更新 C
        //
        // C(k,r) =
        //
        //     sum_ij
        //
        //         X(i,j,k)
        //         * A(i,r)
        //         * B(j,r)
        //
        // ====================================================

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


            // ------------------------------------------------
            // Iterate j inside tile
            // ------------------------------------------------

            // A tile 一次性载入共享内存（j 不变）
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


                // ============================================
                // X(:,j,:)
                //
                // rows = k
                // cols = i
                // ============================================

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


                // ============================================
                // WMMA
                // ============================================

                wmma::fill_fragment(
                    c_frag,
                    0.0f);


                // K = 0..7

                wmma::load_matrix_sync(
                    a_frag_0,
                    s_slice_k,
                    16);

                wmma::mma_sync(
                    c_frag,
                    a_frag_0,
                    b_frag_0,
                    c_frag);


                // K = 8..15

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


                // ============================================
                // Multiply B
                // ============================================

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


            // ================================================
            // Atomic update C
            // ================================================

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


// ============================================================
// Dense Tile MTTKRP - COMPACT shared-memory variant
//
// 不物化完整 16^3 tile：每个 k 切片通过扫描 tile 条目
// （全局池）构建。共享内存仅 3KB/block，使合并内核中
// 稠密/稀疏 block 能共驻 SM（17 blocks/SM）。
//
// 注意：本变体比完整 tile 变体慢（逐 k 扫描全局池），
// 仅用于合并内核 —— 其耗时藏在稀疏 44ms 之下。
// ============================================================

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

    // k 分组偏移（相对 tile 池段起点）
    const int* k_off =
        d_k_offsets
            ? d_k_offsets + tile_index * 16
            : nullptr;

    // WMMA fragments (TF32 input, FP32 accumulator)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> a_frag_0;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> a_frag_1;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> b_frag_0;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> b_frag_1;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // rank 分块：一次处理 16 个 rank
    for (int r_base = 0;
         r_base < R;
         r_base += WMMA_N)
    {
        // Stage factor rows into smem (B for mode 0, A for modes 1/2)
        //
        // 注意：因子行索引 = tile 块原点 + 块内局部索引
        // （mode 0: B 的 j 行 -> b1；mode 1/2: A 的 i 行 -> b0）
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
            // ==================================================
            // Mode 2: 更新 C
            //
            // C(k,r) = sum_ij X(i,j,k) * A(i,r) * B(j,r)
            //
            // k 分组池下改为：
            //   D(i,r) = sum_j X(i,j,k) * B(j,r)   （同 mode 0 切片）
            //   C(k,r) += sum_i A(i,r) * D(i,r)
            //
            // 只用 k 分组（1x 池扫描），不再按 j 扫描。
            // ==================================================

            // 先载入 B 因子到 s_factor（mode 2 的 factor 是 B，k 无关）
            {
                for (int i = lane_id;
                     i < 256;
                     i += 32)
                {
                    const int o_row = i / 16;   // j
                    const int r_col = i % 16;   // rank

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

                // slice X(i,j,k)：rows=i, cols=j（同 mode 0）
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

                // WMMA: D(i, r) += X(i,j,k) * B(j, r)
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

                // C(k,r) += sum_i A(i,r) * D(i,r)
                for (int i = lane_id;
                     i < 256;
                     i += 32)
                {
                    const int row = i / 16;   // i
                    const int col = i % 16;   // rank

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

            continue;  // 下一 rank 块
        }

        for (int k = 0;
             k < TILE_DIM;
             ++k)
        {
            if ((b2 + k) >= dimK)
                continue;

            // Build X(:,:,k) slice by scanning tile entries
            for (int i = lane_id;
                 i < 256;
                 i += 32)
            {
                s_slice_k[i] = 0.0f;
            }

            __syncwarp();

            if (mode == 0)
            {
                // rows = i, cols = j, fix k
                // k 分组池：只扫描 [k_off[k], k_off[k+1]) 连续区间
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
                // rows = j, cols = i, fix k
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
                // mode 2 已在独立循环处理，这里不可达
            }

            __syncwarp();

            // WMMA: C(i, rank) += X(i,j,k) * F(j, rank)
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

            // C = sum over k: FtV[row][rank] += C_out * C_factor[k][rank]
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


// ============================================================
// Standalone dense MTTKRP kernel (1 warp per block)
// ============================================================

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

    // 完整 tile 方案（实测 13.8ms/模式，双流串行下最优）
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


// ============================================================
// MERGED MTTKRP kernel (dense tiles + sparse segments in ONE grid)
//
// 目标：本驱动（560.35.05）上两条独立流的内核从不并行
// （work distributor 网格级串行，8 组微基准验证），
// 唯一能真正并行的是单网格内按 blockIdx 分工：
//
//     blockIdx.x % 2 == 0  -> 稠密 tile（张量核心 WMMA）
//     blockIdx.x % 2 == 1  -> 稀疏段（FP64 CUDA 核）
//
// 强制分 SM：通过 block 数量比例控制两路各占多少 SM 槽位。
// 合并后寄存器取两路最大值（稠密 116），但实测稀疏核在
// 8 warps/SM 下吞吐不变（带宽受限而非延迟受限），
// 因此合并不会伤稀疏性能。
//
// 本内核只处理 R == 32 的情况（1 warp 恰好覆盖 32 个 rank，
// 稀疏段无需跨 warp 归约）。R 其它取值走原双流路径。
// ============================================================

// ============================================================
// Packed-coordinate decode（与 sparse.cu 同款，合并内核内联使用）
// ============================================================

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


// ============================================================
// 1-warp sparse segment（32 线程 = 32 rank，lane 即 rank）
//
// 32 线程/block 的合并内核使用：无需跨 warp 归约、无需 s_red，
// 共享内存占用为 0 —— 这是让合并内核每 SM 驻留 33 个 block 的关键。
// ============================================================

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

    // lane 即 rank：每个 lane 只累加自己的 rank
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

    // 跨 warp 归约（与 sparse.cu 的 coalesced kernel 相同）
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


// 合并内核：每个 block 128 线程（4 warps）。
//
// blockIdx 偶数 -> 稠密 block：4 个 warp 各处理一个稠密 tile
//                 （warp w 处理 tiles[4*work_id + w]），
//                 每 warp 3KB 共享内存（紧凑 tile），共 12KB。
//
// blockIdx 奇数 -> 稀疏 block：128 线程协作处理一个稀疏段
//                 （与 sparse.cu 的 coalesced kernel 同结构，
//                  跨 warp 用共享内存归约），1KB 共享内存。
//
// 共享内存统一分配 12KB（稠密路径的上限）：
//     100KB / 12KB = 8 blocks/SM；
//     寄存器取两路最大值 ~116 -> 4 blocks/SM（16 warps/SM）。
//     稀疏部分约占 8 warps/SM —— 实测稀疏核在该占用率下吞吐不变。
//
// 本内核只处理 R == 32。

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

    int dense_every,

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

    // -------- role assignment (load-balanced fused grid) --------
    //  * both sides present: dense blocks occupy every dense_every-th
    //    position (bi % dense_every == 0), sparse blocks fill the rest;
    //    host sizes the grid so that ALL dense tiles and ALL sparse
    //    segments are covered, guards below just idle the few leftovers.
    //  * one-sided grids (only dense or only sparse) map directly.
    const int bi = blockIdx.x;

    bool is_dense = false;
    int  work_id  = 0;

    if (num_sparse_segments <= 0)
    {
        is_dense = true;
        work_id  = bi;
    }
    else if (num_dense_tiles <= 0)
    {
        is_dense = false;
        work_id  = bi;
    }
    else
    {
        is_dense = ((bi % dense_every) == 0);
        if (is_dense)
            work_id = bi / dense_every;
        else
            work_id = bi - (bi + dense_every - 1) / dense_every;
    }

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
        // 稠密 block：1 个 warp 处理 1 个 tile（紧凑 tile，3KB smem）
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
        // 稀疏 block：1 个 warp 处理 1 个段（lane 即 rank，0 smem）
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


// ============================================================
// Async dense MTTKRP launcher
//
// Important:
//
//     NO cudaDeviceSynchronize()
//
// All CUDA operations are submitted to caller-provided stream.
//
// ============================================================

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


    // ========================================================
    // Device tile metadata
    // ========================================================

    DenseTile16* d_tiles =
        nullptr;

    const size_t tile_bytes =
        static_cast<size_t>(
            hybrid.num_dense_tiles) *
        sizeof(DenseTile16);


    // ========================================================
    // Stream ordered allocation
    // ========================================================

    CHECK_CUDA(
        cudaMallocAsync(
            &d_tiles,
            tile_bytes,
            stream));


    // ========================================================
    // Copy tile metadata
    //
    // 注意：
    //
    // DenseTile16 中的 d_coords / d_values
    // 本身是 GPU pointer。
    //
    // 这些 pointer 指向：
    //
    //     hybrid.d_dense_coords_pool
    //     hybrid.d_dense_values_pool
    //
    // 因此这里仅复制 metadata。
    // ========================================================

    CHECK_CUDA(
        cudaMemcpyAsync(
            d_tiles,

            hybrid.dense_tiles.data(),

            tile_bytes,

            cudaMemcpyHostToDevice,

            stream));


    // ========================================================
    // Kernel configuration
    // ========================================================

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


    // ========================================================
    // Shared memory
    // ========================================================

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


    // ========================================================
    // Launch
    // ========================================================

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


    // ========================================================
    // Stream ordered free
    // ========================================================

    CHECK_CUDA(
        cudaFreeAsync(
            d_tiles,
            stream));
}


// ============================================================
// MERGED MTTKRP launcher
//
// 稠密 tile（张量核心）+ 稀疏段（CUDA 核）合并在单网格中，
// 按 blockIdx 奇偶分工 —— 本驱动唯一能真正并行两路负载的方式。
//
// 仅支持 R == 32。其它 R 走原双流路径（调用方保证）。
// ============================================================

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

    // 实验开关：ALS_MERGED_NO_DENSE=1 时跳过稠密（测稀疏路径本身）
    static const bool no_dense =
        std::getenv("ALS_MERGED_NO_DENSE") != nullptr;

    // ALS_MERGED_NO_SPARSE=1 时跳过稀疏（测稠密路径本身）
    static const bool no_sparse =
        std::getenv("ALS_MERGED_NO_SPARSE") != nullptr;

    if (hybrid.num_dense_tiles == 0 &&
        hybrid.mttkrp_blocks[mode] == 0)
    {
        return;
    }

    // ========================================================
    // Device tile metadata (dense tiles only)
    // ========================================================

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

    // ========================================================
    // Load-balanced grid sizing
    // ========================================================

    // 稠密 block：每个 1 个 tile（1 warp）
    const int num_dense_blocks =
        no_dense
            ? 0
            : hybrid.num_dense_tiles;

    const int num_sparse_segments =
        no_sparse
            ? 0
            : static_cast<int>(
                hybrid.mttkrp_blocks[mode]);

    if (num_dense_blocks == 0 &&
        num_sparse_segments == 0)
    {
        return;
    }

    // dense_every = k  -> 每 k 个 block 中 1 个是稠密 tile
    //（其余 k-1 个是稀疏段），实现两类工作在同一 grid 内按
    // 硬件调度交错（真正的 TC/CUDA-core 并发）。
    static const int stride_override =
        std::getenv("ALS_MERGED_STRIDE")
            ? std::atoi(std::getenv("ALS_MERGED_STRIDE"))
            : 0;

    int dense_every = 1;
    int grid = 0;

    if (num_sparse_segments <= 0)
    {
        grid = num_dense_blocks;              // dense-only grid
    }
    else if (num_dense_blocks <= 0)
    {
        grid = num_sparse_segments;           // sparse-only grid
    }
    else
    {
        int k = stride_override;

        if (k <= 0)
        {
            const double frac_dense =
                static_cast<double>(num_dense_blocks) /
                (static_cast<double>(num_dense_blocks) +
                 static_cast<double>(num_sparse_segments));

            // 稠密块约每 (1/frac_dense) 个槽位出现一次
            k = std::max(
                2,
                static_cast<int>(
                    std::round(1.0 / frac_dense)));
        }

        dense_every = k;

        // 总网格 T 必须完整覆盖所有稠密 tile 与所有稀疏段，
        // 内核守卫只让少量尾部槽位空转。
        size_t T = std::max(
            static_cast<size_t>(num_dense_blocks) +
                static_cast<size_t>(num_sparse_segments),
            static_cast<size_t>(k) *
                    (static_cast<size_t>(num_dense_blocks) - 1) +
                1);

        while (T - (T + static_cast<size_t>(k) - 1) /
                       static_cast<size_t>(k) <
               static_cast<size_t>(num_sparse_segments))
        {
            ++T;
        }

        grid = static_cast<int>(T);
    }

    if (grid <= 0)
    {
        return;
    }

    // ========================================================
    // Shared memory: 1 warp x 3KB compact tile = 3KB
    // （33 blocks/SM，稀疏 1-warp 路径 0 smem）
    // ========================================================

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

    // ========================================================
    // Sparse views
    // ========================================================

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

    // ========================================================
    // Launch
    // ========================================================

    mttkrp_merged_kernel<<<
        grid,
        32,
        shared_mem_size,
        stream>>>(
            d_tiles,

            num_dense_blocks,

            mode,

            hybrid.d_dense_k_offsets,

            coords,
            sval,

            bro,
            bst,
            bcn,

            num_sparse_segments,

            dense_every,

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

    // ========================================================
    // Stream ordered free
    // ========================================================

    if (d_tiles != nullptr)
    {
        CHECK_CUDA(
            cudaFreeAsync(
                d_tiles,
                stream));
    }
}


// ============================================================
// Backward-compatible synchronous dense MTTKRP
// ============================================================

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
