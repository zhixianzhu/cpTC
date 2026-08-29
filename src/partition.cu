#include "partition.hpp"
#include "sparse.hpp"
#include "common.hpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// ============================================================
// GPU Kernel 1
//
// 为每个 NNZ 计算 block_id
//
// 原始坐标：
//
//     (m0, m1, m2)
//
// 转换：
//
//     (block_i, block_j, block_k)
//              ↓
//           block_id
//
// block_id:
//
//     block_id =
//         (block_i * num_blocks_j + block_j)
//         * num_blocks_k + block_k
//
// 同时保存原始 NNZ index，后续 RadixSort 按 block_id 排序。
// ============================================================

__global__ void generate_block_keys_kernel(
    const int* __restrict__ d_m0,
    const int* __restrict__ d_m1,
    const int* __restrict__ d_m2,

    uint64_t* __restrict__ d_keys,
    uint32_t* __restrict__ d_indices,

    size_t nnz,

    uint64_t num_blocks_j,
    uint64_t num_blocks_k)
{
    size_t i =
        static_cast<size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    if (i >= nnz)
        return;

    // --------------------------------------------------------
    // Block coordinates
    // --------------------------------------------------------

    uint64_t block_i =
        static_cast<uint64_t>(d_m0[i]) /
        TILE_DIM;

    uint64_t block_j =
        static_cast<uint64_t>(d_m1[i]) /
        TILE_DIM;

    uint64_t block_k =
        static_cast<uint64_t>(d_m2[i]) /
        TILE_DIM;

    // --------------------------------------------------------
    // Encode:
    //
    // (block_i, block_j, block_k)
    //             ↓
    //          block_id
    // --------------------------------------------------------

    uint64_t block_id =
        (
            block_i *
            num_blocks_j +
            block_j
        ) *
        num_blocks_k +
        block_k;

    d_keys[i] = block_id;

    // 当前项目要求 NNZ <= UINT32_MAX
    d_indices[i] =
        static_cast<uint32_t>(i);
}


// ============================================================
// GPU Kernel 2
//
// 每个 CUDA block 处理一个 block run。
//
// 一个 run：
//
//     sorted[start ... start + count)
//
// Dense:
//
//     local_id + value
//
// Sparse:
//
//     block_id + local_id + value
//
// ============================================================

__global__ void scatter_partition_kernel(
    const uint32_t* __restrict__ d_sorted_indices,

    const uint64_t* __restrict__ d_unique_keys,

    const int* __restrict__ d_run_counts,

    const size_t* __restrict__ d_run_starts,

    const int* __restrict__ d_dense_flags,

    const size_t* __restrict__ d_dense_offsets,

    const size_t* __restrict__ d_sparse_offsets,

    size_t num_runs,

    const int* __restrict__ d_m0,
    const int* __restrict__ d_m1,
    const int* __restrict__ d_m2,

    const double* __restrict__ d_val,

    // --------------------------------------------------------
    // Dense output
    // --------------------------------------------------------

    int* __restrict__ d_dense_coords_pool,

    double* __restrict__ d_dense_values_pool,

    // --------------------------------------------------------
    // Sparse output
    //
    // NEW:
    //
    //     完整坐标 (i, j, k) + value
    //
    // MTTKRP kernel 不再需要 64 位 block_id 除法解码。
    // --------------------------------------------------------

    uint32_t* __restrict__ d_sp_i,

    uint32_t* __restrict__ d_sp_j,

    uint32_t* __restrict__ d_sp_k,

    double* __restrict__ d_sp_val)
{
    size_t run =
        static_cast<size_t>(blockIdx.x);

    if (run >= num_runs)
        return;

    int count =
        d_run_counts[run];

    size_t sorted_start =
        d_run_starts[run];

    bool is_dense =
        d_dense_flags[run] != 0;

    size_t output_offset =
        is_dense
            ? d_dense_offsets[run]
            : d_sparse_offsets[run];

    // --------------------------------------------------------
    // 当前 run 对应的 block_id
    // --------------------------------------------------------

    uint64_t block_id =
        d_unique_keys[run];

    // --------------------------------------------------------
    // 一个 CUDA block 处理一个 tensor block
    // --------------------------------------------------------

    for (int k = threadIdx.x;
         k < count;
         k += blockDim.x)
    {
        size_t sorted_pos =
            sorted_start +
            static_cast<size_t>(k);

        uint32_t original_idx =
            d_sorted_indices[sorted_pos];

        // ----------------------------------------------------
        // 原始 global coordinate
        // ----------------------------------------------------

        int i =
            d_m0[original_idx];

        int j =
            d_m1[original_idx];

        int l =
            d_m2[original_idx];

        // ----------------------------------------------------
        // Local coordinate
        //
        // 0 <= local_* < TILE_DIM
        // ----------------------------------------------------

        int local_i =
            i % TILE_DIM;

        int local_j =
            j % TILE_DIM;

        int local_k =
            l % TILE_DIM;

        // ----------------------------------------------------
        // Linear local_id
        //
        // local_id =
        //
        //     (local_i * TILE_DIM + local_j)
        //     * TILE_DIM + local_k
        //
        // TILE_DIM = 16:
        //
        //     local_id in [0, 4095]
        // ----------------------------------------------------

        uint16_t local_id =
            static_cast<uint16_t>(
                (
                    local_i * TILE_DIM +
                    local_j
                ) *
                TILE_DIM +
                local_k
            );

        size_t dst =
            output_offset +
            static_cast<size_t>(k);

        // ----------------------------------------------------
        // Dense
        //
        // Dense pool 仍然使用 int，
        // 但里面存储的已经是 linear local_id。
        // ----------------------------------------------------

        if (is_dense)
        {
            d_dense_coords_pool[dst] =
                static_cast<int>(local_id);

            d_dense_values_pool[dst] =
                d_val[original_idx];
        }

        // ----------------------------------------------------
        // Sparse
        //
        // 直接保存完整坐标 (i, j, k)。
        // ----------------------------------------------------

        else
        {
            d_sp_i[dst] =
                static_cast<uint32_t>(i);

            d_sp_j[dst] =
                static_cast<uint32_t>(j);

            d_sp_k[dst] =
                static_cast<uint32_t>(l);

            d_sp_val[dst] =
                d_val[original_idx];
        }
    }
}


// ============================================================
// GPU partition
// ============================================================

// ============================================================
// Dense pool k-grouping
//
// 把每个稠密 tile 池段内的条目按 local_k 重排（k 分组），
// 并记录每个 k 的起始偏移。这样 WMMA 内核构建 slice k 时
// 只需扫描 [k_off[k], k_off[k+1]) 连续区间（1 次池读），
// 而不是对每个 k 全池扫描（16 次池读）。
//
// 每个 CUDA block 处理一个稠密 tile。
// ============================================================

__global__ void dense_k_group_kernel(
    int* __restrict__ d_coords_pool,
    double* __restrict__ d_values_pool,
    int* __restrict__ d_k_offsets,
    const int* __restrict__ d_tile_nnz,
    const size_t* __restrict__ d_tile_offsets,
    int num_tiles)
{
    const int t = blockIdx.x;

    if (t >= num_tiles)
        return;

    const int nnz =
        d_tile_nnz[t];

    if (nnz == 0)
        return;

    const size_t base =
        d_tile_offsets[t];

    // 16 个 k 的直方图 + 前缀和 + 放置游标
    __shared__ int s_hist[16];
    __shared__ int s_start[16];
    __shared__ int s_cursor[16];

    for (int i = threadIdx.x;
         i < 16;
         i += blockDim.x)
    {
        s_hist[i] = 0;
        s_start[i] = 0;
        s_cursor[i] = 0;
    }

    __syncthreads();

    // Pass 1: 统计每个 k 的条目数
    for (int e = threadIdx.x;
         e < nnz;
         e += blockDim.x)
    {
        const int lid =
            d_coords_pool[base + e];

        const int k =
            lid & 15;

        atomicAdd(&s_hist[k], 1);
    }

    __syncthreads();

    // 前缀和（16 元素，单线程串行即可）
    if (threadIdx.x == 0)
    {
        int acc = 0;
        for (int k = 0; k < 16; ++k)
        {
            s_start[k] = acc;
            s_cursor[k] = acc;
            acc += s_hist[k];
        }
    }

    __syncthreads();

    // Pass 2: 重排到临时区（同一 block 内先写 s_scratch）
    extern __shared__ char s_scratch_raw[];
    int* s_scratch_coords =
        reinterpret_cast<int*>(s_scratch_raw);

    // double 需要 8 字节对齐
    const size_t int_bytes =
        static_cast<size_t>(nnz) * sizeof(int);

    const size_t aligned_off =
        (int_bytes + 7u) & ~size_t(7u);

    double* s_scratch_vals =
        reinterpret_cast<double*>(
            s_scratch_raw + aligned_off);

    for (int e = threadIdx.x;
         e < nnz;
         e += blockDim.x)
    {
        const int lid =
            d_coords_pool[base + e];

        const int k =
            lid & 15;

        const int dst =
            atomicAdd(&s_cursor[k], 1);

        s_scratch_coords[dst] =
            lid;

        s_scratch_vals[dst] =
            d_values_pool[base + e];
    }

    __syncthreads();

    // 写回池段（k 分组有序）
    for (int e = threadIdx.x;
         e < nnz;
         e += blockDim.x)
    {
        d_coords_pool[base + e] =
            s_scratch_coords[e];

        d_values_pool[base + e] =
            s_scratch_vals[e];
    }

    // 记录每个 k 的起始偏移（相对 tile 起点）
    for (int k = threadIdx.x;
         k < 16;
         k += blockDim.x)
    {
        d_k_offsets[
            t * 16 + k] =
            s_start[k];
    }
}


// 启动 dense pool k-grouping（在 partition 之后、视图构建之前）
static void k_group_dense_pool(
    HybridCOOTensor& hybrid,
    cudaStream_t stream)
{
    if (hybrid.num_dense_tiles == 0 ||
        hybrid.d_dense_coords_pool == nullptr)
    {
        return;
    }

    const int nt =
        hybrid.num_dense_tiles;

    // tile 偏移表（host 端已有序：dense_tiles 按创建顺序）
    std::vector<size_t> h_off(
        static_cast<size_t>(nt) + 1,
        0);

    for (int t = 0; t < nt; ++t)
    {
        h_off[t + 1] =
            h_off[t] +
            static_cast<size_t>(
                hybrid.dense_tiles[t].nnz);
    }

    size_t* d_off = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_off,
        (static_cast<size_t>(nt) + 1) *
            sizeof(size_t)));

    CHECK_CUDA(cudaMemcpyAsync(
        d_off,
        h_off.data(),
        (static_cast<size_t>(nt) + 1) *
            sizeof(size_t),
        cudaMemcpyHostToDevice,
        stream));

    int* d_nnz = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_nnz,
        static_cast<size_t>(nt) *
            sizeof(int)));

    // 最大 tile nnz（决定动态共享内存大小）
    size_t max_nnz = 0;

    for (int t = 0; t < nt; ++t)
    {
        const size_t n =
            static_cast<size_t>(
                hybrid.dense_tiles[t].nnz);

        if (n > max_nnz)
            max_nnz = n;
    }

    if (max_nnz == 0)
        return;

    // k 偏移表
    CHECK_CUDA(cudaMalloc(
        &hybrid.d_dense_k_offsets,
        static_cast<size_t>(nt) *
            16 *
            sizeof(int)));

    // 把 nnz 拷到设备（用 cudaMemcpyAsync 逐项不便，用一次 memcpy）
    std::vector<int> h_nnz(
        static_cast<size_t>(nt));

    for (int t = 0; t < nt; ++t)
    {
        h_nnz[t] =
            hybrid.dense_tiles[t].nnz;
    }

    CHECK_CUDA(cudaMemcpyAsync(
        d_nnz,
        h_nnz.data(),
        static_cast<size_t>(nt) *
            sizeof(int),
        cudaMemcpyHostToDevice,
        stream));

    // 共享内存：coords int[] + values double[]，最多 4096 条目
    const size_t int_bytes =
        max_nnz * sizeof(int);

    const size_t aligned_off =
        (int_bytes + 7u) & ~size_t(7u);

    const size_t scratch_bytes =
        aligned_off +
        max_nnz * sizeof(double);

    CHECK_CUDA(
        cudaFuncSetAttribute(
            dense_k_group_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(
                scratch_bytes)));

    dense_k_group_kernel<<<
        nt,
        256,
        scratch_bytes,
        stream>>>(
            hybrid.d_dense_coords_pool,
            hybrid.d_dense_values_pool,
            hybrid.d_dense_k_offsets,
            d_nnz,
            d_off,
            nt);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaFree(d_off);
    cudaFree(d_nnz);
}




void partition_tensor_hybrid_gpu(
    const COOTensor& tensor,
    HybridCOOTensor& hybrid,
    double dense_threshold,
    cudaStream_t stream)
{
    std::cout
        << "[Partition] GPU partitioning for NNZ: "
        << tensor.nnz
        << "..."
        << std::endl;

    // ========================================================
    // 0. Basic checks
    // ========================================================

    if (tensor.nnz == 0)
    {
        free_hybrid_tensor(hybrid);

        hybrid.dims[0] = tensor.dims[0];
        hybrid.dims[1] = tensor.dims[1];
        hybrid.dims[2] = tensor.dims[2];

        hybrid.num_blocks[0] =
            (tensor.dims[0] + TILE_DIM - 1) /
            TILE_DIM;

        hybrid.num_blocks[1] =
            (tensor.dims[1] + TILE_DIM - 1) /
            TILE_DIM;

        hybrid.num_blocks[2] =
            (tensor.dims[2] + TILE_DIM - 1) /
            TILE_DIM;

        hybrid.num_dense_tiles = 0;
        hybrid.sparse_nnz = 0;

        std::cout
            << "[Partition] Empty tensor."
            << std::endl;

        return;
    }

    if (tensor.nnz >
        static_cast<size_t>(
            std::numeric_limits<uint32_t>::max()))
    {
        std::cerr
            << "[Partition] ERROR: NNZ exceeds "
               "uint32_t range."
            << std::endl;

        std::exit(EXIT_FAILURE);
    }

    // ========================================================
    // 1. Clear old Hybrid
    // ========================================================

    free_hybrid_tensor(hybrid);

    hybrid.dims[0] = tensor.dims[0];
    hybrid.dims[1] = tensor.dims[1];
    hybrid.dims[2] = tensor.dims[2];

    hybrid.num_dense_tiles = 0;
    hybrid.sparse_nnz = 0;

    hybrid.dense_tiles.clear();

    // ========================================================
    // 2. Tile / Block grid
    // ========================================================

    const size_t num_blocks_i =
        (tensor.dims[0] + TILE_DIM - 1) /
        TILE_DIM;

    const size_t num_blocks_j =
        (tensor.dims[1] + TILE_DIM - 1) /
        TILE_DIM;

    const size_t num_blocks_k =
        (tensor.dims[2] + TILE_DIM - 1) /
        TILE_DIM;

    hybrid.num_blocks[0] =
        num_blocks_i;

    hybrid.num_blocks[1] =
        num_blocks_j;

    hybrid.num_blocks[2] =
        num_blocks_k;

    const uint64_t num_blocks_j_u64 =
        static_cast<uint64_t>(
            num_blocks_j);

    const uint64_t num_blocks_k_u64 =
        static_cast<uint64_t>(
            num_blocks_k);

    // ========================================================
    // 3. Dense threshold
    //
    // 一个 16^3 block：
    //
    //     4096 positions
    //
    // dense_threshold > 0 时：cutoff = 4096 * threshold（如 0.03 -> 122）
    // dense_threshold <= 0 时：自动平衡模式——
    //   选择 cutoff 使"稠密 tile 交给张量核(WMMA)的工作量"与
    //   "稀疏残差交给 CUDA 核的工作量"近似相等。
    //
    // 实测速率（RTX 4060, R=32）：
    //   张量核 WMMA：约 1.20 ns/元素
    //   CUDA 核合并内核：约 0.688 ns/元素
    //   平衡点：稠密占比 = 0.688/(0.688+1.20) ≈ 36.4%
    // ========================================================

    const double block_volume =
        static_cast<double>(
            TILE_DIM *
            TILE_DIM *
            TILE_DIM);

    int dense_nnz_cutoff = 0;

    if (dense_threshold > 0.0)
    {
        dense_nnz_cutoff =
            static_cast<int>(
                block_volume *
                dense_threshold);

        if (dense_nnz_cutoff < 1)
            dense_nnz_cutoff = 1;

        if (dense_nnz_cutoff >
            TILE_DIM * TILE_DIM * TILE_DIM)
        {
            dense_nnz_cutoff =
                TILE_DIM * TILE_DIM * TILE_DIM;
        }

        std::cout
            << "[Partition] Dense cutoff: "
            << dense_nnz_cutoff
            << " / "
            << TILE_DIM *
               TILE_DIM *
               TILE_DIM
            << std::endl;
    }

    // ========================================================
    // 4. Allocate block keys / original indices
    // ========================================================

    uint64_t* d_keys_in = nullptr;
    uint64_t* d_keys_out = nullptr;

    uint32_t* d_indices_in = nullptr;
    uint32_t* d_indices_out = nullptr;

    const size_t nnz_bytes_u64 =
        tensor.nnz *
        sizeof(uint64_t);

    const size_t nnz_bytes_u32 =
        tensor.nnz *
        sizeof(uint32_t);

    CHECK_CUDA(cudaMalloc(
        &d_keys_in,
        nnz_bytes_u64));

    CHECK_CUDA(cudaMalloc(
        &d_keys_out,
        nnz_bytes_u64));

    CHECK_CUDA(cudaMalloc(
        &d_indices_in,
        nnz_bytes_u32));

    CHECK_CUDA(cudaMalloc(
        &d_indices_out,
        nnz_bytes_u32));

    // ========================================================
    // 5. Generate block_id
    // ========================================================

    constexpr int threads = 256;

    const int blocks =
        static_cast<int>(
            (
                tensor.nnz +
                threads -
                1
            ) /
            threads);

    generate_block_keys_kernel<<<
        blocks,
        threads,
        0,
        stream>>>(
            tensor.d_m0,
            tensor.d_m1,
            tensor.d_m2,

            d_keys_in,
            d_indices_in,

            tensor.nnz,

            num_blocks_j_u64,
            num_blocks_k_u64
        );

    CHECK_CUDA(cudaGetLastError());

    // ========================================================
    // 6. CUB Radix Sort
    //
    // sort key:
    //
    //     block_id
    //
    // value:
    //
    //     original NNZ index
    // ========================================================

    void* d_sort_temp = nullptr;

    size_t sort_temp_bytes = 0;

    CHECK_CUDA(
        cub::DeviceRadixSort::SortPairs(
            d_sort_temp,
            sort_temp_bytes,

            d_keys_in,
            d_keys_out,

            d_indices_in,
            d_indices_out,

            static_cast<int>(
                tensor.nnz),

            0,
            64,

            stream
        )
    );

    CHECK_CUDA(cudaMalloc(
        &d_sort_temp,
        sort_temp_bytes));

    CHECK_CUDA(
        cub::DeviceRadixSort::SortPairs(
            d_sort_temp,
            sort_temp_bytes,

            d_keys_in,
            d_keys_out,

            d_indices_in,
            d_indices_out,

            static_cast<int>(
                tensor.nnz),

            0,
            64,

            stream
        )
    );

    // 输入 buffer 不再需要
    CHECK_CUDA(cudaFree(d_keys_in));
    CHECK_CUDA(cudaFree(d_indices_in));

    d_keys_in = nullptr;
    d_indices_in = nullptr;

    // ========================================================
    // 7. Run Length Encode
    //
    // sorted block_id:
    //
    //     3 3 3 5 5 9 9 9 9
    //
    // =>
    //
    // unique:
    //
    //     3 5 9
    //
    // count:
    //
    //     3 2 4
    // ========================================================

    uint64_t* d_unique_keys = nullptr;
    int* d_run_counts = nullptr;
    int* d_num_runs = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_unique_keys,
        nnz_bytes_u64));

    CHECK_CUDA(cudaMalloc(
        &d_run_counts,
        tensor.nnz *
        sizeof(int)));

    CHECK_CUDA(cudaMalloc(
        &d_num_runs,
        sizeof(int)));

    void* d_rle_temp = nullptr;

    size_t rle_temp_bytes = 0;

    CHECK_CUDA(
        cub::DeviceRunLengthEncode::Encode(
            d_rle_temp,
            rle_temp_bytes,

            d_keys_out,

            d_unique_keys,
            d_run_counts,
            d_num_runs,

            static_cast<int>(
                tensor.nnz),

            stream
        )
    );

    CHECK_CUDA(cudaMalloc(
        &d_rle_temp,
        rle_temp_bytes));

    CHECK_CUDA(
        cub::DeviceRunLengthEncode::Encode(
            d_rle_temp,
            rle_temp_bytes,

            d_keys_out,

            d_unique_keys,
            d_run_counts,
            d_num_runs,

            static_cast<int>(
                tensor.nnz),

            stream
        )
    );

    int h_num_runs = 0;

    CHECK_CUDA(cudaMemcpyAsync(
        &h_num_runs,
        d_num_runs,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));

    CHECK_CUDA(
        cudaStreamSynchronize(stream));

    if (h_num_runs <= 0)
    {
        cudaFree(d_keys_out);
        cudaFree(d_indices_out);
        cudaFree(d_unique_keys);
        cudaFree(d_run_counts);
        cudaFree(d_num_runs);
        cudaFree(d_sort_temp);
        cudaFree(d_rle_temp);

        return;
    }

    // ========================================================
    // 8. Copy block metadata to CPU
    //
    // 这里只复制：
    //
    //     unique block_id
    //     count
    //
    // 而不是复制全部 NNZ。
    // ========================================================

    std::vector<uint64_t> h_unique_keys(
        static_cast<size_t>(
            h_num_runs));

    std::vector<int> h_run_counts(
        static_cast<size_t>(
            h_num_runs));

    CHECK_CUDA(cudaMemcpyAsync(
        h_unique_keys.data(),
        d_unique_keys,

        static_cast<size_t>(
            h_num_runs) *
        sizeof(uint64_t),

        cudaMemcpyDeviceToHost,
        stream));

    CHECK_CUDA(cudaMemcpyAsync(
        h_run_counts.data(),
        d_run_counts,

        static_cast<size_t>(
            h_num_runs) *
        sizeof(int),

        cudaMemcpyDeviceToHost,
        stream));

    CHECK_CUDA(
        cudaStreamSynchronize(stream));

    // ========================================================
    // 8b. 自动平衡：选择 cutoff 使张量核与 CUDA 核工作量相等
    // ========================================================

    if (dense_threshold <= 0.0)
    {
        // 基准实测速率（RTX 4060 Laptop, 24 SM）：
        //   张量核 WMMA：约 1.20 ns/元素
        //   CUDA 核合并内核：约 0.688 ns/元素
        // 按当前设备的 SM 数相对基准缩放（如 RTX 4090 128 SM，
        // 吞吐约为 24 SM 的 5.3 倍），使自动平衡适配目标 GPU。
        const double rate_wmma_base = 1.20;    // ns/elem @ 4060
        const double rate_cuda_base = 0.688;   // ns/elem @ 4060

        int dev_sm_count = 24;   // 基准设备 SM 数
        int dev_id = 0;
        cudaGetDevice(&dev_id);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev_id) == cudaSuccess &&
            prop.multiProcessorCount > 0)
        {
            dev_sm_count = prop.multiProcessorCount;
        }

        const double sm_scale =
            static_cast<double>(dev_sm_count) / 24.0;

        const double rate_wmma  = rate_wmma_base / sm_scale;
        const double rate_cuda  = rate_cuda_base / sm_scale;
        const double frac_dense =
            rate_cuda / (rate_wmma + rate_cuda);

        const size_t target_dense_nnz =
            static_cast<size_t>(
                static_cast<double>(tensor.nnz) *
                frac_dense);

        // 按 run 计数降序累加，找到达到目标稠密量的 cutoff
        std::vector<int> counts_sorted(
            h_run_counts.begin(),
            h_run_counts.begin() + h_num_runs);

        std::sort(
            counts_sorted.begin(),
            counts_sorted.end(),
            std::greater<int>());

        size_t acc = 0;
        int auto_cutoff = 1;

        for (int c : counts_sorted)
        {
            acc += static_cast<size_t>(c);

            if (acc >= target_dense_nnz)
            {
                auto_cutoff = c;
                break;
            }
        }

        dense_nnz_cutoff = auto_cutoff;

        std::cout
            << "[Partition] Auto-balance: dense cutoff = "
            << dense_nnz_cutoff
            << " / "
            << TILE_DIM * TILE_DIM * TILE_DIM
            << " (target dense nnz ~ "
            << target_dense_nnz
            << ", " << frac_dense * 100.0 << "%)"
            << std::endl;
    }

    // ========================================================
    // 9. CPU classify Dense / Sparse
    // ========================================================

    std::vector<int> h_dense_flags(
        static_cast<size_t>(
            h_num_runs),
        0);

    std::vector<size_t> h_run_starts(
        static_cast<size_t>(
            h_num_runs));

    std::vector<size_t> h_dense_offsets(
        static_cast<size_t>(
            h_num_runs),
        0);

    std::vector<size_t> h_sparse_offsets(
        static_cast<size_t>(
            h_num_runs),
        0);

    size_t dense_total_nnz = 0;
    size_t sparse_total_nnz = 0;
    size_t running_start = 0;

    for (int r = 0;
         r < h_num_runs;
         ++r)
    {
        const int count =
            h_run_counts[
                static_cast<size_t>(r)];

        h_run_starts[
            static_cast<size_t>(r)] =
            running_start;

        running_start +=
            static_cast<size_t>(
                count);

        if (count >= dense_nnz_cutoff)
        {
            h_dense_flags[
                static_cast<size_t>(r)] = 1;

            h_dense_offsets[
                static_cast<size_t>(r)] =
                dense_total_nnz;

            dense_total_nnz +=
                static_cast<size_t>(
                    count);
        }
        else
        {
            h_dense_flags[
                static_cast<size_t>(r)] = 0;

            h_sparse_offsets[
                static_cast<size_t>(r)] =
                sparse_total_nnz;

            sparse_total_nnz +=
                static_cast<size_t>(
                    count);
        }
    }

    // ========================================================
    // 10. Allocate Dense pools
    // ========================================================

    hybrid.dense_coords_capacity =
        dense_total_nnz;

    hybrid.dense_values_capacity =
        dense_total_nnz;

    if (dense_total_nnz > 0)
    {
        CHECK_CUDA(cudaMalloc(
            &hybrid.d_dense_coords_pool,

            dense_total_nnz *
            sizeof(int)));

        CHECK_CUDA(cudaMalloc(
            &hybrid.d_dense_values_pool,

            dense_total_nnz *
            sizeof(double)));
    }

    // ========================================================
    // 11. Allocate Sparse pools
    //
    // NEW:
    //
    //     uint32_t i
    //     uint32_t j
    //     uint32_t k
    //     double   value
    // ========================================================

    hybrid.sparse_nnz =
        sparse_total_nnz;

    if (sparse_total_nnz > 0)
    {
        CHECK_CUDA(cudaMalloc(
            &hybrid.d_sp_i,

            sparse_total_nnz *
            sizeof(uint32_t)));

        CHECK_CUDA(cudaMalloc(
            &hybrid.d_sp_j,

            sparse_total_nnz *
            sizeof(uint32_t)));

        CHECK_CUDA(cudaMalloc(
            &hybrid.d_sp_k,

            sparse_total_nnz *
            sizeof(uint32_t)));

        CHECK_CUDA(cudaMalloc(
            &hybrid.d_sp_val,

            sparse_total_nnz *
            sizeof(double)));
    }

    // ========================================================
    // 12. Create DenseTile metadata
    // ========================================================

    hybrid.num_dense_tiles = 0;

    hybrid.dense_tiles.reserve(
        static_cast<size_t>(
            h_num_runs));

    for (int r = 0;
         r < h_num_runs;
         ++r)
    {
        if (!h_dense_flags[
                static_cast<size_t>(r)])
        {
            continue;
        }

        const uint64_t block_id =
            h_unique_keys[
                static_cast<size_t>(r)];

        DenseTile16 tile{};

        // ----------------------------------------------------
        // Decode:
        //
        // block_id
        //     ↓
        // (block_i, block_j, block_k)
        // ----------------------------------------------------

        const uint64_t bk =
            block_id %
            num_blocks_k_u64;

        const uint64_t tmp =
            block_id /
            num_blocks_k_u64;

        const uint64_t bj =
            tmp %
            num_blocks_j_u64;

        const uint64_t bi =
            tmp /
            num_blocks_j_u64;

        tile.block_idx[0] =
            static_cast<int>(bi);

        tile.block_idx[1] =
            static_cast<int>(bj);

        tile.block_idx[2] =
            static_cast<int>(bk);

        tile.nnz =
            h_run_counts[
                static_cast<size_t>(r)];

        const size_t offset =
            h_dense_offsets[
                static_cast<size_t>(r)];

        tile.d_coords =
            hybrid.d_dense_coords_pool
            ? hybrid.d_dense_coords_pool +
              offset
            : nullptr;

        tile.d_values =
            hybrid.d_dense_values_pool
            ? hybrid.d_dense_values_pool +
              offset
            : nullptr;

        hybrid.dense_tiles.push_back(
            tile);

        ++hybrid.num_dense_tiles;
    }

    // ========================================================
    // 13. Copy metadata to GPU
    // ========================================================

    size_t runs_bytes =
        static_cast<size_t>(
            h_num_runs);

    size_t* d_run_starts = nullptr;
    int* d_dense_flags = nullptr;
    size_t* d_dense_offsets = nullptr;
    size_t* d_sparse_offsets = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_run_starts,
        runs_bytes *
        sizeof(size_t)));

    CHECK_CUDA(cudaMalloc(
        &d_dense_flags,
        runs_bytes *
        sizeof(int)));

    CHECK_CUDA(cudaMalloc(
        &d_dense_offsets,
        runs_bytes *
        sizeof(size_t)));

    CHECK_CUDA(cudaMalloc(
        &d_sparse_offsets,
        runs_bytes *
        sizeof(size_t)));

    CHECK_CUDA(cudaMemcpyAsync(
        d_run_starts,
        h_run_starts.data(),

        runs_bytes *
        sizeof(size_t),

        cudaMemcpyHostToDevice,
        stream));

    CHECK_CUDA(cudaMemcpyAsync(
        d_dense_flags,
        h_dense_flags.data(),

        runs_bytes *
        sizeof(int),

        cudaMemcpyHostToDevice,
        stream));

    CHECK_CUDA(cudaMemcpyAsync(
        d_dense_offsets,
        h_dense_offsets.data(),

        runs_bytes *
        sizeof(size_t),

        cudaMemcpyHostToDevice,
        stream));

    CHECK_CUDA(cudaMemcpyAsync(
        d_sparse_offsets,
        h_sparse_offsets.data(),

        runs_bytes *
        sizeof(size_t),

        cudaMemcpyHostToDevice,
        stream));

    // ========================================================
    // 14. GPU Scatter
    // ========================================================

    constexpr int scatter_threads = 256;

    scatter_partition_kernel<<<
        h_num_runs,
        scatter_threads,
        0,
        stream>>>(
            d_indices_out,

            d_unique_keys,
            d_run_counts,
            d_run_starts,

            d_dense_flags,
            d_dense_offsets,
            d_sparse_offsets,

            static_cast<size_t>(
                h_num_runs),

            tensor.d_m0,
            tensor.d_m1,
            tensor.d_m2,

            tensor.d_val,

            // Dense
            hybrid.d_dense_coords_pool,
            hybrid.d_dense_values_pool,

            // Sparse
            hybrid.d_sp_i,
            hybrid.d_sp_j,
            hybrid.d_sp_k,
            hybrid.d_sp_val
        );

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(
        cudaStreamSynchronize(stream));

    // 14b. （停用）k 分组：本驱动合并内核实测更慢；且与 FP32 稀疏实验
    //      相互干扰（dense 池重排后 residual 内核读错数据）。双流串行
    //      最优路径不需要它。

    // ========================================================
    // 15. Free temporary GPU buffers
    // ========================================================

    CHECK_CUDA(cudaFree(
        d_keys_out));

    CHECK_CUDA(cudaFree(
        d_indices_out));

    CHECK_CUDA(cudaFree(
        d_unique_keys));

    CHECK_CUDA(cudaFree(
        d_run_counts));

    CHECK_CUDA(cudaFree(
        d_num_runs));

    CHECK_CUDA(cudaFree(
        d_sort_temp));

    CHECK_CUDA(cudaFree(
        d_rle_temp));

    CHECK_CUDA(cudaFree(
        d_run_starts));

    CHECK_CUDA(cudaFree(
        d_dense_flags));

    CHECK_CUDA(cudaFree(
        d_dense_offsets));

    CHECK_CUDA(cudaFree(
        d_sparse_offsets));

    // ========================================================
    // 16. Result
    // ========================================================

    std::cout
        << "[Partition] GPU Complete. "
        << "Dense Tiles: "
        << hybrid.num_dense_tiles
        << " | Sparse Residue NNZ: "
        << hybrid.sparse_nnz
        << std::endl;
}


// ============================================================
// Default entry
// ============================================================

void partition_tensor_hybrid(
    const COOTensor& tensor,
    HybridCOOTensor& hybrid,
    double dense_threshold)
{
    partition_tensor_hybrid_gpu(
        tensor,
        hybrid,
        dense_threshold,
        0);
}


// ============================================================
// Free Hybrid Tensor
// ============================================================

void free_hybrid_tensor(
    HybridCOOTensor& hybrid)
{
    // --------------------------------------------------------
    // Dense pool
    // --------------------------------------------------------

    if (hybrid.d_dense_coords_pool)
    {
        cudaFree(
            hybrid.d_dense_coords_pool);

        hybrid.d_dense_coords_pool =
            nullptr;
    }

    if (hybrid.d_dense_values_pool)
    {
        cudaFree(
            hybrid.d_dense_values_pool);

        hybrid.d_dense_values_pool =
            nullptr;
    }

    if (hybrid.d_dense_k_offsets)
    {
        cudaFree(
            hybrid.d_dense_k_offsets);

        hybrid.d_dense_k_offsets =
            nullptr;
    }

    // --------------------------------------------------------
    // Sparse coordinate arrays
    // --------------------------------------------------------

    if (hybrid.d_sp_i)
    {
        cudaFree(
            hybrid.d_sp_i);

        hybrid.d_sp_i =
            nullptr;
    }

    if (hybrid.d_sp_j)
    {
        cudaFree(
            hybrid.d_sp_j);

        hybrid.d_sp_j =
            nullptr;
    }

    if (hybrid.d_sp_k)
    {
        cudaFree(
            hybrid.d_sp_k);

        hybrid.d_sp_k =
            nullptr;
    }

    // --------------------------------------------------------
    // Sparse values
    // --------------------------------------------------------

    if (hybrid.d_sp_val)
    {
        cudaFree(
            hybrid.d_sp_val);

        hybrid.d_sp_val =
            nullptr;
    }

    // --------------------------------------------------------
    // MTTKRP row-sorted views
    // --------------------------------------------------------

    for (int m = 0; m < 3; ++m)
    {
        if (hybrid.d_sp2_coords[m])
        {
            cudaFree(
                hybrid.d_sp2_coords[m]);

            hybrid.d_sp2_coords[m] =
                nullptr;
        }

        if (hybrid.d_sp2_val[m])
        {
            cudaFree(
                hybrid.d_sp2_val[m]);

            hybrid.d_sp2_val[m] =
                nullptr;
        }

        if (hybrid.d_blk_row[m])
        {
            cudaFree(
                hybrid.d_blk_row[m]);

            hybrid.d_blk_row[m] =
                nullptr;
        }

        if (hybrid.d_blk_start[m])
        {
            cudaFree(
                hybrid.d_blk_start[m]);

            hybrid.d_blk_start[m] =
                nullptr;
        }

        if (hybrid.d_blk_cnt[m])
        {
            cudaFree(
                hybrid.d_blk_cnt[m]);

            hybrid.d_blk_cnt[m] =
                nullptr;
        }
    }

    hybrid.mttkrp_blocks[0] = 0;
    hybrid.mttkrp_blocks[1] = 0;
    hybrid.mttkrp_blocks[2] = 0;

    hybrid.mttkrp_ready = false;

    // --------------------------------------------------------
    // Host metadata
    // --------------------------------------------------------

    hybrid.dense_tiles.clear();

    hybrid.num_dense_tiles = 0;

    hybrid.sparse_nnz = 0;

    hybrid.dense_coords_capacity = 0;

    hybrid.dense_values_capacity = 0;

    hybrid.num_blocks[0] = 0;
    hybrid.num_blocks[1] = 0;
    hybrid.num_blocks[2] = 0;
}

// ============================================================
// MTTKRP row-sorted view construction
//
// 为每个 mode 建立按目标坐标排序的稀疏视图：
//
//     d_sp2_perm[mode][pos] : 排序后位置 pos 处的原始元素号
//     d_sp2_coords[idx]     : 打包坐标 (i<<30)|(j<<15)|k
//     d_sp2_val[idx]        : 值
//
// 同时构建 block 映射：
//
//     d_blk_row[mode][b]    : block b 对应的目标行
//     d_blk_start[mode][b]  : block b 在排序数组中的起始位置
//     d_blk_cnt[mode][b]    : block b 处理的元素个数
//
// 调用时机：partition_tensor_hybrid_gpu() 之后。
// 构建完成后释放 partition 阶段的临时坐标数组 (d_sp_i/j/k)。
// ============================================================

__global__ void pack_coords_kernel(
    const uint32_t* __restrict__ d_i,
    const uint32_t* __restrict__ d_j,
    const uint32_t* __restrict__ d_k,
    uint64_t* __restrict__ d_packed,
    size_t nnz)
{
    size_t idx =
        static_cast<size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    if (idx >= nnz)
        return;

    d_packed[idx] =
        (static_cast<uint64_t>(d_i[idx]) << 30) |
        (static_cast<uint64_t>(d_j[idx]) << 15) |
        static_cast<uint64_t>(d_k[idx]);
}

__global__ void extract_sort_keys_kernel(
    int mode,
    const uint64_t* __restrict__ d_packed,
    uint64_t* __restrict__ d_key,
    uint32_t* __restrict__ d_idx,
    size_t nnz)
{
    size_t idx =
        static_cast<size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    if (idx >= nnz)
        return;

    const uint64_t pc =
        d_packed[idx];

    const uint32_t i =
        static_cast<uint32_t>((pc >> 30) & 0x7FFFu);
    const uint32_t j =
        static_cast<uint32_t>((pc >> 15) & 0x7FFFu);
    const uint32_t k =
        static_cast<uint32_t>(pc & 0x7FFFu);

    // 排序键 = (target << 30) | (o1 << 15) | o0
    //
    // 次级字段 o1 对应的因子行被寄存器缓存（run 内复用），
    // 三级字段 o0 对应的因子行按元素读取。
    //
    // 选择：
    //   o0 = 维度较小的因子（B 2.35MB / A 3.1MB），
    //   使按元素读取的因子工作集更小、更容易常驻 L2。
    uint64_t key = 0;

    if (mode == 0)
        key = (static_cast<uint64_t>(i) << 30) |
              (static_cast<uint64_t>(k) << 15) |
              static_cast<uint64_t>(j);
    else if (mode == 1)
        key = (static_cast<uint64_t>(j) << 30) |
              (static_cast<uint64_t>(k) << 15) |
              static_cast<uint64_t>(i);
    else
        key = (static_cast<uint64_t>(k) << 30) |
              (static_cast<uint64_t>(j) << 15) |
              static_cast<uint64_t>(i);

    d_key[idx] = key;

    if (d_idx != nullptr)
    {
        d_idx[idx] = static_cast<uint32_t>(idx);
    }
}

// 从排序后的 key 中提取目标行坐标（用于按行分块）
__global__ void extract_target_from_sorted_kernel(
    const uint64_t* __restrict__ d_key,
    uint32_t* __restrict__ d_tgt,
    size_t nnz)
{
    size_t idx =
        static_cast<size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    if (idx >= nnz)
        return;

    d_tgt[idx] =
        static_cast<uint32_t>(d_key[idx] >> 30);
}

__global__ void gather_sorted_kernel(
    const uint32_t* __restrict__ d_perm,
    const uint64_t* __restrict__ d_coords,
    const double* __restrict__ d_vals,
    uint64_t* __restrict__ d_sorted_coords,
    double* __restrict__ d_sorted_vals,
    size_t nnz)
{
    size_t pos =
        static_cast<size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    if (pos >= nnz)
        return;

    const uint32_t idx =
        d_perm[pos];

    d_sorted_coords[pos] =
        d_coords[idx];

    d_sorted_vals[pos] =
        d_vals[idx];
}

// 把稠密 tile 的元素解码为全局坐标并打包
// （local_id -> (li,lj,lk)，加 block 原点 -> 全局 (i,j,k)）
__global__ void pack_dense_tiles_kernel(
    const DenseTile16* __restrict__ d_tiles,
    const size_t* __restrict__ d_tile_offsets,
    int num_tiles,
    uint64_t* __restrict__ d_packed,
    double* __restrict__ d_vals)
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

    const size_t base =
        d_tile_offsets[tile_id];

    for (int e = threadIdx.x;
         e < t.nnz;
         e += blockDim.x)
    {
        const int lid =
            t.d_coords[e];

        const int li = lid >> 8;
        const int lj = (lid >> 4) & 15;
        const int lk = lid & 15;

        const int i = b0 + li;
        const int j = b1 + lj;
        const int k = b2 + lk;

        const uint64_t packed =
            (static_cast<uint64_t>(i) << 30) |
            (static_cast<uint64_t>(j) << 15) |
            static_cast<uint64_t>(k);

        d_packed[base + static_cast<size_t>(e)] =
            packed;

        d_vals[base + static_cast<size_t>(e)] =
            t.d_values[e];
    }
}

void build_mttkrp_views(
    HybridCOOTensor& hybrid,
    cudaStream_t stream)
{
    hybrid.mttkrp_ready = false;

    const size_t nnz =
        hybrid.sparse_nnz;

    if (nnz == 0)
        return;

    if (hybrid.d_sp_i == nullptr ||
        hybrid.d_sp_j == nullptr ||
        hybrid.d_sp_k == nullptr)
    {
        std::cerr
            << "[MTTKRP] sparse coordinates missing; "
               "run partition first."
            << std::endl;

        return;
    }

    const uint32_t dims[3] =
    {
        static_cast<uint32_t>(hybrid.dims[0]),
        static_cast<uint32_t>(hybrid.dims[1]),
        static_cast<uint32_t>(hybrid.dims[2])
    };

    if (dims[0] >= 32768 ||
        dims[1] >= 32768 ||
        dims[2] >= 32768)
    {
        std::cerr
            << "[MTTKRP] dims >= 32768 not supported "
               "by packed 15-bit coordinate layout."
            << std::endl;

        return;
    }

    constexpr int threads = 256;

    const int blocks =
        static_cast<int>(
            (nnz + threads - 1) /
            threads);

    // ========================================================
    // 1. Pack coordinates (i,j,k) -> uint64
    // ========================================================

    uint64_t* d_packed = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_packed,
        nnz * sizeof(uint64_t)));

    pack_coords_kernel<<<
        blocks,
        threads,
        0,
        stream>>>(
            hybrid.d_sp_i,
            hybrid.d_sp_j,
            hybrid.d_sp_k,
            d_packed,
            nnz);

    CHECK_CUDA(cudaGetLastError());

    // ========================================================
    // 2. 释放 partition 阶段的临时稀疏坐标数组
    //
    // 注意：d_sp_val 保留到所有 mode 的 gather 完成后再释放。
    // ========================================================

    cudaFree(hybrid.d_sp_i);
    cudaFree(hybrid.d_sp_j);
    cudaFree(hybrid.d_sp_k);

    hybrid.d_sp_i = nullptr;
    hybrid.d_sp_j = nullptr;
    hybrid.d_sp_k = nullptr;

    // ========================================================
    // 4. 临时 buffer
    // ========================================================

    uint64_t* d_key_in = nullptr;
    uint64_t* d_key_out = nullptr;

    uint32_t* d_unique = nullptr;
    int* d_counts = nullptr;
    int* d_num_runs = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_key_in,
        nnz * sizeof(uint64_t)));

    CHECK_CUDA(cudaMalloc(
        &d_key_out,
        nnz * sizeof(uint64_t)));

    CHECK_CUDA(cudaMalloc(
        &d_unique,
        nnz * sizeof(uint32_t)));

    CHECK_CUDA(cudaMalloc(
        &d_counts,
        nnz * sizeof(int)));

    CHECK_CUDA(cudaMalloc(
        &d_num_runs,
        sizeof(int)));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    std::cout
        << "[MTTKRP] Building row-sorted views "
           "for 3 modes..."
        << std::endl;

    // ========================================================
    // 5. 逐 mode 构建
    // ========================================================

    for (int mode = 0; mode < 3; ++mode)
    {
        // 独立 index buffer（排序 value = 原始元素号）
        uint32_t* d_idx_in = nullptr;
        uint32_t* d_idx_out = nullptr;

        CHECK_CUDA(cudaMalloc(
            &d_idx_in,
            nnz * sizeof(uint32_t)));

        CHECK_CUDA(cudaMalloc(
            &d_idx_out,
            nnz * sizeof(uint32_t)));

        // 提取排序键 (target<<30)|(o0<<15)|o1 + 初始化 identity index
        extract_sort_keys_kernel<<<
            blocks,
            threads,
            0,
            stream>>>(
                mode,
                d_packed,
                d_key_in,
                d_idx_in,
                nnz);

        CHECK_CUDA(cudaGetLastError());

        // CUB radix sort: (key=(target,o0,o1), value=original index)
        CHECK_CUDA(
            cub::DeviceRadixSort::SortPairs(
                d_temp,
                temp_bytes,

                d_key_in,
                d_key_out,

                d_idx_in,
                d_idx_out,

                static_cast<int>(nnz),

                0,
                48,

                stream));

        if (d_temp == nullptr)
        {
            CHECK_CUDA(cudaMalloc(
                &d_temp,
                temp_bytes));
        }

        CHECK_CUDA(
            cub::DeviceRadixSort::SortPairs(
                d_temp,
                temp_bytes,

                d_key_in,
                d_key_out,

                d_idx_in,
                d_idx_out,

                static_cast<int>(nnz),

                0,
                48,

                stream));

        // 从排序后的 key 中提取目标行（写入 d_idx_in 暂存，
        // key 排序结果已不再需要）
        extract_target_from_sorted_kernel<<<
            blocks,
            threads,
            0,
            stream>>>(
                d_key_out,
                d_idx_in,
                nnz);

        CHECK_CUDA(cudaGetLastError());

        // RLE on sorted targets
        // 注意：RLE 的临时空间需求可能大于 radix sort 的 temp_bytes，
        // 必须先查询再（按需）重新分配，否则 CUB 返回 invalid argument。
        // 查询时必须传 d_temp=nullptr（CUB 约定），传非空指针会执行。
        {
            size_t rle_bytes = 0;
            CHECK_CUDA(
                cub::DeviceRunLengthEncode::Encode(
                    nullptr,
                    rle_bytes,

                    d_idx_in,

                    d_unique,
                    d_counts,
                    d_num_runs,

                    static_cast<int>(nnz),

                    stream));

            if (rle_bytes > temp_bytes)
            {
                if (d_temp)
                    cudaFree(d_temp);
                CHECK_CUDA(cudaMalloc(
                    &d_temp,
                    rle_bytes));
                temp_bytes = rle_bytes;
            }
        }

        CHECK_CUDA(
            cub::DeviceRunLengthEncode::Encode(
                d_temp,
                temp_bytes,

                d_idx_in,

                d_unique,
                d_counts,
                d_num_runs,

                static_cast<int>(nnz),

                stream));

        CHECK_CUDA(
            cudaStreamSynchronize(stream));

        int h_num_runs = 0;

        CHECK_CUDA(cudaMemcpyAsync(
            &h_num_runs,
            d_num_runs,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));

        CHECK_CUDA(
            cudaStreamSynchronize(stream));

        std::vector<uint32_t> h_unique(
            static_cast<size_t>(h_num_runs));

        std::vector<int> h_counts(
            static_cast<size_t>(h_num_runs));

        CHECK_CUDA(cudaMemcpyAsync(
            h_unique.data(),
            d_unique,
            static_cast<size_t>(h_num_runs) *
                sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream));

        CHECK_CUDA(cudaMemcpyAsync(
            h_counts.data(),
            d_counts,
            static_cast<size_t>(h_num_runs) *
                sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));

        CHECK_CUDA(
            cudaStreamSynchronize(stream));

        // ====================================================
        // 构建 block 映射
        // ====================================================

        std::vector<uint32_t> blk_row;
        std::vector<uint32_t> blk_start;
        std::vector<uint32_t> blk_cnt;

        size_t cum = 0;

        for (int r = 0; r < h_num_runs; ++r)
        {
            const uint32_t tgt =
                h_unique[static_cast<size_t>(r)];

            const int cnt =
                h_counts[static_cast<size_t>(r)];

            const int chunks =
                (cnt + MTTKRP_CHUNK - 1) /
                MTTKRP_CHUNK;

            for (int c = 0; c < chunks; ++c)
            {
                const uint32_t s =
                    static_cast<uint32_t>(
                        cum +
                        static_cast<size_t>(c) *
                            MTTKRP_CHUNK);

                const uint32_t n =
                    static_cast<uint32_t>(
                        std::min(
                            MTTKRP_CHUNK,
                            cnt -
                                c * MTTKRP_CHUNK));

                blk_row.push_back(tgt);
                blk_start.push_back(s);
                blk_cnt.push_back(n);
            }

            cum += static_cast<size_t>(cnt);
        }

        hybrid.mttkrp_blocks[mode] =
            blk_row.size();

        if (!blk_row.empty())
        {
            CHECK_CUDA(cudaMalloc(
                &hybrid.d_blk_row[mode],
                blk_row.size() *
                    sizeof(uint32_t)));

            CHECK_CUDA(cudaMalloc(
                &hybrid.d_blk_start[mode],
                blk_start.size() *
                    sizeof(uint32_t)));

            CHECK_CUDA(cudaMalloc(
                &hybrid.d_blk_cnt[mode],
                blk_cnt.size() *
                    sizeof(uint32_t)));

            CHECK_CUDA(cudaMemcpyAsync(
                hybrid.d_blk_row[mode],
                blk_row.data(),
                blk_row.size() *
                    sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream));

            CHECK_CUDA(cudaMemcpyAsync(
                hybrid.d_blk_start[mode],
                blk_start.data(),
                blk_start.size() *
                    sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream));

            CHECK_CUDA(cudaMemcpyAsync(
                hybrid.d_blk_cnt[mode],
                blk_cnt.data(),
                blk_cnt.size() *
                    sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream));

            CHECK_CUDA(
                cudaStreamSynchronize(stream));
        }

        // ====================================================
        // 用排序后的 permutation 收集坐标与值，
        // 生成该 mode 的直接排序数组（kernel 中完全合并读取）。
        // ====================================================

        uint64_t* d_sorted_coords = nullptr;
        double* d_sorted_vals = nullptr;

        CHECK_CUDA(cudaMalloc(
            &d_sorted_coords,
            nnz * sizeof(uint64_t)));

        CHECK_CUDA(cudaMalloc(
            &d_sorted_vals,
            nnz * sizeof(double)));

        gather_sorted_kernel<<<
            blocks,
            threads,
            0,
            stream>>>(
                d_idx_out,
                d_packed,
                hybrid.d_sp_val,
                d_sorted_coords,
                d_sorted_vals,
                nnz);

        CHECK_CUDA(cudaGetLastError());

        hybrid.d_sp2_coords[mode] =
            d_sorted_coords;

        hybrid.d_sp2_val[mode] =
            d_sorted_vals;

        CHECK_CUDA(cudaFree(d_idx_in));
        CHECK_CUDA(cudaFree(d_idx_out));
    }

    // ========================================================
    // 6. 释放临时 buffer
    // ========================================================

    CHECK_CUDA(cudaFree(d_key_in));
    CHECK_CUDA(cudaFree(d_key_out));
    CHECK_CUDA(cudaFree(d_unique));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_num_runs));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_packed));

    // 释放 partition 阶段的稀疏值数组
    if (hybrid.d_sp_val)
    {
        cudaFree(hybrid.d_sp_val);
        hybrid.d_sp_val = nullptr;
    }

    hybrid.mttkrp_ready = true;

    std::cout
        << "[MTTKRP] Views ready. blocks per mode: "
        << hybrid.mttkrp_blocks[0]
        << " / "
        << hybrid.mttkrp_blocks[1]
        << " / "
        << hybrid.mttkrp_blocks[2]
        << std::endl;
}

// ============================================================
// 稠密 tile 解码：把稠密池元素解码为完整坐标 (i,j,k,val)
// ============================================================

__global__ void decode_dense_tiles_kernel(
    const DenseTile16* __restrict__ d_tiles,
    const size_t* __restrict__ d_tile_offsets,
    int num_tiles,
    uint32_t* __restrict__ d_i,
    uint32_t* __restrict__ d_j,
    uint32_t* __restrict__ d_k,
    double* __restrict__ d_val)
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

    const size_t base =
        d_tile_offsets[tile_id];

    for (int e = threadIdx.x;
         e < t.nnz;
         e += blockDim.x)
    {
        const int lid =
            t.d_coords[e];

        d_i[base + static_cast<size_t>(e)] =
            static_cast<uint32_t>(b0 + (lid >> 8));

        d_j[base + static_cast<size_t>(e)] =
            static_cast<uint32_t>(b1 + ((lid >> 4) & 15));

        d_k[base + static_cast<size_t>(e)] =
            static_cast<uint32_t>(b2 + (lid & 15));

        d_val[base + static_cast<size_t>(e)] =
            t.d_values[e];
    }
}

// ============================================================
// 构建稠密部分的 MTTKRP 视图（与稀疏残差走同一流水线）
//
// 调用后 hybrid 的稠密池被释放（元素已并入 dense_out 的视图）。
// ============================================================

void build_dense_mttkrp_views(
    HybridCOOTensor& hybrid,
    HybridCOOTensor& dense_out,
    cudaStream_t stream)
{
    const size_t nnz =
        hybrid.dense_coords_capacity;

    dense_out = HybridCOOTensor{};

    dense_out.dims[0] = hybrid.dims[0];
    dense_out.dims[1] = hybrid.dims[1];
    dense_out.dims[2] = hybrid.dims[2];

    dense_out.num_blocks[0] = hybrid.num_blocks[0];
    dense_out.num_blocks[1] = hybrid.num_blocks[1];
    dense_out.num_blocks[2] = hybrid.num_blocks[2];

    if (nnz == 0 || hybrid.num_dense_tiles == 0)
        return;

    std::cout
        << "[MTTKRP] Building dense-tile views (NNZ="
        << nnz
        << ")..."
        << std::endl;

    // 设备端 tile 元数据
    DenseTile16* d_tiles = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_tiles,
        static_cast<size_t>(hybrid.num_dense_tiles) *
            sizeof(DenseTile16)));

    CHECK_CUDA(cudaMemcpy(
        d_tiles,
        hybrid.dense_tiles.data(),
        static_cast<size_t>(hybrid.num_dense_tiles) *
            sizeof(DenseTile16),
        cudaMemcpyHostToDevice));

    // host 端累计 offset
    std::vector<size_t> h_offsets(
        static_cast<size_t>(hybrid.num_dense_tiles) + 1,
        0);

    for (int t = 0; t < hybrid.num_dense_tiles; ++t)
    {
        h_offsets[t + 1] =
            h_offsets[t] +
            static_cast<size_t>(hybrid.dense_tiles[t].nnz);
    }

    size_t* d_tile_offsets = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_tile_offsets,
        (static_cast<size_t>(hybrid.num_dense_tiles) + 1) *
            sizeof(size_t)));

    CHECK_CUDA(cudaMemcpy(
        d_tile_offsets,
        h_offsets.data(),
        (static_cast<size_t>(hybrid.num_dense_tiles) + 1) *
            sizeof(size_t),
        cudaMemcpyHostToDevice));

    // 稠密坐标数组
    CHECK_CUDA(cudaMalloc(
        &dense_out.d_sp_i,
        nnz * sizeof(uint32_t)));

    CHECK_CUDA(cudaMalloc(
        &dense_out.d_sp_j,
        nnz * sizeof(uint32_t)));

    CHECK_CUDA(cudaMalloc(
        &dense_out.d_sp_k,
        nnz * sizeof(uint32_t)));

    CHECK_CUDA(cudaMalloc(
        &dense_out.d_sp_val,
        nnz * sizeof(double)));

    dense_out.sparse_nnz = nnz;

    decode_dense_tiles_kernel<<<
        hybrid.num_dense_tiles,
        256,
        0,
        stream>>>(
            d_tiles,
            d_tile_offsets,
            hybrid.num_dense_tiles,
            dense_out.d_sp_i,
            dense_out.d_sp_j,
            dense_out.d_sp_k,
            dense_out.d_sp_val);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaFree(d_tiles);
    cudaFree(d_tile_offsets);

    // 复用视图构建流水线
    build_mttkrp_views(dense_out, stream);

    // 释放稠密池（元素已并入 dense_out 视图）
    if (hybrid.d_dense_coords_pool)
    {
        cudaFree(hybrid.d_dense_coords_pool);
        hybrid.d_dense_coords_pool = nullptr;
    }

    if (hybrid.d_dense_values_pool)
    {
        cudaFree(hybrid.d_dense_values_pool);
        hybrid.d_dense_values_pool = nullptr;
    }

    hybrid.dense_tiles.clear();
    hybrid.dense_coords_capacity = 0;
    hybrid.dense_values_capacity = 0;
    hybrid.num_dense_tiles = 0;

    std::cout
        << "[MTTKRP] Dense views ready."
        << std::endl;
}

