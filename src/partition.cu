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

    uint64_t block_i =
        static_cast<uint64_t>(d_m0[i]) /
        TILE_DIM;

    uint64_t block_j =
        static_cast<uint64_t>(d_m1[i]) /
        TILE_DIM;

    uint64_t block_k =
        static_cast<uint64_t>(d_m2[i]) /
        TILE_DIM;

    uint64_t block_id =
        (
            block_i *
            num_blocks_j +
            block_j
        ) *
        num_blocks_k +
        block_k;

    d_keys[i] = block_id;

    d_indices[i] =
        static_cast<uint32_t>(i);
}

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

    int* __restrict__ d_dense_coords_pool,

    double* __restrict__ d_dense_values_pool,

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

    uint64_t block_id =
        d_unique_keys[run];

    for (int k = threadIdx.x;
         k < count;
         k += blockDim.x)
    {
        size_t sorted_pos =
            sorted_start +
            static_cast<size_t>(k);

        uint32_t original_idx =
            d_sorted_indices[sorted_pos];

        int i =
            d_m0[original_idx];

        int j =
            d_m1[original_idx];

        int l =
            d_m2[original_idx];

        int local_i =
            i % TILE_DIM;

        int local_j =
            j % TILE_DIM;

        int local_k =
            l % TILE_DIM;

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

        if (is_dense)
        {
            d_dense_coords_pool[dst] =
                static_cast<int>(local_id);

            d_dense_values_pool[dst] =
                d_val[original_idx];
        }

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

    extern __shared__ char s_scratch_raw[];
    int* s_scratch_coords =
        reinterpret_cast<int*>(s_scratch_raw);

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

    for (int e = threadIdx.x;
         e < nnz;
         e += blockDim.x)
    {
        d_coords_pool[base + e] =
            s_scratch_coords[e];

        d_values_pool[base + e] =
            s_scratch_vals[e];
    }

    for (int k = threadIdx.x;
         k < 16;
         k += blockDim.x)
    {
        d_k_offsets[
            t * 16 + k] =
            s_start[k];
    }
}

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

    CHECK_CUDA(cudaMalloc(
        &hybrid.d_dense_k_offsets,
        static_cast<size_t>(nt) *
            16 *
            sizeof(int)));

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

    free_hybrid_tensor(hybrid);

    hybrid.dims[0] = tensor.dims[0];
    hybrid.dims[1] = tensor.dims[1];
    hybrid.dims[2] = tensor.dims[2];

    hybrid.num_dense_tiles = 0;
    hybrid.sparse_nnz = 0;

    hybrid.dense_tiles.clear();

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

    CHECK_CUDA(cudaFree(d_keys_in));
    CHECK_CUDA(cudaFree(d_indices_in));

    d_keys_in = nullptr;
    d_indices_in = nullptr;

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

    if (dense_threshold <= 0.0)
    {

        const double rate_wmma  = 1.20;
        const double rate_cuda  = 0.688;
        const double frac_dense =
            rate_cuda / (rate_wmma + rate_cuda);

        const size_t target_dense_nnz =
            static_cast<size_t>(
                static_cast<double>(tensor.nnz) *
                frac_dense);

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

            hybrid.d_dense_coords_pool,
            hybrid.d_dense_values_pool,

            hybrid.d_sp_i,
            hybrid.d_sp_j,
            hybrid.d_sp_k,
            hybrid.d_sp_val
        );

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(
        cudaStreamSynchronize(stream));

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

    std::cout
        << "[Partition] GPU Complete. "
        << "Dense Tiles: "
        << hybrid.num_dense_tiles
        << " | Sparse Residue NNZ: "
        << hybrid.sparse_nnz
        << std::endl;
}

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

void free_hybrid_tensor(
    HybridCOOTensor& hybrid)
{

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

    if (hybrid.d_sp_val)
    {
        cudaFree(
            hybrid.d_sp_val);

        hybrid.d_sp_val =
            nullptr;
    }

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

    hybrid.dense_tiles.clear();

    hybrid.num_dense_tiles = 0;

    hybrid.sparse_nnz = 0;

    hybrid.dense_coords_capacity = 0;

    hybrid.dense_values_capacity = 0;

    hybrid.num_blocks[0] = 0;
    hybrid.num_blocks[1] = 0;
    hybrid.num_blocks[2] = 0;
}

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

    cudaFree(hybrid.d_sp_i);
    cudaFree(hybrid.d_sp_j);
    cudaFree(hybrid.d_sp_k);

    hybrid.d_sp_i = nullptr;
    hybrid.d_sp_j = nullptr;
    hybrid.d_sp_k = nullptr;

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

    for (int mode = 0; mode < 3; ++mode)
    {

        uint32_t* d_idx_in = nullptr;
        uint32_t* d_idx_out = nullptr;

        CHECK_CUDA(cudaMalloc(
            &d_idx_in,
            nnz * sizeof(uint32_t)));

        CHECK_CUDA(cudaMalloc(
            &d_idx_out,
            nnz * sizeof(uint32_t)));

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

        extract_target_from_sorted_kernel<<<
            blocks,
            threads,
            0,
            stream>>>(
                d_key_out,
                d_idx_in,
                nnz);

        CHECK_CUDA(cudaGetLastError());

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

    CHECK_CUDA(cudaFree(d_key_in));
    CHECK_CUDA(cudaFree(d_key_out));
    CHECK_CUDA(cudaFree(d_unique));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_num_runs));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_packed));

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

    build_mttkrp_views(dense_out, stream);

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

