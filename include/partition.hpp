#ifndef PARTITION_HPP
#define PARTITION_HPP

#include "tensor.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#ifndef TILE_DIM
#define TILE_DIM 16
#endif

// ============================================================
// Tile coordinate encoding
//
// Global coordinate:
//
//     (i, j, k)
//
// is represented by:
//
//     (block_id, local_id)
//
// where:
//
//     block_id
//         identifies the TILE_DIM^3 block
//
//     local_id
//         identifies the position inside the block
//
// For TILE_DIM = 16:
//
//     local_id = (local_i * 16 + local_j) * 16 + local_k
//
// Therefore:
//
//     0 <= local_id < 16^3 = 4096
//
// local_id can theoretically be stored in uint16_t.
// ============================================================

using BlockId  = std::uint64_t;
using LocalId  = std::uint16_t;

// ============================================================
// Block grid
//
// Number of blocks along each tensor dimension:
//
//     num_blocks_i = ceil(dim_i / TILE_DIM)
//     num_blocks_j = ceil(dim_j / TILE_DIM)
//     num_blocks_k = ceil(dim_k / TILE_DIM)
// ============================================================

inline std::size_t num_blocks_for_dim(
    int dim)
{
    return (
        static_cast<std::size_t>(dim) +
        TILE_DIM -
        1
    ) / TILE_DIM;
}

// ============================================================
// Encode block coordinates
//
//     (block_i, block_j, block_k)
//             ↓
//          block_id
//
// block_id is laid out in row-major order:
//
//     block_id =
//         (block_i * num_blocks_j + block_j)
//         * num_blocks_k
//         + block_k
// ============================================================

inline BlockId encode_block_id(
    std::size_t block_i,
    std::size_t block_j,
    std::size_t block_k,
    std::size_t num_blocks_j,
    std::size_t num_blocks_k)
{
    return
        (
            static_cast<BlockId>(block_i) *
            static_cast<BlockId>(num_blocks_j) +
            static_cast<BlockId>(block_j)
        ) *
        static_cast<BlockId>(num_blocks_k) +
        static_cast<BlockId>(block_k);
}

// ============================================================
// Decode block_id
//
//     block_id
//         ↓
//     (block_i, block_j, block_k)
//
// ============================================================

inline void decode_block_id(
    BlockId block_id,
    std::size_t num_blocks_j,
    std::size_t num_blocks_k,
    std::size_t& block_i,
    std::size_t& block_j,
    std::size_t& block_k)
{
    block_k =
        static_cast<std::size_t>(
            block_id %
            static_cast<BlockId>(num_blocks_k)
        );

    BlockId tmp =
        block_id /
        static_cast<BlockId>(num_blocks_k);

    block_j =
        static_cast<std::size_t>(
            tmp %
            static_cast<BlockId>(num_blocks_j)
        );

    block_i =
        static_cast<std::size_t>(
            tmp /
            static_cast<BlockId>(num_blocks_j)
        );
}

// ============================================================
// Encode local coordinate
//
//     (local_i, local_j, local_k)
//              ↓
//           local_id
//
// For TILE_DIM = 16:
//
//     local_id =
//         local_i * 256
//       + local_j * 16
//       + local_k
//
// Range:
//
//     0 <= local_id < 4096
//
// ============================================================

constexpr LocalId encode_local_id(
    int local_i,
    int local_j,
    int local_k)
{
    return static_cast<LocalId>(
        (
            local_i * TILE_DIM +
            local_j
        ) * TILE_DIM +
        local_k
    );
}

// ============================================================
// Decode local_id
//
//     local_id
//         ↓
//     (local_i, local_j, local_k)
// ============================================================

constexpr void decode_local_id(
    LocalId local_id,
    int& local_i,
    int& local_j,
    int& local_k)
{
    const int id =
        static_cast<int>(local_id);

    local_k =
        id % TILE_DIM;

    local_j =
        (id / TILE_DIM) % TILE_DIM;

    local_i =
        id / (TILE_DIM * TILE_DIM);
}

// ============================================================
// Encode global coordinate
//
//     (i, j, k)
//        ↓
//     block_id
//     local_id
//
// ============================================================

inline void encode_coordinate(
    int i,
    int j,
    int k,
    std::size_t num_blocks_j,
    std::size_t num_blocks_k,
    BlockId& block_id,
    LocalId& local_id)
{
    const std::size_t block_i =
        static_cast<std::size_t>(i) /
        TILE_DIM;

    const std::size_t block_j =
        static_cast<std::size_t>(j) /
        TILE_DIM;

    const std::size_t block_k =
        static_cast<std::size_t>(k) /
        TILE_DIM;

    const int local_i =
        i % TILE_DIM;

    const int local_j =
        j % TILE_DIM;

    const int local_k =
        k % TILE_DIM;

    block_id =
        encode_block_id(
            block_i,
            block_j,
            block_k,
            num_blocks_j,
            num_blocks_k
        );

    local_id =
        encode_local_id(
            local_i,
            local_j,
            local_k
        );
}

// ============================================================
// Decode global coordinate
//
//     block_id + local_id
//              ↓
//            (i,j,k)
// ============================================================

inline void decode_coordinate(
    BlockId block_id,
    LocalId local_id,
    std::size_t num_blocks_j,
    std::size_t num_blocks_k,
    int& i,
    int& j,
    int& k)
{
    std::size_t block_i;
    std::size_t block_j;
    std::size_t block_k;

    decode_block_id(
        block_id,
        num_blocks_j,
        num_blocks_k,
        block_i,
        block_j,
        block_k
    );

    int local_i;
    int local_j;
    int local_k;

    decode_local_id(
        local_id,
        local_i,
        local_j,
        local_k
    );

    i =
        static_cast<int>(
            block_i * TILE_DIM +
            local_i
        );

    j =
        static_cast<int>(
            block_j * TILE_DIM +
            local_j
        );

    k =
        static_cast<int>(
            block_k * TILE_DIM +
            local_k
        );
}

// ============================================================
// GPU Hybrid Partition
//
// Partition strategy:
//
//     COO (i,j,k,value)
//
//         ↓
//
//     (block_id, local_id, value)
//
//         ↓ sort by block_id
//
//     Dense blocks
//     Sparse blocks
//
// ============================================================

void partition_tensor_hybrid(
    const COOTensor& tensor,
    HybridCOOTensor& hybrid,
    double dense_threshold = 0.03
);

// ============================================================
// CUDA Stream version
// ============================================================

void partition_tensor_hybrid_gpu(
    const COOTensor& tensor,
    HybridCOOTensor& hybrid,
    double dense_threshold,
    cudaStream_t stream
);

// ============================================================
// Free
// ============================================================

void free_hybrid_tensor(
    HybridCOOTensor& hybrid
);

// ============================================================
// Build MTTKRP row-sorted views
//
// 在 partition_tensor_hybrid_gpu 之后调用一次。
//
// 为每个 mode 建立按目标行排序的稀疏视图，
// 并释放 partition 阶段的临时稀疏坐标数组。
// ============================================================

void build_mttkrp_views(
    HybridCOOTensor& hybrid,
    cudaStream_t stream
);

// 构建稠密 tile 部分的 MTTKRP 视图（解码稠密池并复用同一流水线）
void build_dense_mttkrp_views(
    HybridCOOTensor& hybrid,
    HybridCOOTensor& dense_out,
    cudaStream_t stream
);

// ============================================================
// Compatibility with old code
// ============================================================

inline void create_hybrid_format(
    const COOTensor& coo,
    HybridCOOTensor& hybrid,
    double density_threshold = 0.03
)
{
    partition_tensor_hybrid(
        coo,
        hybrid,
        density_threshold
    );
}

inline void free_hybrid_format(
    HybridCOOTensor& hybrid
)
{
    free_hybrid_tensor(hybrid);
}

#endif // PARTITION_HPP