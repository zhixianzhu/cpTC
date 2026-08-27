#ifndef PARTITION_HPP
#define PARTITION_HPP

#include "tensor.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#ifndef TILE_DIM
#define TILE_DIM 16
#endif

using BlockId  = std::uint64_t;
using LocalId  = std::uint16_t;

inline std::size_t num_blocks_for_dim(
    int dim)
{
    return (
        static_cast<std::size_t>(dim) +
        TILE_DIM -
        1
    ) / TILE_DIM;
}

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

void partition_tensor_hybrid(
    const COOTensor& tensor,
    HybridCOOTensor& hybrid,
    double dense_threshold = 0.03
);

void partition_tensor_hybrid_gpu(
    const COOTensor& tensor,
    HybridCOOTensor& hybrid,
    double dense_threshold,
    cudaStream_t stream
);

void free_hybrid_tensor(
    HybridCOOTensor& hybrid
);

void build_mttkrp_views(
    HybridCOOTensor& hybrid,
    cudaStream_t stream
);

void build_dense_mttkrp_views(
    HybridCOOTensor& hybrid,
    HybridCOOTensor& dense_out,
    cudaStream_t stream
);

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

#endif