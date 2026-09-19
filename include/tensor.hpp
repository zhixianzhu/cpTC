#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// ============================================================
// 基础 COO 格式张量
//
// 原始格式：
//
//     (m0, m1, m2, value)
//
// 这里仍然保存完整三维坐标。
// 原始 Tensor / TNS loader 使用这个格式。
//
// ============================================================

struct COOTensor {

    // --------------------------------------------------------
    // Tensor dimensions
    // --------------------------------------------------------

    size_t dims[3] = {0, 0, 0};

    // Number of nonzeros
    size_t nnz = 0;

    // --------------------------------------------------------
    // GPU COO
    // --------------------------------------------------------

    int* d_m0 = nullptr;
    int* d_m1 = nullptr;
    int* d_m2 = nullptr;

    double* d_val = nullptr;

    // --------------------------------------------------------
    // CPU COO
    // --------------------------------------------------------

    std::vector<int> h_m0;
    std::vector<int> h_m1;
    std::vector<int> h_m2;
    std::vector<double> h_val;
};


// ============================================================
// Dense Tile
//
// 一个 Dense Tile 对应一个 TILE_DIM^3 block。
//
// 当前项目：
//
//     TILE_DIM = 16
//
// 因此：
//
//     16 × 16 × 16 = 4096
//
// Dense Tile 内部仍然使用 contiguous pool。
//
// ============================================================

struct DenseTile16 {

    // --------------------------------------------------------
    // Block coordinate
    //
    // block_idx[0] = block_i
    // block_idx[1] = block_j
    // block_idx[2] = block_k
    //
    // 注意：
    // 这里保存的是 Dense Tile 的三维 block 坐标，
    // 不是 Sparse entry 的 global coordinate。
    // --------------------------------------------------------

    int block_idx[3] = {0, 0, 0};

    // --------------------------------------------------------
    // Number of nonzeros in this tile
    // --------------------------------------------------------

    int nnz = 0;

    // --------------------------------------------------------
    // Pointers into contiguous GPU pools
    // --------------------------------------------------------

    // Packed local coordinate
    //
    // 编码：
    //
    //     local_id =
    //         (local_i << 8) |
    //         (local_j << 4) |
    //         local_k
    //
    // TILE_DIM = 16 时：
    //
    //     local_i : 0 ~ 15
    //     local_j : 0 ~ 15
    //     local_k : 0 ~ 15
    //
    // Dense kernel 当前仍然使用 int，
    // 因此这里暂时保持 int*。
    //

    int* d_coords = nullptr;

    // Values
    double* d_values = nullptr;
};

using DenseTile = DenseTile16;


// ============================================================
// Hybrid COO / Dense Tile
//
// 整体结构：
//
//                         Hybrid Tensor
//                              |
//                ┌─────────────┴─────────────┐
//                │                           │
//          Dense Tiles                 Sparse Residue
//                │                           │
//          block_idx[3]              block_id + local_id
//                │                           │
//          dense pool                    values
//
// ============================================================

struct HybridCOOTensor {

    // ========================================================
    // Tensor dimensions
    // ========================================================

    size_t dims[3] = {0, 0, 0};

    // ========================================================
    // Number of dense tiles
    // ========================================================

    int num_dense_tiles = 0;

    // ========================================================
    // Number of sparse nonzeros
    // ========================================================

    size_t sparse_nnz = 0;


    // ========================================================
    // Sparse residue
    //
    // 原来的格式：
    //
    //     d_sp_m0
    //     d_sp_m1
    //     d_sp_m2
    //     d_sp_val
    //
    // 修改后：
    //
    //     d_sp_block_id
    //     d_sp_local_id
    //     d_sp_val
    //
    // 即：
    //
    //     (i, j, k, value)
    //
    //          ↓
    //
    //     (block_id, local_id, value)
    //
    // ========================================================


    // --------------------------------------------------------
    // Block ID
    //
    // 一个 uint64_t 表示三维 block 坐标：
    //
    //     (block_i, block_j, block_k)
    //
    // 编码：
    //
    //     block_id =
    //         (block_i * num_blocks_j + block_j)
    //         * num_blocks_k
    //         + block_k
    //
    // 解码：
    //
    //     block_k = block_id % num_blocks_k
    //
    //     tmp = block_id / num_blocks_k
    //
    //     block_j = tmp % num_blocks_j
    //
    //     block_i = tmp / num_blocks_j
    //
    // 使用 uint64_t 是为了避免大规模 Tensor
    // 在 block 数量较大时发生 int 溢出。
    // --------------------------------------------------------

    // 优化后的稀疏残差直接保存完整三维坐标 (i,j,k)，
    // MTTKRP kernel 不再需要 64 位 block_id 除法解码。

    uint32_t* d_sp_i = nullptr;
    uint32_t* d_sp_j = nullptr;
    uint32_t* d_sp_k = nullptr;


    // --------------------------------------------------------
    // Local ID
    //
    // 一个 uint16_t 表示 block 内部的三维相对坐标：
    //
    //     (local_i, local_j, local_k)
    //
    // 编码：
    //
    //     local_id =
    //         (local_i * TILE_DIM + local_j)
    //         * TILE_DIM
    //         + local_k
    //
    // TILE_DIM = 16 时：
    //
    //     0 <= local_i < 16
    //     0 <= local_j < 16
    //     0 <= local_k < 16
    //
    // 因此：
    //
    //     0 <= local_id < 4096
    //
    // uint16_t 最大值为 65535，
    // 足够保存 4096 个 block 内位置。
    //
    // --------------------------------------------------------

    // 旧格式 local_id 已被完整坐标取代（见上方 d_sp_i/j/k）。


    // --------------------------------------------------------
    // Sparse values
    // --------------------------------------------------------

    double* d_sp_val = nullptr;


    // ========================================================
    // MTTKRP row-sorted views (built once by build_mttkrp_views)
    //
    // 为每个 mode 建立按目标坐标排序的视图：
    //
    //     d_sp2_coords[mode][pos] : 排序后位置 pos 处的打包坐标
    //     d_sp2_val[mode][pos]    : 排序后位置 pos 处的值
    //
    // 直接存储排序后的数组（不是 permutation 间接寻址），
    // 使 MTTKRP kernel 的坐标/值读取完全合并。
    //
    // 每个 mode 的 block 映射：
    //
    //     d_blk_row[mode][b]    : block b 对应的目标行
    //     d_blk_start[mode][b]  : block b 在排序数组中的起始位置
    //     d_blk_cnt[mode][b]    : block b 处理的元素个数
    //
    // kernel 中每个 block 对该行做一次归约，
    // 原子操作数量从 nnz*R 降至 blocks*R。
    // ========================================================

    uint64_t* d_sp2_coords[3] = {nullptr, nullptr, nullptr};

    double* d_sp2_val[3] = {nullptr, nullptr, nullptr};

    uint32_t* d_blk_row[3] = {nullptr, nullptr, nullptr};
    uint32_t* d_blk_start[3] = {nullptr, nullptr, nullptr};
    uint32_t* d_blk_cnt[3] = {nullptr, nullptr, nullptr};

    size_t mttkrp_blocks[3] = {0, 0, 0};

    bool mttkrp_ready = false;


    // ========================================================
    // Block metadata
    //
    // Tensor 在三个维度上分别被划分成多少个 block。
    //
    //     num_blocks[0] = ceil(dims[0] / TILE_DIM)
    //     num_blocks[1] = ceil(dims[1] / TILE_DIM)
    //     num_blocks[2] = ceil(dims[2] / TILE_DIM)
    //
    // block_id 的编码和解码都依赖这三个值。
    // ========================================================

    size_t num_blocks[3] = {0, 0, 0};


    // ========================================================
    // Dense tile metadata
    // ========================================================

    std::vector<DenseTile16> dense_tiles;


    // ========================================================
    // Dense tile contiguous memory pool
    //
    // 所有 Dense Tile 共用两个 GPU buffer，
    // 避免大量 cudaMalloc。
    // ========================================================

    int* d_dense_coords_pool = nullptr;

    double* d_dense_values_pool = nullptr;

    // 每个稠密 tile 的 k 分组起始偏移（num_dense_tiles x 16，int）
    // 池内条目按 local_k 重排后，slice k 只需扫描连续区间。
    int* d_dense_k_offsets = nullptr;

    size_t dense_coords_capacity = 0;

    size_t dense_values_capacity = 0;
};


// ============================================================
// Tensor API
// ============================================================

void load_tns_file(
    const std::string& filename,
    COOTensor& tensor
);

void free_coo_tensor(
    COOTensor& tensor
);


#endif // TENSOR_HPP