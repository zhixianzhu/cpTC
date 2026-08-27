#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cuda_runtime.h>

struct COOTensor {

    size_t dims[3] = {0, 0, 0};

    size_t nnz = 0;

    int* d_m0 = nullptr;
    int* d_m1 = nullptr;
    int* d_m2 = nullptr;

    double* d_val = nullptr;

    std::vector<int> h_m0;
    std::vector<int> h_m1;
    std::vector<int> h_m2;
    std::vector<double> h_val;
};

struct DenseTile16 {

    int block_idx[3] = {0, 0, 0};

    int nnz = 0;

    int* d_coords = nullptr;

    double* d_values = nullptr;
};

using DenseTile = DenseTile16;

struct HybridCOOTensor {

    size_t dims[3] = {0, 0, 0};

    int num_dense_tiles = 0;

    size_t sparse_nnz = 0;

    uint32_t* d_sp_i = nullptr;
    uint32_t* d_sp_j = nullptr;
    uint32_t* d_sp_k = nullptr;

    double* d_sp_val = nullptr;

    uint64_t* d_sp2_coords[3] = {nullptr, nullptr, nullptr};

    double* d_sp2_val[3] = {nullptr, nullptr, nullptr};

    uint32_t* d_blk_row[3] = {nullptr, nullptr, nullptr};
    uint32_t* d_blk_start[3] = {nullptr, nullptr, nullptr};
    uint32_t* d_blk_cnt[3] = {nullptr, nullptr, nullptr};

    size_t mttkrp_blocks[3] = {0, 0, 0};

    bool mttkrp_ready = false;

    size_t num_blocks[3] = {0, 0, 0};

    std::vector<DenseTile16> dense_tiles;

    int* d_dense_coords_pool = nullptr;

    double* d_dense_values_pool = nullptr;

    int* d_dense_k_offsets = nullptr;

    size_t dense_coords_capacity = 0;

    size_t dense_values_capacity = 0;
};

void load_tns_file(
    const std::string& filename,
    COOTensor& tensor
);

void free_coo_tensor(
    COOTensor& tensor
);

#endif