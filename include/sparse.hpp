#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <cuda_runtime.h>
#include <cstddef>

#ifndef MTTKRP_BLOCK
#define MTTKRP_BLOCK 128
#endif

#ifndef MTTKRP_ITEMS
#define MTTKRP_ITEMS 8
#endif

#ifndef MTTKRP_CHUNK
#define MTTKRP_CHUNK (MTTKRP_BLOCK * MTTKRP_ITEMS)
#endif

struct HybridCOOTensor;

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
    size_t count);

void compute_sparse_mttkrp_async(
    const HybridCOOTensor& hybrid,
    int target_mode,

    const double* d_A,
    const double* d_B,
    const double* d_C,

    double* d_FtV,
    double* d_FtF,

    int R,

    cudaStream_t stream);

void compute_sparse_mttkrp(
    const HybridCOOTensor& hybrid,
    int target_mode,

    const double* d_A,
    const double* d_B,
    const double* d_C,

    double* d_FtV,
    double* d_FtF,

    int R);

#endif