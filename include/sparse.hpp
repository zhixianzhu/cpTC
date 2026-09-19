#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <cuda_runtime.h>
#include <cstddef>

// ============================================================
// Row-sorted MTTKRP kernel configuration
//
// 每个 CUDA block 处理 MTTKRP_CHUNK 个元素（同一目标行），
// 在 block 内做归约，最后对该行每个 rank 只做一次 atomicAdd。
// ============================================================

#ifndef MTTKRP_BLOCK
#define MTTKRP_BLOCK 128
#endif

#ifndef MTTKRP_ITEMS
#define MTTKRP_ITEMS 8
#endif

#ifndef MTTKRP_CHUNK
#define MTTKRP_CHUNK (MTTKRP_BLOCK * MTTKRP_ITEMS)
#endif

// ============================================================
// Forward declaration
// ============================================================

struct HybridCOOTensor;

// ============================================================
// Sparse MTTKRP - Async
//
// 这个接口不会调用 cudaDeviceSynchronize()。
// CUDA kernel 会提交到指定 stream，
// 可以和 dense stream 上的计算并发执行。
//
// start_idx:
//     sparse COO 起始位置
//
// count:
//     本次处理的 sparse 元素数量
// ============================================================

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


// ============================================================
// Sparse MTTKRP - Async Full
//
// 自动处理：
//     [0, hybrid.sparse_nnz)
//
// 不进行 cudaDeviceSynchronize()
//
// 可以直接放到 sparse_stream 上，
// 与 dense_stream 并发。
// ============================================================

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


// ============================================================
// Sparse MTTKRP - Synchronous
//
// 兼容原来的调用方式。
//
// 内部创建独立 CUDA stream，
// 提交 sparse kernel 后等待完成。
// ============================================================

void compute_sparse_mttkrp(
    const HybridCOOTensor& hybrid,
    int target_mode,

    const double* d_A,
    const double* d_B,
    const double* d_C,

    double* d_FtV,
    double* d_FtF,

    int R);

#endif // SPARSE_HPP