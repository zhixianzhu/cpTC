#include "cadr.hpp"
#include "common.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

__global__ void unpermute_factor_kernel(const double* __restrict__ d_Factor_perm,
                                        double* __restrict__ d_Factor_orig,
                                        const int* __restrict__ d_perm,
                                        size_t dim_len, int R) {
    size_t old_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_idx < dim_len) {
        int new_idx = d_perm[old_idx];
        for (int r = 0; r < R; ++r) {
            d_Factor_orig[old_idx * R + r] = d_Factor_perm[new_idx * R + r];
        }
    }
}

TensorPermutation compute_cadr_reordering(const COOTensor& tensor) {
    std::cout << "[CADR] Computing degree-based mode reordering mapping..." << std::endl;

    TensorPermutation perm;
    perm.perm0.resize(tensor.dims[0]);
    perm.perm1.resize(tensor.dims[1]);
    perm.perm2.resize(tensor.dims[2]);

    std::vector<size_t> deg0(tensor.dims[0], 0);
    std::vector<size_t> deg1(tensor.dims[1], 0);
    std::vector<size_t> deg2(tensor.dims[2], 0);

    for (size_t i = 0; i < tensor.nnz; ++i) {
        deg0[tensor.h_m0[i]]++;
        deg1[tensor.h_m1[i]]++;
        deg2[tensor.h_m2[i]]++;
    }

    auto build_perm = [](const std::vector<size_t>& deg, std::vector<int>& perm) {
        std::vector<int> sorted_indices(deg.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
            return deg[a] > deg[b];
        });
        for (int new_idx = 0; new_idx < static_cast<int>(sorted_indices.size()); ++new_idx) {
            int old_idx = sorted_indices[new_idx];
            perm[old_idx] = new_idx;
        }
    };

    build_perm(deg0, perm.perm0);
    build_perm(deg1, perm.perm1);
    build_perm(deg2, perm.perm2);

    CHECK_CUDA(cudaMalloc(&perm.d_perm0, tensor.dims[0] * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&perm.d_perm1, tensor.dims[1] * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&perm.d_perm2, tensor.dims[2] * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(perm.d_perm0, perm.perm0.data(), tensor.dims[0] * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(perm.d_perm1, perm.perm1.data(), tensor.dims[1] * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(perm.d_perm2, perm.perm2.data(), tensor.dims[2] * sizeof(int), cudaMemcpyHostToDevice));

    return perm;
}

void apply_cadr_reordering(COOTensor& tensor, const TensorPermutation& perm) {
    std::cout << "[CADR] Applying mode permutation to COO tensor coordinates..." << std::endl;

    for (size_t i = 0; i < tensor.nnz; ++i) {
        tensor.h_m0[i] = perm.perm0[tensor.h_m0[i]];
        tensor.h_m1[i] = perm.perm1[tensor.h_m1[i]];
        tensor.h_m2[i] = perm.perm2[tensor.h_m2[i]];
    }

    CHECK_CUDA(cudaMemcpy(tensor.d_m0, tensor.h_m0.data(), tensor.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(tensor.d_m1, tensor.h_m1.data(), tensor.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(tensor.d_m2, tensor.h_m2.data(), tensor.nnz * sizeof(int), cudaMemcpyHostToDevice));
}

void unpermute_factor_matrices(double* d_A_perm, double* d_B_perm, double* d_C_perm,
                               double** d_factors_orig,
                               const TensorPermutation& perm,
                               const size_t* dims, int R) {
    std::cout << "[CADR] Unpermuting factor matrices back to original order..." << std::endl;

    int threads = 256;

    int blocks0 = (dims[0] + threads - 1) / threads;
    unpermute_factor_kernel<<<blocks0, threads>>>(d_A_perm, d_factors_orig[0], perm.d_perm0, dims[0], R);

    int blocks1 = (dims[1] + threads - 1) / threads;
    unpermute_factor_kernel<<<blocks1, threads>>>(d_B_perm, d_factors_orig[1], perm.d_perm1, dims[1], R);

    int blocks2 = (dims[2] + threads - 1) / threads;
    unpermute_factor_kernel<<<blocks2, threads>>>(d_C_perm, d_factors_orig[2], perm.d_perm2, dims[2], R);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void free_cadr_permutation(TensorPermutation& perm) {
    if (perm.d_perm0) cudaFree(perm.d_perm0);
    if (perm.d_perm1) cudaFree(perm.d_perm1);
    if (perm.d_perm2) cudaFree(perm.d_perm2);
}
