#ifndef CADR_HPP
#define CADR_HPP

#include "tensor.hpp"
#include <vector>

// CADR 重排映射结构体
struct TensorPermutation {
    std::vector<int> perm0; // perm0[old_idx] = new_idx
    std::vector<int> perm1;
    std::vector<int> perm2;

    // GPU 设备端映射指针
    int* d_perm0 = nullptr;
    int* d_perm1 = nullptr;
    int* d_perm2 = nullptr;
};

// 1. 根据切片度数计算 CADR 重排映射
TensorPermutation compute_cadr_reordering(const COOTensor& tensor);

// 2. 将重排映射应用到 COO 张量上
void apply_cadr_reordering(COOTensor& tensor, const TensorPermutation& perm);

// 3. 在 GPU 端将求解出的因子矩阵逆映射还原回原始顺序
void unpermute_factor_matrices(double* d_A_perm, double* d_B_perm, double* d_C_perm,
                               double** d_factors_orig,
                               const TensorPermutation& perm,
                               const size_t* dims, int R);

// 4. 释放重排资源
void free_cadr_permutation(TensorPermutation& perm);

#endif // CADR_HPP