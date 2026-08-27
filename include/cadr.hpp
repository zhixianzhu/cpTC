#ifndef CADR_HPP
#define CADR_HPP

#include "tensor.hpp"
#include <vector>

struct TensorPermutation {
    std::vector<int> perm0;
    std::vector<int> perm1;
    std::vector<int> perm2;

    int* d_perm0 = nullptr;
    int* d_perm1 = nullptr;
    int* d_perm2 = nullptr;
};

TensorPermutation compute_cadr_reordering(const COOTensor& tensor);

void apply_cadr_reordering(COOTensor& tensor, const TensorPermutation& perm);

void unpermute_factor_matrices(double* d_A_perm, double* d_B_perm, double* d_C_perm,
                               double** d_factors_orig,
                               const TensorPermutation& perm,
                               const size_t* dims, int R);

void free_cadr_permutation(TensorPermutation& perm);

#endif