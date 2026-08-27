#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <cstddef>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

struct SVDWorkspace {
    cusolverDnHandle_t cusolver = nullptr;
    gesvdjInfo_t gesvd_info = nullptr;

    double* d_U = nullptr;
    double* d_V = nullptr;
    double* d_Ainv = nullptr;
    double* d_S = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    size_t work_len = 0;
    int R = 0;
};

void init_svd_workspace(
    SVDWorkspace& ws,
    int R);

void free_svd_workspace(
    SVDWorkspace& ws);

bool solve_als_system_svd(
    const SVDWorkspace& ws,
    cublasHandle_t cublas,
    double* d_FtF,
    double* d_FtV,
    double* d_Factor,
    size_t dim_len,
    int R);

void solve_als_system_magma(
    double* d_FtF,
    double* d_FtV,
    double* d_Factor,
    size_t dim_len,
    int R,
    double lambda_reg);

#endif
