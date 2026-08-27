#include "compute_fit.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

double compute_fit(
    const HybridCOOTensor& h1,
    const HybridCOOTensor& h2,
    const double* d_A,
    const double* d_B,
    const double* d_C,
    const double* d_lambda,
    int R)
{
    const size_t nnz_total =
        h1.sparse_nnz +
        h2.sparse_nnz +
        h1.dense_coords_capacity +
        h2.dense_coords_capacity;

    if (nnz_total == 0)
    {
        std::cerr
            << "[compute_fit] nnz == 0"
            << std::endl;

        return 0.0;
    }

    if (d_A == nullptr ||
        d_B == nullptr ||
        d_C == nullptr ||
        R <= 0)
    {
        std::cerr
            << "[compute_fit] Invalid factor pointer or rank"
            << std::endl;

        return -1.0;
    }

    double h_res_norm_sq = 0.0;
    double h_X_norm_sq = 0.0;

    compute_residual_and_norm_sq(
        h1, h2,
        d_A,
        d_B,
        d_C,
        d_lambda,
        (d_lambda != nullptr),
        R,
        h_res_norm_sq,
        h_X_norm_sq);

    const double X_norm =
        std::sqrt(h_X_norm_sq);

    if (!std::isfinite(X_norm) ||
        X_norm < 1e-300)
    {
        std::cerr
            << "[compute_fit] Invalid ||X||_F = "
            << X_norm
            << std::endl;

        return 0.0;
    }

    if (!std::isfinite(h_res_norm_sq))
    {
        std::cerr
            << "[compute_fit] Residual norm is NaN/Inf"
            << std::endl;

        return 0.0;
    }

    if (h_res_norm_sq < 0.0)
        h_res_norm_sq = 0.0;

    const double res_norm =
        std::sqrt(h_res_norm_sq);

    double fit =
        1.0 - res_norm / X_norm;

    if (fit < 0.0 && fit > -1e-10)
        fit = 0.0;

    if (fit > 1.0 && fit < 1.0 + 1e-10)
        fit = 1.0;

    const double implied_rmse =
        std::sqrt(
            h_res_norm_sq /
            static_cast<double>(nnz_total));

    std::cout
        << "[Fit] ||X||_F = "
        << X_norm
        << " ||X-Xhat||_F = "
        << res_norm
        << " RMSE = "
        << implied_rmse
        << " fit = "
        << fit
        << std::endl;

    return fit;
}

void print_fit_report(
    const HybridCOOTensor& h1,
    const HybridCOOTensor& h2,
    const double* d_A,
    const double* d_B,
    const double* d_C,
    const double* d_lambda,
    int R,
    double rmse,
    const char* tag)
{
    if (tag == nullptr)
        tag = "";

    const double fit =
        compute_fit(
            h1, h2,
            d_A,
            d_B,
            d_C,
            d_lambda,
            R);

    std::cout
        << "\n[Fit Report] "
        << tag
        << " R="
        << R
        << " RMSE="
        << rmse
        << " fit="
        << fit
        << " ("
        << fit * 100.0
        << "% Frobenius energy explained)"
        << "\n"
        << std::endl;
}
