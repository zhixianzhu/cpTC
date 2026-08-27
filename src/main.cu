#include <als.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#include <cuda_runtime.h>
#include <magma_v2.h>

#include "tensor.hpp"
#include "cadr.hpp"
#include "partition.hpp"
#include "common.hpp"
#include "sgd.hpp"

__global__ void fill_value_kernel(double* x, size_t n, double val);
#include <chrono>

int main(int argc, char** argv)
{

    magma_init();

    debug_cuda_device_state();

    int R = 32;
    int max_iters = 20;

    double lambda_reg = 1e-4;

    double threshold = 0.03;

    std::string filename = "nell-2.tns";

    if (argc > 1)
    {
        filename = argv[1];
    }

    if (argc > 2)
    {
        max_iters = std::atoi(argv[2]);
    }

    if (argc > 3)
    {
        R = std::atoi(argv[3]);
    }

    if (argc > 4)
    {
        threshold = std::atof(argv[4]);
    }

    if (max_iters <= 0) max_iters = 1;
    if (R <= 0) R = 1;

    std::cout
        << "[Main] Loading real tensor dataset from file: "
        << filename
        << std::endl;

    COOTensor tensor;

    auto t_load0 = std::chrono::high_resolution_clock::now();

    load_tns_file(
        filename,
        tensor);

    auto t_load1 = std::chrono::high_resolution_clock::now();

    std::cout << "[Main] Load time = "
              << std::chrono::duration<double, std::milli>(t_load1 - t_load0).count()
              << " ms" << std::endl;

    std::cout
        << "[Main] Tensor Dimensions: "
        << tensor.dims[0]
        << " x "
        << tensor.dims[1]
        << " x "
        << tensor.dims[2]
        << " | NNZ: "
        << tensor.nnz
        << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    TensorPermutation perm =
        compute_cadr_reordering(tensor);

    apply_cadr_reordering(
        tensor,
        perm);

    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            end - start).count();

    std::cout << "[CADR] Total reordering time = "
              << elapsed_ms << " ms\n";

    double* d_factors_perm[3] =
    {
        nullptr,
        nullptr,
        nullptr
    };

    init_factors_random(
        d_factors_perm,
        tensor.dims,
        R);

    {
        const size_t n0 = tensor.dims[0] * static_cast<size_t>(R);
        const size_t n1 = tensor.dims[1] * static_cast<size_t>(R);
        const size_t n2 = tensor.dims[2] * static_cast<size_t>(R);

        double* d_factors_combined = nullptr;

        CHECK_CUDA(cudaMalloc(
            &d_factors_combined,
            (n0 + n1 + n2) * sizeof(double)));

        CHECK_CUDA(cudaMemcpy(
            d_factors_combined,
            d_factors_perm[0],
            n0 * sizeof(double),
            cudaMemcpyDeviceToDevice));

        CHECK_CUDA(cudaMemcpy(
            d_factors_combined + n0,
            d_factors_perm[1],
            n1 * sizeof(double),
            cudaMemcpyDeviceToDevice));

        CHECK_CUDA(cudaMemcpy(
            d_factors_combined + n0 + n1,
            d_factors_perm[2],
            n2 * sizeof(double),
            cudaMemcpyDeviceToDevice));

        cudaFree(d_factors_perm[0]);
        cudaFree(d_factors_perm[1]);
        cudaFree(d_factors_perm[2]);

        d_factors_perm[0] = d_factors_combined;
        d_factors_perm[1] = d_factors_combined + n0;
        d_factors_perm[2] = d_factors_combined + n0 + n1;
    }

    HybridCOOTensor hybrid{};

    create_hybrid_format(
        tensor,
        hybrid,
        threshold);

    if (tensor.d_m0)
    {
        cudaFree(tensor.d_m0);
        tensor.d_m0 = nullptr;
    }
    if (tensor.d_m1)
    {
        cudaFree(tensor.d_m1);
        tensor.d_m1 = nullptr;
    }
    if (tensor.d_m2)
    {
        cudaFree(tensor.d_m2);
        tensor.d_m2 = nullptr;
    }
    if (tensor.d_val)
    {
        cudaFree(tensor.d_val);
        tensor.d_val = nullptr;
    }

    {
        auto t_v0 = std::chrono::high_resolution_clock::now();

        build_mttkrp_views(hybrid, 0);

        auto t_v1 = std::chrono::high_resolution_clock::now();

        std::cout << "[Main] MTTKRP views build time = "
                  << std::chrono::duration<double, std::milli>(t_v1 - t_v0).count()
                  << " ms" << std::endl;
    }

    HybridCOOTensor hybrid_dense{};

    const size_t dense_nnz =
        hybrid.dense_coords_capacity;

    const size_t sparse_nnz =
        hybrid.sparse_nnz;

    const size_t num_dense_tiles =
        static_cast<size_t>(hybrid.num_dense_tiles);

    constexpr size_t TILE_VOLUME =
        TILE_DIM * TILE_DIM * TILE_DIM;

    const double avg_tile_fill =
        (num_dense_tiles > 0 && TILE_VOLUME > 0)
            ? static_cast<double>(dense_nnz) /
              (static_cast<double>(num_dense_tiles) *
               static_cast<double>(TILE_VOLUME))
            : 0.0;

    const double eff_dense =
        static_cast<double>(dense_nnz) * avg_tile_fill;

    const double dense_ratio =
        (dense_nnz + sparse_nnz) > 0
            ? static_cast<double>(dense_nnz) /
              static_cast<double>(dense_nnz + sparse_nnz)
            : 0.0;

    const double fill_min =
        std::getenv("ALS_FILL_MIN")
            ? std::atof(std::getenv("ALS_FILL_MIN"))
            : 0.11;

    bool use_wmma;

    if (std::getenv("ALS_FORCE_WMMA"))
    {
        use_wmma = true;
    }
    else if (std::getenv("ALS_FORCE_CUDA"))
    {
        use_wmma = false;
    }
    else
    {
        use_wmma = (avg_tile_fill >= fill_min);
    }

    std::cout << "[Main] density check: dense_nnz=" << dense_nnz
              << " sparse_nnz=" << sparse_nnz
              << " tiles=" << num_dense_tiles
              << " avg_fill=" << avg_tile_fill
              << " eff_dense=" << eff_dense
              << " dense_ratio=" << dense_ratio
              << " (threshold " << fill_min << ")"
              << " -> " << (use_wmma ? "dual-stream(WMMA+FP64)" : "CUDA-core-only(FP64)")
              << std::endl;

    if (!use_wmma)
    {

        build_dense_mttkrp_views(
            hybrid,
            hybrid_dense,
            0);
    }

    double* d_lambda = nullptr;

    CHECK_CUDA(
        cudaMalloc(
            &d_lambda,
            static_cast<size_t>(R) *
            sizeof(double)));

    CHECK_CUDA(
        cudaMemset(
            d_lambda,
            0,
            static_cast<size_t>(R) *
            sizeof(double)));

    if (std::getenv("ALS_SGD")) {
        fill_value_kernel<<<(R + 255) / 256, 256>>>(
            d_lambda, R, 1.0);
        CHECK_CUDA(cudaGetLastError());
    }

    if (std::getenv("ALS_SGD")) {

        const double gamma =
            std::getenv("ALS_SGD_GAMMA")
                ? std::atof(std::getenv("ALS_SGD_GAMMA"))
                : 1e-9;

        const int batch =
            std::getenv("ALS_SGD_BATCH")
                ? std::atoi(std::getenv("ALS_SGD_BATCH"))
                : (1 << 20);

        execute_sgd_decomposition(
            hybrid,
            d_factors_perm[0],
            d_factors_perm[1],
            d_factors_perm[2],
            d_lambda,
            R,
            max_iters,
            gamma,
            batch,
            42);
    } else {
        execute_als_decomposition(
            tensor,
            hybrid,
            hybrid_dense,

            d_factors_perm[0],
            d_factors_perm[1],
            d_factors_perm[2],

            d_lambda,

            R,
            max_iters,
            lambda_reg);
    }

    double* d_factors_orig[3] =
    {
        nullptr,
        nullptr,
        nullptr
    };

    for (int i = 0; i < 3; ++i)
    {
        CHECK_CUDA(
            cudaMalloc(
                &d_factors_orig[i],
                tensor.dims[i] *
                static_cast<size_t>(R) *
                sizeof(double)));
    }

    unpermute_factor_matrices(
        d_factors_perm[0],
        d_factors_perm[1],
        d_factors_perm[2],
        d_factors_orig,
        perm,
        tensor.dims,
        R);

    std::vector<double> h_lambda(R);

    CHECK_CUDA(
        cudaMemcpy(
            h_lambda.data(),
            d_lambda,
            static_cast<size_t>(R) *
            sizeof(double),
            cudaMemcpyDeviceToHost));

    std::cout
        << "[Main] Final CP weights:";

    for (int r = 0; r < R; ++r)
    {
        std::cout
            << " "
            << h_lambda[r];
    }

    std::cout
        << std::endl;

    if (const char* exp = std::getenv("ALS_EXPORT_FACTORS"))
    {
        const char* names[3] = {"0", "1", "2"};
        const size_t dims3[3] = {tensor.dims[0], tensor.dims[1], tensor.dims[2]};
        for (int n = 0; n < 3; ++n)
        {
            std::string path = std::string(exp) + "." + names[n] + ".out";
            FILE* fp = fopen(path.c_str(), "w");
            fprintf(fp, "matrix\n2\n%zu %d\n", dims3[n], R);
            std::vector<double> h_f(dims3[n] * R);

            cudaMemcpy(h_f.data(), d_factors_orig[n],
                       dims3[n] * R * sizeof(double), cudaMemcpyDeviceToHost);

            for (size_t j = 0; j < dims3[n]; ++j)
                for (int i = 0; i < R; ++i)
                    fprintf(fp, "%.20lf\n", h_f[j * R + i]);
            fclose(fp);
        }
        std::string lpath = std::string(exp) + ".lambda.out";
        FILE* lf = fopen(lpath.c_str(), "w");
        fprintf(lf, "vector\n1\n%d\n", R);
        for (int r = 0; r < R; ++r)
            fprintf(lf, "%.20lf\n", h_lambda[r]);
        fclose(lf);
        std::cout << "[Main] factors exported to " << exp << std::endl;
    }

    free_cadr_permutation(perm);

    free_hybrid_format(hybrid);
    free_hybrid_format(hybrid_dense);

    for (int i = 0; i < 3; ++i)
    {
        if (d_factors_orig[i] != nullptr)
        {
            CHECK_CUDA(
                cudaFree(
                    d_factors_orig[i]));

            d_factors_orig[i] = nullptr;
        }
    }

    if (d_factors_perm[0] != nullptr)
    {
        CHECK_CUDA(
            cudaFree(
                d_factors_perm[0]));

        d_factors_perm[0] = nullptr;
        d_factors_perm[1] = nullptr;
        d_factors_perm[2] = nullptr;
    }

    if (d_lambda != nullptr)
    {
        CHECK_CUDA(
            cudaFree(d_lambda));

        d_lambda = nullptr;
    }

    free_coo_tensor(tensor);

    magma_finalize();

    std::cout
        << "[Main] Real tensor execution complete "
        << "with CADR Reordering & Unpermutation!"
        << std::endl;

    return 0;
}
