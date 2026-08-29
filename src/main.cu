#include <als.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <magma_v2.h>

#include "tensor.hpp"
#include "cadr.hpp"
#include "partition.hpp"
#include "common.hpp"
#include "sgd.hpp"

// als.cu 内部内核（SGD 路径复用 λ 初始化）
__global__ void fill_value_kernel(double* x, size_t n, double val);
#include <chrono>

int main(int argc, char** argv)
{
    // ============================================================
    // 1. 初始化 MAGMA 环境与 CUDA 设备
    // ============================================================

    magma_init();

    debug_cuda_device_state();


    // ============================================================
    // CP-ALS / CADR / Hybrid 超参数
    // ============================================================

    int R = 32;
    int max_iters = 20;

    // ALS normal equation 的 L2 正则化系数
    double lambda_reg = 1e-4;

    // Hybrid partition 密度阈值
    double threshold = 0.03;


    // ============================================================
    // 2. 获取真实张量数据集路径
    //
    // 用法：
    //
    //     ./总项目 <tns_file> [max_iters] [R] [dense_threshold]
    //
    // ============================================================

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


    // ============================================================
    // 3. 读取 .tns 文件
    // ============================================================

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


    // ============================================================
    // 4. CADR Mode Reordering
    // ============================================================

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


    // ============================================================
    // 5. 初始化重排空间下的 CP 因子
    //
    // d_factors_perm:
    //
    //   [0] -> A
    //   [1] -> B
    //   [2] -> C
    //
    // 三个 factor 均使用 row-major:
    //
    //   factor[i * R + r]
    // ============================================================

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

    // ============================================================
    // 5b. 合并三个因子到连续显存
    //
    // 用于 L2 持久化窗口（保护因子行不被流式坐标数据挤出缓存）。
    // d_factors_perm[0] 指向合并缓冲区的基地址。
    // ============================================================

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


    // ============================================================
    // 6. Hybrid COO / Dense Partition
    // ============================================================

    HybridCOOTensor hybrid{};

    create_hybrid_format(
        tensor,
        hybrid,
        threshold);

    // 视图构建只依赖 hybrid 的稀疏坐标数组（d_sp_i/j/k），
    // COO 设备数组此后不再需要（RMSE/fit 均基于视图计算）。
    // 在构建视图之前释放，避免峰值显存超限。
    // host 端 h_* 向量保留（CADR/unpermute/调试使用）。
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

    // 构建 MTTKRP 按行排序视图（仅一次，后续 ALS 迭代复用）
    {
        auto t_v0 = std::chrono::high_resolution_clock::now();

        build_mttkrp_views(hybrid, 0);

        auto t_v1 = std::chrono::high_resolution_clock::now();

        std::cout << "[Main] MTTKRP views build time = "
                  << std::chrono::duration<double, std::milli>(t_v1 - t_v0).count()
                  << " ms" << std::endl;
    }

    // 稠密 tile 处理策略（自适应）：
    //
    // 在重排 + 分区后检测张量稠密度。判据 = 平均 tile 填充率：
    //
    //     avg_fill = dense_nnz / (num_dense_tiles * 4096)
    //
    // 物理含义：
    //   WMMA 张量核的固定成本是 per-tile 的（每个 tile 清空 4096 槽
    //   + MMA 流水线），每元素成本随 tile 填充率下降，与元素总量
    //   基本无关。因此"谁快"由 avg_fill 决定，而不是 dense_ratio
    //   或元素总量。
    //
    // 实测（RTX 4060, R=32, 合成块结构张量 + NELL-2）：
    //   avg_fill <  ~0.09 -> 纯 CUDA core 更快（s1/s2/s6/NELL-2）
    //   avg_fill >= ~0.13 -> 双流串行（WMMA+FP64）更快（s8/s9/s11）
    //   分界约 0.11，默认阈值 0.11，可用 ALS_FILL_MIN 覆盖。
    //
    // 可用 ALS_FORCE_WMMA=1 / ALS_FORCE_CUDA=1 强制覆盖。
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

    // 填充率自适应阈值：
    //   基准 0.11 在 RTX 4060 Laptop（24 SM）上实测校准。
    //   RTX 4090（128 SM）等更多 SM 的设备上，张量核 WMMA 网格
    //   并行性更强，每 tile 固定开销摊薄更快，阈值随 SM 数降低
    //   （sqrt 缩放，保守起见不降过 0.08）。
    double fill_min = 0.11;
    if (!std::getenv("ALS_FILL_MIN"))
    {
        int dev_id = 0;
        cudaGetDevice(&dev_id);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev_id) == cudaSuccess &&
            prop.multiProcessorCount > 0)
        {
            const double sm_scale =
                static_cast<double>(prop.multiProcessorCount) / 24.0;
            fill_min = 0.11 / std::sqrt(sm_scale);
            if (fill_min < 0.08) fill_min = 0.08;
            std::cout << "[Main] device=" << prop.name
                      << " SMs=" << prop.multiProcessorCount
                      << " fill_threshold=" << fill_min
                      << " (auto-scaled from 0.11 @24SM)" << std::endl;
        }
    }
    else
    {
        fill_min = std::atof(std::getenv("ALS_FILL_MIN"));
    }

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

    std::cout << "[Main] 稠密度检测: dense_nnz=" << dense_nnz
              << " sparse_nnz=" << sparse_nnz
              << " tiles=" << num_dense_tiles
              << " avg_fill=" << avg_tile_fill
              << " eff_dense=" << eff_dense
              << " dense_ratio=" << dense_ratio
              << " (阈值 " << fill_min << ")"
              << " -> " << (use_wmma ? "双流串行(WMMA+FP64)" : "纯 CUDA core(全 FP64)")
              << std::endl;

    if (!use_wmma)
    {
        // 纯 CUDA core：构建 FP64 稠密视图（稠密池被释放，
        // 全部走 FP64 稀疏内核）
        build_dense_mttkrp_views(
            hybrid,
            hybrid_dense,
            0);
    }


    // ============================================================
    // 7. 分配 CP 权重 lambda
    //
    // 注意：
    //
    // 这里的 d_lambda 不是 ALS normal equation 的
    // 正则化系数。
    //
    // lambda_reg:
    //     ALS 求解过程使用的正则化参数
    //
    // d_lambda:
    //     最终 CP decomposition 的 component weights
    //
    // 最终模型：
    //
    //     X ≈ Σ_r lambda[r]
    //               A[:,r]
    //               B[:,r]
    //               C[:,r]
    // ============================================================

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

    // SGD 路径需要 λ 初始为 1（ALS 在 als.cu 内部自行初始化）
    if (std::getenv("ALS_SGD")) {
        fill_value_kernel<<<(R + 255) / 256, 256>>>(
            d_lambda, R, 1.0);
        CHECK_CUDA(cudaGetLastError());
    }


    // ============================================================
    // 8. 执行 CP-ALS / CP-SGD
    // ============================================================

    if (std::getenv("ALS_SGD")) {
        // CP-SGD：稀疏残差 SGD 更新 + 稠密 tile WMMA 贡献
        // 参数：ALS_SGD=1, ALS_SGD_GAMMA=<lr>, ALS_SGD_BATCH=<size>
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


    // ============================================================
    // 9. 将重排空间下的 factor
    //    逆映射回原始坐标顺序
    // ============================================================

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


    // ============================================================
    // 10. 输出最终 CP 权重
    //
    // 这里暂时只做调试输出。
    //
    // 如果后续要保存模型，可以在这里把：
    //
    //   d_factors_orig[0]
    //   d_factors_orig[1]
    //   d_factors_orig[2]
    //   d_lambda
    //
    // 一起写入文件。
    // ============================================================

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

    // ============================================================
    // 因子导出（BLCO 兼容格式，供统一评估器对比）
    //   环境变量 ALS_EXPORT_FACTORS=/tmp/my_f 时导出到 /tmp/my_f.{0,1,2}.out
    // ============================================================

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
            // 使用 unpermute 后的因子（原始张量顺序）
            cudaMemcpy(h_f.data(), d_factors_orig[n],
                       dims3[n] * R * sizeof(double), cudaMemcpyDeviceToHost);
            // col-major 输出（与 BLCO 相同：U[n][j*rank+i]）
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


    // ============================================================
    // 11. 释放 GPU / Host 资源
    // ============================================================

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

    // d_factors_perm 已合并为单一缓冲区，只释放基地址
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


    // ============================================================
    // 12. 终结 MAGMA
    // ============================================================

    magma_finalize();

    std::cout
        << "[Main] Real tensor execution complete "
        << "with CADR Reordering & Unpermutation!"
        << std::endl;

    return 0;
}
