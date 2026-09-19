#ifndef GENERIC_HPP
#define GENERIC_HPP

// ============================================================
// cpTC generic higher-order (N >= 2) sparse tensor CP path.
//
// Kept fully separate from the 3-mode hybrid/WMMA pipeline so
// that N == 3 continues to use the original (Tensor-Core aware)
// code path unchanged.  The generic path is pure FP64 ALS and
// works for any order up to the in-file column count of the .tns
// (practical limit: ~8 modes; runtime R <= 32 in the current
// kernel, see g_mttkrp_kernel).
// ============================================================

#include <cstddef>
#include <vector>
#include <string>

struct GTensor {
    int nmodes = 0;                    // order of the tensor
    std::vector<size_t> dims;          // size per mode (0-based max+1)
    size_t nnz = 0;

    std::vector<double> h_vals;        // element values
    std::vector<std::vector<uint32_t>> h_coords; // h_coords[m][e]
};

// Probe: number of modes (value columns) of a .tns file.
// Returns -1 on error.  Only needs to read the first data lines.
int probe_tns_nmodes(const char* path);

// Load an arbitrary-order .tns into GTensor (1-based -> 0-based).
// Returns false on error.
bool load_tns_generic(const char* path, GTensor& t, double* load_ms_out);

// Standalone unit test for the N-order dense (Tensor Core) path.
int run_dense_selftest();

// Run CP-ALS on an N-order sparse tensor (any N >= 2) and print
// per-iteration gram fit + final exact fit / RMSE.
// seed is used for the U[0,1) factor initialization (mt19937).
int run_generic_cp(GTensor& t, int max_iters, int R,
                   unsigned int seed);

#endif
