// ============================================================
// cpTC generic higher-order sparse tensor CP (pure FP64 ALS).
// Works for any order N >= 2 with runtime rank R <= 32.
// The 3-mode Tensor-Core pipeline is intentionally untouched.
// ============================================================

#include "generic.hpp"
#include "solver.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

using namespace nvcuda;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <string>
#include <fstream>

#define GEN_CHECK(call)                                                     \
    do {                                                                    \
        cudaError_t e_ = (call);                                            \
        if (e_ != cudaSuccess) {                                            \
            std::cerr << "[GEN] CUDA error " << e_ << " at " << __FILE__    \
                      << ":" << __LINE__ << " \""                           \
                      << cudaGetErrorString(e_) << "\"" << std::endl;       \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

static inline uint64_t splitbit_hash(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}


// ---------------- tiny generic kernels ----------------

__global__ void g_fill_kernel(double* x, size_t n, double v)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x)
        x[i] = v;
}

__global__ void g_mul_kernel(const double* a, const double* b,
                             double* c, size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x)
        c[i] = a[i] * b[i];
}

__global__ void g_sum_kernel(const double* x, size_t n, double* out)
{
    double s = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x)
        s += x[i];

    __shared__ double sh[256];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    sh[tid] = s;
    __syncthreads();
    for (int off = nt / 2; off > 0; off >>= 1)
    {
        if (tid < off)
            sh[tid] += sh[tid + off];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(out, sh[0]);
}

// ---------------- MTTKRP (warp per element, lane per rank) ----------------
// Each block is one warp that owns one element; lane l accumulates rank l.
// Requires R <= 32.

__global__ void g_mttkrp_kernel(
    const double* __restrict__ vals,
    const uint32_t* __restrict__ coords,   // [m*nnz + e]
    size_t nnz,
    int nmodes,
    int target,
    int R,
    const double* const* __restrict__ F,   // nmodes factor pointers
    double* __restrict__ FtV)              // dim[target] * R, row-major
{
    const size_t e = blockIdx.x;
    if (e >= nnz)
        return;

    const int lane = threadIdx.x;          // blockDim.x == 32
    const uint32_t row = coords[(size_t)target * nnz + e];
    const double v = vals[e];

    if (lane < R)
    {
        double acc = v;
        for (int m = 0; m < nmodes; ++m)
        {
            if (m == target)
                continue;
            const uint32_t om = coords[(size_t)m * nnz + e];
            acc *= F[m][(size_t)om * R + lane];
        }
        atomicAdd(&FtV[(size_t)row * R + lane], acc);
    }
}

// ---------------- MTTKRP v2: row-segment blocks ----------------
// One 128-thread block per (row, segment<=SEG elements) of the
// target-mode-sorted view.  Each thread owns one rank (lane) and
// accumulates several elements in registers; 4 warps are reduced
// through shared memory, then exactly ONE atomicAdd per (row,rank)
// per segment.  Requires R <= 32.

#define GEN_SEG 1024

__global__ void g_mttkrp_seg_kernel(
    const double* __restrict__ svals,       // nnz, sorted by target row
    const uint32_t* __restrict__ scoords,   // (nmodes-1)*nnz, others ascending
    size_t nnz,
    int nmodes,
    int target,
    int R,
    const uint32_t* __restrict__ seg_row,
    const uint32_t* __restrict__ seg_start,
    const uint32_t* __restrict__ seg_cnt,
    const double* const* __restrict__ F,
    double* __restrict__ FtV)
{
    const int seg = blockIdx.x;
    const uint32_t row = seg_row[seg];
    const uint32_t s0  = seg_start[seg];
    const uint32_t cnt = seg_cnt[seg];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;      // 4 warps (blockDim 128)

    double acc = 0.0;

    if (lane < R)
    {
        for (uint32_t e = s0 + (uint32_t)warp;
             e < s0 + cnt;
             e += 4u)
        {
            double prod = svals[e];

            int q = 0;
            for (int m = 0; m < nmodes; ++m)
            {
                if (m == target)
                    continue;
                const uint32_t om = scoords[(size_t)q * nnz + e];
                prod *= F[m][(size_t)om * R + lane];
                ++q;
            }
            acc += prod;
        }
    }

    __shared__ double s_red[4][32];
    s_red[warp][lane] = acc;
    __syncthreads();

    if (threadIdx.x < 32 && threadIdx.x < R)
    {
        const double total =
            s_red[0][threadIdx.x] + s_red[1][threadIdx.x] +
            s_red[2][threadIdx.x] + s_red[3][threadIdx.x];

        if (total != 0.0)
            atomicAdd(&FtV[(size_t)row * R + threadIdx.x], total);
    }
}

// ---------------- Gram of one factor (R <= 32) ----------------
// G = F^T F computed with a row tile staged in shared memory: the
// factor is read from global exactly once, and only R*(R+1)/2
// accumulators + one atomicAdd per element of G per block are used.
// For small R this beats cuBLAS DGEMM by a wide margin on factors
// with millions of rows (LBNL: 868k x 16).

#define GEN_GRAM_SMEM_BYTES (64 * 1024)

__global__ void g_gram_kernel(
    const double* __restrict__ F,
    size_t rows,
    int R,
    int rows_per_block,
    double* __restrict__ G)
{
    extern __shared__ double smem[];
    double* s_tile = smem;                                  // rows_per_block * R
    double* s_G    = smem + (size_t)rows_per_block * R;     // R * R

    const size_t start = (size_t)blockIdx.x * (size_t)rows_per_block;
    if (start >= rows)
        return;

    const size_t nrows =
        ((size_t)rows_per_block < rows - start)
            ? (size_t)rows_per_block
            : (rows - start);

    for (int i = threadIdx.x; i < R * R; i += blockDim.x)
        s_G[i] = 0.0;

    for (size_t i = threadIdx.x; i < nrows * (size_t)R; i += blockDim.x)
        s_tile[i] = F[start * (size_t)R + i];

    __syncthreads();

    const int npairs = R * (R + 1) / 2;

    for (int p = threadIdx.x; p < npairs; p += blockDim.x)
    {
        int a = 0, rem = p;
        while (rem >= R - a)
        {
            rem -= (R - a);
            ++a;
        }
        const int b = a + rem;

        const double* pa = s_tile + a;
        const double* pb = s_tile + b;

        double acc = 0.0;
        for (size_t i = 0; i < nrows; ++i)
            acc += pa[i * (size_t)R] * pb[i * (size_t)R];

        s_G[a * R + b] = acc;
        s_G[b * R + a] = acc;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < R * R; i += blockDim.x)
        atomicAdd(&G[i], s_G[i]);
}

// v3 MTTKRP: same row-segment scheme, but EPW elements are packed into
// each warp (lanes r, r+R, ... carry different elements of the same
// segment).  For R <= 16 half of a warp would otherwise sit idle, so
// EPW = 32/R keeps every lane busy.  All elements of a segment share the
// target row, so the packed partials are summed before the single
// atomicAdd per (row, rank) per segment.
//
// seg_single[e] != 0 means the segment is the ONLY segment of its row:
// then the result can be *stored* instead of atomically added.  Tensors
// with many short rows (e.g. Flickr: 28M rows for 113M nnz) are
// dominated by atomic traffic otherwise.
template <int EPW>
__global__ void g_mttkrp_seg_pack_kernel(
    const double* __restrict__ svals,
    const uint32_t* __restrict__ scoords,
    size_t nnz,
    int nmodes,
    int target,
    int R,
    const uint32_t* __restrict__ seg_row,
    const uint32_t* __restrict__ seg_start,
    const uint32_t* __restrict__ seg_cnt,
    const uint8_t* __restrict__ seg_single,
    const uint8_t* __restrict__ skip,      // nullable: elements handled by the dense path
    const double* const* __restrict__ F,
    double* __restrict__ FtV)
{
    const int seg = blockIdx.x;
    const uint32_t row = seg_row[seg];
    const uint32_t s0  = seg_start[seg];
    const uint32_t cnt = seg_cnt[seg];
    const bool single  = (seg_single[seg] != 0);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;      // 4 warps (blockDim 128)

    const int r    = lane % R;              // rank
    const int slot = lane / R;              // packed element slot

    double acc = 0.0;

    for (uint32_t e = s0 + (uint32_t)(warp * EPW + slot);
         e < s0 + cnt;
         e += (uint32_t)(4 * EPW))
    {
        if (skip != nullptr && skip[e])
            continue;

        double prod = svals[e];

        int q = 0;
        for (int m = 0; m < nmodes; ++m)
        {
            if (m == target)
                continue;
            const uint32_t om = scoords[(size_t)q * nnz + e];
            prod *= F[m][(size_t)om * R + r];
            ++q;
        }
        acc += prod;
    }

    // sum the EPW slots inside the warp (same rank, lanes r, r+R, ...)
    #pragma unroll
    for (int off = R; off < 32; off <<= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    __shared__ double s_red[4][32];
    if (lane < R)
        s_red[warp][lane] = acc;
    __syncthreads();

    if (threadIdx.x < R)
    {
        const double total =
            s_red[0][threadIdx.x] + s_red[1][threadIdx.x] +
            s_red[2][threadIdx.x] + s_red[3][threadIdx.x];

        if (single)
            FtV[(size_t)row * R + threadIdx.x] = total;
        else if (total != 0.0)
            atomicAdd(&FtV[(size_t)row * R + threadIdx.x], total);
    }
}

// Column norms of a row-major factor, bandwidth friendly: each thread
// walks whole rows (R contiguous doubles) and keeps R accumulators, so
// the factor is read exactly once with coalesced accesses.  (A naively
// column-parallel kernel reads one double per 16/32-double row and
// wastes ~16-32x of the memory bandwidth -- this dominated the runtime
// of tensors with very large mode dimensions, e.g. Flickr mode 1 with
// 28M rows.)
__global__ void g_col_norms2_kernel(
    const double* __restrict__ F,
    size_t rows,
    int R,
    double* __restrict__ norms)
{
    extern __shared__ double s_acc[];       // blockDim.x * R

    double acc[32];
    for (int r = 0; r < R; ++r)
        acc[r] = 0.0;

    const size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < rows;
         i += stride)
    {
        const double* row = F + i * (size_t)R;
        for (int r = 0; r < R; ++r)
        {
            const double v = row[r];
            acc[r] += v * v;
        }
    }

    double* mine = s_acc + (size_t)threadIdx.x * R;
    for (int r = 0; r < R; ++r)
        mine[r] = acc[r];

    __syncthreads();

    if (threadIdx.x < R)
    {
        double sum = 0.0;
        for (int t = 0; t < (int)blockDim.x; ++t)
            sum += s_acc[(size_t)t * R + threadIdx.x];
        atomicAdd(&norms[threadIdx.x], sum);
    }
}

__global__ void g_norms_finalize_kernel(double* norms, int R)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < R)
    {
        double n = std::sqrt(norms[r]);
        if (!std::isfinite(n) || n < 1e-14)
            n = 1.0;
        norms[r] = n;
    }
}


// ============================================================
// N-order dense-block (Tensor Core) path.
//
// A "unit" is a 16^3 tile of the three INNER modes with the remaining
// (outer) modes fixed to one coordinate tuple.  For an inner target mode n
// the MTTKRP contribution is exactly the 3rd-order one times the per-rank
// scalar  w_r(c) = prod_{m outer} F_m[c_m, r]:
//
//   Y_n[i,r] += sum_{j,k} X[i,j,k,c] * F_p[j,r] * F_q[k,r] * w_r(c)
//
// so the existing WMMA scheme applies verbatim: fold F_p into the MMA,
// multiply by F_q[k,r]*w_r(c) in FP64 at the output, accumulate across the
// 16 slabs in registers and issue one atomicAdd per (row, rank) per unit.
// Outer target modes stay on the FP64 sparse kernel.
// ============================================================

struct GDUnit {
    uint32_t bi, bj, bk;      // tile coords on inner slots 0,1,2
    uint32_t nnz;             // elements in the unit
    uint32_t start;           // offset into the pool
    uint32_t outer_off;       // offset into outer coord array ((N-3) per unit)
};

__global__ void g_dense_tile_mttkrp_kernel(
    const GDUnit* __restrict__ units,
    const uint32_t* __restrict__ lid,        // local ids (a*256+b*16+c)
    const double* __restrict__ vals,
    const uint32_t* __restrict__ outer_coords,
    const int* __restrict__ outer_modes,
    int n_outer,
    const int* __restrict__ slot_dims,   // dims[inner[0]], dims[inner[1]], dims[inner[2]]
    int nmodes,
    int target,                              // inner slot 0..2
    int R,
    const double* const* __restrict__ F,
    double* __restrict__ FtV)
{
    const int unit_id = blockIdx.x;
    const int lane = threadIdx.x & 31;

    extern __shared__ float sm[];
    float* tile  = sm;            // 4096
    float* slice = sm + 4096;     // 256
    float* fac   = sm + 4352;     // 256
    float* gout  = sm + 4608;     // 256

    const GDUnit u = units[unit_id];

    for (int i = lane; i < 4096; i += 32) tile[i] = 0.0f;
    __syncwarp();
    // accumulate: duplicate coordinates inside a unit must sum (the FP64
    // kernel sums them too); a plain store would silently keep only the last
    for (uint32_t e = (uint32_t)lane; e < u.nnz; e += 32)
        atomicAdd(&tile[lid[u.start + e]], (float)vals[u.start + e]);
    __syncwarp();

    // slot roles for this target: li -> slot t, lj -> slot p, lk -> slot q
    // slot roles: li -> slot t, lj -> slot p, lk -> slot q.
    // The smem tile is laid out as slot0*256 + slot1*16 + slot2, so each of
    // li/lj/lk must be weighted by the *stride of its slot*.
    const int slot_li = target;
    const int slot_lj = (target + 1) % 3;
    const int slot_lk = (target + 2) % 3;
    auto slot_stride = [](int slot) { return (slot == 0) ? 256 : (slot == 1) ? 16 : 1; };
    const int coef_li = slot_stride(slot_li);
    const int coef_lj = slot_stride(slot_lj);
    const int coef_lk = slot_stride(slot_lk);

    // outer weight w_r(c) (FP64), per rank
    double w[32];
    for (int r = 0; r < R && r < 32; ++r) w[r] = 1.0;
    for (int q = 0; q < n_outer; ++q)
    {
        const uint32_t c =
            outer_coords[u.outer_off + (uint32_t)q];
        const double* Fm = F[outer_modes[q]];
        for (int r = 0; r < R && r < 32; ++r)
            w[r] *= Fm[(size_t)c * R + r];
    }

    const int pm = (target + 1) % 3;          // folded factor = inner slot p
    const int qm = (target + 2) % 3;          // outer multiply = inner slot q

    wmma::fragment<wmma::matrix_a, 16, 16, 8,
                   wmma::precision::tf32, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 8,
                   wmma::precision::tf32, wmma::row_major> b0, b1;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> cfr;

    for (int r_base = 0; r_base < R; r_base += 16)
    {
        double acc[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.0;

        // factor(-slot p) rows: fac[lj*16 + rr] = F_p[(bp*16+lj)*R + r_base+rr]
        for (int i = lane; i < 256; i += 32)
        {
            const int lj = i / 16, rr = i % 16;
            const int gr = r_base + rr;
            const uint64_t pcoord =
                (pm == 0) ? u.bi : (pm == 1) ? u.bj : u.bk;
            const uint32_t grow = (uint32_t)pcoord * 16u + (uint32_t)lj;
            fac[i] = (gr < R && (int)grow < slot_dims[pm])
                         ? (float)F[pm][(size_t)grow * R + gr] : 0.0f;
        }
        __syncwarp();

        for (int lk = 0; lk < 16; ++lk)
        {
            for (int i = lane; i < 256; i += 32)
            {
                const int li = i / 16, lj = i % 16;
                slice[i] = tile[li * coef_li + lj * coef_lj + lk * coef_lk];
            }
            __syncwarp();

            wmma::fill_fragment(cfr, 0.0f);
            wmma::load_matrix_sync(a0, slice, 16);
            wmma::load_matrix_sync(b0, fac, 16);
            wmma::mma_sync(cfr, a0, b0, cfr);
            wmma::load_matrix_sync(a1, slice + 8, 16);
            wmma::load_matrix_sync(b1, fac + 128, 16);
            wmma::mma_sync(cfr, a1, b1, cfr);
            wmma::store_matrix_sync(gout, cfr, 16, wmma::mem_row_major);
            __syncwarp();

            const uint64_t qcoord = (qm == 0) ? u.bi : (qm == 1) ? u.bj : u.bk;
            const uint32_t qrow = (uint32_t)qcoord * 16u + (uint32_t)lk;
            const bool q_ok = ((int)qrow < slot_dims[qm]);

            for (int it = 0; it < 8; ++it)
            {
                const int idx = it * 32 + lane;
                const int li = idx / 16, rr = idx % 16;
                const int gr = r_base + rr;
                if (gr < R && q_ok)
                    acc[it] += (double)gout[idx] *
                               F[qm][(size_t)qrow * R + gr] * w[gr];
            }
            __syncwarp();
        }

        const uint64_t tcoord = (target == 0) ? u.bi : (target == 1) ? u.bj : u.bk;
        for (int it = 0; it < 8; ++it)
        {
            const int idx = it * 32 + lane;
            const int li = idx / 16, rr = idx % 16;
            const int gr = r_base + rr;
            const size_t trow = (size_t)tcoord * 16u + (size_t)li;
            if (gr < R && trow < (size_t)slot_dims[target] && acc[it] != 0.0)
                atomicAdd(&FtV[trow * R + gr], acc[it]);
        }
        __syncthreads();
    }
}

// Host side: pick dense units over inner slots (0,1,2) and build the pool.
struct GDense {
    GDUnit*   d_units   = nullptr;
    uint32_t* d_lid     = nullptr;
    double*   d_vals    = nullptr;
    uint32_t* d_outer   = nullptr;
    uint8_t*  d_is_dense = nullptr;    // per element flag (original order)
    int*      d_outer_modes = nullptr; // device copy of outer mode ids
    std::vector<uint8_t> h_is_dense;   // host copy, used while building views
    std::vector<int> outer_modes;      // which modes are outer (in stored order)
    int n_units = 0;
    long long dense_nnz = 0;
    bool enabled = false;
};

static void build_dense_nd(GTensor& t, double threshold, int* inner,
                           GDense& gd, double* ms_out)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    const int N = t.nmodes;
    const size_t nnz = t.nnz;
    const int cutoff = (int)(4096.0 * threshold);

    gd.outer_modes.clear();
    for (int m = 0; m < N; ++m)
        if (m != inner[0] && m != inner[1] && m != inner[2])
            gd.outer_modes.push_back(m);

    std::unordered_map<uint64_t, uint32_t> cnt;
    cnt.reserve(nnz / 4 + 1);

    std::vector<uint64_t> key(nnz);
    for (size_t e = 0; e < nnz; ++e)
    {
        const uint32_t a = t.h_coords[(size_t)inner[0]][e];
        const uint32_t b = t.h_coords[(size_t)inner[1]][e];
        const uint32_t c = t.h_coords[(size_t)inner[2]][e];
        uint64_t o = 0;
        for (int m = 0; m < N; ++m)
            if (m != inner[0] && m != inner[1] && m != inner[2])
                o = o * 1000003ull + t.h_coords[(size_t)m][e];
        const uint64_t tile = ((uint64_t)(a / 16) << 40) |
                              ((uint64_t)(b / 16) << 20) | (uint64_t)(c / 16);
        key[e] = splitbit_hash(tile ^ (o * 0x9E3779B97F4A7C15ull));
        ++cnt[key[e]];
    }

    std::vector<char> dense_key;
    std::unordered_map<uint64_t, uint32_t> uniq;
    // second pass capturing the actual (tile, outer) metadata of dense units
    gd.d_is_dense = nullptr;
    std::vector<uint8_t> is_d;

    // collect the key -> (bi,bj,bk,outer) mapping for dense keys
    std::unordered_map<uint64_t, uint32_t> dense_index;
    std::vector<GDUnit> units;
    std::vector<std::vector<uint32_t>> outer_of_unit;

    for (size_t e = 0; e < nnz; ++e)
    {
        auto it = cnt.find(key[e]);
        if (it == cnt.end() || it->second < (uint32_t)cutoff) continue;
        if (dense_index.find(key[e]) == dense_index.end())
        {
            GDUnit u{};
            u.bi = t.h_coords[(size_t)inner[0]][e] / 16;
            u.bj = t.h_coords[(size_t)inner[1]][e] / 16;
            u.bk = t.h_coords[(size_t)inner[2]][e] / 16;
            u.nnz = 0;
            u.start = 0;
            u.outer_off = (uint32_t)(outer_of_unit.size() * (size_t)(N - 3));
            std::vector<uint32_t> oc;
            for (int m = 0; m < N; ++m)
                if (m != inner[0] && m != inner[1] && m != inner[2])
                    oc.push_back(t.h_coords[(size_t)m][e]);
            outer_of_unit.push_back(oc);
            dense_index[key[e]] = (uint32_t)units.size();
            units.push_back(u);
        }
    }

    gd.n_units = (int)units.size();
    if (gd.n_units == 0)
    {
        if (ms_out) *ms_out =
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
        return;
    }

    // allocate pool
    size_t total = 0;
    for (size_t e = 0; e < nnz; ++e)
    {
        auto it = dense_index.find(key[e]);
        if (it != dense_index.end())
        {
            units[it->second].nnz++;
            ++total;
        }
    }
    {
        uint32_t off = 0;
        for (auto& u : units) { u.start = off; off += u.nnz; }
    }

    std::vector<uint32_t> lid(total);
    std::vector<double>   val(total);
    is_d.assign(nnz, 0);
    {
        std::vector<uint32_t> cur(units.size());
        for (size_t i = 0; i < units.size(); ++i) cur[i] = units[i].start;
        for (size_t e = 0; e < nnz; ++e)
        {
            auto it = dense_index.find(key[e]);
            if (it == dense_index.end()) continue;
            const uint32_t ui = it->second;
            const uint32_t la = t.h_coords[(size_t)inner[0]][e] % 16;
            const uint32_t lb = t.h_coords[(size_t)inner[1]][e] % 16;
            const uint32_t lc = t.h_coords[(size_t)inner[2]][e] % 16;
            lid[cur[ui]] = la * 256 + lb * 16 + lc;
            val[cur[ui]] = t.h_vals[e];
            ++cur[ui];
            is_d[e] = 1;
        }
    }

    std::vector<uint32_t> outer_flat;
    for (auto& v : outer_of_unit)
        for (uint32_t x : v) outer_flat.push_back(x);

    GEN_CHECK(cudaMalloc(&gd.d_units, units.size() * sizeof(GDUnit)));
    GEN_CHECK(cudaMemcpy(gd.d_units, units.data(),
                         units.size() * sizeof(GDUnit),
                         cudaMemcpyHostToDevice));
    GEN_CHECK(cudaMalloc(&gd.d_lid, total * sizeof(uint32_t)));
    GEN_CHECK(cudaMemcpy(gd.d_lid, lid.data(), total * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
    GEN_CHECK(cudaMalloc(&gd.d_vals, total * sizeof(double)));
    GEN_CHECK(cudaMemcpy(gd.d_vals, val.data(), total * sizeof(double),
                         cudaMemcpyHostToDevice));
    GEN_CHECK(cudaMalloc(&gd.d_outer, outer_flat.size() * sizeof(uint32_t)));
    GEN_CHECK(cudaMemcpy(gd.d_outer, outer_flat.data(),
                         outer_flat.size() * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
    GEN_CHECK(cudaMalloc(&gd.d_is_dense, nnz * sizeof(uint8_t)));
    GEN_CHECK(cudaMemcpy(gd.d_is_dense, is_d.data(), nnz * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));

    GEN_CHECK(cudaMalloc(&gd.d_outer_modes,
                         gd.outer_modes.size() * sizeof(int)));
    GEN_CHECK(cudaMemcpy(gd.d_outer_modes, gd.outer_modes.data(),
                         gd.outer_modes.size() * sizeof(int),
                         cudaMemcpyHostToDevice));

    gd.dense_nnz = (long long)total;
    gd.h_is_dense = is_d;                 // keep for view-order skip flags
    gd.enabled = true;

    if (ms_out) *ms_out =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
}

static void free_dense_nd(GDense& gd)
{
    if (gd.d_units) cudaFree(gd.d_units);
    if (gd.d_lid) cudaFree(gd.d_lid);
    if (gd.d_vals) cudaFree(gd.d_vals);
    if (gd.d_outer) cudaFree(gd.d_outer);
    if (gd.d_is_dense) cudaFree(gd.d_is_dense);
    if (gd.d_outer_modes) cudaFree(gd.d_outer_modes);
    gd = GDense{};
}

// ---------------- Cholesky solve of the R x R normal equations ----------------
// (FtF + eps I) x = FtV, solved per factor row.  FtF is the Hadamard
// product of R x R Gram matrices (symmetric PSD); for the small R used
// here a shared-memory Cholesky + triangular solves is far cheaper
// than a cuSOLVER gesvdj call (which dominated the per-iteration time
// on real tensors).  A tiny relative ridge keeps the system invertible
// when a Gram is rank deficient (e.g. empty target rows).

__global__ void g_chol_solve_kernel(
    const double* __restrict__ FtF,
    const double* __restrict__ FtV,
    double* __restrict__ F,
    size_t dim,
    int R)
{
    extern __shared__ double sL[];          // R*R, row-major lower part

    if (threadIdx.x == 0)
    {
        double tr = 0.0;
        for (int i = 0; i < R; ++i)
            tr += FtF[i * R + i];

        const double eps =
            1e-12 * ((tr > 0.0) ? tr / (double)R : 1.0) + 1e-300;

        for (int i = 0; i < R; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                double s = FtF[i * R + j];
                if (i == j)
                    s += eps;

                for (int k = 0; k < j; ++k)
                    s -= sL[i * R + k] * sL[j * R + k];

                if (i == j)
                {
                    sL[i * R + j] = (s > 0.0) ? sqrt(s) : sqrt(eps);
                }
                else
                {
                    sL[i * R + j] = s / sL[j * R + j];
                }
            }
        }
    }
    __syncthreads();

    double y[32];
    double x[32];

    for (size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         row < dim;
         row += (size_t)gridDim.x * blockDim.x)
    {
        const double* b = FtV + row * (size_t)R;

        // forward substitution: L y = b
        for (int i = 0; i < R; ++i)
        {
            double s = b[i];
            for (int k = 0; k < i; ++k)
                s -= sL[i * R + k] * y[k];
            y[i] = s / sL[i * R + i];
        }

        // back substitution: L^T x = y
        for (int i = R - 1; i >= 0; --i)
        {
            double s = y[i];
            for (int k = i + 1; k < R; ++k)
                s -= sL[k * R + i] * x[k];
            x[i] = s / sL[i * R + i];
        }

        double* f = F + row * (size_t)R;
        for (int i = 0; i < R; ++i)
            f[i] = x[i];
    }
}

// ---------------- lambda / normalization ----------------

__global__ void g_div_lambda_kernel(double* F, size_t dim, int R,
                                    const double* lambda)
{
    const size_t total = dim * (size_t)R;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x)
        F[idx] /= lambda[idx % R];
}

__global__ void g_col_norms_kernel(const double* F, size_t dim, int R,
                                   double* norms)
{
    const int r = blockIdx.x;              // one block per rank
    if (r >= R)
        return;

    double s = 0.0;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x)
    {
        const double v = F[i * (size_t)R + r];
        s += v * v;
    }

    __shared__ double sh[256];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    sh[tid] = s;
    __syncthreads();
    for (int off = nt / 2; off > 0; off >>= 1)
    {
        if (tid < off)
            sh[tid] += sh[tid + off];
        __syncthreads();
    }
    if (tid == 0)
    {
        double n = std::sqrt(sh[0]);
        if (!std::isfinite(n) || n < 1e-14)
            n = 1.0;
        norms[r] = n;
    }
}

__global__ void g_normalize_kernel(double* F, size_t dim, int R,
                                   const double* norms)
{
    const size_t total = dim * (size_t)R;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x)
        F[idx] /= norms[idx % R];
}

__global__ void g_set_lambda_kernel(double* lambda, int R,
                                    const double* norms)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < R)
        lambda[r] = norms[r];
}

// dot of factor F (dim x R) with FtV (dim x R) per rank: out[r]
__global__ void g_dot_kernel(const double* F, const double* FtV,
                             size_t dim, int R, double* out)
{
    const int r = blockIdx.x;
    if (r >= R)
        return;

    double s = 0.0;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x)
        s += F[i * (size_t)R + r] * FtV[i * (size_t)R + r];

    __shared__ double sh[256];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    sh[tid] = s;
    __syncthreads();
    for (int off = nt / 2; off > 0; off >>= 1)
    {
        if (tid < off)
            sh[tid] += sh[tid + off];
        __syncthreads();
    }
    if (tid == 0)
        out[r] = sh[0];
}

__global__ void g_dot_lambda_kernel(const double* lambda,
                                    const double* out, int R,
                                    double* scalar)
{
    double s = 0.0;
    for (int r = threadIdx.x; r < R; r += blockDim.x)
        s += lambda[r] * out[r];

    __shared__ double sh[256];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    sh[tid] = s;
    __syncthreads();
    for (int off = nt / 2; off > 0; off >>= 1)
    {
        if (tid < off)
            sh[tid] += sh[tid + off];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(scalar, sh[0]);
}

// final exact residual: res^2 over all nnz
__global__ void g_exact_residual_kernel(
    const double* __restrict__ vals,
    const uint32_t* __restrict__ coords,
    size_t nnz,
    int nmodes,
    int R,
    const double* const* __restrict__ F,
    const double* __restrict__ lambda,
    double* __restrict__ d_res2)
{
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         e < nnz;
         e += (size_t)gridDim.x * blockDim.x)
    {
        const double v = vals[e];
        double xhat = 0.0;

        for (int r = 0; r < R; ++r)
        {
            double prod = lambda[r];
            for (int m = 0; m < nmodes; ++m)
            {
                const uint32_t om = coords[(size_t)m * nnz + e];
                prod *= F[m][(size_t)om * R + r];
            }
            xhat += prod;
        }

        const double d = v - xhat;
        atomicAdd(d_res2, d * d);
    }
}

// ---------------- host helpers ----------------

static int fast_int(const char* p, const char** end)
{
    int v = 0;
    while (*p >= '0' && *p <= '9')
    {
        v = v * 10 + (*p - '0');
        ++p;
    }
    *end = p;
    return v;
}

int probe_tns_nmodes(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return -1;

    char buf[1 << 16];
    int nmodes = -1;

    while (fgets(buf, sizeof buf, f))
    {
        if (buf[0] == '#' || buf[0] == '\n' || buf[0] == '\r')
            continue;

        char* p = buf;
        while (*p == ' ' || *p == '\t')
            ++p;

        if (!(*p >= '0' && *p <= '9'))
            continue;              // skip header lines

        int fields = 0;
        const char* q = p;
        while (*q && *q != '\n' && *q != '\r')
        {
            while (*q == ' ' || *q == '\t')
                ++q;
            if (*q == '\n' || *q == '\r' || !*q)
                break;
            ++fields;
            while (*q && *q != ' ' && *q != '\t')
                ++q;
        }
        nmodes = fields - 1;       // indices columns = fields - 1 (value)
        break;
    }
    fclose(f);
    return nmodes;
}

bool load_tns_generic(const char* path, GTensor& t, double* load_ms_out)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    FILE* f = fopen(path, "rb");
    if (!f)
    {
        std::cerr << "[GEN] cannot open " << path << std::endl;
        return false;
    }

    // pass 1: count lines for reserve
    constexpr size_t BUF = 1u << 22;
    std::vector<char> buf(BUF);
    size_t n = 0;
    size_t est = 0;
    while ((n = fread(buf.data(), 1, BUF, f)) > 0)
        for (size_t i = 0; i < n; ++i)
            if (buf[i] == '\n')
                ++est;
    rewind(f);

    const int nmodes = probe_tns_nmodes(path);
    if (nmodes < 2)
    {
        std::cerr << "[GEN] invalid tensor order: " << nmodes << std::endl;
        fclose(f);
        return false;
    }

    t.nmodes = nmodes;
    t.dims.assign(nmodes, 0);
    t.h_coords.assign(nmodes, {});
    for (int m = 0; m < nmodes; ++m)
        t.h_coords[m].reserve(est);
    t.h_vals.reserve(est);

    // pass 2: parse
    std::vector<int> idx(nmodes);
    std::vector<size_t> mx(nmodes, 0);

    auto process_line = [&](const char* s, size_t len)
    {
        if (len == 0)
            return;
        if (s[0] == '#')
            return;

        const char* p = s;
        const char* end = nullptr;

        for (int m = 0; m < nmodes; ++m)
        {
            while (*p == ' ' || *p == '\t')
                ++p;
            if (!(*p >= '0' && *p <= '9'))
                return;
            idx[m] = fast_int(p, &end);
            p = end;
        }
        while (*p == ' ' || *p == '\t')
            ++p;

        char* endc = nullptr;
        const double v = std::strtod(p, &endc);
        if (endc == p)
            return;

        for (int m = 0; m < nmodes; ++m)
        {
            const uint32_t z = (idx[m] > 0) ? (uint32_t)(idx[m] - 1) : 0;
            t.h_coords[m].push_back(z);
            if ((size_t)idx[m] > mx[m])
                mx[m] = (size_t)idx[m];
        }
        t.h_vals.push_back(v);
    };

    // pass 2: block reads with correct cross-block line handling.
    // (The carried partial line stays at the head of the buffer and the
    //  next read appends *after* it -- otherwise one line is dropped
    //  every BUF bytes.)
    size_t carry = 0;
    while (true)
    {
        const size_t got = fread(buf.data() + carry, 1, BUF - carry, f);
        const size_t total = carry + got;

        if (total == 0)
            break;

        size_t pos = 0;
        while (pos < total)
        {
            const char* nl =
                (const char*)memchr(buf.data() + pos, '\n', total - pos);
            if (!nl)
                break;
            process_line(buf.data() + pos, (size_t)(nl - (buf.data() + pos)));
            pos = (size_t)(nl - buf.data()) + 1;
        }

        carry = total - pos;
        if (carry > 0)
            memmove(buf.data(), buf.data() + pos, carry);

        if (got == 0)
            break;                     // EOF

        if (carry == BUF)
        {
            std::cerr << "[GEN] line longer than read buffer (" << BUF
                      << " bytes); aborting" << std::endl;
            fclose(f);
            return false;
        }
    }
    if (carry > 0)
        process_line(buf.data(), carry);

    fclose(f);

    for (int m = 0; m < nmodes; ++m)
        t.dims[m] = (mx[m] > 0) ? mx[m] : 1;

    t.nnz = t.h_vals.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    if (load_ms_out)
        *load_ms_out =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

// ---------------- CP-ALS driver ----------------

// ============================================================
// Index reordering for the N-order path.
//
//   GEN_REORDER=hits (default): iterated tensor hub scores
//        s_m(i) = sum_{e: i} prod_{m'!=m} s_m'(o_m'), T rounds
//        (GEN_REORDER_T, default 3), then sort indices by score desc.
//        Cheap (O(T*N*nnz) on a bounded sample) and measured to give the
//        best dense-block coverage among the tested heuristics.
//   GEN_REORDER=cadr : degree-descending per mode (cpTC's CADR) -- the
//        fallback / 3rd-order convention.
//   GEN_REORDER=none : identity (no reordering).
//
// A reordering is a bijection per mode, so fit/RMSE are invariant; it only
// changes memory locality (and, for a dense-block pathway, which 16^3 tiles
// are dense).
// ============================================================

// index maps from the last reordering (old -> new); used so the random factor
// initialization stays consistent with the relabeled data rows.
static std::vector<std::vector<uint32_t>> g_rank;

static void apply_reordering(GTensor& t, const char* how, int rounds,
                             double* ms_out,
                             std::vector<std::vector<uint32_t>>* rank_out)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    const int N = t.nmodes;
    const size_t nnz = t.nnz;

    std::vector<std::vector<uint32_t>> rank((size_t)N);
    for (int m = 0; m < N; ++m)
    {
        rank[(size_t)m].resize(t.dims[m]);
        for (size_t i = 0; i < t.dims[m]; ++i)
            rank[(size_t)m][i] = (uint32_t)i;
    }

    // ---- auto: probe first, reorder only when it can actually lift the
    //      dense-16^3-tile fill above the Tensor-Core break-even (tau=0.11).
    //      (a) cheap raw probe on a sample; if the tensor is hopeless there
    //          (<0.02) skip reordering entirely;  (b) otherwise try HITS and
    //          keep it only if the probed fill reaches tau.
    if (std::strcmp(how, "auto") == 0)
    {
        const size_t budget = 2000000;
        const size_t st = (nnz > budget) ? (nnz / budget) : 1;
        const uint64_t nbj = (t.dims[1] + 15) / 16;
        const uint64_t nbk = (t.dims[2] + 15) / 16;

        auto fill_of = [&](const std::vector<std::vector<uint32_t>>& rk,
                           bool use_rank) -> double
        {
            std::unordered_map<uint64_t, uint32_t> cnt;
            cnt.reserve(budget / 2 + 1);
            for (size_t e = 0; e < nnz; e += st)
            {
                const uint64_t a = use_rank ? rk[0][t.h_coords[0][e]] : t.h_coords[0][e];
                const uint64_t b = use_rank ? rk[1][t.h_coords[1][e]] : t.h_coords[1][e];
                const uint64_t c = use_rank ? rk[2][t.h_coords[2][e]] : t.h_coords[2][e];
                const uint64_t tile = ((a / 16) * nbj + b / 16) * nbk + c / 16;
                uint64_t o = 0;
                for (int m = 3; m < N; ++m)
                    o = o * 1000003ull +
                        (use_rank ? rk[(size_t)m][t.h_coords[(size_t)m][e]]
                                  : t.h_coords[(size_t)m][e]);
                ++cnt[splitbit_hash(tile ^ (o * 0x9E3779B97F4A7C15ull))];
            }
            // counts are from a strided sample: scale back before applying
            // the (full-tensor) dense cutoff of 122 nonzeros per 16^3 tile
            long long tiles = 0, nnz_in = 0;
            for (const auto& kv : cnt)
                if ((long long)kv.second * (long long)st >= 122)
                { ++tiles; nnz_in += (long long)kv.second * (long long)st; }
            return tiles ? (double)nnz_in / ((double)tiles * 4096.0) : 0.0;
        };

        const auto t_probe0 = std::chrono::high_resolution_clock::now();

        // (a) raw probe (no permutation applied yet)
        std::vector<std::vector<uint32_t>> ident;
        const double f_raw = fill_of(ident, false);

        if (f_raw < 0.02)
        {
            const double pms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_probe0).count();
            std::cout << "[GEN] reorder=auto -> none (raw probe fill " << f_raw
                      << " < 0.02, stride=" << st << "; skipped HITS probe)"
                      << std::endl;
            g_rank.clear();
            if (ms_out) *ms_out = pms;
            return;
        }

        // (b) tentative HITS, keep only if it reaches tau
        double hits_ms = 0.0;
        g_rank.clear();
        apply_reordering(t, "hits", rounds, &hits_ms, &g_rank);
        const std::vector<std::vector<uint32_t>> rk_try = g_rank;

        double f_hits = 0.0;
        bool ok = ((int)rk_try.size() == N);
        if (ok) f_hits = fill_of(rk_try, true);

        const double pms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_probe0).count();

        if (ok && f_hits >= 0.11)
        {
            std::cout << "[GEN] reorder=auto -> hits (probe fill " << f_raw
                      << " -> " << f_hits << " >= 0.11)" << std::endl;
            if (ms_out) *ms_out = pms;
            return;
        }

        // revert the tentative permutation, keep identity
        for (int m = 0; m < N; ++m)
        {
            std::vector<uint32_t> inv(t.dims[m]);
            for (size_t a = 0; a < t.dims[m]; ++a)
                inv[rk_try[(size_t)m][a]] = (uint32_t)a;
            for (size_t e = 0; e < nnz; ++e)
                t.h_coords[(size_t)m][e] = inv[t.h_coords[(size_t)m][e]];
        }
        std::cout << "[GEN] reorder=auto -> none (probe fill " << f_raw
                  << " -> " << f_hits << " < 0.11)" << std::endl;
        g_rank.clear();
        if (ms_out) *ms_out = pms;
        return;
    }

    if (std::strcmp(how, "hits") == 0)
    {
        size_t sample_budget = 8000000;
        if (const char* e = std::getenv("GEN_REORDER_SAMPLE"))
            sample_budget = (size_t)std::strtoull(e, nullptr, 10);
        const size_t stride = (nnz > sample_budget && sample_budget > 0)
                                  ? (nnz / sample_budget) : 1;

        std::vector<std::vector<double>> s((size_t)N), ns((size_t)N);
        for (int m = 0; m < N; ++m)
            s[(size_t)m].assign(t.dims[m], 1.0);

        for (int it = 0; it < rounds; ++it)
        {
            for (int m = 0; m < N; ++m)
                ns[(size_t)m].assign(t.dims[m], 0.0);

            for (size_t e = 0; e < nnz; e += stride)
            {
                for (int m = 0; m < N; ++m)
                {
                    double p = 1.0;
                    for (int q = 0; q < N; ++q)
                        if (q != m)
                            p *= s[(size_t)q][t.h_coords[(size_t)q][e]];
                    ns[(size_t)m][t.h_coords[(size_t)m][e]] += p;
                }
            }
            for (int m = 0; m < N; ++m)
            {
                double mx = 0.0;
                for (double v : ns[(size_t)m]) mx = std::max(mx, v);
                if (mx > 0.0)
                    for (double& v : ns[(size_t)m]) v /= mx;
                s[(size_t)m].swap(ns[(size_t)m]);
            }
        }

        for (int m = 0; m < N; ++m)
        {
            std::vector<uint32_t> ord(t.dims[m]);
            for (size_t i = 0; i < t.dims[m]; ++i) ord[i] = (uint32_t)i;
            std::sort(ord.begin(), ord.end(),
                      [&](uint32_t a, uint32_t b)
                      {
                          const double sa = s[(size_t)m][a];
                          const double sb = s[(size_t)m][b];
                          return (sa != sb) ? (sa > sb) : (a < b);
                      });
            for (size_t i = 0; i < t.dims[m]; ++i)
                rank[(size_t)m][ord[i]] = (uint32_t)i;
        }
    }
    else if (std::strcmp(how, "cadr") == 0)
    {
        for (int m = 0; m < N; ++m)
        {
            std::vector<uint32_t> deg(t.dims[m], 0);
            for (size_t e = 0; e < nnz; ++e)
                ++deg[t.h_coords[(size_t)m][e]];

            std::vector<uint32_t> ord(t.dims[m]);
            for (size_t i = 0; i < t.dims[m]; ++i) ord[i] = (uint32_t)i;
            std::sort(ord.begin(), ord.end(),
                      [&](uint32_t a, uint32_t b)
                      { return deg[a] != deg[b] ? deg[a] > deg[b] : a < b; });
            for (size_t i = 0; i < t.dims[m]; ++i)
                rank[(size_t)m][ord[i]] = (uint32_t)i;
        }
    }
    else
    {
        if (ms_out) *ms_out = 0.0;
        return;                                  // "none"
    }

    for (int m = 0; m < N; ++m)
        for (size_t e = 0; e < nnz; ++e)
            t.h_coords[(size_t)m][e] = rank[(size_t)m][t.h_coords[(size_t)m][e]];

    if (rank_out)
        *rank_out = rank;      // old index -> new index

    auto t1 = std::chrono::high_resolution_clock::now();
    if (ms_out)
        *ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
}


// ============================================================
// GEN_SELFTEST=1 : standalone unit test for the N-order dense path.
//
// Builds a small 4th-order tensor with (a) fully/partially filled 16^3
// tiles, (b) boundary tiles whose dims are not multiples of 16, and
// (c) scattered residual elements.  Then compares, for every inner mode,
//   * dense-only FtV (threshold 0 -> every element belongs to a unit)
//   * dense + FP64-residual FtV (threshold 0.01 -> mixed)
// against an exact FP64 CPU reference of the MTTKRP.
// ============================================================
int run_dense_selftest()
{
    const int N = 4;
    const int dims[4] = {40, 24, 36, 3};      // 40 = 2.5 tiles, 24 = 1.5, 36 = 2.25
    const int R = 8;
    const int inner[3] = {0, 1, 2};

    std::vector<std::vector<uint32_t>> co(N);
    std::vector<double> vals;

    std::mt19937 gen(1234);
    std::uniform_real_distribution<double> uv(0.0, 1.0);

    auto push = [&](int i, int j, int k, int l, double v)
    {
        co[0].push_back((uint32_t)i); co[1].push_back((uint32_t)j);
        co[2].push_back((uint32_t)k); co[3].push_back((uint32_t)l);
        vals.push_back(v);
    };

    // dense tiles (≈50% fill): two tiles per outer slice
    const int tiles[2][3] = {{0, 0, 0}, {1, 0, 1}};
    for (int l = 0; l < dims[3]; ++l)
        for (int t = 0; t < 2; ++t)
            for (int a = 0; a < 16; ++a)
                for (int b = 0; b < 16; ++b)
                    for (int c = 0; c < 16; ++c)
                        if (uv(gen) < 0.5)
                            push(tiles[t][0] * 16 + a, tiles[t][1] * 16 + b,
                                 tiles[t][2] * 16 + c, l, 1.0 + uv(gen));

    // boundary tiles: last partial tile in each mode (exercises the guards)
    for (int l = 0; l < dims[3]; ++l)
        for (int a = 32; a < dims[0]; ++a)          // i = 32..39  (partial tile bi=2)
            for (int b = 16; b < dims[1]; ++b)      // j = 16..23  (partial tile bj=1)
                for (int c = 32; c < dims[2]; ++c)  // k = 32..35  (partial tile bk=2)
                    if (uv(gen) < 0.6)
                        push(a, b, c, l, 1.0 + uv(gen));

    // residual scatter
    for (int e = 0; e < 4000; ++e)
        push((int)(uv(gen) * dims[0]) % dims[0],
             (int)(uv(gen) * dims[1]) % dims[1],
             (int)(uv(gen) * dims[2]) % dims[2],
             (int)(uv(gen) * dims[3]) % dims[3],
             1.0 + uv(gen));

    const size_t nnz = vals.size();
    printf("[SELFTEST] N=%d dims=%d,%d,%d,%d nnz=%zu R=%d\n",
           N, dims[0], dims[1], dims[2], dims[3], nnz, R);

    // factors (host) and exact CPU reference
    std::vector<std::vector<double>> F(N);
    for (int m = 0; m < N; ++m)
    {
        F[m].resize((size_t)dims[m] * R);
        for (auto& x : F[m]) x = uv(gen);
    }

    // FtV is written base-relative for every mode (the ALS loop reuses the
    // same buffer, zeroed per mode), so keep one reference per mode.
    std::vector<std::vector<double>> ref((size_t)N);
    for (int n = 0; n < N; ++n)
    {
        ref[(size_t)n].assign((size_t)R * dims[n], 0.0);
        for (size_t e = 0; e < nnz; ++e)
            for (int r = 0; r < R; ++r)
            {
                double q = vals[e];
                for (int m = 0; m < N; ++m)
                    if (m != n) q *= F[m][(size_t)co[m][e] * R + r];
                ref[(size_t)n][(size_t)co[n][e] * R + r] += q;
            }
    }

    // ---- device setup ----
    uint32_t* d_co = nullptr;
    double*   d_va = nullptr;
    double*   d_F[8] = {nullptr};
    double*   d_FtV = nullptr;
    const size_t per_mode = (size_t)R * (dims[0] + dims[1] + dims[2]);

    GEN_CHECK(cudaMalloc(&d_co, (size_t)N * nnz * sizeof(uint32_t)));
    GEN_CHECK(cudaMalloc(&d_va, nnz * sizeof(double)));
    GEN_CHECK(cudaMalloc(&d_FtV, per_mode * sizeof(double)));

    std::vector<uint32_t> flat((size_t)N * nnz);
    for (int m = 0; m < N; ++m)
        std::copy(co[m].begin(), co[m].end(), flat.begin() + (size_t)m * nnz);
    GEN_CHECK(cudaMemcpy(d_co, flat.data(), (size_t)N * nnz * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
    GEN_CHECK(cudaMemcpy(d_va, vals.data(), nnz * sizeof(double),
                         cudaMemcpyHostToDevice));

    double** d_Fp = nullptr;
    GEN_CHECK(cudaMalloc(&d_Fp, N * sizeof(double*)));
    for (int m = 0; m < N; ++m)
    {
        GEN_CHECK(cudaMalloc(&d_F[m], (size_t)dims[m] * R * sizeof(double)));
        GEN_CHECK(cudaMemcpy(d_F[m], F[m].data(),
                             (size_t)dims[m] * R * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
    GEN_CHECK(cudaMemcpy(d_Fp, d_F, N * sizeof(double*), cudaMemcpyHostToDevice));

    int* d_dims = nullptr;
    GEN_CHECK(cudaMalloc(&d_dims, N * sizeof(int)));
    GEN_CHECK(cudaMemcpy(d_dims, dims, N * sizeof(int), cudaMemcpyHostToDevice));

    int* d_slot_dims = nullptr;
    GEN_CHECK(cudaMalloc(&d_slot_dims, 3 * sizeof(int)));

    // trivial per-element views (one element per segment) for the FP64 kernel
    struct TV { uint32_t* sv; double* va; uint32_t* row; uint32_t* st;
                uint32_t* cnt; uint8_t* single; uint8_t* skip; };
    std::vector<TV> tv((size_t)3, TV{});

    auto build_tv = [&](int n, const std::vector<uint8_t>& skip_flags)
    {
        std::vector<uint32_t> ord(nnz);
        for (size_t e = 0; e < nnz; ++e) ord[e] = (uint32_t)e;
        std::sort(ord.begin(), ord.end(),
                  [&](uint32_t a, uint32_t b) { return co[n][a] < co[n][b]; });

        std::vector<uint32_t> sv((size_t)(N - 1) * nnz);
        std::vector<double>   va(nnz);
        std::vector<uint32_t> row(nnz), st(nnz), cnt(nnz, 1);
        std::vector<uint8_t>  single(nnz, 0), sk(nnz, 0);   // accumulate
        for (size_t i = 0; i < nnz; ++i)
        {
            const uint32_t e = ord[i];
            va[i] = vals[e];
            row[i] = co[n][e];
            st[i] = (uint32_t)i;
            sk[i] = skip_flags.empty() ? 0 : skip_flags[e];
            int q = 0;
            for (int m = 0; m < N; ++m)
                if (m != n) sv[(size_t)q++ * nnz + i] = co[m][e];
        }
        TV& t = tv[(size_t)n];
        GEN_CHECK(cudaMalloc(&t.sv, sv.size() * sizeof(uint32_t)));
        GEN_CHECK(cudaMalloc(&t.va, nnz * sizeof(double)));
        GEN_CHECK(cudaMalloc(&t.row, nnz * sizeof(uint32_t)));
        GEN_CHECK(cudaMalloc(&t.st, nnz * sizeof(uint32_t)));
        GEN_CHECK(cudaMalloc(&t.cnt, nnz * sizeof(uint32_t)));
        GEN_CHECK(cudaMalloc(&t.single, nnz * sizeof(uint8_t)));
        GEN_CHECK(cudaMalloc(&t.skip, nnz * sizeof(uint8_t)));
        GEN_CHECK(cudaMemcpy(t.sv, sv.data(), sv.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(t.va, va.data(), nnz * sizeof(double),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(t.row, row.data(), nnz * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(t.st, st.data(), nnz * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(t.cnt, cnt.data(), nnz * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(t.single, single.data(), nnz * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(t.skip, sk.data(), nnz * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));
    };

    int fails = 0;
    const int inner_sets[4][3] = {{0,1,2},{1,2,3},{0,1,3},{0,2,3}};
    for (int iset = 0; iset < 4; ++iset)
    for (int pass = 0; pass < 2; ++pass)
    {
        int inner[3] = { inner_sets[iset][0], inner_sets[iset][1], inner_sets[iset][2] };
        const double thr = (pass == 0) ? 0.0 : 0.01;   // dense-only vs mixed
        printf("[SELFTEST] --- inner=(%d,%d,%d) pass %d (%s) ---\n",
               inner[0], inner[1], inner[2], pass,
               pass == 0 ? "dense-only" : "mixed dense+FP64");

        // build dense config on host from the same coordinates
        GTensor t;
        t.nmodes = N;
        t.dims.assign(dims, dims + N);
        t.h_coords = co;
        t.h_vals = vals;
        t.nnz = nnz;

        GDense gd;
        double dms = 0.0;
        std::vector<int> inner_v(inner, inner + 3);
        build_dense_nd(t, thr, inner_v.data(), gd, &dms);
        printf("[SELFTEST]   units=%d dense_nnz=%lld (%.1f%%)\n", gd.n_units,
               gd.dense_nnz, 100.0 * (double)gd.dense_nnz / (double)nnz);

        for (int ii = 0; ii < 3; ++ii)
        {
            const int n = inner[ii];
            GEN_CHECK(cudaMemset(d_FtV, 0, per_mode * sizeof(double)));

            // dense part
            const size_t dsmem = (size_t)(4096 + 256 + 256 + 256) * sizeof(float);
            GEN_CHECK(cudaFuncSetAttribute(
                g_dense_tile_mttkrp_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsmem));
            {
                const int sd[3] = { dims[inner[0]], dims[inner[1]], dims[inner[2]] };
                GEN_CHECK(cudaMemcpy(d_slot_dims, sd, 3 * sizeof(int),
                                     cudaMemcpyHostToDevice));
            }
            g_dense_tile_mttkrp_kernel<<<gd.n_units, 32, dsmem>>>(
                gd.d_units, gd.d_lid, gd.d_vals, gd.d_outer,
                gd.d_outer_modes, (int)gd.outer_modes.size(),
                d_slot_dims, N, (ii), R, d_Fp, d_FtV);
            GEN_CHECK(cudaGetLastError());

            // residual part: only in the mixed pass, skipping dense elements
            if (pass == 1)
            {
                std::vector<uint8_t> sk(nnz, 0);
                for (size_t e = 0; e < nnz; ++e) sk[e] = gd.h_is_dense[e];
                build_tv(n, sk);
                const TV& t2 = tv[(size_t)n];
                g_mttkrp_seg_pack_kernel<4><<<(unsigned)nnz, 128>>>(
                    t2.va, t2.sv, nnz, N, n, R, t2.row, t2.st, t2.cnt,
                    t2.single, t2.skip, d_Fp, d_FtV);
                GEN_CHECK(cudaGetLastError());
            }
            GEN_CHECK(cudaDeviceSynchronize());

            // compare with the reference
            std::vector<double> got((size_t)R * dims[n]);
            GEN_CHECK(cudaMemcpy(got.data(), d_FtV,
                                 got.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost));

            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < got.size(); ++i)
            {
                const double d = got[i] - ref[(size_t)n][i];
                num += d * d;
                den += ref[(size_t)n][i] * ref[(size_t)n][i];
            }
            const double rel = (den > 0.0) ? std::sqrt(num / den) : 0.0;
            printf("[SELFTEST]   mode %d: relative L2 error = %.3e  %s\n",
                   n, rel, (rel < 2e-3 ? "OK" : "FAIL"));
            if (!(rel < 2e-3)) ++fails;
        }

        free_dense_nd(gd);
    }

    printf("[SELFTEST] %s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}

int run_generic_cp(GTensor& t, int max_iters, int R,
                   unsigned int seed)
{
    const int nmodes = t.nmodes;
    const size_t nnz = t.nnz;

    // ---------------- default reordering (N >= 4: HITS, else CADR) --------
    {
        const char* how = std::getenv("GEN_REORDER");
        if (!how) how = (nmodes >= 4) ? "hits" : "cadr";

        if (std::strcmp(how, "cadr") != 0 &&
            std::strcmp(how, "hits") != 0 &&
            std::strcmp(how, "none") != 0 &&
            std::strcmp(how, "auto") != 0)
        {
            std::cerr << "[GEN] unknown GEN_REORDER=" << how
                      << " -> falling back to cadr" << std::endl;
            how = "cadr";
        }

        int rounds = 3;
        if (const char* e = std::getenv("GEN_REORDER_T"))
            rounds = std::atoi(e);
        if (rounds < 1) rounds = 1;

        double rms = 0.0;
        apply_reordering(t, how, rounds, &rms, &g_rank);
        std::cout << "[GEN] reorder=" << how;
        if (std::strcmp(how, "hits") == 0)
            std::cout << "(T=" << rounds << ")";
        std::cout << " prep=" << rms << " ms" << std::endl;
    }

    if (nmodes < 2 || nnz == 0 || R <= 0 || R > 32)
    {
        std::cerr << "[GEN] invalid run config (nmodes=" << nmodes
                  << " nnz=" << nnz << " R=" << R << ")" << std::endl;
        return -1;
    }
    if (nnz > (size_t)0x7FFFFFFF)
    {
        std::cerr << "[GEN] nnz too large for 1D grid (" << nnz << ")"
                  << std::endl;
        return -1;
    }
    if (max_iters <= 0)
        max_iters = 1;

    size_t max_dim = 0;
    for (int m = 0; m < nmodes; ++m)
        max_dim = std::max(max_dim, t.dims[m]);

    std::cout << "[GEN] N-order CP-ALS: order=" << nmodes
              << " R=" << R << " iters=" << max_iters
              << " nnz=" << nnz << std::endl;
    std::cout << "[GEN] dims = ";
    for (int m = 0; m < nmodes; ++m)
        std::cout << (m ? " x " : "") << t.dims[m];
    std::cout << std::endl;

    cublasHandle_t cublas = nullptr;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "[GEN] cublasCreate failed" << std::endl;
        return -1;
    }

    const size_t RR = (size_t)R * R;

    // device arrays
    uint32_t* d_coords = nullptr;
    double*   d_vals   = nullptr;
    GEN_CHECK(cudaMalloc(&d_coords, (size_t)nmodes * nnz * sizeof(uint32_t)));
    GEN_CHECK(cudaMalloc(&d_vals,   nnz * sizeof(double)));

    std::vector<double*> h_F(nmodes, nullptr);
    double** d_F = nullptr;
    GEN_CHECK(cudaMalloc(&d_F, nmodes * sizeof(double*)));

    std::vector<int> h_dims((size_t)nmodes);
    for (int m = 0; m < nmodes; ++m) h_dims[(size_t)m] = (int)t.dims[m];
    int* d_dims = nullptr;
    GEN_CHECK(cudaMalloc(&d_dims, (size_t)nmodes * sizeof(int)));
    GEN_CHECK(cudaMemcpy(d_dims, h_dims.data(), (size_t)nmodes * sizeof(int),
                         cudaMemcpyHostToDevice));


    double* d_lambda = nullptr;
    double* d_FtV    = nullptr;
    double* d_FtF    = nullptr;
    double* d_Gtmp   = nullptr;
    double* d_model  = nullptr;
    double* d_work   = nullptr;   // R doubles (norms / dot out)
    double* d_scalar = nullptr;   // 2 doubles

    GEN_CHECK(cudaMalloc(&d_lambda, R * sizeof(double)));
    GEN_CHECK(cudaMalloc(&d_FtV, max_dim * (size_t)R * sizeof(double)));
    GEN_CHECK(cudaMalloc(&d_FtF, RR * sizeof(double)));
    GEN_CHECK(cudaMalloc(&d_Gtmp, RR * sizeof(double)));
    GEN_CHECK(cudaMalloc(&d_model, RR * sizeof(double)));
    GEN_CHECK(cudaMalloc(&d_work, R * sizeof(double)));
    GEN_CHECK(cudaMalloc(&d_scalar, 2 * sizeof(double)));

    // upload coordinates / values
    std::vector<uint32_t> h_flat((size_t)nmodes * nnz);
    for (int m = 0; m < nmodes; ++m)
        std::copy(t.h_coords[m].begin(), t.h_coords[m].end(),
                  h_flat.begin() + (size_t)m * nnz);
    GEN_CHECK(cudaMemcpy(d_coords, h_flat.data(),
                         (size_t)nmodes * nnz * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
    GEN_CHECK(cudaMemcpy(d_vals, t.h_vals.data(), nnz * sizeof(double),
                         cudaMemcpyHostToDevice));

    // initialize factors U[0,1) and lambda = 1
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int m = 0; m < nmodes; ++m)
    {
        const size_t dim = t.dims[m];
        GEN_CHECK(cudaMalloc(&h_F[m], dim * (size_t)R * sizeof(double)));

        std::vector<double> u(dim * R);
        for (auto& x : u)
            x = dis(gen);

        // Reordering must not change the model: permute the initial factor
        // rows with the same index map so row a of the data keeps the random
        // vector it would have had without reordering.
        std::vector<double> h_f(dim * R);
        const bool has_rank =
            (m < (int)g_rank.size()) && (g_rank[(size_t)m].size() == dim);
        for (size_t a = 0; a < dim; ++a)
        {
            const size_t nw = has_rank ? (size_t)g_rank[(size_t)m][a] : a;
            for (int r = 0; r < R; ++r)
                h_f[nw * (size_t)R + r] = u[a * (size_t)R + r];
        }

        GEN_CHECK(cudaMemcpy(h_F[m], h_f.data(),
                             dim * (size_t)R * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
    GEN_CHECK(cudaMemcpy(d_F, h_F.data(), nmodes * sizeof(double*),
                         cudaMemcpyHostToDevice));

    g_fill_kernel<<<(R + 255) / 256, 256>>>(d_lambda, R, 1.0);
    GEN_CHECK(cudaGetLastError());

    // ========================================================
    // Dense-block (Tensor Core) configuration for the N-order path
    //   GEN_DENSE=0 disables; GEN_DENSE_THRESHOLD (default 0.03) is the
    //   per-(16^3 tile, outer tuple) density cutoff.
    // ========================================================
    static int gd_inner[3] = {0, 1, 2};
    if (const char* e = std::getenv("GEN_DENSE_INNER"))
    {
        int k = 0;
        const char* q = e;
        while (*q && k < 3)
        {
            gd_inner[k++] = atoi(q);
            while (*q && *q != ',') ++q;
            if (*q == ',') ++q;
        }
    }

    GDense gd;
    {
        const bool want = (std::getenv("GEN_DENSE") == nullptr) ||
                          (std::atoi(std::getenv("GEN_DENSE")) != 0);
        double thr = 0.03;
        if (const char* e = std::getenv("GEN_DENSE_THRESHOLD"))
            thr = std::atof(e);

        // The dense kernel is validated (self-test, relative L2 error ~6e-4)
        // only for the default inner set (0,1,2); other inner sets still show
        // large errors, so refuse them rather than compute silently wrong
        // results.
        const bool inner_supported =
            (gd_inner[0] == 0 && gd_inner[1] == 1 && gd_inner[2] == 2);
        if (want && !inner_supported)
        {
            std::cerr << "[GEN] dense path disabled: GEN_DENSE_INNER="
                      << gd_inner[0] << "," << gd_inner[1] << ","
                      << gd_inner[2] << " is not yet supported (only 0,1,2)"
                      << std::endl;
        }

        if (want && inner_supported && nmodes >= 4 && thr > 0.0)
        {
            double dms = 0.0;
            build_dense_nd(t, thr, gd_inner, gd, &dms);
            std::cout << "[GEN] dense: units=" << gd.n_units
                      << " dense_nnz=" << gd.dense_nnz
                      << " (" << (nnz ? 100.0 * (double)gd.dense_nnz / (double)nnz : 0.0)
                      << "% of nnz) cutoff=" << (int)(4096.0 * thr)
                      << " prep=" << dms << " ms" << std::endl;
        }
    }

    // ========================================================
    // Per-mode row-sorted views for the segment MTTKRP (built once)
    //   for each mode n: elements sorted by coordinate n, so a row
    //   occupies a contiguous run; runs are cut into <= GEN_SEG
    //   element segments -> one block per segment.
    // ========================================================
    struct ModeView {
        uint32_t* d_scoords = nullptr;   // (nmodes-1) * nnz
        double*   d_svals   = nullptr;   // nnz
        uint32_t* d_row     = nullptr;
        uint32_t* d_start   = nullptr;
        uint32_t* d_cnt     = nullptr;
        uint8_t*  d_single  = nullptr;   // 1 = only segment of its row
        uint8_t*  d_skip    = nullptr;   // 1 = handled by the dense path
        int       nseg      = 0;
    };

    std::vector<ModeView> views((size_t)nmodes);

    for (int n = 0; n < nmodes; ++n)
    {
        const size_t dim_n = t.dims[n];

        std::vector<size_t> start(dim_n + 1, 0);
        for (size_t e = 0; e < nnz; ++e)
            ++start[(size_t)t.h_coords[n][e] + 1];
        for (size_t i = 1; i <= dim_n; ++i)
            start[i] += start[i - 1];

        std::vector<uint32_t> cursor(start.begin(), start.end() - 1);
        std::vector<uint32_t> order(nnz);
        for (size_t e = 0; e < nnz; ++e)
        {
            const uint32_t c = t.h_coords[n][e];
            order[cursor[c]++] = (uint32_t)e;
        }

        std::vector<uint32_t> hsc((size_t)(nmodes - 1) * nnz);
        std::vector<double>   hsv(nnz);
        std::vector<uint8_t>  hskip(nnz, 0);
        for (size_t i = 0; i < nnz; ++i)
        {
            const uint32_t e = order[i];
            hsv[i] = t.h_vals[e];
            if (!gd.h_is_dense.empty()) hskip[i] = gd.h_is_dense[e];
            int q = 0;
            for (int m = 0; m < nmodes; ++m)
            {
                if (m == n)
                    continue;
                hsc[(size_t)q * nnz + i] = t.h_coords[m][e];
                ++q;
            }
        }

        std::vector<uint32_t> vrow, vstart, vcnt;
        std::vector<uint8_t>  vsingle;
        for (size_t r = 0; r < dim_n; ++r)
        {
            const size_t rs = start[r], re = start[r + 1];
            const size_t nseg_row =
                (re > rs) ? ((re - rs + GEN_SEG - 1) / GEN_SEG) : 0;
            for (size_t s = rs; s < re; s += GEN_SEG)
            {
                const size_t len = std::min((size_t)GEN_SEG, re - s);
                vrow.push_back((uint32_t)r);
                vstart.push_back((uint32_t)s);
                vcnt.push_back((uint32_t)len);
                vsingle.push_back((uint8_t)(nseg_row == 1 ? 1 : 0));
            }
        }

        // If the dense path is active for this mode, a row's elements can be
        // split between a dense unit (atomicAdd) and the residual segment
        // (plain store when it is the row's only segment) -- the store would
        // silently drop the dense contribution, so force atomic accumulation
        // for dense-covered inner modes.
        if (gd.enabled &&
            (n == gd_inner[0] || n == gd_inner[1] || n == gd_inner[2]))
            for (auto& s8 : vsingle) s8 = 0;

        ModeView& v = views[(size_t)n];
        v.nseg = (int)vrow.size();

        GEN_CHECK(cudaMalloc(&v.d_scoords,
                             (size_t)(nmodes - 1) * nnz * sizeof(uint32_t)));
        GEN_CHECK(cudaMalloc(&v.d_svals, nnz * sizeof(double)));
        GEN_CHECK(cudaMalloc(&v.d_row, vrow.size() * sizeof(uint32_t)));
        GEN_CHECK(cudaMalloc(&v.d_start, vstart.size() * sizeof(uint32_t)));
        GEN_CHECK(cudaMalloc(&v.d_cnt, vcnt.size() * sizeof(uint32_t)));
        GEN_CHECK(cudaMalloc(&v.d_single, vsingle.size() * sizeof(uint8_t)));
        GEN_CHECK(cudaMalloc(&v.d_skip, hskip.size() * sizeof(uint8_t)));

        GEN_CHECK(cudaMemcpy(v.d_scoords, hsc.data(),
                             (size_t)(nmodes - 1) * nnz * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(v.d_svals, hsv.data(), nnz * sizeof(double),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(v.d_row, vrow.data(),
                             vrow.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(v.d_start, vstart.data(),
                             vstart.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(v.d_cnt, vcnt.data(),
                             vcnt.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(v.d_single, vsingle.data(),
                             vsingle.size() * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));
        GEN_CHECK(cudaMemcpy(v.d_skip, hskip.data(),
                             hskip.size() * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));

        std::cout << "[GEN] mode " << n << " view: " << v.nseg
                  << " segments (" << nnz << " nnz)" << std::endl;
    }

    // per-slot dimensions for the dense kernel's boundary guards
    int h_slot_dims[3] = {0, 0, 0};
    for (int q = 0; q < 3; ++q)
        h_slot_dims[q] = (int)t.dims[(size_t)gd_inner[q]];
    int* d_slot_dims = nullptr;
    GEN_CHECK(cudaMalloc(&d_slot_dims, 3 * sizeof(int)));
    GEN_CHECK(cudaMemcpy(d_slot_dims, h_slot_dims, 3 * sizeof(int),
                         cudaMemcpyHostToDevice));

    // ========================================================
    // Cached per-mode Gram matrices (refreshed right after the
    // corresponding factor is updated -> N gram passes per iteration
    // instead of N*(N-1)).  Gram uses the custom shared-memory tiled
    // kernel: for R <= 32 it reads each factor exactly once.
    // ========================================================
    const int gram_rows_per_block =
        std::max(32, (int)((GEN_GRAM_SMEM_BYTES - R * R * (int)sizeof(double)) /
                           (R * (int)sizeof(double))) / 32 * 32);
    const size_t gram_smem =
        (size_t)gram_rows_per_block * (size_t)R * sizeof(double) +
        (size_t)R * R * sizeof(double);

    GEN_CHECK(cudaFuncSetAttribute(
        g_gram_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)gram_smem));

    std::vector<double*> d_G((size_t)nmodes, nullptr);
    for (int m = 0; m < nmodes; ++m)
    {
        GEN_CHECK(cudaMalloc(&d_G[(size_t)m], RR * sizeof(double)));
        GEN_CHECK(cudaMemset(d_G[(size_t)m], 0, RR * sizeof(double)));
        const int gblocks_m =
            (int)((t.dims[m] + (size_t)gram_rows_per_block - 1) /
                  (size_t)gram_rows_per_block);
        g_gram_kernel<<<gblocks_m, 256, gram_smem>>>(
            h_F[m], t.dims[m], R, gram_rows_per_block, d_G[(size_t)m]);
        GEN_CHECK(cudaGetLastError());
    }
    GEN_CHECK(cudaDeviceSynchronize());

    // ||X||_F^2 on host (cheap single pass)
    double X2 = 0.0;
    for (const double v : t.h_vals)
        X2 += v * v;
    const double Xnorm = std::sqrt(X2);

    // SVD solver workspace
    SVDWorkspace svd_ws;
    init_svd_workspace(svd_ws, R);

    const int grid_rr = (int)((RR + 255) / 256);
    const int grid_nnz = (int)((nnz + 255) / 256);
    const unsigned int grid_mtt = (unsigned int)nnz;   // warp per element

    // ---- optional phase profiling: GEN_PROFILE=1 ----
    static const bool g_prof = (std::getenv("GEN_PROFILE") != nullptr);
    const int n_marks = 2 + 4 * nmodes;          // start + 4/mode + end
    std::vector<cudaEvent_t> ev((size_t)(g_prof ? n_marks : 0));
    if (g_prof)
        for (auto& e : ev)
            cudaEventCreate(&e);
    double prof[5] = {0, 0, 0, 0, 0};            // gram mttkrp solve norm fit

    auto mark = [&](int i)
    {
        if (g_prof && i < n_marks)
            cudaEventRecord(ev[(size_t)i]);
    };

    auto t_iter0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iters; ++iter)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        int mk = 0;
        mark(mk++);

        // -------- per-mode ALS updates --------
        for (int n = 0; n < nmodes; ++n)
        {
            // FtF = Hadamard product of the CACHED grams of all other
            // modes (each gram refreshed once per iteration, right
            // after its own factor update).
            g_fill_kernel<<<grid_rr, 256>>>(d_FtF, RR, 1.0);
            GEN_CHECK(cudaGetLastError());

            for (int m = 0; m < nmodes; ++m)
            {
                if (m == n)
                    continue;
                g_mul_kernel<<<grid_rr, 256>>>(d_FtF, d_G[(size_t)m], d_FtF, RR);
            }
            mark(mk++);   // FtF done

            const size_t dim_n = t.dims[n];

            // zero FtV rows for this mode and run the segment MTTKRP
            g_fill_kernel<<<(int)((dim_n * (size_t)R + 255) / 256), 256>>>(
                d_FtV, dim_n * (size_t)R, 0.0);

            // ---- dense (Tensor Core) contribution for inner modes ----
            const bool mode_is_inner =
                (n == gd_inner[0] || n == gd_inner[1] || n == gd_inner[2]);
            const uint8_t* skip_ptr = nullptr;

            if (gd.enabled && mode_is_inner && R <= 32)
            {
                int tslot = (n == gd_inner[0]) ? 0 : (n == gd_inner[1]) ? 1 : 2;
                const size_t dsmem = (size_t)(4096 + 256 + 256 + 256) * sizeof(float);
                GEN_CHECK(cudaFuncSetAttribute(
                    g_dense_tile_mttkrp_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsmem));
                g_dense_tile_mttkrp_kernel<<<gd.n_units, 32, dsmem>>>(
                    gd.d_units, gd.d_lid, gd.d_vals, gd.d_outer,
                    gd.d_outer_modes, (int)gd.outer_modes.size(),
                    d_slot_dims, nmodes, tslot, R, d_F, d_FtV);
                GEN_CHECK(cudaGetLastError());
                skip_ptr = views[(size_t)n].d_skip;  // view-order flags
            }

            const ModeView& v = views[(size_t)n];
            if (v.nseg > 0)
            {
                if (R <= 8)
                {
                    g_mttkrp_seg_pack_kernel<4><<<v.nseg, 128>>>(
                        v.d_svals, v.d_scoords, nnz, nmodes, n, R,
                        v.d_row, v.d_start, v.d_cnt, v.d_single, skip_ptr,
                        d_F, d_FtV);
                }
                else if (R <= 16)
                {
                    g_mttkrp_seg_pack_kernel<2><<<v.nseg, 128>>>(
                        v.d_svals, v.d_scoords, nnz, nmodes, n, R,
                        v.d_row, v.d_start, v.d_cnt, v.d_single, skip_ptr,
                        d_F, d_FtV);
                }
                else
                {
                    g_mttkrp_seg_pack_kernel<1><<<v.nseg, 128>>>(
                        v.d_svals, v.d_scoords, nnz, nmodes, n, R,
                        v.d_row, v.d_start, v.d_cnt, v.d_single, skip_ptr,
                        d_F, d_FtV);
                }
            }
            GEN_CHECK(cudaGetLastError());
            mark(mk++);   // MTTKRP done (stream-ordered; no sync needed)

            // solve min ||X_(n) - Fn (KR others)^T||
            //   default: shared-memory Cholesky of the R x R normal
            //   equations (cheap; the gesvdj call dominated per-iteration
            //   time on real tensors)
            //   GEN_SOLVER=svd: truncated-SVD pseudoinverse instead
            static const bool use_svd =
                (std::getenv("GEN_SOLVER") != nullptr) &&
                (std::string(std::getenv("GEN_SOLVER")) == "svd");

            if (use_svd)
            {
                if (!solve_als_system_svd_nosync(svd_ws, cublas, d_FtF,
                                                 d_FtV, h_F[n], dim_n, R))
                {
                    std::cerr << "[GEN] solve failed at mode " << n
                              << std::endl;
                    return -1;
                }
            }
            else
            {
                const size_t chol_smem = (size_t)R * R * sizeof(double);
                int chol_blocks = (int)((dim_n + 255) / 256);
                if (chol_blocks > 16384) chol_blocks = 16384;
                g_chol_solve_kernel<<<chol_blocks, 256, chol_smem>>>(
                    d_FtF, d_FtV, h_F[n], dim_n, R);
                GEN_CHECK(cudaGetLastError());
            }
            mark(mk++);   // solve done

            // column-normalize the new factor and record its column
            // norms as the Kruskal weights.  (The legacy divide-by-lambda
            // + normalize + absorb sequence is algebraically identical to
            // this: the old lambda cancels out, so one pass over the
            // factor is saved per mode.)
            {
                GEN_CHECK(cudaMemset(d_work, 0, R * sizeof(double)));
                const int nth = (R <= 16) ? 256 : 128;
                int nb = (int)((dim_n + nth - 1) / nth);
                if (nb > 4096) nb = 4096;
                g_col_norms2_kernel<<<nb, nth, (size_t)nth * R * sizeof(double)>>>(
                    h_F[n], dim_n, R, d_work);
                GEN_CHECK(cudaGetLastError());
                g_norms_finalize_kernel<<<(R + 255) / 256, 256>>>(d_work, R);
            }
            g_normalize_kernel<<<
                (int)((dim_n * (size_t)R + 255) / 256), 256>>>(
                h_F[n], dim_n, R, d_work);
            g_set_lambda_kernel<<<(R + 255) / 256, 256>>>(
                d_lambda, R, d_work);
            GEN_CHECK(cudaGetLastError());

            // refresh the cached gram of this mode (factor just changed)
            {
                GEN_CHECK(cudaMemset(d_G[(size_t)n], 0, RR * sizeof(double)));
                const int gb =
                    (int)((dim_n + (size_t)gram_rows_per_block - 1) /
                          (size_t)gram_rows_per_block);
                g_gram_kernel<<<gb, 256, gram_smem>>>(
                    h_F[n], dim_n, R, gram_rows_per_block, d_G[(size_t)n]);
                GEN_CHECK(cudaGetLastError());
            }
            mark(mk++);   // normalize + gram refresh done
        }

        // -------- gram-based fit (O(R^2 N), reuses cached grams) --------
        g_fill_kernel<<<grid_rr, 256>>>(d_model, RR, 0.0);
        const double one = 1.0;
        cublasStatus_t st = cublasDger(
            cublas, R, R, &one, d_lambda, 1, d_lambda, 1, d_model, R);
        if (st != CUBLAS_STATUS_SUCCESS)
        {
            cudaError_t ce = cudaGetLastError();
            std::cerr << "[GEN] dger failed: cublas status=" << (int)st
                      << " last_cuda_error=" << (int)ce << " \""
                      << cudaGetErrorString(ce) << "\"" << std::endl;
            return -1;
        }

        for (int m = 0; m < nmodes; ++m)
            g_mul_kernel<<<grid_rr, 256>>>(d_model, d_G[(size_t)m], d_model, RR);

        // iprod = <X, Xhat> = sum_r lambda_r * (F_{last} . FtV_{last})
        GEN_CHECK(cudaMemset(d_work, 0, R * sizeof(double)));
        g_dot_kernel<<<R, 256>>>(h_F[nmodes - 1], d_FtV,
                                 t.dims[nmodes - 1], R, d_work);
        GEN_CHECK(cudaMemset(d_scalar, 0, sizeof(double)));
        g_dot_lambda_kernel<<<1, 256>>>(d_lambda, d_work, R, d_scalar);

        // model^2 sum
        GEN_CHECK(cudaMemset(d_scalar + 1, 0, sizeof(double)));
        g_sum_kernel<<<grid_rr ? grid_rr : 1, 256>>>(d_model, RR,
                                                     d_scalar + 1);

        double h[3];
        GEN_CHECK(cudaMemcpy(h, d_scalar, 2 * sizeof(double),
                             cudaMemcpyDeviceToHost));
        const double iprod = h[0];
        const double model2 = h[1];

        const double residual_sq =
            std::max(X2 - 2.0 * iprod + model2, 0.0);
        const double fit =
            (Xnorm > 0.0)
                ? 1.0 - std::sqrt(residual_sq) / Xnorm
                : 0.0;

        mark(mk++);   // fit done

        if (g_prof)
        {
            GEN_CHECK(cudaEventSynchronize(ev[(size_t)mk - 1]));

            double d[5] = {0, 0, 0, 0, 0};
            for (int i = 0; i + 1 < mk; ++i)
            {
                float msq = 0.0f;
                cudaEventElapsedTime(&msq, ev[(size_t)i], ev[(size_t)i + 1]);
                const int phase = (i < 4 * nmodes) ? (i % 4) : 4;
                d[phase] += msq;
            }
            for (int q = 0; q < 5; ++q)
                prof[q] += d[q];

            const double sum = d[0] + d[1] + d[2] + d[3] + d[4];
            std::cout << "[GEN][profile] iter " << iter + 1
                      << " gram=" << d[0]
                      << " mttkrp=" << d[1]
                      << " solve=" << d[2]
                      << " norm=" << d[3]
                      << " fit=" << d[4]
                      << " sum=" << sum << " ms" << std::endl;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[GEN] iter " << iter + 1 << "/" << max_iters
                  << " time=" << ms << " ms"
                  << " iprod=" << iprod
                  << " model2=" << model2
                  << " fit=" << fit << std::endl;
    }

    // -------- optional factor dump (for unified external evaluation) --------
    if (const char* gd_dump = std::getenv("GEN_DUMP"))
    {
        const std::string base(gd_dump);
        std::ofstream fo((base + ".factors.txt").c_str());
        fo.precision(17);
        fo << "# cpTC factor dump (factors are column-normalised; lambda holds the norms)\n";
        fo << "order " << nmodes << "\nrank " << R << "\ndims";
        for (int m = 0; m < nmodes; ++m) fo << " " << t.dims[m];
        fo << "\n";
        std::vector<double> fbuf;
        for (int m = 0; m < nmodes; ++m)
        {
            const size_t n = (size_t)t.dims[m] * (size_t)R;
            fbuf.assign(n, 0.0);
            GEN_CHECK(cudaMemcpy(fbuf.data(), h_F[m], n * sizeof(double),
                                 cudaMemcpyDeviceToHost));
            fo << "mode " << m << " " << t.dims[m] << " " << R << "\n";
            for (size_t i = 0; i < (size_t)t.dims[m]; ++i)
            {
                for (int r = 0; r < R; ++r)
                    fo << (r ? " " : "") << fbuf[i * (size_t)R + (size_t)r];
                fo << "\n";
            }
        }
        std::vector<double> lam((size_t)R, 1.0);
        GEN_CHECK(cudaMemcpy(lam.data(), d_lambda, (size_t)R * sizeof(double),
                             cudaMemcpyDeviceToHost));
        fo << "lambda " << R << "\n";
        for (int r = 0; r < R; ++r) fo << (r ? " " : "") << lam[(size_t)r];
        fo << "\n";
        fo.close();
        std::cout << "[GEN] factors dumped to " << base << ".factors.txt"
                  << std::endl;
    }

    auto t_iter1 = std::chrono::high_resolution_clock::now();
    std::cout << "[GEN] ALS total = "
              << std::chrono::duration<double, std::milli>(t_iter1 - t_iter0)
                     .count()
              << " ms" << std::endl;

    if (g_prof && max_iters > 0)
    {
        const double n = (double)max_iters;
        const double tot = prof[0] + prof[1] + prof[2] + prof[3] + prof[4];
        std::cout << "[GEN][profile] avg/iter over " << max_iters << " iters:"
                  << " gram=" << prof[0] / n
                  << " mttkrp=" << prof[1] / n
                  << " solve=" << prof[2] / n
                  << " norm=" << prof[3] / n
                  << " fit=" << prof[4] / n
                  << " (sum=" << tot / n << " ms)" << std::endl;
        if (tot > 0.0)
        {
            std::cout << "[GEN][profile] share:"
                      << " gram=" << 100.0 * prof[0] / tot << "%"
                      << " mttkrp=" << 100.0 * prof[1] / tot << "%"
                      << " solve=" << 100.0 * prof[2] / tot << "%"
                      << " norm=" << 100.0 * prof[3] / tot << "%"
                      << " fit=" << 100.0 * prof[4] / tot << "%" << std::endl;
        }
        for (auto& e : ev)
            cudaEventDestroy(e);
    }

    // -------- final exact residual --------
    GEN_CHECK(cudaMemset(d_scalar, 0, sizeof(double)));
    const int exact_blocks = (int)((nnz + 255) / 256);
    g_exact_residual_kernel<<<exact_blocks, 256>>>(
        d_vals, d_coords, nnz, nmodes, R, d_F, d_lambda, d_scalar);

    double h_res2 = 0.0;
    GEN_CHECK(cudaMemcpy(&h_res2, d_scalar, sizeof(double),
                         cudaMemcpyDeviceToHost));

    const double res = std::sqrt(std::max(h_res2, 0.0));
    const double rmse = (nnz > 0) ? res / std::sqrt((double)nnz) : 0.0;
    const double fit_exact =
        (Xnorm > 0.0) ? 1.0 - res / Xnorm : 0.0;
    std::cout << "[GEN] exact: ||X-Xhat||_F=" << res
              << " RMSE=" << rmse
              << " fit=" << fit_exact
              << " (||X||_F=" << Xnorm << ")" << std::endl;

    std::cout << "[GEN] lambda = ";
    {
        std::vector<double> h_l(R);
        GEN_CHECK(cudaMemcpy(h_l.data(), d_lambda, R * sizeof(double),
                             cudaMemcpyDeviceToHost));
        for (int r = 0; r < R; ++r)
            std::cout << (r ? " " : "") << h_l[r];
        std::cout << std::endl;
    }

    // cleanup
    free_dense_nd(gd);
    free_svd_workspace(svd_ws);
    for (int m = 0; m < nmodes; ++m)
        if (h_F[m])
            cudaFree(h_F[m]);
    cudaFree(d_F);
    cudaFree(d_coords);
    cudaFree(d_vals);
    cudaFree(d_lambda);
    cudaFree(d_FtV);
    cudaFree(d_FtF);
    cudaFree(d_Gtmp);
    cudaFree(d_model);
    cudaFree(d_work);
    cudaFree(d_scalar);
    cublasDestroy(cublas);
    return 0;
}
