#include "tensor.hpp"
#include "common.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cmath>

static inline int fast_parse_int(const char* s, const char** end)
{
    int v = 0;
    const char* p = s;

    while (*p >= '0' && *p <= '9')
    {
        v = v * 10 + (*p - '0');
        ++p;
    }

    *end = p;
    return v;
}

void load_tns_file(
    const std::string& filename,
    COOTensor& tensor)
{
    FILE* f = fopen(filename.c_str(), "rb");

    if (!f)
    {
        std::cerr
            << "[Error] Cannot open file: "
            << filename
            << std::endl;

        exit(EXIT_FAILURE);
    }

    tensor.h_m0.clear();
    tensor.h_m1.clear();
    tensor.h_m2.clear();
    tensor.h_val.clear();

    constexpr size_t BUF_SIZE = 1u << 22;

    std::vector<char> buf(BUF_SIZE);

    size_t est_lines = 0;

    size_t n = 0;

    while ((n = fread(buf.data(), 1, BUF_SIZE, f)) > 0)
    {
        for (size_t i = 0; i < n; ++i)
        {
            if (buf[i] == '\n')
                ++est_lines;
        }
    }

    rewind(f);

    tensor.h_m0.reserve(est_lines);
    tensor.h_m1.reserve(est_lines);
    tensor.h_m2.reserve(est_lines);
    tensor.h_val.reserve(est_lines);

    size_t max_m0 = 0, max_m1 = 0, max_m2 = 0;

    char* const b = buf.data();

    size_t carry = 0;

    auto process_line =
        [&](const char* s, size_t len)
    {
        if (len == 0)
            return;

        if (s[0] == '#')
            return;

        const char* const line_end =
            s + len;

        const char* p = s;

        while (p < line_end &&
               (*p == ' ' || *p == '\t'))
            ++p;

        if (p >= line_end)
            return;

        const char* end = nullptr;

        const int m0 =
            fast_parse_int(p, &end);

        p = end;

        while (p < line_end &&
               (*p == ' ' || *p == '\t'))
            ++p;

        if (p >= line_end)
            return;

        const int m1 =
            fast_parse_int(p, &end);

        p = end;

        while (p < line_end &&
               (*p == ' ' || *p == '\t'))
            ++p;

        if (p >= line_end)
            return;

        const int m2 =
            fast_parse_int(p, &end);

        p = end;

        while (p < line_end &&
               (*p == ' ' || *p == '\t'))
            ++p;

        if (p >= line_end)
            return;

        char* endc = nullptr;

        const double val =
            std::strtod(p, &endc);

        if (endc == p)
            return;

        const int m0z = m0 > 0 ? m0 - 1 : 0;
        const int m1z = m1 > 0 ? m1 - 1 : 0;
        const int m2z = m2 > 0 ? m2 - 1 : 0;

        tensor.h_m0.push_back(m0z);
        tensor.h_m1.push_back(m1z);
        tensor.h_m2.push_back(m2z);
        tensor.h_val.push_back(val);

        if (static_cast<size_t>(m0z) > max_m0)
            max_m0 = static_cast<size_t>(m0z);

        if (static_cast<size_t>(m1z) > max_m1)
            max_m1 = static_cast<size_t>(m1z);

        if (static_cast<size_t>(m2z) > max_m2)
            max_m2 = static_cast<size_t>(m2z);
    };

    while (true)
    {
        const size_t got =
            fread(b + carry, 1, BUF_SIZE - carry, f);

        const size_t avail =
            carry + got;

        if (avail == 0)
            break;

        size_t line_start = 0;
        bool saw_nl = false;

        for (size_t i = 0; i < avail; ++i)
        {
            if (b[i] == '\n')
            {
                process_line(
                    b + line_start,
                    i - line_start);

                line_start = i + 1;
                saw_nl = true;
            }
        }

        if (line_start < avail)
        {

            if (got < BUF_SIZE - carry)
            {

                process_line(
                    b + line_start,
                    avail - line_start);

                break;
            }

            carry = avail - line_start;

            memmove(b, b + line_start, carry);
        }
        else
        {
            carry = 0;
        }

        (void)saw_nl;

        if (got == 0)
            break;
    }

    fclose(f);

    tensor.nnz =
        tensor.h_m0.size();

    tensor.dims[0] =
        max_m0 + 1;

    tensor.dims[1] =
        max_m1 + 1;

    tensor.dims[2] =
        max_m2 + 1;

    std::cout
        << tensor.nnz
        << " non-zero elements."
        << std::endl;

    CHECK_CUDA(cudaMalloc(&tensor.d_m0, tensor.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&tensor.d_m1, tensor.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&tensor.d_m2, tensor.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&tensor.d_val, tensor.nnz * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(tensor.d_m0, tensor.h_m0.data(), tensor.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(tensor.d_m1, tensor.h_m1.data(), tensor.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(tensor.d_m2, tensor.h_m2.data(), tensor.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(tensor.d_val, tensor.h_val.data(), tensor.nnz * sizeof(double), cudaMemcpyHostToDevice));
}

void free_coo_tensor(COOTensor& tensor)
{
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

    tensor.h_m0.clear();
    tensor.h_m1.clear();
    tensor.h_m2.clear();
    tensor.h_val.clear();

    tensor.nnz = 0;
    tensor.dims[0] = 0;
    tensor.dims[1] = 0;
    tensor.dims[2] = 0;
}
