#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void BetweenResJacFirst(
    float *a, unsigned int a_num_alloc, SharedIndex *a_indices, float *b,
    unsigned int b_num_alloc, SharedIndex *b_indices, float *d,
    unsigned int d_num_alloc, float *out_res, unsigned int out_res_num_alloc,
    float *const out_rTr, float *out_a_jac, unsigned int out_a_jac_num_alloc,
    float *const out_a_njtr, unsigned int out_a_njtr_num_alloc,
    float *const out_a_precond_diag, unsigned int out_a_precond_diag_num_alloc,
    float *const out_a_precond_tril, unsigned int out_a_precond_tril_num_alloc,
    float *out_b_jac, unsigned int out_b_jac_num_alloc, float *const out_b_njtr,
    unsigned int out_b_njtr_num_alloc, float *const out_b_precond_diag,
    unsigned int out_b_precond_diag_num_alloc, float *const out_b_precond_tril,
    unsigned int out_b_precond_tril_num_alloc, size_t problem_size);

} // namespace caspar