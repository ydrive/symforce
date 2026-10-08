#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void WpriorResJacFirst(
    float *pt, unsigned int pt_num_alloc, SharedIndex *pt_indices, float *tw,
    unsigned int tw_num_alloc, float *out_res, unsigned int out_res_num_alloc,
    float *const out_rTr, float *const out_pt_njtr,
    unsigned int out_pt_njtr_num_alloc, float *const out_pt_precond_diag,
    unsigned int out_pt_precond_diag_num_alloc,
    float *const out_pt_precond_tril,
    unsigned int out_pt_precond_tril_num_alloc, size_t problem_size);

} // namespace caspar