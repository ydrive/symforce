#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void WpriorJtjnjtrDirect(float *pt_njtr, unsigned int pt_njtr_num_alloc,
                         SharedIndex *pt_njtr_indices, float *pt_jac,
                         unsigned int pt_jac_num_alloc,
                         float *const out_pt_njtr,
                         unsigned int out_pt_njtr_num_alloc,
                         size_t problem_size);

} // namespace caspar