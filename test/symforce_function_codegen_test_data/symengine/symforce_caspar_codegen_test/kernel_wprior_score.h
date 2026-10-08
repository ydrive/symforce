#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void WpriorScore(float *pt, unsigned int pt_num_alloc, SharedIndex *pt_indices,
                 float *tw, unsigned int tw_num_alloc, float *const out_rTr,
                 size_t problem_size);

} // namespace caspar