#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void BetweenScore(float *a, unsigned int a_num_alloc, SharedIndex *a_indices,
                  float *b, unsigned int b_num_alloc, SharedIndex *b_indices,
                  float *d, unsigned int d_num_alloc, float *const out_rTr,
                  size_t problem_size);

} // namespace caspar