#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_wprior_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    WpriorJtjnjtrDirectKernel(float *pt_njtr, unsigned int pt_njtr_num_alloc,
                              SharedIndex *pt_njtr_indices, float *pt_jac,
                              unsigned int pt_jac_num_alloc,
                              float *const out_pt_njtr,
                              unsigned int out_pt_njtr_num_alloc,
                              size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pt_njtr_indices_loc[1024];
  pt_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pt_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
}

void WpriorJtjnjtrDirect(float *pt_njtr, unsigned int pt_njtr_num_alloc,
                         SharedIndex *pt_njtr_indices, float *pt_jac,
                         unsigned int pt_jac_num_alloc,
                         float *const out_pt_njtr,
                         unsigned int out_pt_njtr_num_alloc,
                         size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  WpriorJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pt_njtr, pt_njtr_num_alloc, pt_njtr_indices, pt_jac, pt_jac_num_alloc,
      out_pt_njtr, out_pt_njtr_num_alloc, problem_size);
}

} // namespace caspar