#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_between_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) BetweenJtjnjtrDirectKernel(
    float *a_njtr, unsigned int a_njtr_num_alloc, SharedIndex *a_njtr_indices,
    float *a_jac, unsigned int a_jac_num_alloc, float *b_njtr,
    unsigned int b_njtr_num_alloc, SharedIndex *b_njtr_indices, float *b_jac,
    unsigned int b_jac_num_alloc, float *const out_a_njtr,
    unsigned int out_a_njtr_num_alloc, float *const out_b_njtr,
    unsigned int out_b_njtr_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex a_njtr_indices_loc[1024];
  a_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? a_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex b_njtr_indices_loc[1024];
  b_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? b_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3;
  LoadShared<3, float, float>(b_njtr, 0 * b_njtr_num_alloc, b_njtr_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       b_njtr_indices_loc[threadIdx.x].target, r0, r1, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = -1.00000000000000000e+00;
    r0 = r0 * r3;
    r1 = r1 * r3;
    r2 = r2 * r3;
    WriteSum3<float, float>((float *)inout_shared, r0, r1, r2);
  };
  FlushSumShared<3, float>(out_a_njtr, 0 * out_a_njtr_num_alloc,
                           a_njtr_indices_loc, (float *)inout_shared);
  LoadShared<3, float, float>(a_njtr, 0 * a_njtr_num_alloc, a_njtr_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       a_njtr_indices_loc[threadIdx.x].target, r2, r1, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = r2 * r3;
    r1 = r1 * r3;
    r3 = r0 * r3;
    WriteSum3<float, float>((float *)inout_shared, r2, r1, r3);
  };
  FlushSumShared<3, float>(out_b_njtr, 0 * out_b_njtr_num_alloc,
                           b_njtr_indices_loc, (float *)inout_shared);
}

void BetweenJtjnjtrDirect(
    float *a_njtr, unsigned int a_njtr_num_alloc, SharedIndex *a_njtr_indices,
    float *a_jac, unsigned int a_jac_num_alloc, float *b_njtr,
    unsigned int b_njtr_num_alloc, SharedIndex *b_njtr_indices, float *b_jac,
    unsigned int b_jac_num_alloc, float *const out_a_njtr,
    unsigned int out_a_njtr_num_alloc, float *const out_b_njtr,
    unsigned int out_b_njtr_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  BetweenJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      a_njtr, a_njtr_num_alloc, a_njtr_indices, a_jac, a_jac_num_alloc, b_njtr,
      b_njtr_num_alloc, b_njtr_indices, b_jac, b_jac_num_alloc, out_a_njtr,
      out_a_njtr_num_alloc, out_b_njtr, out_b_njtr_num_alloc, problem_size);
}

} // namespace caspar