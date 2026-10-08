#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_between_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    BetweenScoreKernel(float *a, unsigned int a_num_alloc,
                       SharedIndex *a_indices, float *b,
                       unsigned int b_num_alloc, SharedIndex *b_indices,
                       float *d, unsigned int d_num_alloc, float *const out_rTr,
                       size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex a_indices_loc[1024];
  a_indices_loc[threadIdx.x] = (global_thread_idx < problem_size
                                    ? a_indices[global_thread_idx]
                                    : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex b_indices_loc[1024];
  b_indices_loc[threadIdx.x] = (global_thread_idx < problem_size
                                    ? b_indices[global_thread_idx]
                                    : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;
  LoadShared<3, float, float>(b, 0 * b_num_alloc, b_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared, b_indices_loc[threadIdx.x].target,
                       r0, r1, r2);
  };
  __syncthreads();
  LoadShared<3, float, float>(a, 0 * a_num_alloc, a_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared, a_indices_loc[threadIdx.x].target,
                       r3, r4, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = -1.00000000000000000e+00;
    r5 = fmaf(r5, r6, r2);
    ReadIdx3<1024, float, float, float4>(d, 0 * d_num_alloc, global_thread_idx,
                                         r2, r7, r8);
    r5 = fmaf(r8, r6, r5);
    r3 = fmaf(r3, r6, r0);
    r3 = fmaf(r2, r6, r3);
    r3 = fmaf(r3, r3, r5 * r5);
    r4 = fmaf(r4, r6, r1);
    r4 = fmaf(r7, r6, r4);
    r3 = fmaf(r4, r4, r3);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r3);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void BetweenScore(float *a, unsigned int a_num_alloc, SharedIndex *a_indices,
                  float *b, unsigned int b_num_alloc, SharedIndex *b_indices,
                  float *d, unsigned int d_num_alloc, float *const out_rTr,
                  size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  BetweenScoreKernel<<<n_blocks, 1024>>>(a, a_num_alloc, a_indices, b,
                                         b_num_alloc, b_indices, d, d_num_alloc,
                                         out_rTr, problem_size);
}

} // namespace caspar