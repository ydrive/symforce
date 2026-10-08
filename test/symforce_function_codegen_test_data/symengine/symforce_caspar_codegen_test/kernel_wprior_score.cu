#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_wprior_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    WpriorScoreKernel(float *pt, unsigned int pt_num_alloc,
                      SharedIndex *pt_indices, float *tw,
                      unsigned int tw_num_alloc, float *const out_rTr,
                      size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pt_indices_loc[1024];
  pt_indices_loc[threadIdx.x] = (global_thread_idx < problem_size
                                     ? pt_indices[global_thread_idx]
                                     : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7;
  LoadShared<3, float, float>(pt, 0 * pt_num_alloc, pt_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pt_indices_loc[threadIdx.x].target, r0, r1, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(tw, 0 * tw_num_alloc,
                                         global_thread_idx, r3, r4, r5, r6);
    r7 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r7, r0);
    r3 = r3 * r3;
    r6 = r6 * r6;
    r5 = fmaf(r5, r7, r2);
    r5 = r5 * r5;
    ReadIdx2<1024, float, float, float2>(tw, 4 * tw_num_alloc,
                                         global_thread_idx, r2, r0);
    r0 = r0 * r0;
    r0 = fmaf(r5, r0, r3 * r6);
    r7 = fmaf(r4, r7, r1);
    r7 = r7 * r7;
    r2 = r2 * r2;
    r0 = fmaf(r7, r2, r0);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r0);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void WpriorScore(float *pt, unsigned int pt_num_alloc, SharedIndex *pt_indices,
                 float *tw, unsigned int tw_num_alloc, float *const out_rTr,
                 size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  WpriorScoreKernel<<<n_blocks, 1024>>>(pt, pt_num_alloc, pt_indices, tw,
                                        tw_num_alloc, out_rTr, problem_size);
}

} // namespace caspar