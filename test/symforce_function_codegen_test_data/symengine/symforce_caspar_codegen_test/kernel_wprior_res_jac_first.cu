#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_wprior_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) WpriorResJacFirstKernel(
    float *pt, unsigned int pt_num_alloc, SharedIndex *pt_indices, float *tw,
    unsigned int tw_num_alloc, float *out_res, unsigned int out_res_num_alloc,
    float *const out_rTr, float *const out_pt_njtr,
    unsigned int out_pt_njtr_num_alloc, float *const out_pt_precond_diag,
    unsigned int out_pt_precond_diag_num_alloc,
    float *const out_pt_precond_tril,
    unsigned int out_pt_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pt_indices_loc[1024];
  pt_indices_loc[threadIdx.x] = (global_thread_idx < problem_size
                                     ? pt_indices[global_thread_idx]
                                     : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(tw, 0 * tw_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
  };
  LoadShared<3, float, float>(pt, 0 * pt_num_alloc, pt_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pt_indices_loc[threadIdx.x].target, r4, r5, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = -1.00000000000000000e+00;
    r0 = fmaf(r0, r7, r4);
    r4 = r3 * r0;
    ReadIdx2<1024, float, float, float2>(tw, 4 * tw_num_alloc,
                                         global_thread_idx, r8, r9);
    r1 = fmaf(r1, r7, r5);
    r5 = r8 * r1;
    r2 = fmaf(r2, r7, r6);
    r6 = r9 * r2;
    WriteIdx3<1024, float, float, float4>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r4, r5, r6);
    r4 = r3 * r4;
    r6 = r9 * r6;
    r2 = fmaf(r2, r6, r0 * r4);
    r5 = r8 * r5;
    r2 = fmaf(r1, r5, r2);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r2);
  if (global_thread_idx < problem_size) {
    r4 = r7 * r4;
    r5 = r7 * r5;
    r6 = r7 * r6;
    WriteSum3<float, float>((float *)inout_shared, r4, r5, r6);
  };
  FlushSumShared<3, float>(out_pt_njtr, 0 * out_pt_njtr_num_alloc,
                           pt_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r3 * r3;
    r9 = r9 * r9;
    r8 = r8 * r8;
    WriteSum3<float, float>((float *)inout_shared, r3, r8, r9);
  };
  FlushSumShared<3, float>(out_pt_precond_diag,
                           0 * out_pt_precond_diag_num_alloc, pt_indices_loc,
                           (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void WpriorResJacFirst(
    float *pt, unsigned int pt_num_alloc, SharedIndex *pt_indices, float *tw,
    unsigned int tw_num_alloc, float *out_res, unsigned int out_res_num_alloc,
    float *const out_rTr, float *const out_pt_njtr,
    unsigned int out_pt_njtr_num_alloc, float *const out_pt_precond_diag,
    unsigned int out_pt_precond_diag_num_alloc,
    float *const out_pt_precond_tril,
    unsigned int out_pt_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  WpriorResJacFirstKernel<<<n_blocks, 1024>>>(
      pt, pt_num_alloc, pt_indices, tw, tw_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pt_njtr, out_pt_njtr_num_alloc,
      out_pt_precond_diag, out_pt_precond_diag_num_alloc, out_pt_precond_tril,
      out_pt_precond_tril_num_alloc, problem_size);
}

} // namespace caspar