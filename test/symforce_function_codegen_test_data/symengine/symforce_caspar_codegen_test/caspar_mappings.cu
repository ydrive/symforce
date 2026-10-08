#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "caspar_mappings.h"

namespace cg = cooperative_groups;

// We use shared memory to improve the memory access.
// A smaller block size of 32 allows for larger nodetypes.
constexpr int block_size = 32;

namespace caspar {

__global__ __launch_bounds__(block_size, 1) void DeltaStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void DeltaCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t DeltaStackedToCaspar(const float *stacked_data, float *cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  DeltaStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t DeltaCasparToStacked(const float *cas_data, float *stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  DeltaCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void PointStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void PointCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t PointStackedToCaspar(const float *stacked_data, float *cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PointCasparToStacked(const float *cas_data, float *stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void TargetWeightStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 6];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 6 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 6;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 6] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 6;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void TargetWeightCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 6];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 6;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 6 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 6;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 6];
  }
}

cudaError_t TargetWeightStackedToCaspar(const float *stacked_data,
                                        float *cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  TargetWeightStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t TargetWeightCasparToStacked(const float *cas_data,
                                        float *stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  TargetWeightCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

} // namespace caspar