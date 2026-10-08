#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t DeltaStackedToCaspar(const float *stacked_data, float *cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t DeltaCasparToStacked(const float *cas_data, float *stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t PointStackedToCaspar(const float *stacked_data, float *cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t PointCasparToStacked(const float *cas_data, float *stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t TargetWeightStackedToCaspar(const float *stacked_data,
                                        float *cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t TargetWeightCasparToStacked(const float *cas_data,
                                        float *stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

} // namespace caspar