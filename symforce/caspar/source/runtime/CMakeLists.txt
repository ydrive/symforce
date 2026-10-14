# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# The runtime of the generated caspar libraries: index sorting and sharing, and the solver
# helpers, which instantiate both scalar types. It is copied next to each generated library,
# whose CMakeLists.txt adds this directory unless a caspar_runtime target already exists.
find_package(CUDAToolkit REQUIRED)

file(GLOB RUNTIME_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu" "*.cc")
list(FILTER RUNTIME_SOURCES EXCLUDE REGEX "pybind.*")
file(GLOB RUNTIME_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cuh" "*.h")
list(FILTER RUNTIME_HEADERS EXCLUDE REGEX "pybind.*")

add_library(caspar_runtime STATIC ${RUNTIME_SOURCES} ${RUNTIME_HEADERS})
target_include_directories(caspar_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(caspar_runtime PUBLIC CUDA::cudart)
target_compile_options(caspar_runtime PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
set_target_properties(caspar_runtime PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
