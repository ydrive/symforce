#pragma once

#include "caspar_mappings.h"
#include "pybind_array_tools.h"

namespace caspar {

void add_casmappings_pybindings(pybind11::module_ module) {
  module.def("delta_stacked_to_caspar", [](pybind11::object stacked_data,
                                           pybind11::object cas_data) {
    if (GetNumCols(stacked_data) != 3) {
      throw std::runtime_error("The stacked data must have 3 columns.");
    }
    if (GetNumRows(cas_data) != 4) {
      throw std::runtime_error("The caspar data must have 4 rows.");
    }
    int num_objects = GetNumRows(stacked_data);
    int cas_stride = GetNumCols(cas_data);
    if (cas_stride < num_objects) {
      throw std::runtime_error("The caspar data must have at least as many "
                               "columns as stacked_data has rows.");
    }
    cudaSetDevice(GetDeviceId(stacked_data));
    DeltaStackedToCaspar(AsFloatPtr(stacked_data), AsFloatPtr(cas_data),
                         cas_stride, 0, num_objects);
  });
  module.def("delta_caspar_to_stacked", [](pybind11::object cas_data,
                                           pybind11::object stacked_data) {
    if (GetNumCols(stacked_data) != 3) {
      throw std::runtime_error("The stacked data must have 3 columns.");
    }
    if (GetNumRows(cas_data) != 4) {
      throw std::runtime_error("The caspar data must have 4 rows.");
    }
    int num_objects = GetNumRows(stacked_data);
    int cas_stride = GetNumCols(cas_data);
    if (cas_stride < num_objects) {
      throw std::runtime_error("The caspar data must have at least as many "
                               "columns as stacked_data has rows.");
    }
    cudaSetDevice(GetDeviceId(cas_data));
    DeltaCasparToStacked(AsFloatPtr(cas_data), AsFloatPtr(stacked_data),
                         cas_stride, 0, num_objects);
  });
  module.def("point_stacked_to_caspar", [](pybind11::object stacked_data,
                                           pybind11::object cas_data) {
    if (GetNumCols(stacked_data) != 3) {
      throw std::runtime_error("The stacked data must have 3 columns.");
    }
    if (GetNumRows(cas_data) != 4) {
      throw std::runtime_error("The caspar data must have 4 rows.");
    }
    int num_objects = GetNumRows(stacked_data);
    int cas_stride = GetNumCols(cas_data);
    if (cas_stride < num_objects) {
      throw std::runtime_error("The caspar data must have at least as many "
                               "columns as stacked_data has rows.");
    }
    cudaSetDevice(GetDeviceId(stacked_data));
    PointStackedToCaspar(AsFloatPtr(stacked_data), AsFloatPtr(cas_data),
                         cas_stride, 0, num_objects);
  });
  module.def("point_caspar_to_stacked", [](pybind11::object cas_data,
                                           pybind11::object stacked_data) {
    if (GetNumCols(stacked_data) != 3) {
      throw std::runtime_error("The stacked data must have 3 columns.");
    }
    if (GetNumRows(cas_data) != 4) {
      throw std::runtime_error("The caspar data must have 4 rows.");
    }
    int num_objects = GetNumRows(stacked_data);
    int cas_stride = GetNumCols(cas_data);
    if (cas_stride < num_objects) {
      throw std::runtime_error("The caspar data must have at least as many "
                               "columns as stacked_data has rows.");
    }
    cudaSetDevice(GetDeviceId(cas_data));
    PointCasparToStacked(AsFloatPtr(cas_data), AsFloatPtr(stacked_data),
                         cas_stride, 0, num_objects);
  });
  module.def(
      "target_weight_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 6) {
          throw std::runtime_error("The stacked data must have 6 columns.");
        }
        if (GetNumRows(cas_data) != 6) {
          throw std::runtime_error("The caspar data must have 6 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error("The caspar data must have at least as many "
                                   "columns as stacked_data has rows.");
        }
        cudaSetDevice(GetDeviceId(stacked_data));
        TargetWeightStackedToCaspar(AsFloatPtr(stacked_data),
                                    AsFloatPtr(cas_data), cas_stride, 0,
                                    num_objects);
      });
  module.def(
      "target_weight_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 6) {
          throw std::runtime_error("The stacked data must have 6 columns.");
        }
        if (GetNumRows(cas_data) != 6) {
          throw std::runtime_error("The caspar data must have 6 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error("The caspar data must have at least as many "
                                   "columns as stacked_data has rows.");
        }
        cudaSetDevice(GetDeviceId(cas_data));
        TargetWeightCasparToStacked(AsFloatPtr(cas_data),
                                    AsFloatPtr(stacked_data), cas_stride, 0,
                                    num_objects);
      });
}

} // namespace caspar