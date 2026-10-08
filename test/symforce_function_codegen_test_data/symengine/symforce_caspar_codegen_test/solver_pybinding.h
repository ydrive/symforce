#include <pybind11/stl.h>

#include "pybind_array_tools.h"
#include "solver.h"

namespace caspar {

inline void add_solver_pybinding(pybind11::module_ module) {
  py::enum_<ExitReason>(module, "ExitReason")
      .value("MAX_ITERATIONS", ExitReason::MAX_ITERATIONS)
      .value("CONVERGED_SCORE_THRESHOLD", ExitReason::CONVERGED_SCORE_THRESHOLD)
      .value("CONVERGED_DIAG_EXIT", ExitReason::CONVERGED_DIAG_EXIT)
      .export_values();

  py::class_<IterationData>(module, "IterationData")
      .def(py::init<>())
      .def_readwrite("solver_iter", &IterationData::solver_iter)
      .def_readwrite("pcg_iter", &IterationData::pcg_iter)
      .def_readwrite("score_current", &IterationData::score_current)
      .def_readwrite("score_best", &IterationData::score_best)
      .def_readwrite("step_quality", &IterationData::step_quality)
      .def_readwrite("diag", &IterationData::diag)
      .def_readwrite("dt_inc", &IterationData::dt_inc)
      .def_readwrite("dt_tot", &IterationData::dt_tot)
      .def_readwrite("step_accepted", &IterationData::step_accepted)
      .def_readwrite("traced", &IterationData::traced)
      .def_readwrite("lm_accepted", &IterationData::lm_accepted)
      .def_readwrite("pcg_exit", &IterationData::pcg_exit)
      .def_readwrite("r0_norm2", &IterationData::r0_norm2)
      .def_readwrite("stale_rkp1_at_entry", &IterationData::stale_rkp1_at_entry)
      .def_readwrite("pred_decrease", &IterationData::pred_decrease)
      .def_readwrite("pcg_r_norm2", &IterationData::pcg_r_norm2)
      .def_readwrite("pcg_alpha", &IterationData::pcg_alpha)
      .def_readwrite("pcg_beta", &IterationData::pcg_beta)
      .def_readwrite("pcg_rho", &IterationData::pcg_rho)
      .def_readwrite("pcg_pAp", &IterationData::pcg_pAp);

  py::class_<SolveResult>(module, "SolveResult")
      .def(py::init<>())
      .def_readwrite("initial_score", &SolveResult::initial_score)
      .def_readwrite("final_score", &SolveResult::final_score)
      .def_readwrite("iteration_count", &SolveResult::iteration_count)
      .def_readwrite("runtime", &SolveResult::runtime)
      .def_readwrite("exit_reason", &SolveResult::exit_reason)
      .def_readwrite("iterations", &SolveResult::iterations);
  py::class_<GraphSolver>(module, "GraphSolver",
                          "Class for solving Factor Graphs.")
      .def(py::init<SolverParams<double>, size_t, size_t, size_t, int>(),
           py::arg("params"), py::kw_only(), py::arg("Point_num_max") = 0,
           py::arg("wprior_num_max") = 0, py::arg("between_num_max") = 0,
           py::arg("device_id") = 0)

      .def("set_params", &GraphSolver::set_params)
      .def("solve", &GraphSolver::solve,
           py::call_guard<py::gil_scoped_release>(),
           py::arg("print_progress") = false,
           py::arg("verbose_logging") = false)
      .def("finish_indices", &GraphSolver::finish_indices)
      .def("get_allocation_size", &GraphSolver::get_allocation_size)

      .def("set_Point_num", &GraphSolver::SetPointNum)
      .def(
          "set_Point_nodes_from_stacked_host",
          [](GraphSolver &solver, pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.SetPointNodesFromStackedHost(
                AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"), pybind11::arg("offset") = 0)
      .def(
          "set_Point_nodes_from_stacked_device",
          [](GraphSolver &solver, pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.SetPointNodesFromStackedDevice(
                AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"), pybind11::arg("offset") = 0)
      .def(
          "get_Point_nodes_to_stacked_host",
          [](GraphSolver &solver, pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.GetPointNodesToStackedHost(AsFloatPtr(stacked_data), offset,
                                              GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"), pybind11::arg("offset") = 0)
      .def(
          "get_Point_nodes_to_stacked_device",
          [](GraphSolver &solver, pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.GetPointNodesToStackedDevice(
                AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"), pybind11::arg("offset") = 0)

      .def("set_wprior_num", &GraphSolver::SetWpriorNum)

      .def("set_wprior_pt_indices_from_host",
           [](GraphSolver &solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.SetWpriorPtIndicesFromHost(AsUintPtr(indices),
                                               GetNumRows(indices));
           })
      .def("set_wprior_pt_indices_from_device",
           [](GraphSolver &solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.SetWpriorPtIndicesFromDevice(AsUintPtr(indices),
                                                 GetNumRows(indices));
           })
      .def(
          "set_wprior_tw_data_from_stacked_device",
          [](GraphSolver &solver, pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.SetWpriorTwDataFromStackedDevice(
                AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"), pybind11::arg("offset") = 0)
      .def(
          "set_wprior_tw_data_from_stacked_host",
          [](GraphSolver &solver, pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.SetWpriorTwDataFromStackedHost(
                AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"), pybind11::arg("offset") = 0)
      .def("set_between_num", &GraphSolver::SetBetweenNum)

      .def("set_between_a_indices_from_host",
           [](GraphSolver &solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.SetBetweenAIndicesFromHost(AsUintPtr(indices),
                                               GetNumRows(indices));
           })
      .def("set_between_a_indices_from_device",
           [](GraphSolver &solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.SetBetweenAIndicesFromDevice(AsUintPtr(indices),
                                                 GetNumRows(indices));
           })
      .def("set_between_b_indices_from_host",
           [](GraphSolver &solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.SetBetweenBIndicesFromHost(AsUintPtr(indices),
                                               GetNumRows(indices));
           })
      .def("set_between_b_indices_from_device",
           [](GraphSolver &solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.SetBetweenBIndicesFromDevice(AsUintPtr(indices),
                                                 GetNumRows(indices));
           })
      .def(
          "set_between_d_data_from_stacked_device",
          [](GraphSolver &solver, pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.SetBetweenDDataFromStackedDevice(
                AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"), pybind11::arg("offset") = 0)
      .def(
          "set_between_d_data_from_stacked_host",
          [](GraphSolver &solver, pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.SetBetweenDDataFromStackedHost(
                AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"), pybind11::arg("offset") = 0);
}

} // namespace caspar