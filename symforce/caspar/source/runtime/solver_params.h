/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

namespace caspar {

template <typename T>
struct SolverParams {
  T diag_init = 1.0;
  T diag_scaling_up = 2.0;
  T diag_scaling_down = 0.333333f;
  T diag_exit_value = 1e3;
  T diag_min = 1e-12;
  T score_exit_value = 0.0f;
  int solver_iter_max = 100;
  int pcg_iter_max = 20;
  T pcg_rel_error_exit = 1e-4;
  T pcg_rel_score_exit = -1.0f;    // disabled if == -1.0f
  T pcg_rel_decrease_min = -1.0f;  // disabled if == -1.0f
  T solver_rel_decrease_min = 1.0f;
  // Stops once a candidate step changes the score by at most this fraction of the
  // current score, like Ceres' function_tolerance. Disabled if <= 0.
  // The score is accumulated in the solver's storage type with GPU atomics, so its
  // relative error grows with the number of residuals and varies between runs. A
  // tolerance at or below that noise is not meaningful: in a float solver, values near
  // or below float epsilon (~1.2e-7) fire only when the score stops changing in float,
  // and somewhat larger values can be swamped by summation noise on large problems.
  T solver_function_tolerance = 0.0f;
  // Records per-PCG-iteration data in IterationData when > 0 and verbose_logging is on.
  int trace_pcg = 0;

  // Templated conversion constructor to enable automatic type conversion
  SolverParams() = default;

  template <typename U>
  SolverParams(const SolverParams<U>& other)
      : diag_init(static_cast<T>(other.diag_init)),
        diag_scaling_up(static_cast<T>(other.diag_scaling_up)),
        diag_scaling_down(static_cast<T>(other.diag_scaling_down)),
        diag_exit_value(static_cast<T>(other.diag_exit_value)),
        diag_min(static_cast<T>(other.diag_min)),
        score_exit_value(static_cast<T>(other.score_exit_value)),
        solver_iter_max(other.solver_iter_max),
        pcg_iter_max(other.pcg_iter_max),
        pcg_rel_error_exit(static_cast<T>(other.pcg_rel_error_exit)),
        pcg_rel_score_exit(static_cast<T>(other.pcg_rel_score_exit)),
        pcg_rel_decrease_min(static_cast<T>(other.pcg_rel_decrease_min)),
        solver_rel_decrease_min(static_cast<T>(other.solver_rel_decrease_min)),
        solver_function_tolerance(static_cast<T>(other.solver_function_tolerance)),
        trace_pcg(other.trace_pcg) {}
};
}  // namespace caspar
