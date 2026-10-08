#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include <vector>

#include "shared_indices.h"
#include "solver_params.h"

namespace caspar {

enum class ExitReason {
  MAX_ITERATIONS,
  CONVERGED_SCORE_THRESHOLD,
  CONVERGED_DIAG_EXIT
};

struct IterationData {
  int solver_iter;
  int pcg_iter;
  double score_current;
  double score_best;
  double step_quality;
  double diag;
  double dt_inc;
  double dt_tot;
  bool step_accepted;

  // Filled only when SolverParams::trace_pcg > 0 and verbose_logging is on.
  enum PcgExit {
    NOT_TRACED = -1,
    ITER_MAX = 0,
    REL_DECREASE = 1,
    REL_SCORE = 2,
    REL_ERROR = 3
  };
  bool traced = false;
  bool lm_accepted = false;
  int pcg_exit = NOT_TRACED;
  double r0_norm2 = 0.0;
  double stale_rkp1_at_entry = 0.0;
  double pred_decrease = 0.0;
  std::vector<double> pcg_r_norm2;
  std::vector<double> pcg_alpha;
  std::vector<double> pcg_beta;
  std::vector<double> pcg_rho;
  std::vector<double> pcg_pAp;
};

struct SolveResult {
  double initial_score;
  double final_score;
  int iteration_count;
  double runtime;
  ExitReason exit_reason;
  std::vector<IterationData> iterations;
};

class GraphSolver {
public:
  /**
   * Base constructor.
   *
   * @param params: The params to use for the solver
   * @param Point_num_max the maximum number of Points
   * @param wprior_num_max the maximum number of wpriors
   * @param between_num_max the maximum number of betweens
   */
  GraphSolver(const SolverParams<double> &params, size_t Point_num_max,
              size_t wprior_num_max, size_t between_num_max, int device_id = 0);

  // This class is managing cuda memory and cannot be copied.
  GraphSolver(const GraphSolver &) = delete;
  GraphSolver &operator=(const GraphSolver &) = delete;

  GraphSolver(GraphSolver &&) = default;
  GraphSolver &operator=(GraphSolver &&) = default;

  ~GraphSolver();

  /**
   * Set the solver parameters.
   */
  void set_params(const SolverParams<double> &params);

  /**
   * Run the solver.
   */
  SolveResult solve(bool print_progress = false, bool verbose_logging = false);

  /**
   * Finish the indices.
   *
   * This function has to be called after all indices are set and before the
   * solve function is called.
   */
  void finish_indices();

  /**
   * Get the number of allocated bytes.
   */
  size_t get_allocation_size();

  /**
   * Set the current value for the Point nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPointNodesFromStackedHost(const float *const data, size_t offset,
                                    size_t num);

  /**
   * Set the current value for the Point nodes from the stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPointNodesFromStackedDevice(const float *const data, size_t offset,
                                      size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output host
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPointNodesToStackedHost(float *const data, size_t offset, size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output device
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPointNodesToStackedDevice(float *const data, size_t offset,
                                    size_t num);

  /**
   * Set the current number of active nodes of type Point.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPointNum(size_t num);

  /**
   * Set the indices for the pt argument for the Wprior factor from host.
   */
  void SetWpriorPtIndicesFromHost(const unsigned int *const indices,
                                  size_t num);

  /**
   * Set the indices for the pt argument for the Wprior factor from device.
   */
  void SetWpriorPtIndicesFromDevice(const unsigned int *const indices,
                                    size_t num);

  /**
   * Set the values for the tw consts Wprior factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetWpriorTwDataFromStackedHost(const float *const data, size_t offset,
                                      size_t num);

  /**
   * Set the values for the tw consts Wprior factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetWpriorTwDataFromStackedDevice(const float *const data, size_t offset,
                                        size_t num);

  /**
   * Set the current number of Wprior factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetWpriorNum(size_t num);

  /**
   * Set the indices for the a argument for the Between factor from host.
   */
  void SetBetweenAIndicesFromHost(const unsigned int *const indices,
                                  size_t num);

  /**
   * Set the indices for the a argument for the Between factor from device.
   */
  void SetBetweenAIndicesFromDevice(const unsigned int *const indices,
                                    size_t num);

  /**
   * Set the indices for the b argument for the Between factor from host.
   */
  void SetBetweenBIndicesFromHost(const unsigned int *const indices,
                                  size_t num);

  /**
   * Set the indices for the b argument for the Between factor from device.
   */
  void SetBetweenBIndicesFromDevice(const unsigned int *const indices,
                                    size_t num);

  /**
   * Set the values for the d consts Between factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetBetweenDDataFromStackedHost(const float *const data, size_t offset,
                                      size_t num);

  /**
   * Set the values for the d consts Between factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetBetweenDDataFromStackedDevice(const float *const data, size_t offset,
                                        size_t num);

  /**
   * Set the current number of Between factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetBetweenNum(size_t num);

private:
  SolverParams<float> params_;
  int device_id_;
  uint8_t *origin_ptr_;
  size_t scratch_inout_size_;
  size_t allocation_size_;

  int solver_iter_;
  int pcg_iter_;

  bool indices_valid_;

  float pcg_r_0_norm2_;
  float pcg_r_kp1_norm2_;

  size_t Point_num_;
  size_t Point_num_max_;
  size_t wprior_num_;
  size_t wprior_num_max_;
  size_t between_num_;
  size_t between_num_max_;

  size_t get_nbytes();
  float LinearizeFirst();
  void Linearize();
  float DoResJacFirst();
  void DoResJac();
  void DoNormalize();
  void DoJtjpDirect();
  void DoAlphaFirst();
  void DoAlpha();
  void DoUpdateStepFirst();
  void DoUpdateStep();
  void DoUpdateRFirst();
  void DoUpdateR();
  float DoRetractScore();
  void DoBeta();
  void DoUpdateP();
  void DoUpdateMp();
  float GetPredDecrease();

  float *marker__start_;
  float *nodes__Point__storage_current_;
  float *nodes__Point__storage_check_;
  float *nodes__Point__storage_new_best_;
  SharedIndex *facs__wprior__args__pt__idx_shared_;
  float *facs__wprior__args__tw__data_;
  SharedIndex *facs__between__args__a__idx_shared_;
  SharedIndex *facs__between__args__b__idx_shared_;
  float *facs__between__args__d__data_;
  float *marker__scratch_inout_;
  float *facs__wprior__res_;
  float *facs__between__res_;
  float *facs__wprior__args__pt__jac_;
  float *facs__between__args__a__jac_;
  float *facs__between__args__b__jac_;
  float *nodes__Point__z_;
  float *nodes__Point__z_end__;
  float *nodes__Point__p_;
  float *nodes__Point__p_end__;
  float *nodes__Point__step_;
  float *nodes__Point__step_end__;
  float *marker__w_start_;
  float *nodes__Point__w_;
  float *marker__w_end_;
  float *marker__r_0_start_;
  float *nodes__Point__r_0_;
  float *marker__r_0_end_;
  float *marker__r_k_start_;
  float *nodes__Point__r_k_;
  float *marker__r_k_end_;
  float *marker__Mp_start_;
  float *nodes__Point__Mp_;
  float *marker__Mp_end_;
  float *marker__precond_start_;
  float *nodes__Point__precond_diag_;
  float *nodes__Point__precond_tril_;
  float *marker__precond_end_;
  float *marker__jp_start_;
  float *facs__wprior__jp_;
  float *facs__between__jp_;
  float *marker__jp_end_;
  float *solver__current_diag_;
  float *solver__alpha_numerator_;
  float *solver__alpha_denominator_;
  float *solver__alpha_;
  float *solver__neg_alpha_;
  float *solver__beta_numerator_;
  float *solver__beta_;
  float *solver__r_0_norm2_tot_;
  float *solver__r_kp1_norm2_tot_;
  float *solver__pred_decrease_tot_;
  float *solver__res_tot_;
};

} // namespace caspar