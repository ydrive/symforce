#include "solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "caspar_mappings.h"
#include "shared_indices.h"
#include "solver_tools.h"
#include "sort_indices.h"

#include "kernel_Point_alpha_denominator_or_beta_numerator.h"
#include "kernel_Point_alpha_numerator_denominator.h"
#include "kernel_Point_normalize.h"
#include "kernel_Point_pred_decrease_times_two.h"
#include "kernel_Point_retract.h"
#include "kernel_Point_start_w.h"
#include "kernel_Point_start_w_contribute.h"
#include "kernel_Point_update_Mp.h"
#include "kernel_Point_update_p.h"
#include "kernel_Point_update_r.h"
#include "kernel_Point_update_r_first.h"
#include "kernel_Point_update_step.h"
#include "kernel_Point_update_step_first.h"
#include "kernel_between_jtjnjtr_direct.h"
#include "kernel_between_res_jac.h"
#include "kernel_between_res_jac_first.h"
#include "kernel_between_score.h"
#include "kernel_wprior_jtjnjtr_direct.h"
#include "kernel_wprior_res_jac.h"
#include "kernel_wprior_res_jac_first.h"
#include "kernel_wprior_score.h"

namespace {

void make_aligned(size_t &offset, size_t alignment_bytes) {
  offset = ((offset + alignment_bytes - 1) / alignment_bytes) * alignment_bytes;
}

template <typename T>
void increment_offset(size_t &offset, size_t num_elements,
                      size_t alignment_elements) {
  make_aligned(offset, alignment_elements * sizeof(T));
  offset += num_elements * sizeof(T);
}

template <typename T>
T *assign_and_increment(uint8_t *origin_ptr, size_t &offset,
                        size_t num_elements, size_t alignment_elements) {
  make_aligned(offset, alignment_elements * sizeof(T));
  size_t old_offset = offset;
  offset += num_elements * sizeof(T);
  return reinterpret_cast<T *>(origin_ptr + old_offset);
}

} // namespace

namespace caspar {

GraphSolver::GraphSolver(const SolverParams<double> &params,
                         size_t Point_num_max, size_t wprior_num_max,
                         size_t between_num_max, int device_id)
    : params_(params), device_id_(device_id), Point_num_(Point_num_max),
      Point_num_max_(Point_num_max), wprior_num_(wprior_num_max),
      wprior_num_max_(wprior_num_max), between_num_(between_num_max),
      between_num_max_(between_num_max) {
  indices_valid_ = false;
  if (params.pcg_rel_error_exit <= 0.0f) {
    throw std::runtime_error("params.pcg_rel_error_exit must be positive");
  }
  if (params.diag_init < 0.0f) {
    throw std::runtime_error("params.diag_init must be positive");
  }
  allocation_size_ = get_nbytes();

  if (device_id_ < 0) {
    throw std::runtime_error("Invalid CUDA device id: " +
                             std::to_string(device_id_));
  }
  if (device_id_ != 0) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= device_id_) {
      throw std::runtime_error("CUDA detected " + std::to_string(deviceCount) +
                               " devices, but device " +
                               std::to_string(device_id_) +
                               " was requested (0-indexed)");
    }
  }
  cudaSetDevice(device_id_);
  cudaMalloc(&origin_ptr_, allocation_size_);

  size_t offset = 0;
  marker__start_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__storage_current_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__storage_check_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__storage_new_best_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  facs__wprior__args__pt__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * wprior_num_, 4);
  facs__wprior__args__tw__data_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * wprior_num_, 4);
  facs__between__args__a__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * between_num_, 4);
  facs__between__args__b__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * between_num_, 4);
  facs__between__args__d__data_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * between_num_, 4);
  marker__scratch_inout_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  facs__wprior__res_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * wprior_num_, 4);
  facs__between__res_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * between_num_, 4);
  facs__wprior__args__pt__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * wprior_num_, 4);
  facs__between__args__a__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * between_num_, 4);
  facs__between__args__b__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * between_num_, 4);
  nodes__Point__z_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__p_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__step_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__w_start_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__w_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  marker__w_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 1);
  marker__r_0_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__r_0_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  marker__r_0_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__r_k_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__r_k_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  marker__r_k_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__Mp_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__Mp_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  marker__Mp_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__precond_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__precond_diag_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__precond_tril_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  marker__precond_end_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 1);
  marker__jp_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  facs__wprior__jp_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * wprior_num_, 4);
  facs__between__jp_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * between_num_, 4);
  marker__jp_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 1);
  solver__current_diag_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_numerator_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_denominator_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_ = assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__neg_alpha_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__beta_numerator_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__beta_ = assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__r_0_norm2_tot_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__r_kp1_norm2_tot_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__pred_decrease_tot_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__res_tot_ = assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);

  scratch_inout_size_ = offset; // sorting, sum,
}

GraphSolver::~GraphSolver() {
  cudaSetDevice(device_id_);
  cudaFree(origin_ptr_);
}

void GraphSolver::set_params(const SolverParams<double> &params) {
  this->params_ = params;
}

size_t GraphSolver::get_allocation_size() { return allocation_size_; }

SolveResult GraphSolver::solve(bool print_progress, bool verbose_logging) {
  cudaSetDevice(device_id_);
  SolveResult result;
  result.exit_reason = ExitReason::MAX_ITERATIONS;
  float score_best;
  float score_best_pcg;
  float diag = params_.diag_init;
  cudaMemcpy(solver__current_diag_, &diag, sizeof(float),
             cudaMemcpyHostToDevice);

  float up_scale = params_.diag_scaling_up;
  float quality;

  std::chrono::time_point<std::chrono::steady_clock> t0 =
      std::chrono::steady_clock::now();
  std::chrono::time_point<std::chrono::steady_clock> t_prev = t0;
  score_best = DoResJacFirst();
  result.initial_score = score_best;
  if (print_progress) {
    printf("                                 score_init: % .6e\n", score_best);
  }

  for (solver_iter_ = 0; solver_iter_ < params_.solver_iter_max;
       solver_iter_++) {
    if (solver_iter_ != 0) {
      DoResJac();
    }
    score_best_pcg = score_best;
    const bool trace = verbose_logging && params_.trace_pcg > 0;
    IterationData trace_data;
    if (trace) {
      trace_data.traced = true;
      trace_data.pcg_exit = IterationData::ITER_MAX;
    }
    for (pcg_iter_ = 0; pcg_iter_ < params_.pcg_iter_max; pcg_iter_++) {
      DoNormalize();

      if (pcg_iter_ == 0) {
        Copy(marker__r_k_start_, marker__r_k_end_, marker__w_start_);
        DoJtjpDirect();
        DoAlphaFirst();
        DoUpdateStepFirst();
        if (trace) {
          trace_data.stale_rkp1_at_entry = ReadCuMem(solver__r_kp1_norm2_tot_);
        }
        DoUpdateRFirst();
        if (trace) {
          trace_data.r0_norm2 = pcg_r_0_norm2_;
        }
      } else {
        DoBeta();
        DoUpdateP();
        DoUpdateMp();
        DoJtjpDirect();
        DoAlpha();
        DoUpdateStep();
        DoUpdateR();
      }
      if (trace) {
        trace_data.pcg_r_norm2.push_back(pcg_r_kp1_norm2_);
        trace_data.pcg_alpha.push_back(ReadCuMem(solver__alpha_));
        trace_data.pcg_pAp.push_back(ReadCuMem(solver__alpha_denominator_));
        trace_data.pcg_rho.push_back(ReadCuMem(pcg_iter_ == 0
                                                   ? solver__alpha_numerator_
                                                   : solver__beta_numerator_));
        trace_data.pcg_beta.push_back(
            pcg_iter_ == 0 ? std::nan("") : ReadCuMem(solver__beta_));
      }
      if (params_.pcg_rel_decrease_min != -1.0f ||
          params_.pcg_rel_score_exit != -1.0f) {
        float score_new_pcg = DoRetractScore();
        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {
          trace_data.pcg_exit =
              trace ? IterationData::REL_DECREASE : trace_data.pcg_exit;
          break;
        }
        std::swap(nodes__Point__storage_check_,
                  nodes__Point__storage_new_best_);
        score_best_pcg = score_new_pcg;
        if (params_.pcg_rel_score_exit != -1.0f &&
            score_best_pcg < score_best * params_.pcg_rel_score_exit) {
          trace_data.pcg_exit =
              trace ? IterationData::REL_SCORE : trace_data.pcg_exit;
          break;
        }
      }
      if (pcg_r_kp1_norm2_ < pcg_r_0_norm2_ * params_.pcg_rel_error_exit) {
        trace_data.pcg_exit =
            trace ? IterationData::REL_ERROR : trace_data.pcg_exit;
        break;
      }
    }
    pcg_iter_ = std::min(pcg_iter_, params_.pcg_iter_max - 1);

    if (params_.pcg_rel_decrease_min == -1.0f &&
        params_.pcg_rel_score_exit == -1.0f) {
      score_best_pcg = DoRetractScore();
      std::swap(nodes__Point__storage_check_, nodes__Point__storage_new_best_);
    }

    const float diag_current = diag;
    bool step_accepted = false;
    if (score_best_pcg < score_best * params_.solver_rel_decrease_min) {
      step_accepted = true;
      const float pred_decrease = GetPredDecrease();
      quality = (score_best - score_best_pcg) / pred_decrease;
      if (trace) {
        trace_data.lm_accepted = true;
        trace_data.pred_decrease = pred_decrease;
      }
      const float quality_tmp = 2 * quality - 1;
      float scale = std::max(params_.diag_scaling_down,
                             1.0f - quality_tmp * quality_tmp * quality_tmp);
      diag = std::max(params_.diag_min, diag * scale);
      cudaMemcpy(solver__current_diag_, &diag, sizeof(float),
                 cudaMemcpyHostToDevice);
      up_scale = params_.diag_scaling_up;
      score_best = score_best_pcg;
      std::swap(nodes__Point__storage_current_,
                nodes__Point__storage_new_best_);

    } else {
      quality = 0.0f;
      diag = diag * up_scale;
      if (diag > params_.diag_exit_value) {
        result.exit_reason = ExitReason::CONVERGED_DIAG_EXIT;
        break;
      }
      cudaMemcpy(solver__current_diag_, &diag, sizeof(float),
                 cudaMemcpyHostToDevice);
      up_scale *= 2;
    }
    const auto t_now = std::chrono::steady_clock::now();
    const double dt_inc = std::chrono::duration<double>(t_now - t_prev).count();
    const double dt_tot = std::chrono::duration<double>(t_now - t0).count();

    if (verbose_logging) {
      if (trace && !trace_data.lm_accepted) {
        trace_data.pred_decrease = std::nan("");
      }
      IterationData iter_data = std::move(trace_data);
      iter_data.solver_iter = solver_iter_;
      iter_data.pcg_iter = pcg_iter_;
      iter_data.score_current = score_best_pcg;
      iter_data.score_best = score_best;
      iter_data.step_quality = quality;
      iter_data.diag = diag_current;
      iter_data.dt_inc = dt_inc;
      iter_data.dt_tot = dt_tot;
      iter_data.step_accepted = step_accepted;
      result.iterations.push_back(std::move(iter_data));
    }

    if (print_progress) {
      printf("solver_iter: % 3d  ", solver_iter_);
      printf("pcg_iter: % 3d  ", pcg_iter_);
      printf("score_current: % 13.6e  ", score_best_pcg);
      printf("score_best: % 13.6e  ", score_best);
      printf("step_quality: % 7.3f  ", quality);
      printf("diag: % 6.3e  ", diag_current);
      printf("dt_inc: % 10.6f  ", dt_inc);
      printf("dt_tot: % 10.6f  ", dt_tot);
      t_prev = t_now;
      printf("\n");
    }
    if (score_best <= params_.score_exit_value) {
      result.exit_reason = ExitReason::CONVERGED_SCORE_THRESHOLD;
      break;
    }
  }

  const auto t_final = std::chrono::steady_clock::now();
  result.final_score = score_best;
  result.iteration_count = solver_iter_;
  result.runtime = std::chrono::duration<double>(t_final - t0).count();
  return result;
}

float GraphSolver::DoResJacFirst() {
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  Zero(marker__r_0_start_, marker__precond_end_);

  WpriorResJacFirst(nodes__Point__storage_current_, Point_num_max_,
                    facs__wprior__args__pt__idx_shared_,
                    facs__wprior__args__tw__data_, wprior_num_max_,

                    facs__wprior__res_, wprior_num_, solver__res_tot_,
                    nodes__Point__r_k_, Point_num_, nodes__Point__precond_diag_,
                    Point_num_, nodes__Point__precond_tril_, Point_num_,
                    wprior_num_);

  BetweenResJacFirst(
      nodes__Point__storage_current_, Point_num_max_,
      facs__between__args__a__idx_shared_, nodes__Point__storage_current_,
      Point_num_max_, facs__between__args__b__idx_shared_,
      facs__between__args__d__data_, between_num_max_,

      facs__between__res_, between_num_, solver__res_tot_,
      facs__between__args__a__jac_, between_num_, nodes__Point__r_k_,
      Point_num_, nodes__Point__precond_diag_, Point_num_,
      nodes__Point__precond_tril_, Point_num_, facs__between__args__b__jac_,
      between_num_, nodes__Point__r_k_, Point_num_, nodes__Point__precond_diag_,
      Point_num_, nodes__Point__precond_tril_, Point_num_, between_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
  return 0.5 * ReadCuMem(solver__res_tot_);
}
void GraphSolver::DoResJac() {
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  Zero(marker__r_0_start_, marker__precond_end_);

  WpriorResJac(nodes__Point__storage_current_, Point_num_max_,
               facs__wprior__args__pt__idx_shared_,
               facs__wprior__args__tw__data_, wprior_num_max_,

               facs__wprior__res_, wprior_num_,

               nodes__Point__r_k_, Point_num_, nodes__Point__precond_diag_,
               Point_num_, nodes__Point__precond_tril_, Point_num_,
               wprior_num_);

  BetweenResJac(
      nodes__Point__storage_current_, Point_num_max_,
      facs__between__args__a__idx_shared_, nodes__Point__storage_current_,
      Point_num_max_, facs__between__args__b__idx_shared_,
      facs__between__args__d__data_, between_num_max_,

      facs__between__res_, between_num_,

      facs__between__args__a__jac_, between_num_, nodes__Point__r_k_,
      Point_num_, nodes__Point__precond_diag_, Point_num_,
      nodes__Point__precond_tril_, Point_num_, facs__between__args__b__jac_,
      between_num_, nodes__Point__r_k_, Point_num_, nodes__Point__precond_diag_,
      Point_num_, nodes__Point__precond_tril_, Point_num_, between_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
}

void GraphSolver::DoNormalize() {
  float *r_k;
  float *z;
  z = pcg_iter_ == 0 ? nodes__Point__p_ : nodes__Point__z_;
  PointNormalize(nodes__Point__precond_diag_, Point_num_,
                 nodes__Point__precond_tril_, Point_num_, nodes__Point__r_k_,
                 Point_num_, solver__current_diag_, z, Point_num_, Point_num_);
}

void GraphSolver::DoUpdateMp() {
  PointUpdateMp(nodes__Point__r_k_, Point_num_, nodes__Point__Mp_, Point_num_,
                solver__beta_, nodes__Point__Mp_, Point_num_, nodes__Point__w_,
                Point_num_, Point_num_);
}

void GraphSolver::DoJtjpDirect() {
  BetweenJtjnjtrDirect(
      nodes__Point__p_, Point_num_, facs__between__args__a__idx_shared_,
      facs__between__args__a__jac_, between_num_, nodes__Point__p_, Point_num_,
      facs__between__args__b__idx_shared_, facs__between__args__b__jac_,
      between_num_, nodes__Point__w_, Point_num_, nodes__Point__w_, Point_num_,
      between_num_);
}

void GraphSolver::DoAlphaFirst() {
  Zero(solver__alpha_numerator_, solver__alpha_denominator_ + 1);
  float *p_kp1;
  float *r_k;
  PointAlphaNumeratorDenominator(
      nodes__Point__p_, Point_num_, nodes__Point__r_k_, Point_num_,
      nodes__Point__w_, Point_num_, solver__alpha_numerator_,
      solver__alpha_denominator_, Point_num_);

  AlphaFromNumDenom(solver__alpha_numerator_, solver__alpha_denominator_,
                    solver__alpha_, solver__neg_alpha_);
}

void GraphSolver::DoAlpha() {
  Zero(solver__alpha_denominator_, solver__alpha_denominator_ + 1);
  PointAlphaDenominatorOrBetaNumerator(nodes__Point__p_, Point_num_,
                                       nodes__Point__w_, Point_num_,
                                       solver__alpha_denominator_, Point_num_);

  AlphaFromNumDenom(solver__beta_numerator_, solver__alpha_denominator_,
                    solver__alpha_, solver__neg_alpha_);
}

void GraphSolver::DoUpdateStepFirst() {
  PointUpdateStepFirst(nodes__Point__p_, Point_num_, solver__alpha_,
                       nodes__Point__step_, Point_num_, Point_num_);
}

void GraphSolver::DoUpdateStep() {
  PointUpdateStep(nodes__Point__step_, Point_num_, nodes__Point__p_, Point_num_,
                  solver__alpha_, nodes__Point__step_, Point_num_, Point_num_);
}

void GraphSolver::DoUpdateRFirst() {
  Zero(solver__r_0_norm2_tot_, solver__r_0_norm2_tot_ + 1);
  Zero(solver__r_kp1_norm2_tot_, solver__r_kp1_norm2_tot_ + 1);

  PointUpdateRFirst(nodes__Point__r_k_, Point_num_, nodes__Point__w_,
                    Point_num_, solver__neg_alpha_, nodes__Point__r_k_,
                    Point_num_, solver__r_0_norm2_tot_,
                    solver__r_kp1_norm2_tot_, Point_num_);

  pcg_r_0_norm2_ = ReadCuMem(solver__r_0_norm2_tot_);
  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);
}

void GraphSolver::DoUpdateR() {
  Zero(solver__r_kp1_norm2_tot_, solver__r_kp1_norm2_tot_ + 1);

  PointUpdateR(nodes__Point__r_k_, Point_num_, nodes__Point__w_, Point_num_,
               solver__neg_alpha_, nodes__Point__r_k_, Point_num_,
               solver__r_kp1_norm2_tot_, Point_num_);
  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);
}

float GraphSolver::DoRetractScore() {
  PointRetract(nodes__Point__storage_current_, Point_num_max_,
               nodes__Point__step_, Point_num_, nodes__Point__storage_check_,
               Point_num_max_, Point_num_);
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  WpriorScore(nodes__Point__storage_check_, Point_num_max_,
              facs__wprior__args__pt__idx_shared_,
              facs__wprior__args__tw__data_, wprior_num_max_, solver__res_tot_,
              wprior_num_);
  BetweenScore(nodes__Point__storage_check_, Point_num_max_,
               facs__between__args__a__idx_shared_,
               nodes__Point__storage_check_, Point_num_max_,
               facs__between__args__b__idx_shared_,
               facs__between__args__d__data_, between_num_max_,
               solver__res_tot_, between_num_);
  return 0.5 * ReadCuMem(solver__res_tot_);
}

void GraphSolver::DoBeta() {
  Zero(solver__beta_numerator_, solver__beta_numerator_ + 1);

  PointAlphaDenominatorOrBetaNumerator(nodes__Point__r_k_, Point_num_,
                                       nodes__Point__z_, Point_num_,
                                       solver__beta_numerator_, Point_num_);
  BetaFromNumDenom(solver__beta_numerator_, solver__alpha_numerator_,
                   solver__beta_);
  // Keep rho_k as the denominator of the next beta (rho_(k+1) / rho_k).
  Copy(solver__beta_numerator_, solver__beta_numerator_ + 1,
       solver__alpha_numerator_);
}

void GraphSolver::DoUpdateP() {
  PointUpdateP(nodes__Point__z_, Point_num_, nodes__Point__p_, Point_num_,
               solver__beta_, nodes__Point__p_, Point_num_, Point_num_);
}

float GraphSolver::GetPredDecrease() {
  Zero(solver__pred_decrease_tot_, solver__pred_decrease_tot_ + 1);
  PointPredDecreaseTimesTwo(nodes__Point__step_, Point_num_,
                            nodes__Point__precond_diag_, Point_num_,
                            solver__current_diag_, nodes__Point__r_0_,
                            Point_num_, solver__pred_decrease_tot_, Point_num_);
  return 0.5 * ReadCuMem(solver__pred_decrease_tot_);
}

void GraphSolver::finish_indices() { indices_valid_ = true; }

void GraphSolver::SetPointNum(const size_t num) {
  cudaSetDevice(device_id_);
  if (num > Point_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > Point_num_max_");
  }
  Point_num_ = num;
}

void GraphSolver::SetPointNodesFromStackedHost(const float *const data,
                                               const size_t offset,
                                               const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  PointStackedToCaspar(marker__scratch_inout_, nodes__Point__storage_current_,
                       Point_num_max_, offset, num);
}

void GraphSolver::SetPointNodesFromStackedDevice(const float *const data,
                                                 const size_t offset,
                                                 const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  PointStackedToCaspar(data, nodes__Point__storage_current_, Point_num_max_,
                       offset, num);
}

void GraphSolver::GetPointNodesToStackedHost(float *const data,
                                             const size_t offset,
                                             const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  PointCasparToStacked(nodes__Point__storage_current_, marker__scratch_inout_,
                       Point_num_max_, offset, num);
  cudaMemcpy(data, marker__scratch_inout_, 3 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPointNodesToStackedDevice(float *const data,
                                               const size_t offset,
                                               const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  PointCasparToStacked(nodes__Point__storage_current_, data, Point_num_max_,
                       offset, num);
}

void GraphSolver::SetWpriorNum(const size_t num) {
  if (num > wprior_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > wprior_num_max_");
  }
  wprior_num_ = num;
}
void GraphSolver::SetWpriorPtIndicesFromHost(const unsigned int *const indices,
                                             size_t num) {
  cudaSetDevice(device_id_);
  if (num != wprior_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != wprior_num_. Use SetwpriorNum before setting indices.");
  }
  cudaMemcpy((unsigned int *)marker__scratch_inout_, indices,
             num * sizeof(unsigned int), cudaMemcpyHostToDevice);
  SetWpriorPtIndicesFromDevice((unsigned int *)marker__scratch_inout_, num);
}

void GraphSolver::SetWpriorPtIndicesFromDevice(
    const unsigned int *const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != wprior_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != wprior_num_. Use SetwpriorNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__wprior__args__pt__idx_shared_, num);
}
void GraphSolver::SetWpriorTwDataFromStackedHost(const float *const data,
                                                 size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > wprior_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > wprior_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 6 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  TargetWeightStackedToCaspar(marker__scratch_inout_,
                              facs__wprior__args__tw__data_, wprior_num_max_,
                              offset, num);
}

void GraphSolver::SetWpriorTwDataFromStackedDevice(const float *const data,
                                                   size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > wprior_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > wprior_num_max_");
  }
  TargetWeightStackedToCaspar(data, facs__wprior__args__tw__data_,
                              wprior_num_max_, offset, num);
}
void GraphSolver::SetBetweenNum(const size_t num) {
  if (num > between_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > between_num_max_");
  }
  between_num_ = num;
}
void GraphSolver::SetBetweenAIndicesFromHost(const unsigned int *const indices,
                                             size_t num) {
  cudaSetDevice(device_id_);
  if (num != between_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != between_num_. Use SetbetweenNum before setting indices.");
  }
  cudaMemcpy((unsigned int *)marker__scratch_inout_, indices,
             num * sizeof(unsigned int), cudaMemcpyHostToDevice);
  SetBetweenAIndicesFromDevice((unsigned int *)marker__scratch_inout_, num);
}

void GraphSolver::SetBetweenAIndicesFromDevice(
    const unsigned int *const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != between_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != between_num_. Use SetbetweenNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__between__args__a__idx_shared_, num);
}
void GraphSolver::SetBetweenBIndicesFromHost(const unsigned int *const indices,
                                             size_t num) {
  cudaSetDevice(device_id_);
  if (num != between_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != between_num_. Use SetbetweenNum before setting indices.");
  }
  cudaMemcpy((unsigned int *)marker__scratch_inout_, indices,
             num * sizeof(unsigned int), cudaMemcpyHostToDevice);
  SetBetweenBIndicesFromDevice((unsigned int *)marker__scratch_inout_, num);
}

void GraphSolver::SetBetweenBIndicesFromDevice(
    const unsigned int *const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != between_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != between_num_. Use SetbetweenNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__between__args__b__idx_shared_, num);
}
void GraphSolver::SetBetweenDDataFromStackedHost(const float *const data,
                                                 size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > between_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > between_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  DeltaStackedToCaspar(marker__scratch_inout_, facs__between__args__d__data_,
                       between_num_max_, offset, num);
}

void GraphSolver::SetBetweenDDataFromStackedDevice(const float *const data,
                                                   size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > between_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > between_num_max_");
  }
  DeltaStackedToCaspar(data, facs__between__args__d__data_, between_num_max_,
                       offset, num);
}

size_t GraphSolver::get_nbytes() {
  size_t offset = 0;
  size_t at_least = 0;
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * wprior_num_, 4);
  increment_offset<float>(offset, 6 * wprior_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * between_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * between_num_, 4);
  increment_offset<float>(offset, 4 * between_num_, 4);
  at_least = std::max(at_least,
                      offset + std::max({3 * Point_num_max_}) * sizeof(float));
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * wprior_num_, 4);
  increment_offset<float>(offset, 4 * between_num_, 4);
  increment_offset<float>(offset, 4 * wprior_num_, 4);
  increment_offset<float>(offset, 0 * between_num_, 4);
  increment_offset<float>(offset, 0 * between_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 1);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 1);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * wprior_num_, 4);
  increment_offset<float>(offset, 4 * between_num_, 4);
  increment_offset<float>(offset, 0 * 0, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);

  return std::max(offset, at_least);
}

} // namespace caspar