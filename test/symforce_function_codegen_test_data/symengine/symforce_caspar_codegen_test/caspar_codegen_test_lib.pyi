from __future__ import annotations

import typing as T

"""
Any object with a valid __array_interface__

https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface
"""
Array = T.Any

"""
Any object with a valid __cuda_array_interface__

https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface
"""
CudaArray = T.Any

class ExitReason:
    MAX_ITERATIONS: int
    CONVERGED_SCORE_THRESHOLD: int
    CONVERGED_DIAG_EXIT: int

class IterationData:
    solver_iter: int
    pcg_iter: int
    score_current: float
    score_best: float
    step_quality: float
    diag: float
    dt_inc: float
    dt_tot: float
    step_accepted: bool
    traced: bool
    lm_accepted: bool
    pcg_exit: int
    r0_norm2: float
    stale_rkp1_at_entry: float
    pred_decrease: float
    pcg_r_norm2: T.List[float]
    pcg_alpha: T.List[float]
    pcg_beta: T.List[float]
    pcg_rho: T.List[float]
    pcg_pAp: T.List[float]

class SolveResult:
    initial_score: float
    final_score: float
    iteration_count: int
    runtime: float
    exit_reason: ExitReason
    iterations: T.List[IterationData]

class SolverParams:
    solver_iter_max: int
    pcg_iter_max: int

    diag_init: float
    diag_scaling_up: float
    diag_scaling_down: float
    diag_exit_value: float
    diag_min: float

    solver_rel_decrease_min: float
    score_exit_value: float

    pcg_rel_decrease_min: float
    pcg_rel_error_exit: float
    pcg_rel_score_exit: float

class GraphSolver:
    def __init__(self, params: SolverParams,
                 *,
                 Point_num_max: int = 0,
                 wprior_num_max: int = 0,
                 between_num_max: int = 0,
                 device_id: int = 0,
    ): ...

    def set_params(self, params: SolverParams) -> None:
        """
        Set the solver parameters.
        """

    def solve(self, print_progress: bool = False, verbose_logging: bool = False) -> SolveResult:
        """
        Run the solver.
        """

    def finish_indices(self) -> None:
        """
        Finish the indices.

        This function has to be called after all indices are set and before the solve function is called.
        """

    def get_allocation_size(self) -> int:
        """
        Get the number of allocated bytes.
        """

    def set_Point_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the Point nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_Point_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the Point nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_Point_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the Point nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_Point_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the Point nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_Point_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type Point.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """

    def set_wprior_pt_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pt argument for the wprior factor from host.
        """

    def set_wprior_pt_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pt argument for the wprior factor from device.
        """

    def set_wprior_tw_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the tw consts wprior factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_wprior_tw_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the tw consts wprior factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_wprior_num(self, num: int) -> None:
        """
        Set the current number of wprior factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_between_a_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the a argument for the between factor from host.
        """

    def set_between_a_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the a argument for the between factor from device.
        """
    def set_between_b_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the b argument for the between factor from host.
        """

    def set_between_b_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the b argument for the between factor from device.
        """

    def set_between_d_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the d consts between factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_between_d_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the d consts between factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_between_num(self, num: int) -> None:
        """
        Set the current number of between factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """

def delta_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked Delta data to the caspar data format.
    """

def delta_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar Delta data to the stacked data format.
    """

def point_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked Point data to the caspar data format.
    """

def point_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar Point data to the stacked data format.
    """

def target_weight_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked TargetWeight data to the caspar data format.
    """

def target_weight_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar TargetWeight data to the stacked data format.
    """


def shared_indices(indices: CudaArray, out_shared: CudaArray) -> None:
    """
    Calculate shared indices from the indices.
    """

def wprior_res_jac_first(
    pt: CudaArray,
    pt_indices: CudaArray,
    tw: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pt_njtr: CudaArray,
    out_pt_precond_diag: CudaArray,
    out_pt_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def wprior_res_jac(
    pt: CudaArray,
    pt_indices: CudaArray,
    tw: CudaArray,
    out_res: CudaArray,
    out_pt_njtr: CudaArray,
    out_pt_precond_diag: CudaArray,
    out_pt_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def wprior_score(
    pt: CudaArray,
    pt_indices: CudaArray,
    tw: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def wprior_jtjnjtr_direct(
    pt_njtr: CudaArray,
    pt_njtr_indices: CudaArray,
    pt_jac: CudaArray,
    out_pt_njtr: CudaArray,
    problem_size: int
) -> None: ...

def between_res_jac_first(
    a: CudaArray,
    a_indices: CudaArray,
    b: CudaArray,
    b_indices: CudaArray,
    d: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_a_jac: CudaArray,
    out_a_njtr: CudaArray,
    out_a_precond_diag: CudaArray,
    out_a_precond_tril: CudaArray,
    out_b_jac: CudaArray,
    out_b_njtr: CudaArray,
    out_b_precond_diag: CudaArray,
    out_b_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def between_res_jac(
    a: CudaArray,
    a_indices: CudaArray,
    b: CudaArray,
    b_indices: CudaArray,
    d: CudaArray,
    out_res: CudaArray,
    out_a_jac: CudaArray,
    out_a_njtr: CudaArray,
    out_a_precond_diag: CudaArray,
    out_a_precond_tril: CudaArray,
    out_b_jac: CudaArray,
    out_b_njtr: CudaArray,
    out_b_precond_diag: CudaArray,
    out_b_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def between_score(
    a: CudaArray,
    a_indices: CudaArray,
    b: CudaArray,
    b_indices: CudaArray,
    d: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def between_jtjnjtr_direct(
    a_njtr: CudaArray,
    a_njtr_indices: CudaArray,
    a_jac: CudaArray,
    b_njtr: CudaArray,
    b_njtr_indices: CudaArray,
    b_jac: CudaArray,
    out_a_njtr: CudaArray,
    out_b_njtr: CudaArray,
    problem_size: int
) -> None: ...

def Point_retract(
    Point: CudaArray,
    delta: CudaArray,
    out_Point_retracted: CudaArray,
    problem_size: int
) -> None: ...

def Point_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def Point_start_w(
    Point_precond_diag: CudaArray,
    diag: CudaArray,
    Point_p: CudaArray,
    out_Point_w: CudaArray,
    problem_size: int
) -> None: ...

def Point_start_w_contribute(
    Point_precond_diag: CudaArray,
    diag: CudaArray,
    Point_p: CudaArray,
    out_Point_w: CudaArray,
    problem_size: int
) -> None: ...

def Point_alpha_numerator_denominator(
    Point_p_kp1: CudaArray,
    Point_r_k: CudaArray,
    Point_w: CudaArray,
    Point_total_ag: CudaArray,
    Point_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def Point_alpha_denominator_or_beta_numerator(
    Point_p_kp1: CudaArray,
    Point_w: CudaArray,
    Point_out: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_r_first(
    Point_r_k: CudaArray,
    Point_w: CudaArray,
    negalpha: CudaArray,
    out_Point_r_kp1: CudaArray,
    out_Point_r_0_norm2_tot: CudaArray,
    out_Point_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_r(
    Point_r_k: CudaArray,
    Point_w: CudaArray,
    negalpha: CudaArray,
    out_Point_r_kp1: CudaArray,
    out_Point_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_step_first(
    Point_p_kp1: CudaArray,
    alpha: CudaArray,
    out_Point_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_step(
    Point_step_k: CudaArray,
    Point_p_kp1: CudaArray,
    alpha: CudaArray,
    out_Point_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_p(
    Point_z: CudaArray,
    Point_p_k: CudaArray,
    beta: CudaArray,
    out_Point_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_Mp(
    Point_r_k: CudaArray,
    Point_Mp: CudaArray,
    beta: CudaArray,
    out_Point_Mp_kp1: CudaArray,
    out_Point_w: CudaArray,
    problem_size: int
) -> None: ...

def Point_pred_decrease_times_two(
    Point_step: CudaArray,
    Point_precond_diag: CudaArray,
    diag: CudaArray,
    Point_njtr: CudaArray,
    out_Point_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

