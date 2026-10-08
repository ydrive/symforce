# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_number(1e-6)

import importlib.util
import shutil
import sysconfig
import tempfile
import unittest
from pathlib import Path

import numpy as np

from symforce import typing as T
from symforce.test_util import TestCase
from symforce.test_util import symengine_only

HAS_CUDA = all(shutil.which(tool) for tool in ("nvcc", "nvidia-smi", "cmake"))

LIB_NAME = "caspar_solver_test_lib"


def build_library(output_dir: Path) -> T.Any:
    """
    Generate, compile and import a float Caspar library with two linear factors on 3D points:

    - wprior(pt, tw): w * (pt - t) elementwise, with tw = [t, w]
    - between(a, b, d): b - a - d
    """
    import symforce.symbolic as sf
    from symforce.caspar import CasparLibrary
    from symforce.caspar import memory as mem

    class Point(sf.V3): ...

    class TargetWeight(sf.V6): ...

    class Delta(sf.V3): ...

    caslib = CasparLibrary(name=LIB_NAME, dtype=mem.DType.FLOAT)

    @caslib.add_factor
    def wprior(
        pt: T.Annotated[Point, mem.TunableShared],
        tw: T.Annotated[TargetWeight, mem.ConstantSequential],
    ) -> sf.V3:
        return sf.V3([tw[3 + i] * (pt[i] - tw[i]) for i in range(3)])

    @caslib.add_factor
    def between(
        a: T.Annotated[Point, mem.TunableShared],
        b: T.Annotated[Point, mem.TunableShared],
        d: T.Annotated[Delta, mem.ConstantSequential],
    ) -> sf.V3:
        return sf.V3(b - a - d)

    caslib.generate(output_dir)
    caslib.compile(output_dir)

    library_path = output_dir / f"{LIB_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"
    spec = importlib.util.spec_from_file_location(LIB_NAME, library_path)
    assert spec is not None and spec.loader is not None
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib


class ChainProblem(T.NamedTuple):
    """
    Points with a weighted prior each and, if coupled, a between factor to the next point.
    """

    start: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    deltas: np.ndarray

    @staticmethod
    def random(num_points: int, seed: int, coupled: bool = True) -> "ChainProblem":
        rng = np.random.default_rng(seed)
        truth = np.cumsum(rng.normal(size=(num_points, 3)), axis=0)
        weights = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=(num_points, 3)))
        targets = truth + 0.01 * rng.normal(size=truth.shape)
        if coupled:
            deltas = np.diff(truth, axis=0) + 0.01 * rng.normal(size=(num_points - 1, 3))
        else:
            deltas = np.zeros((0, 3))
        start = truth + rng.normal(size=truth.shape)
        return ChainProblem(start, targets, weights, deltas)

    def optimum(self) -> np.ndarray:
        """
        The exact least-squares solution, solved per axis since the residuals do not couple axes.
        """
        num_points = len(self.start)
        num_deltas = len(self.deltas)
        solution = np.zeros((num_points, 3))
        for axis in range(3):
            jacobian = np.zeros((num_points + num_deltas, num_points))
            rhs = np.zeros(num_points + num_deltas)
            jacobian[np.arange(num_points), np.arange(num_points)] = self.weights[:, axis]
            rhs[:num_points] = self.weights[:, axis] * self.targets[:, axis]
            jacobian[num_points + np.arange(num_deltas), np.arange(num_deltas)] = -1.0
            jacobian[num_points + np.arange(num_deltas), np.arange(num_deltas) + 1] = 1.0
            rhs[num_points:] = self.deltas[:, axis]
            solution[:, axis] = np.linalg.lstsq(jacobian, rhs, rcond=None)[0]
        return solution


@symengine_only
@unittest.skipIf(not HAS_CUDA, "Requires nvcc, cmake and an NVIDIA GPU")
class SymforceCasparSolverTest(TestCase):
    """
    Runs the generated Caspar solver on small linear problems with known answers.

    One library is generated and compiled for the whole class.
    """

    lib: T.Any = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        output_dir = Path(tempfile.mkdtemp(prefix="symforce_caspar_solver_test_"))
        cls.addClassCleanup(shutil.rmtree, output_dir, ignore_errors=True)
        cls.lib = build_library(output_dir)

    def make_solver(self, problem: ChainProblem, **params: T.Any) -> T.Any:
        """
        A solver with `problem` uploaded. Params not given keep their defaults.
        """
        solver_params = self.lib.SolverParams()
        for name, value in params.items():
            setattr(solver_params, name, value)

        num_points = len(problem.start)
        num_deltas = len(problem.deltas)
        solver = self.lib.GraphSolver(
            solver_params,
            Point_num_max=num_points,
            wprior_num_max=num_points,
            between_num_max=num_deltas,
        )
        solver.set_Point_num(num_points)
        solver.set_Point_nodes_from_stacked_host(problem.start.astype(np.float32))
        solver.set_wprior_num(num_points)
        solver.set_wprior_pt_indices_from_host(np.arange(num_points, dtype=np.int32))
        solver.set_wprior_tw_data_from_stacked_host(
            np.hstack([problem.targets, problem.weights]).astype(np.float32)
        )
        solver.set_between_num(num_deltas)
        if num_deltas:
            solver.set_between_a_indices_from_host(np.arange(num_deltas, dtype=np.int32))
            solver.set_between_b_indices_from_host(np.arange(1, num_deltas + 1, dtype=np.int32))
            solver.set_between_d_data_from_stacked_host(problem.deltas.astype(np.float32))
        return solver

    @staticmethod
    def read_points(solver: T.Any, problem: ChainProblem) -> np.ndarray:
        points = np.zeros(problem.start.shape, dtype=np.float32)
        solver.get_Point_nodes_to_stacked_host(points)
        return points

    def test_tracing_does_not_change_the_solve(self) -> None:
        """
        Turning on trace_pcg changes nothing the solver reports or the state it ends in.
        """
        problem = ChainProblem.random(num_points=30, seed=1)
        runs = []
        for trace_pcg in (0, 1):
            solver = self.make_solver(problem, solver_iter_max=10, trace_pcg=trace_pcg)
            result = solver.solve(False, True)
            iterations = [
                (it.pcg_iter, it.score_current, it.score_best, it.step_quality, it.diag)
                for it in result.iterations
            ]
            runs.append((result.final_score, iterations, self.read_points(solver, problem)))

        (
            (untraced_score, untraced_iterations, untraced_points),
            (
                traced_score,
                traced_iterations,
                traced_points,
            ),
        ) = runs
        self.assertEqual(untraced_score, traced_score)
        self.assertEqual(untraced_iterations, traced_iterations)
        np.testing.assert_array_equal(untraced_points, traced_points)

    def test_iteration_budget_is_used_in_full(self) -> None:
        """
        A solve limited to N iterations ends where a longer solve is after N iterations.

        Constant damping makes every iteration a partial step, so each iteration improves the
        score and a solve that wastes its last iteration is detectably behind.
        """
        problem = ChainProblem.random(num_points=20, seed=2)
        partial_steps = dict(diag_init=1.0 / 3.0, diag_scaling_down=1.0)
        long_result = self.make_solver(problem, solver_iter_max=8, **partial_steps).solve(
            False, True
        )
        score_after = [long_result.initial_score] + [it.score_best for it in long_result.iterations]

        for budget in range(1, 7):
            with self.subTest(budget=budget):
                solver = self.make_solver(problem, solver_iter_max=budget, **partial_steps)
                result = solver.solve(False, False)
                self.assertLess(score_after[budget], score_after[budget - 1])
                np.testing.assert_allclose(result.final_score, score_after[budget], rtol=1e-5)

    def test_step_accepted_reports_acceptance(self) -> None:
        """
        IterationData.step_accepted is true exactly for the iterations whose step was kept.
        """
        problem = ChainProblem.random(num_points=30, seed=3)
        result = self.make_solver(problem, solver_iter_max=10, trace_pcg=1).solve(False, True)

        accepted = [it.lm_accepted for it in result.iterations]
        self.assertTrue(any(accepted))
        self.assertEqual([it.step_accepted for it in result.iterations], accepted)

    def test_exact_preconditioner_needs_one_pcg_iteration(self) -> None:
        """
        Independent points make the block-Jacobi preconditioner exact, so every LM iteration
        solves its linear system in one PCG iteration and never produces a NaN score.
        """
        problem = ChainProblem.random(num_points=64, seed=4, coupled=False)
        result = self.make_solver(problem, solver_iter_max=10, trace_pcg=1).solve(False, True)

        for it in result.iterations:
            with self.subTest(solver_iter=it.solver_iter):
                self.assertEqual(len(it.pcg_r_norm2), 1)
                self.assertTrue(np.isfinite(it.score_current))

    def test_pcg_beta_is_ratio_of_consecutive_residuals(self) -> None:
        """
        PCG uses the conjugate-gradient update beta_k = rho_k / rho_(k-1), where rho_k is the
        preconditioned residual norm r_k^T M^-1 r_k.
        """
        problem = ChainProblem.random(num_points=50, seed=5)
        result = self.make_solver(
            problem, solver_iter_max=1, pcg_iter_max=20, pcg_rel_error_exit=1e-12, trace_pcg=1
        ).solve(False, True)

        rho = np.array(result.iterations[0].pcg_rho)
        beta = np.array(result.iterations[0].pcg_beta)
        self.assertGreater(len(rho), 5)
        np.testing.assert_allclose(beta[1:], rho[1:] / rho[:-1], rtol=1e-5)

    def test_one_lm_step_with_strong_pcg_reaches_linear_optimum(self) -> None:
        """
        On a linear problem with negligible damping, one LM step solved to a tight PCG
        tolerance lands on the least-squares optimum.

        Weak priors on a long chain make the system ill-conditioned enough to need about 100
        PCG iterations, which only converge if the search directions stay conjugate.
        """
        problem = ChainProblem.random(num_points=200, seed=6)
        problem = problem._replace(weights=0.1 * problem.weights)
        solver = self.make_solver(
            problem,
            solver_iter_max=1,
            diag_init=1e-8,
            pcg_iter_max=200,
            pcg_rel_error_exit=1e-12,
        )
        solver.solve(False, False)

        np.testing.assert_allclose(
            self.read_points(solver, problem), problem.optimum(), rtol=0, atol=2e-3
        )


if __name__ == "__main__":
    TestCase.main()
