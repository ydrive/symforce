# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_number(1e-6)

from symforce import path_util
from symforce import typing as T
from symforce.test_util import TestCase
from symforce.test_util import symengine_only

TEST_DATA_DIR = (
    path_util.symforce_data_root(__file__)
    / "test"
    / "symforce_function_codegen_test_data"
    / symforce.get_symbolic_api()
    / "symforce_caspar_codegen_test"
)


class SymforceCasparCodegenTest(TestCase):
    """
    Tests the code Caspar generates for a small solver library. Does not compile or run it.
    """

    @symengine_only
    def test_codegen(self) -> None:
        import symforce.symbolic as sf
        from symforce.caspar import CasparLibrary
        from symforce.caspar import memory as mem

        class Point(sf.V3): ...

        class TargetWeight(sf.V6): ...

        class Delta(sf.V3): ...

        caslib = CasparLibrary(name="caspar_codegen_test_lib", dtype=mem.DType.FLOAT)

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

        output_dir = self.make_output_dir("symforce_caspar_codegen_test_")
        caslib.generate(output_dir)

        # Runtime files are copied verbatim from source/runtime; only compare rendered templates
        runtime_dir = path_util.symforce_data_root(__file__) / "symforce/caspar/source/runtime"
        for runtime_file in runtime_dir.iterdir():
            (output_dir / runtime_file.name).unlink(missing_ok=True)

        self.compare_or_update_directory(output_dir, TEST_DATA_DIR)


if __name__ == "__main__":
    TestCase.main()
