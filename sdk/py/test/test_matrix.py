import unittest
from importlib.metadata import version

import oa


class MatrixTest(unittest.TestCase):
	def test_package_version_matches_distribution_metadata(self) -> None:
		self.assertEqual(oa.__version__, version("oapython"))

	def test_three_line_facade_executes_on_the_native_engine(self) -> None:
		engine = oa.Engine()
		one = oa.matrix.ones(engine, [2, 3])
		two = oa.matrix.full(engine, [2, 3], 2.0)
		total = oa.matrix.add(one, two)
		self.assertEqual(total.shape, [2, 3])
		self.assertEqual(total.dtype, "f32")
		self.assertEqual(total.read_f32(), [3.0] * 6)
		self.assertEqual((one + two).to_list(), [3.0] * 6)
		self.assertIs(oa.core.Matrix, oa.Matrix)
		self.assertIs(oa.runtime.Engine, oa.Engine)

	def test_mat_mul_nt_matches_the_public_layout_contract(self) -> None:
		engine = oa.Engine()
		left = oa.Matrix.from_f32(engine, [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
		right = oa.matrix.ones(engine, [4, 3])
		output = oa.matrix.mat_mul_nt(left, right)
		self.assertEqual(output.shape, [2, 4])
		self.assertEqual(output.to_list(), [6.0] * 4 + [15.0] * 4)


if __name__ == "__main__":
	unittest.main()
