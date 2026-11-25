import unittest
from importlib.metadata import version

import oa
from oa import _native


class MatrixBindingTest(unittest.TestCase):
	def test_package_version_matches_distribution_metadata(self) -> None:
		self.assertEqual(oa.__version__, version("oapython"))
		self.assertEqual(_native.__version__, version("oapython"))

	def test_three_line_facade_executes_on_the_native_runtime(self) -> None:
		runtime = oa.Runtime()
		one = oa.matrix.ones([2, 3], runtime=runtime)
		two = oa.matrix.full([2, 3], 2.0, runtime=runtime)
		total = oa.matrix.add(one, two)
		self.assertEqual(total.shape, [2, 3])
		self.assertEqual(total.dtype, "f32")
		self.assertEqual(total.read_f32(), [3.0] * 6)
		self.assertEqual((one + two).to_list(), [3.0] * 6)
		self.assertIs(oa.core.Matrix, oa.Matrix)
		self.assertIs(oa.runtime.Runtime, oa.Runtime)

	def test_matmul_nt_matches_the_public_layout_contract(self) -> None:
		runtime = oa.Runtime()
		left = oa.Matrix.new_from_f32([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], runtime=runtime)
		right = oa.matrix.ones([4, 3], runtime=runtime)
		output = oa.matrix.matmul(left, right)
		self.assertEqual(output.shape, [2, 4])
		self.assertEqual(output.to_list(), [6.0] * 4 + [15.0] * 4)

	def test_typed_constructors_and_reshape_preserve_value_contracts(self) -> None:
		runtime = oa.Runtime()
		f32 = oa.Matrix.new_from_f32([2, 2], [1.0, -2.0, 3.5, 0.0], runtime=runtime)
		i32 = oa.Matrix.new_from_i32([2], [-3, 7], runtime=runtime)
		u32 = oa.Matrix.new_from_u32([2], [0, 9], runtime=runtime)
		u8 = oa.Matrix.new_from_u8([3], [0, 127, 255], runtime=runtime)

		self.assertEqual((f32.dtype, f32.num_elements, f32.to_list()), ("f32", 4, [1.0, -2.0, 3.5, 0.0]))
		self.assertEqual((i32.dtype, i32.read_i32()), ("i32", [-3, 7]))
		self.assertEqual((u32.dtype, u32.read_u32()), ("u32", [0, 9]))
		self.assertEqual((u8.dtype, u8.read_u8()), ("u8", [0, 127, 255]))
		self.assertEqual(f32.reshape([4]).shape, [4])
		self.assertEqual(oa.matrix.reshape(f32, [1, 4]).to_list(), [1.0, -2.0, 3.5, 0.0])


if __name__ == "__main__":
	unittest.main()
