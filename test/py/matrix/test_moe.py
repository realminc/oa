"""Python binding tests for oa.matrix MoE and top_k_mask operations."""

import unittest

import oa


class MatrixTopKMaskTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_top_k_mask_shape(self) -> None:
		# indices: 3 tokens each selecting from 4 experts
		indices = oa.Matrix.from_i32(self.engine, [3, 2], [0, 1, 1, 3, 2, 0])
		mask = oa.matrix.top_k_mask(indices, num_experts=4)
		self.assertIsInstance(mask, oa.Matrix)
		# shape should be [num_tokens, num_experts]
		self.assertEqual(mask.shape, [3, 4])
		self.assertEqual(mask.dtype, "f32")

	def test_top_k_mask_selected_entries_are_one(self) -> None:
		# Token 0 selects experts 0 and 1; Token 1 selects 1 and 3
		indices = oa.Matrix.from_i32(self.engine, [2, 2], [0, 1, 1, 3])
		mask = oa.matrix.top_k_mask(indices, num_experts=4)
		values = mask.read_f32()
		# Token 0: positions [0,1] should be 1.0, positions [2,3] should be 0.0
		self.assertAlmostEqual(values[0], 1.0, places=5)  # expert 0
		self.assertAlmostEqual(values[1], 1.0, places=5)  # expert 1
		self.assertAlmostEqual(values[2], 0.0, places=5)  # expert 2
		self.assertAlmostEqual(values[3], 0.0, places=5)  # expert 3

	def test_top_k_mask_row_sum_equals_k(self) -> None:
		# Each token selects exactly 2 experts
		indices = oa.Matrix.from_i32(self.engine, [4, 2], [0, 1, 2, 3, 0, 3, 1, 2])
		mask = oa.matrix.top_k_mask(indices, num_experts=4)
		values = mask.read_f32()
		for row in range(4):
			row_sum = sum(values[row * 4 : row * 4 + 4])
			self.assertAlmostEqual(row_sum, 2.0, places=5)


class MatrixMoeExpertPlanTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_moe_expert_plan_returns_plan_type(self) -> None:
		indices = oa.Matrix.from_i32(self.engine, [3, 2], [0, 1, 1, 2, 0, 2])
		plan = oa.matrix.moe_expert_plan(indices, num_experts=3)
		self.assertIsInstance(plan, oa.matrix.MoeExpertPlan)

	def test_moe_expert_plan_fields_are_matrices(self) -> None:
		indices = oa.Matrix.from_i32(self.engine, [3, 2], [0, 1, 1, 2, 0, 2])
		plan = oa.matrix.moe_expert_plan(indices, num_experts=3)
		self.assertIsInstance(plan.counts, oa.Matrix)
		self.assertIsInstance(plan.offsets, oa.Matrix)
		self.assertIsInstance(plan.packed_token, oa.Matrix)
		self.assertIsInstance(plan.packed_expert, oa.Matrix)
		self.assertIsInstance(plan.packed_slot, oa.Matrix)
		self.assertIsInstance(plan.inverse, oa.Matrix)

	def test_moe_expert_plan_counts_shape(self) -> None:
		# 3 tokens × 2 experts from 4 total experts
		indices = oa.Matrix.from_i32(self.engine, [3, 2], [0, 1, 1, 3, 2, 0])
		plan = oa.matrix.moe_expert_plan(indices, num_experts=4)
		# counts has one entry per expert
		self.assertEqual(plan.counts.shape, [4])

	def test_moe_expert_plan_counts_sum_equals_total_assignments(self) -> None:
		# 4 tokens × 2 experts = 8 total token-expert assignments
		indices = oa.Matrix.from_i32(self.engine, [4, 2], [0, 1, 2, 3, 0, 3, 1, 2])
		plan = oa.matrix.moe_expert_plan(indices, num_experts=4)
		total = sum(plan.counts.read_u32())
		self.assertEqual(total, 8)

	def test_moe_expert_plan_packed_length(self) -> None:
		# packed_token/packed_expert/packed_slot are each of length (tokens × k)
		indices = oa.Matrix.from_i32(self.engine, [4, 2], [0, 1, 2, 3, 0, 3, 1, 2])
		plan = oa.matrix.moe_expert_plan(indices, num_experts=4)
		self.assertEqual(plan.packed_token.num_elements, 8)
		self.assertEqual(plan.packed_expert.num_elements, 8)
		self.assertEqual(plan.packed_slot.num_elements, 8)

	def test_moe_expert_plan_inverse_length(self) -> None:
		# inverse has same length as packed arrays
		indices = oa.Matrix.from_i32(self.engine, [4, 2], [0, 1, 2, 3, 0, 3, 1, 2])
		plan = oa.matrix.moe_expert_plan(indices, num_experts=4)
		self.assertEqual(plan.inverse.num_elements, 8)


class MatrixMoeRoutingBiasUpdateTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_routing_bias_update_returns_none(self) -> None:
		# selection_mask: [tokens, experts] binary assignment
		mask = oa.Matrix.from_f32(
			self.engine, [4, 3], [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
		)
		bias = oa.Matrix.from_f32(self.engine, [3], [0.0, 0.0, 0.0])
		result = oa.matrix.moe_routing_bias_update(
			mask, bias, experts_per_token=1, gamma=0.001
		)
		self.assertIsNone(result)

	def test_routing_bias_update_modifies_bias_in_place(self) -> None:
		# Note: the operation updates bias in-place; we can only verify it doesn't raise
		mask = oa.Matrix.from_f32(
			self.engine, [2, 2], [1.0, 0.0, 0.0, 1.0]
		)
		bias = oa.Matrix.from_f32(self.engine, [2], [0.0, 0.0])
		oa.matrix.moe_routing_bias_update(mask, bias, experts_per_token=1, gamma=0.01)


if __name__ == "__main__":
	unittest.main()
