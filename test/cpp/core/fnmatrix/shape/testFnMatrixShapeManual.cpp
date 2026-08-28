// Manual tests for Core/FnMatrix Shape operations
// Transpose, Broadcast, and other shape manipulation operations

#include <gtest/gtest.h>
#include "../../../oaTest.h"
#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <algorithm>
#include <numeric>
#include <vector>

static oa::Engine* GRt = nullptr;

class TestFnMatrixShapeManual : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrixShapeManual";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to create matrix from host data
static oa::Matrix createMatrixFromHost(const std::vector<float>& data, oa::MatrixShape shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape);
}

// ============================================================================
// Transpose Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Transpose_2D_Simple) {
	// Test simple 2D matrix transpose
	constexpr oa::U32 M = 3, N = 4;
	std::vector<float> data = {
		1.0f,  2.0f,  3.0f,  4.0f,   // Row 0
		5.0f,  6.0f,  7.0f,  8.0f,   // Row 1
		9.0f, 10.0f, 11.0f, 12.0f    // Row 2
	};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{M, N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto transposed = oa::FnMatrix::transpose(a, 0, 1);  // Transpose dims 0 and 1
	
	// Check shape
	EXPECT_EQ(transposed.getShape().rank, 2);
	EXPECT_EQ(transposed.getShape()[0], N);  // Swapped
	EXPECT_EQ(transposed.getShape()[1], M);  // Swapped
	
	// expected: columns become rows
	std::vector<float> expected = {
		1.0f, 5.0f,  9.0f,   // col 0 -> Row 0
		2.0f, 6.0f, 10.0f,   // col 1 -> Row 1
		3.0f, 7.0f, 11.0f,   // col 2 -> Row 2
		4.0f, 8.0f, 12.0f    // col 3 -> Row 3
	};
	
	std::vector<float> got(N * M);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(transposed, got.data(), N * M * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < N * M; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Transpose_2D_Square) {
	// Test square matrix transpose
	constexpr oa::U32 N = 3;
	std::vector<float> data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f
	};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{N, N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto transposed = oa::FnMatrix::transpose(a, 0, 1);
	
	// expected: diagonal unchanged, off-diagonal swapped
	std::vector<float> expected = {
		1.0f, 4.0f, 7.0f,
		2.0f, 5.0f, 8.0f,
		3.0f, 6.0f, 9.0f
	};
	
	std::vector<float> got(N * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(transposed, got.data(), N * N * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < N * N; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Transpose_3D) {
	// Test 3D tensor transpose (swap last two dimensions)
	constexpr oa::U32 B = 2, M = 2, N = 3;
	std::vector<float> data = {
		// Batch 0
		1.0f, 2.0f, 3.0f,   // Row 0
		4.0f, 5.0f, 6.0f,   // Row 1
		// Batch 1
		7.0f,  8.0f,  9.0f,  // Row 0
		10.0f, 11.0f, 12.0f  // Row 1
	};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{B, M, N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto transposed = oa::FnMatrix::transpose(a, 1, 2);  // swap dims 1 and 2
	
	// Check shape
	EXPECT_EQ(transposed.getShape().rank, 3);
	EXPECT_EQ(transposed.getShape()[0], B);  // Unchanged
	EXPECT_EQ(transposed.getShape()[1], N);  // Swapped
	EXPECT_EQ(transposed.getShape()[2], M);  // Swapped
	
	// expected: each batch transposed independently
	std::vector<float> expected = {
		// Batch 0 transposed
		1.0f, 4.0f,
		2.0f, 5.0f,
		3.0f, 6.0f,
		// Batch 1 transposed
		7.0f, 10.0f,
		8.0f, 11.0f,
		9.0f, 12.0f
	};
	
	std::vector<float> got(B * N * M);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(transposed, got.data(), B * N * M * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < B * N * M; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Transpose_Identity) {
	// Test that double transpose returns to original
	constexpr oa::U32 M = 2, N = 3;
	std::vector<float> data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{M, N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto transposed = oa::FnMatrix::transpose(a, 0, 1);
	auto double_transposed = oa::FnMatrix::transpose(transposed, 0, 1);
	
	// Should match original
	std::vector<float> got(M * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(double_transposed, got.data(), M * N * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < M * N; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Transpose_BackwardSwapsGradientAxes) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope ctxScope(ctx);
	auto input = createMatrixFromHost(
		{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
		oa::MatrixShape{2, 3});
	input.setRequiresGrad(true);
	const auto weights = createMatrixFromHost(
		{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
		oa::MatrixShape{3, 2});

	oa::GradientTape tape;
	const auto transposed = oa::FnMatrix::transpose(input);
	const auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(transposed, weights));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	std::vector<float> gradient(6);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		input.gradMatrix(), gradient.data(),
		gradient.size() * sizeof(float)).isOk());
	const std::vector<float> expected = {
		1.0F, 3.0F, 5.0F,
		2.0F, 4.0F, 6.0F,
	};
	EXPECT_EQ(gradient, expected);
}

// ============================================================================
// Concat Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Concat_1D_Simple) {
	// Test concatenating 1D tensors
	std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
	std::vector<float> data2 = {4.0f, 5.0f};
	std::vector<float> data3 = {6.0f, 7.0f, 8.0f, 9.0f};
	
	auto a = createMatrixFromHost(data1, oa::MatrixShape{3});
	auto b = createMatrixFromHost(data2, oa::MatrixShape{2});
	auto c = createMatrixFromHost(data3, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::Vector<oa::Matrix> inputs = {a, b, c};
	auto concatenated = oa::FnMatrix::concat(oa::Span<oa::Matrix>(inputs), 0);
	
	// expected: [1, 2, 3, 4, 5, 6, 7, 8, 9]
	EXPECT_EQ(concatenated.getShape()[0], 9);
	
	std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
	std::vector<float> got(9);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(concatenated, got.data(), 9 * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 9; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Concat_2D_Rows) {
	// Test concatenating along rows (dim 0)
	constexpr oa::U32 N = 3;
	std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 2x3
	std::vector<float> data2 = {7.0f, 8.0f, 9.0f};  // 1x3
	
	auto a = createMatrixFromHost(data1, oa::MatrixShape{2, N});
	auto b = createMatrixFromHost(data2, oa::MatrixShape{1, N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::Vector<oa::Matrix> inputs = {a, b};
	auto concatenated = oa::FnMatrix::concat(oa::Span<oa::Matrix>(inputs), 0);
	
	// expected: 3x3 matrix
	EXPECT_EQ(concatenated.getShape()[0], 3);
	EXPECT_EQ(concatenated.getShape()[1], N);
	
	std::vector<float> expected = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f
	};
	
	std::vector<float> got(9);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(concatenated, got.data(), 9 * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 9; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Concat_2D_Cols) {
	// Test concatenating along columns (dim 1)
	constexpr oa::U32 M = 2;
	std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
	std::vector<float> data2 = {5.0f, 6.0f};  // 2x1
	
	auto a = createMatrixFromHost(data1, oa::MatrixShape{M, 2});
	auto b = createMatrixFromHost(data2, oa::MatrixShape{M, 1});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::Vector<oa::Matrix> inputs = {a, b};
	auto concatenated = oa::FnMatrix::concat(oa::Span<oa::Matrix>(inputs), 1);
	
	// expected: 2x3 matrix
	EXPECT_EQ(concatenated.getShape()[0], M);
	EXPECT_EQ(concatenated.getShape()[1], 3);
	
	std::vector<float> expected = {
		1.0f, 2.0f, 5.0f,
		3.0f, 4.0f, 6.0f
	};
	
	std::vector<float> got(6);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(concatenated, got.data(), 6 * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "index " << i;
	}
}

// ============================================================================
// split Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Split_1D_Equal) {
	// Test splitting 1D tensor into equal parts
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{6});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::Vector<oa::I64> sizes = {2, 2, 2};
	auto splits = oa::FnMatrix::split(a, oa::Span<oa::I64>(sizes), 0);
	
	ASSERT_EQ(splits.size(), 3);
	
	// Check each split
	std::vector<std::vector<float>> expected = {
		{1.0f, 2.0f},
		{3.0f, 4.0f},
		{5.0f, 6.0f}
	};
	
	for (oa::U32 i = 0; i < 3; ++i) {
		EXPECT_EQ(splits[i].getShape()[0], 2);
		std::vector<float> got(2);
		ASSERT_TRUE(oa::FnMatrix::copyToHost(splits[i], got.data(), 2 * sizeof(float)).isOk());
		
		for (oa::U32 j = 0; j < 2; ++j) {
			EXPECT_FLOAT_EQ(got[j], expected[i][j]) << "split " << i << ", index " << j;
		}
	}
}

TEST_VK(TestFnMatrixShapeManual, Split_1D_Unequal) {
	// Test splitting 1D tensor into unequal parts
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{7});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::Vector<oa::I64> sizes = {3, 1, 3};
	auto splits = oa::FnMatrix::split(a, oa::Span<oa::I64>(sizes), 0);
	
	ASSERT_EQ(splits.size(), 3);
	
	// Check sizes
	EXPECT_EQ(splits[0].getShape()[0], 3);
	EXPECT_EQ(splits[1].getShape()[0], 1);
	EXPECT_EQ(splits[2].getShape()[0], 3);
	
	// Check values
	std::vector<float> split0(3), split1(1), split2(3);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(splits[0], split0.data(), 3 * sizeof(float)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(splits[1], split1.data(), 1 * sizeof(float)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(splits[2], split2.data(), 3 * sizeof(float)).isOk());
	
	EXPECT_FLOAT_EQ(split0[0], 1.0f);
	EXPECT_FLOAT_EQ(split0[1], 2.0f);
	EXPECT_FLOAT_EQ(split0[2], 3.0f);
	EXPECT_FLOAT_EQ(split1[0], 4.0f);
	EXPECT_FLOAT_EQ(split2[0], 5.0f);
	EXPECT_FLOAT_EQ(split2[1], 6.0f);
	EXPECT_FLOAT_EQ(split2[2], 7.0f);
}

TEST_VK(TestFnMatrixShapeManual, Split_2D_Rows) {
	// Test splitting 2D tensor along rows
	constexpr oa::U32 N = 3;
	std::vector<float> data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f,
		10.0f, 11.0f, 12.0f
	};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{4, N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::Vector<oa::I64> sizes = {1, 2, 1};
	auto splits = oa::FnMatrix::split(a, oa::Span<oa::I64>(sizes), 0);
	
	ASSERT_EQ(splits.size(), 3);
	
	// Check shapes
	EXPECT_EQ(splits[0].getShape()[0], 1);
	EXPECT_EQ(splits[0].getShape()[1], N);
	EXPECT_EQ(splits[1].getShape()[0], 2);
	EXPECT_EQ(splits[1].getShape()[1], N);
	EXPECT_EQ(splits[2].getShape()[0], 1);
	EXPECT_EQ(splits[2].getShape()[1], N);
	
	// Check first split
	std::vector<float> split0(N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(splits[0], split0.data(), N * sizeof(float)).isOk());
	EXPECT_FLOAT_EQ(split0[0], 1.0f);
	EXPECT_FLOAT_EQ(split0[1], 2.0f);
	EXPECT_FLOAT_EQ(split0[2], 3.0f);
}
// ============================================================================
// permute Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Permute_3D_Simple) {
	// Test permuting dimensions of a 3D tensor
	constexpr oa::U32 B = 2, H = 3, W = 4;
	std::vector<float> data(B * H * W);
	std::iota(data.begin(), data.end(), 1.0f);  // 1, 2, 3, ..., 24

	auto a = createMatrixFromHost(data, oa::MatrixShape{B, H, W});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::Vector<oa::I32> dims = {2, 0, 1};  // W, B, H
	auto permuted = a.permute(oa::Span<const oa::I32>(dims));

	// Check shape: [2,3,4] -> [4,2,3]
	EXPECT_EQ(permuted.getShape().rank, 3);
	EXPECT_EQ(permuted.getShape()[0], W);
	EXPECT_EQ(permuted.getShape()[1], B);
	EXPECT_EQ(permuted.getShape()[2], H);

	// permute yields a lazy strided VIEW; the reordered bytes only exist after
	// materializing. contiguous() gathers through the permuted strides into a
	// row-major buffer that CopyToHost can then read in logical order.
	auto materialized = permuted.contiguous();
	std::vector<float> got(B * H * W);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(materialized, got.data(), B * H * W * sizeof(float)).isOk());
	
	// Original: [B,H,W] = [2,3,4]
	// Permuted: [W,B,H] = [4,2,3]
	// Element at [b,h,w] in original should be at [w,b,h] in permuted
	for (oa::U32 b = 0; b < B; ++b) {
		for (oa::U32 h = 0; h < H; ++h) {
			for (oa::U32 w = 0; w < W; ++w) {
				float original_val = data[b * H * W + h * W + w];
				float permuted_val = got[w * B * H + b * H + h];
				EXPECT_FLOAT_EQ(permuted_val, original_val) 
					<< "Mismatch at [" << b << "," << h << "," << w << "]";
			}
		}
	}
}

TEST_VK(TestFnMatrixShapeManual, Permute_4D_NCHW_to_NHWC) {
	// Test common permutation: NCHW -> NHWC (channels last)
	constexpr oa::U32 N = 2, C = 3, H = 2, W = 2;
	std::vector<float> data(N * C * H * W);
	std::iota(data.begin(), data.end(), 1.0f);
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{N, C, H, W});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::Vector<oa::I32> dims = {0, 2, 3, 1};  // N, H, W, C
	auto permuted = a.permute(oa::Span<const oa::I32>(dims));
	
	// Check shape: [2,3,2,2] -> [2,2,2,3]
	EXPECT_EQ(permuted.getShape()[0], N);
	EXPECT_EQ(permuted.getShape()[1], H);
	EXPECT_EQ(permuted.getShape()[2], W);
	EXPECT_EQ(permuted.getShape()[3], C);
}

TEST_VK(TestFnMatrixShapeManual, Permute_Identity) {
	// Test that identity permutation doesn't change data
	constexpr oa::U32 B = 2, H = 3, W = 4;
	std::vector<float> data(B * H * W);
	std::iota(data.begin(), data.end(), 1.0f);
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{B, H, W});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::Vector<oa::I32> dims = {0, 1, 2};  // Identity
	auto permuted = a.permute(oa::Span<const oa::I32>(dims));
	
	std::vector<float> got(B * H * W);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(permuted, got.data(), B * H * W * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < B * H * W; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}

// ============================================================================
// squeeze Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Squeeze_SingleDim) {
	// Test squeezing a single dimension of size 1
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	// Shape: [1, 4] -> squeeze dim 0 -> [4]
	auto a = createMatrixFromHost(data, oa::MatrixShape{1, 4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto squeezed = a.squeeze(0);
	
	EXPECT_EQ(squeezed.getShape().rank, 1);
	EXPECT_EQ(squeezed.getShape()[0], 4);
	
	std::vector<float> got(4);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(squeezed, got.data(), 4 * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Squeeze_MiddleDim) {
	// Test squeezing a middle dimension
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	
	// Shape: [2, 1, 3] -> squeeze dim 1 -> [2, 3]
	auto a = createMatrixFromHost(data, oa::MatrixShape{2, 1, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto squeezed = a.squeeze(1);
	
	EXPECT_EQ(squeezed.getShape().rank, 2);
	EXPECT_EQ(squeezed.getShape()[0], 2);
	EXPECT_EQ(squeezed.getShape()[1], 3);
	
	std::vector<float> got(6);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(squeezed, got.data(), 6 * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Squeeze_LastDim) {
	// Test squeezing the last dimension
	std::vector<float> data = {1.0f, 2.0f, 3.0f};
	
	// Shape: [3, 1] -> squeeze dim 1 -> [3]
	auto a = createMatrixFromHost(data, oa::MatrixShape{3, 1});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto squeezed = a.squeeze(1);
	
	EXPECT_EQ(squeezed.getShape().rank, 1);
	EXPECT_EQ(squeezed.getShape()[0], 3);
}

// ============================================================================
// unsqueeze Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Unsqueeze_Front) {
	// Test adding dimension at the front
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	// Shape: [4] -> unsqueeze dim 0 -> [1, 4]
	auto a = createMatrixFromHost(data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto unsqueezed = a.unsqueeze(0);
	
	EXPECT_EQ(unsqueezed.getShape().rank, 2);
	EXPECT_EQ(unsqueezed.getShape()[0], 1);
	EXPECT_EQ(unsqueezed.getShape()[1], 4);
	
	std::vector<float> got(4);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(unsqueezed, got.data(), 4 * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Unsqueeze_Middle) {
	// Test adding dimension in the middle
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	
	// Shape: [2, 3] -> unsqueeze dim 1 -> [2, 1, 3]
	auto a = createMatrixFromHost(data, oa::MatrixShape{2, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto unsqueezed = a.unsqueeze(1);
	
	EXPECT_EQ(unsqueezed.getShape().rank, 3);
	EXPECT_EQ(unsqueezed.getShape()[0], 2);
	EXPECT_EQ(unsqueezed.getShape()[1], 1);
	EXPECT_EQ(unsqueezed.getShape()[2], 3);
	
	std::vector<float> got(6);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(unsqueezed, got.data(), 6 * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Unsqueeze_End) {
	// Test adding dimension at the end
	std::vector<float> data = {1.0f, 2.0f, 3.0f};
	
	// Shape: [3] -> unsqueeze dim 1 -> [3, 1]
	auto a = createMatrixFromHost(data, oa::MatrixShape{3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto unsqueezed = a.unsqueeze(1);
	
	EXPECT_EQ(unsqueezed.getShape().rank, 2);
	EXPECT_EQ(unsqueezed.getShape()[0], 3);
	EXPECT_EQ(unsqueezed.getShape()[1], 1);
}

TEST_VK(TestFnMatrixShapeManual, Unsqueeze_Squeeze_RoundTrip) {
	// Test that unsqueeze followed by squeeze returns to original
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto unsqueezed = a.unsqueeze(0);  // [4] -> [1, 4]
	auto squeezed = unsqueezed.squeeze(0);  // [1, 4] -> [4]
	
	EXPECT_EQ(squeezed.getShape().rank, 1);
	EXPECT_EQ(squeezed.getShape()[0], 4);
	
	std::vector<float> got(4);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(squeezed, got.data(), 4 * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}
