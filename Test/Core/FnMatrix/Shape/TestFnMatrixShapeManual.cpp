// Manual tests for Core/FnMatrix Shape operations
// Transpose, Broadcast, and other shape manipulation operations

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <algorithm>
#include <numeric>
#include <vector>

static OaComputeEngine* GRt = nullptr;

class TestFnMatrixShapeManual : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixShapeManual";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to create matrix from host data
static OaMatrix CreateMatrixFromHost(const std::vector<float>& data, OaMatrixShape shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
		shape);
}

// ============================================================================
// Transpose Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Transpose_2D_Simple) {
	// Test simple 2D matrix transpose
	constexpr OaU32 M = 3, N = 4;
	std::vector<float> data = {
		1.0f,  2.0f,  3.0f,  4.0f,   // Row 0
		5.0f,  6.0f,  7.0f,  8.0f,   // Row 1
		9.0f, 10.0f, 11.0f, 12.0f    // Row 2
	};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{M, N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto transposed = OaFnMatrix::Transpose(a, 0, 1);  // Transpose dims 0 and 1
	
	// Check shape
	EXPECT_EQ(transposed.GetShape().Rank, 2);
	EXPECT_EQ(transposed.GetShape()[0], N);  // Swapped
	EXPECT_EQ(transposed.GetShape()[1], M);  // Swapped
	
	// Expected: columns become rows
	std::vector<float> expected = {
		1.0f, 5.0f,  9.0f,   // Col 0 -> Row 0
		2.0f, 6.0f, 10.0f,   // Col 1 -> Row 1
		3.0f, 7.0f, 11.0f,   // Col 2 -> Row 2
		4.0f, 8.0f, 12.0f    // Col 3 -> Row 3
	};
	
	std::vector<float> got(N * M);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(transposed, got.data(), N * M * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < N * M; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Transpose_2D_Square) {
	// Test square matrix transpose
	constexpr OaU32 N = 3;
	std::vector<float> data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f
	};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{N, N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto transposed = OaFnMatrix::Transpose(a, 0, 1);
	
	// Expected: diagonal unchanged, off-diagonal swapped
	std::vector<float> expected = {
		1.0f, 4.0f, 7.0f,
		2.0f, 5.0f, 8.0f,
		3.0f, 6.0f, 9.0f
	};
	
	std::vector<float> got(N * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(transposed, got.data(), N * N * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < N * N; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Transpose_3D) {
	// Test 3D tensor transpose (swap last two dimensions)
	constexpr OaU32 B = 2, M = 2, N = 3;
	std::vector<float> data = {
		// Batch 0
		1.0f, 2.0f, 3.0f,   // Row 0
		4.0f, 5.0f, 6.0f,   // Row 1
		// Batch 1
		7.0f,  8.0f,  9.0f,  // Row 0
		10.0f, 11.0f, 12.0f  // Row 1
	};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{B, M, N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto transposed = OaFnMatrix::Transpose(a, 1, 2);  // Swap dims 1 and 2
	
	// Check shape
	EXPECT_EQ(transposed.GetShape().Rank, 3);
	EXPECT_EQ(transposed.GetShape()[0], B);  // Unchanged
	EXPECT_EQ(transposed.GetShape()[1], N);  // Swapped
	EXPECT_EQ(transposed.GetShape()[2], M);  // Swapped
	
	// Expected: each batch transposed independently
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
	ASSERT_TRUE(OaFnMatrix::CopyToHost(transposed, got.data(), B * N * M * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < B * N * M; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Transpose_Identity) {
	// Test that double transpose returns to original
	constexpr OaU32 M = 2, N = 3;
	std::vector<float> data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{M, N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto transposed = OaFnMatrix::Transpose(a, 0, 1);
	auto double_transposed = OaFnMatrix::Transpose(transposed, 0, 1);
	
	// Should match original
	std::vector<float> got(M * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(double_transposed, got.data(), M * N * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < M * N; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}

// ============================================================================
// Concat Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Concat_1D_Simple) {
	// Test concatenating 1D tensors
	std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
	std::vector<float> data2 = {4.0f, 5.0f};
	std::vector<float> data3 = {6.0f, 7.0f, 8.0f, 9.0f};
	
	auto a = CreateMatrixFromHost(data1, OaMatrixShape{3});
	auto b = CreateMatrixFromHost(data2, OaMatrixShape{2});
	auto c = CreateMatrixFromHost(data3, OaMatrixShape{4});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaVec<OaMatrix> inputs = {a, b, c};
	auto concatenated = OaFnMatrix::Concat(OaSpan<OaMatrix>(inputs), 0);
	
	// Expected: [1, 2, 3, 4, 5, 6, 7, 8, 9]
	EXPECT_EQ(concatenated.GetShape()[0], 9);
	
	std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
	std::vector<float> got(9);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(concatenated, got.data(), 9 * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 9; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Concat_2D_Rows) {
	// Test concatenating along rows (dim 0)
	constexpr OaU32 N = 3;
	std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 2x3
	std::vector<float> data2 = {7.0f, 8.0f, 9.0f};  // 1x3
	
	auto a = CreateMatrixFromHost(data1, OaMatrixShape{2, N});
	auto b = CreateMatrixFromHost(data2, OaMatrixShape{1, N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaVec<OaMatrix> inputs = {a, b};
	auto concatenated = OaFnMatrix::Concat(OaSpan<OaMatrix>(inputs), 0);
	
	// Expected: 3x3 matrix
	EXPECT_EQ(concatenated.GetShape()[0], 3);
	EXPECT_EQ(concatenated.GetShape()[1], N);
	
	std::vector<float> expected = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f
	};
	
	std::vector<float> got(9);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(concatenated, got.data(), 9 * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 9; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Concat_2D_Cols) {
	// Test concatenating along columns (dim 1)
	constexpr OaU32 M = 2;
	std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
	std::vector<float> data2 = {5.0f, 6.0f};  // 2x1
	
	auto a = CreateMatrixFromHost(data1, OaMatrixShape{M, 2});
	auto b = CreateMatrixFromHost(data2, OaMatrixShape{M, 1});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaVec<OaMatrix> inputs = {a, b};
	auto concatenated = OaFnMatrix::Concat(OaSpan<OaMatrix>(inputs), 1);
	
	// Expected: 2x3 matrix
	EXPECT_EQ(concatenated.GetShape()[0], M);
	EXPECT_EQ(concatenated.GetShape()[1], 3);
	
	std::vector<float> expected = {
		1.0f, 2.0f, 5.0f,
		3.0f, 4.0f, 6.0f
	};
	
	std::vector<float> got(6);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(concatenated, got.data(), 6 * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "Index " << i;
	}
}

// ============================================================================
// Split Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Split_1D_Equal) {
	// Test splitting 1D tensor into equal parts
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{6});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaVec<OaI64> sizes = {2, 2, 2};
	auto splits = OaFnMatrix::Split(a, OaSpan<OaI64>(sizes), 0);
	
	ASSERT_EQ(splits.Size(), 3);
	
	// Check each split
	std::vector<std::vector<float>> expected = {
		{1.0f, 2.0f},
		{3.0f, 4.0f},
		{5.0f, 6.0f}
	};
	
	for (OaU32 i = 0; i < 3; ++i) {
		EXPECT_EQ(splits[i].GetShape()[0], 2);
		std::vector<float> got(2);
		ASSERT_TRUE(OaFnMatrix::CopyToHost(splits[i], got.data(), 2 * sizeof(float)).IsOk());
		
		for (OaU32 j = 0; j < 2; ++j) {
			EXPECT_FLOAT_EQ(got[j], expected[i][j]) << "Split " << i << ", Index " << j;
		}
	}
}

TEST_VK(TestFnMatrixShapeManual, Split_1D_Unequal) {
	// Test splitting 1D tensor into unequal parts
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{7});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaVec<OaI64> sizes = {3, 1, 3};
	auto splits = OaFnMatrix::Split(a, OaSpan<OaI64>(sizes), 0);
	
	ASSERT_EQ(splits.Size(), 3);
	
	// Check sizes
	EXPECT_EQ(splits[0].GetShape()[0], 3);
	EXPECT_EQ(splits[1].GetShape()[0], 1);
	EXPECT_EQ(splits[2].GetShape()[0], 3);
	
	// Check values
	std::vector<float> split0(3), split1(1), split2(3);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(splits[0], split0.data(), 3 * sizeof(float)).IsOk());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(splits[1], split1.data(), 1 * sizeof(float)).IsOk());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(splits[2], split2.data(), 3 * sizeof(float)).IsOk());
	
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
	constexpr OaU32 N = 3;
	std::vector<float> data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f,
		10.0f, 11.0f, 12.0f
	};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{4, N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaVec<OaI64> sizes = {1, 2, 1};
	auto splits = OaFnMatrix::Split(a, OaSpan<OaI64>(sizes), 0);
	
	ASSERT_EQ(splits.Size(), 3);
	
	// Check shapes
	EXPECT_EQ(splits[0].GetShape()[0], 1);
	EXPECT_EQ(splits[0].GetShape()[1], N);
	EXPECT_EQ(splits[1].GetShape()[0], 2);
	EXPECT_EQ(splits[1].GetShape()[1], N);
	EXPECT_EQ(splits[2].GetShape()[0], 1);
	EXPECT_EQ(splits[2].GetShape()[1], N);
	
	// Check first split
	std::vector<float> split0(N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(splits[0], split0.data(), N * sizeof(float)).IsOk());
	EXPECT_FLOAT_EQ(split0[0], 1.0f);
	EXPECT_FLOAT_EQ(split0[1], 2.0f);
	EXPECT_FLOAT_EQ(split0[2], 3.0f);
}
// ============================================================================
// Permute Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Permute_3D_Simple) {
	// Test permuting dimensions of a 3D tensor
	constexpr OaU32 B = 2, H = 3, W = 4;
	std::vector<float> data(B * H * W);
	std::iota(data.begin(), data.end(), 1.0f);  // 1, 2, 3, ..., 24

	auto a = CreateMatrixFromHost(data, OaMatrixShape{B, H, W});

	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaVec<OaI32> dims = {2, 0, 1};  // W, B, H
	auto permuted = a.Permute(OaSpan<const OaI32>(dims));

	// Check shape: [2,3,4] -> [4,2,3]
	EXPECT_EQ(permuted.GetShape().Rank, 3);
	EXPECT_EQ(permuted.GetShape()[0], W);
	EXPECT_EQ(permuted.GetShape()[1], B);
	EXPECT_EQ(permuted.GetShape()[2], H);

	// Permute yields a lazy strided VIEW; the reordered bytes only exist after
	// materializing. Contiguous() gathers through the permuted strides into a
	// row-major buffer that CopyToHost can then read in logical order.
	auto materialized = permuted.Contiguous();
	std::vector<float> got(B * H * W);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(materialized, got.data(), B * H * W * sizeof(float)).IsOk());
	
	// Original: [B,H,W] = [2,3,4]
	// Permuted: [W,B,H] = [4,2,3]
	// Element at [b,h,w] in original should be at [w,b,h] in permuted
	for (OaU32 b = 0; b < B; ++b) {
		for (OaU32 h = 0; h < H; ++h) {
			for (OaU32 w = 0; w < W; ++w) {
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
	constexpr OaU32 N = 2, C = 3, H = 2, W = 2;
	std::vector<float> data(N * C * H * W);
	std::iota(data.begin(), data.end(), 1.0f);
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{N, C, H, W});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaVec<OaI32> dims = {0, 2, 3, 1};  // N, H, W, C
	auto permuted = a.Permute(OaSpan<const OaI32>(dims));
	
	// Check shape: [2,3,2,2] -> [2,2,2,3]
	EXPECT_EQ(permuted.GetShape()[0], N);
	EXPECT_EQ(permuted.GetShape()[1], H);
	EXPECT_EQ(permuted.GetShape()[2], W);
	EXPECT_EQ(permuted.GetShape()[3], C);
}

TEST_VK(TestFnMatrixShapeManual, Permute_Identity) {
	// Test that identity permutation doesn't change data
	constexpr OaU32 B = 2, H = 3, W = 4;
	std::vector<float> data(B * H * W);
	std::iota(data.begin(), data.end(), 1.0f);
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{B, H, W});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaVec<OaI32> dims = {0, 1, 2};  // Identity
	auto permuted = a.Permute(OaSpan<const OaI32>(dims));
	
	std::vector<float> got(B * H * W);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(permuted, got.data(), B * H * W * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < B * H * W; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}

// ============================================================================
// Squeeze Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Squeeze_SingleDim) {
	// Test squeezing a single dimension of size 1
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	// Shape: [1, 4] -> squeeze dim 0 -> [4]
	auto a = CreateMatrixFromHost(data, OaMatrixShape{1, 4});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto squeezed = a.Squeeze(0);
	
	EXPECT_EQ(squeezed.GetShape().Rank, 1);
	EXPECT_EQ(squeezed.GetShape()[0], 4);
	
	std::vector<float> got(4);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(squeezed, got.data(), 4 * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Squeeze_MiddleDim) {
	// Test squeezing a middle dimension
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	
	// Shape: [2, 1, 3] -> squeeze dim 1 -> [2, 3]
	auto a = CreateMatrixFromHost(data, OaMatrixShape{2, 1, 3});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto squeezed = a.Squeeze(1);
	
	EXPECT_EQ(squeezed.GetShape().Rank, 2);
	EXPECT_EQ(squeezed.GetShape()[0], 2);
	EXPECT_EQ(squeezed.GetShape()[1], 3);
	
	std::vector<float> got(6);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(squeezed, got.data(), 6 * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Squeeze_LastDim) {
	// Test squeezing the last dimension
	std::vector<float> data = {1.0f, 2.0f, 3.0f};
	
	// Shape: [3, 1] -> squeeze dim 1 -> [3]
	auto a = CreateMatrixFromHost(data, OaMatrixShape{3, 1});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto squeezed = a.Squeeze(1);
	
	EXPECT_EQ(squeezed.GetShape().Rank, 1);
	EXPECT_EQ(squeezed.GetShape()[0], 3);
}

// ============================================================================
// Unsqueeze Tests
// ============================================================================

TEST_VK(TestFnMatrixShapeManual, Unsqueeze_Front) {
	// Test adding dimension at the front
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	// Shape: [4] -> unsqueeze dim 0 -> [1, 4]
	auto a = CreateMatrixFromHost(data, OaMatrixShape{4});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto unsqueezed = a.Unsqueeze(0);
	
	EXPECT_EQ(unsqueezed.GetShape().Rank, 2);
	EXPECT_EQ(unsqueezed.GetShape()[0], 1);
	EXPECT_EQ(unsqueezed.GetShape()[1], 4);
	
	std::vector<float> got(4);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(unsqueezed, got.data(), 4 * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Unsqueeze_Middle) {
	// Test adding dimension in the middle
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	
	// Shape: [2, 3] -> unsqueeze dim 1 -> [2, 1, 3]
	auto a = CreateMatrixFromHost(data, OaMatrixShape{2, 3});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto unsqueezed = a.Unsqueeze(1);
	
	EXPECT_EQ(unsqueezed.GetShape().Rank, 3);
	EXPECT_EQ(unsqueezed.GetShape()[0], 2);
	EXPECT_EQ(unsqueezed.GetShape()[1], 1);
	EXPECT_EQ(unsqueezed.GetShape()[2], 3);
	
	std::vector<float> got(6);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(unsqueezed, got.data(), 6 * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixShapeManual, Unsqueeze_End) {
	// Test adding dimension at the end
	std::vector<float> data = {1.0f, 2.0f, 3.0f};
	
	// Shape: [3] -> unsqueeze dim 1 -> [3, 1]
	auto a = CreateMatrixFromHost(data, OaMatrixShape{3});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto unsqueezed = a.Unsqueeze(1);
	
	EXPECT_EQ(unsqueezed.GetShape().Rank, 2);
	EXPECT_EQ(unsqueezed.GetShape()[0], 3);
	EXPECT_EQ(unsqueezed.GetShape()[1], 1);
}

TEST_VK(TestFnMatrixShapeManual, Unsqueeze_Squeeze_RoundTrip) {
	// Test that unsqueeze followed by squeeze returns to original
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{4});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto unsqueezed = a.Unsqueeze(0);  // [4] -> [1, 4]
	auto squeezed = unsqueezed.Squeeze(0);  // [1, 4] -> [4]
	
	EXPECT_EQ(squeezed.GetShape().Rank, 1);
	EXPECT_EQ(squeezed.GetShape()[0], 4);
	
	std::vector<float> got(4);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(squeezed, got.data(), 4 * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}


