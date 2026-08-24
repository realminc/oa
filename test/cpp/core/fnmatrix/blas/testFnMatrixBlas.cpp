// Manual tests for Core/FnMatrix BLAS operations.
//
// IMPORTANT CONTRACT: oa::FnMatrix::matMulNt(A, B) takes B in [N,K] layout and
// computes  C = A @ Bᵀ  (out[m,n] = Σ_k A[m,k]·B[n,k]). This is the OA weight
// convention shared with Linear and attention — it is NOT the PyTorch-standard
// A @ B with B as [K,N]. Batched / standard A@B is oa::FnMatrix::bmm.
// The CPU reference MatMulNt below matches the MatMul contract exactly.

#include <gtest/gtest.h>
#include <oa/core/fnMatrix.h>
#include <oa/ml/fnMatrix.h>     // Bmm
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <vector>
#include <cmath>

static oa::Engine* GRt = nullptr;

class TestFnMatrixBlas : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrixBlas";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}

	// CPU reference for oa::FnMatrix::matMulNt — B is [N,K], C = A @ Bᵀ:
	//   C[m,n] = Σ_k A[m,k] · B[n,k]
	static void matMulNt(const std::vector<float>& a, const std::vector<float>& b,
	                     std::vector<float>& c, oa::U32 m, oa::U32 k, oa::U32 n) {
		for (oa::U32 i = 0; i < m; ++i) {
			for (oa::U32 j = 0; j < n; ++j) {
				float sum = 0.0f;
				for (oa::U32 p = 0; p < k; ++p) {
					sum += a[i * k + p] * b[j * k + p];  // B[n,k] layout
				}
				c[i * n + j] = sum;
			}
		}
	}

	// Standard per-batch reference for oa::FnMatrix::bmm — B is [K,N], C = A @ B.
	static void matMulStd(const std::vector<float>& a, const std::vector<float>& b,
	                      std::vector<float>& c, oa::U32 m, oa::U32 k, oa::U32 n) {
		for (oa::U32 i = 0; i < m; ++i) {
			for (oa::U32 j = 0; j < n; ++j) {
				float sum = 0.0f;
				for (oa::U32 p = 0; p < k; ++p) sum += a[i * k + p] * b[p * n + j];
				c[i * n + j] = sum;
			}
		}
	}
};

TEST_VK(TestFnMatrixBlas, MatMul_Square) {
	// [M=4,K=4] @ [N=4,K=4]ᵀ = [4,4]
	constexpr oa::U32 M = 4, K = 4, N = 4;

	std::vector<float> a_data(M * K);
	std::vector<float> b_data(N * K);
	for (oa::U32 i = 0; i < M * K; ++i) a_data[i] = static_cast<float>(i + 1);          // 1..16
	for (oa::U32 i = 0; i < N * K; ++i) b_data[i] = static_cast<float>((i % K) + 1);    // 1,2,3,4,...

	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(a_data.data()), M * K * sizeof(float)),
		oa::MatrixShape{M, K});
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(b_data.data()), N * K * sizeof(float)),
		oa::MatrixShape{N, K});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	// Force Fp32: the Auto router otherwise picks a bf16 path that fails the
	// 1e-3 correctness tolerance on non-integer data.
	auto c = oa::FnMatrix::matMulNt(a, b, oa::MatMulPrecision::Fp32);

	std::vector<float> c_want(M * N);
	matMulNt(a_data, b_data, c_want, M, K, N);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(c, c_got.data(), M * N * sizeof(float)).isOk());

	for (oa::U32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-3f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Rectangular) {
	// [M=3,K=5] @ [N=4,K=5]ᵀ = [3,4]
	constexpr oa::U32 M = 3, K = 5, N = 4;

	std::vector<float> a_data(M * K);
	std::vector<float> b_data(N * K);
	for (oa::U32 i = 0; i < M * K; ++i) a_data[i] = static_cast<float>(i + 1) * 0.1f;
	for (oa::U32 i = 0; i < N * K; ++i) b_data[i] = static_cast<float>(i + 1) * 0.2f;

	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(a_data.data()), M * K * sizeof(float)),
		oa::MatrixShape{M, K});
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(b_data.data()), N * K * sizeof(float)),
		oa::MatrixShape{N, K});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	// Force Fp32: the Auto router otherwise picks a bf16 path that fails the
	// 1e-3 correctness tolerance on non-integer data.
	auto c = oa::FnMatrix::matMulNt(a, b, oa::MatMulPrecision::Fp32);

	std::vector<float> c_want(M * N);
	matMulNt(a_data, b_data, c_want, M, K, N);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(c, c_got.data(), M * N * sizeof(float)).isOk());

	for (oa::U32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-3f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Vector) {
	// [M=4,K=8] @ [N=1,K=8]ᵀ = [4,1]  (matrix × vector)
	constexpr oa::U32 M = 4, K = 8, N = 1;

	std::vector<float> a_data(M * K);
	std::vector<float> b_data(N * K);
	for (oa::U32 i = 0; i < M * K; ++i) a_data[i] = static_cast<float>(i) * 0.1f;
	for (oa::U32 i = 0; i < N * K; ++i) b_data[i] = static_cast<float>(i + 1);

	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(a_data.data()), M * K * sizeof(float)),
		oa::MatrixShape{M, K});
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(b_data.data()), N * K * sizeof(float)),
		oa::MatrixShape{N, K});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	// Force Fp32: the Auto router otherwise picks a bf16 path that fails the
	// 1e-3 correctness tolerance on non-integer data.
	auto c = oa::FnMatrix::matMulNt(a, b, oa::MatMulPrecision::Fp32);

	std::vector<float> c_want(M * N);
	matMulNt(a_data, b_data, c_want, M, K, N);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(c, c_got.data(), M * N * sizeof(float)).isOk());

	for (oa::U32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-3f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Large) {
	// [M=64,K=128] @ [N=32,K=128]ᵀ = [64,32]
	constexpr oa::U32 M = 64, K = 128, N = 32;

	auto a = oa::FnMatrix::rand(oa::MatrixShape{M, K});
	auto b = oa::FnMatrix::rand(oa::MatrixShape{N, K});

	std::vector<float> a_data(M * K);
	std::vector<float> b_data(N * K);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(a, a_data.data(), M * K * sizeof(float)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(b, b_data.data(), N * K * sizeof(float)).isOk());

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	// Force Fp32: the Auto router otherwise picks a bf16 path that fails the
	// 1e-3 correctness tolerance on non-integer data.
	auto c = oa::FnMatrix::matMulNt(a, b, oa::MatMulPrecision::Fp32);

	std::vector<float> c_want(M * N);
	matMulNt(a_data, b_data, c_want, M, K, N);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(c, c_got.data(), M * N * sizeof(float)).isOk());

	for (oa::U32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-2f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Identity) {
	// A @ Iᵀ = A (identity is symmetric, so the [N,K] vs [K,N] distinction vanishes)
	constexpr oa::U32 N = 8;

	std::vector<float> a_data(N * N);
	std::vector<float> identity(N * N, 0.0f);
	for (oa::U32 i = 0; i < N; ++i) {
		for (oa::U32 j = 0; j < N; ++j) a_data[i * N + j] = static_cast<float>(i * N + j);
		identity[i * N + i] = 1.0f;
	}

	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(a_data.data()), N * N * sizeof(float)),
		oa::MatrixShape{N, N});
	auto I = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(identity.data()), N * N * sizeof(float)),
		oa::MatrixShape{N, N});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto c = oa::FnMatrix::matMulNt(a, I);

	std::vector<float> c_got(N * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(c, c_got.data(), N * N * sizeof(float)).isOk());

	for (oa::U32 i = 0; i < N * N; ++i) EXPECT_NEAR(c_got[i], a_data[i], 1e-4f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Zeros) {
	// A @ 0ᵀ = 0
	constexpr oa::U32 M = 4, K = 6, N = 5;

	auto a = oa::FnMatrix::rand(oa::MatrixShape{M, K});
	auto zeros = oa::FnMatrix::zeros(oa::MatrixShape{N, K});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto c = oa::FnMatrix::matMulNt(a, zeros);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(c, c_got.data(), M * N * sizeof(float)).isOk());

	for (oa::U32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], 0.0f, 1e-6f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, Bmm_Batch) {
	// Batched standard matmul is oa::FnMatrix::bmm: A[B,M,K] @ B[B,K,N] = [B,M,N].
	// (oa::FnMatrix::matMulNt takes a single shared 2D B in [N,K] layout — it does
	// not do per-batch products, so Bmm is the right op here.)
	constexpr oa::U32 BATCH = 2, M = 3, K = 4, N = 5;

	std::vector<float> a_data(BATCH * M * K);
	std::vector<float> b_data(BATCH * K * N);
	for (oa::U32 i = 0; i < BATCH * M * K; ++i) a_data[i] = static_cast<float>(i) * 0.1f;
	for (oa::U32 i = 0; i < BATCH * K * N; ++i) b_data[i] = static_cast<float>(i) * 0.2f;

	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(a_data.data()), BATCH * M * K * sizeof(float)),
		oa::MatrixShape{BATCH, M, K});
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(b_data.data()), BATCH * K * N * sizeof(float)),
		oa::MatrixShape{BATCH, K, N});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto c = oa::FnMatrix::bmm(a, b);

	std::vector<float> c_want(BATCH * M * N);
	for (oa::U32 batch = 0; batch < BATCH; ++batch) {
		std::vector<float> a_batch(a_data.begin() + batch * M * K, a_data.begin() + (batch + 1) * M * K);
		std::vector<float> b_batch(b_data.begin() + batch * K * N, b_data.begin() + (batch + 1) * K * N);
		std::vector<float> c_batch(M * N);
		matMulStd(a_batch, b_batch, c_batch, M, K, N);
		std::copy(c_batch.begin(), c_batch.end(), c_want.begin() + batch * M * N);
	}

	std::vector<float> c_got(BATCH * M * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(c, c_got.data(), BATCH * M * N * sizeof(float)).isOk());

	for (oa::U32 i = 0; i < BATCH * M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-3f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, BmmNt_BatchTails) {
	constexpr oa::U32 BATCH = 2, M = 15, K = 17, N = 9;
	std::vector<float> a_data(BATCH * M * K);
	std::vector<float> b_data(BATCH * N * K);
	for (oa::Usize i = 0; i < a_data.size(); ++i) {
		a_data[i] = static_cast<float>(static_cast<oa::I32>(i % 19U) - 9) * 0.03125F;
	}
	for (oa::Usize i = 0; i < b_data.size(); ++i) {
		b_data[i] = static_cast<float>(static_cast<oa::I32>(i % 23U) - 11) * 0.015625F;
	}

	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(a_data.data()),
			a_data.size() * sizeof(float)), oa::MatrixShape{BATCH, M, K});
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(b_data.data()),
			b_data.size() * sizeof(float)), oa::MatrixShape{BATCH, N, K});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto c = oa::FnMatrix::bmmNt(a, b);
	std::vector<float> c_got(BATCH * M * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		c, c_got.data(), c_got.size() * sizeof(float)).isOk());

	for (oa::U32 batch = 0; batch < BATCH; ++batch) {
		for (oa::U32 row = 0; row < M; ++row) {
			for (oa::U32 col = 0; col < N; ++col) {
				float expected = 0.0F;
				for (oa::U32 inner = 0; inner < K; ++inner) {
					expected += a_data[(batch * M + row) * K + inner]
						* b_data[(batch * N + col) * K + inner];
				}
				const oa::U32 index = (batch * M + row) * N + col;
				EXPECT_NEAR(c_got[index], expected, 1e-4F) << "i=" << index;
			}
		}
	}
}
