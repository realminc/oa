// Manual tests for Core/FnMatrix BLAS operations.
//
// IMPORTANT CONTRACT: OaFnMatrix::MatMulNt(A, B) takes B in [N,K] layout and
// computes  C = A @ Bᵀ  (out[m,n] = Σ_k A[m,k]·B[n,k]). This is the OA weight
// convention shared with Linear and attention — it is NOT the PyTorch-standard
// A @ B with B as [K,N]. Batched / standard A@B is OaFnMatrix::Bmm.
// The CPU reference MatMulNt below matches the MatMul contract exactly.

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>     // Bmm
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/RuntimeGlobal.h>
#include <vector>
#include <cmath>

static OaComputeEngine* GRt = nullptr;

class TestFnMatrixBlas : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixBlas";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
		OaRuntimeGlobal::SetRuntime(GRt);
	}

	// CPU reference for OaFnMatrix::MatMulNt — B is [N,K], C = A @ Bᵀ:
	//   C[m,n] = Σ_k A[m,k] · B[n,k]
	static void MatMulNt(const std::vector<float>& a, const std::vector<float>& b,
	                     std::vector<float>& c, OaU32 m, OaU32 k, OaU32 n) {
		for (OaU32 i = 0; i < m; ++i) {
			for (OaU32 j = 0; j < n; ++j) {
				float sum = 0.0f;
				for (OaU32 p = 0; p < k; ++p) {
					sum += a[i * k + p] * b[j * k + p];  // B[n,k] layout
				}
				c[i * n + j] = sum;
			}
		}
	}

	// Standard per-batch reference for OaFnMatrix::Bmm — B is [K,N], C = A @ B.
	static void MatMulStd(const std::vector<float>& a, const std::vector<float>& b,
	                      std::vector<float>& c, OaU32 m, OaU32 k, OaU32 n) {
		for (OaU32 i = 0; i < m; ++i) {
			for (OaU32 j = 0; j < n; ++j) {
				float sum = 0.0f;
				for (OaU32 p = 0; p < k; ++p) sum += a[i * k + p] * b[p * n + j];
				c[i * n + j] = sum;
			}
		}
	}
};

TEST_VK(TestFnMatrixBlas, MatMul_Square) {
	// [M=4,K=4] @ [N=4,K=4]ᵀ = [4,4]
	constexpr OaU32 M = 4, K = 4, N = 4;

	std::vector<float> a_data(M * K);
	std::vector<float> b_data(N * K);
	for (OaU32 i = 0; i < M * K; ++i) a_data[i] = static_cast<float>(i + 1);          // 1..16
	for (OaU32 i = 0; i < N * K; ++i) b_data[i] = static_cast<float>((i % K) + 1);    // 1,2,3,4,...

	auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(a_data.data()), M * K * sizeof(float)),
		OaMatrixShape{M, K});
	auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(b_data.data()), N * K * sizeof(float)),
		OaMatrixShape{N, K});

	OaContext::Scope ctx_scope(OaContext::GetDefault());
	// Force Fp32: the Auto router otherwise picks a bf16 path that fails the
	// 1e-3 correctness tolerance on non-integer data.
	auto c = OaFnMatrix::MatMulNt(a, b, OaContextMatMulPrecision::Fp32);

	std::vector<float> c_want(M * N);
	MatMulNt(a_data, b_data, c_want, M, K, N);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, c_got.data(), M * N * sizeof(float)).IsOk());

	for (OaU32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-3f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Rectangular) {
	// [M=3,K=5] @ [N=4,K=5]ᵀ = [3,4]
	constexpr OaU32 M = 3, K = 5, N = 4;

	std::vector<float> a_data(M * K);
	std::vector<float> b_data(N * K);
	for (OaU32 i = 0; i < M * K; ++i) a_data[i] = static_cast<float>(i + 1) * 0.1f;
	for (OaU32 i = 0; i < N * K; ++i) b_data[i] = static_cast<float>(i + 1) * 0.2f;

	auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(a_data.data()), M * K * sizeof(float)),
		OaMatrixShape{M, K});
	auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(b_data.data()), N * K * sizeof(float)),
		OaMatrixShape{N, K});

	OaContext::Scope ctx_scope(OaContext::GetDefault());
	// Force Fp32: the Auto router otherwise picks a bf16 path that fails the
	// 1e-3 correctness tolerance on non-integer data.
	auto c = OaFnMatrix::MatMulNt(a, b, OaContextMatMulPrecision::Fp32);

	std::vector<float> c_want(M * N);
	MatMulNt(a_data, b_data, c_want, M, K, N);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, c_got.data(), M * N * sizeof(float)).IsOk());

	for (OaU32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-3f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Vector) {
	// [M=4,K=8] @ [N=1,K=8]ᵀ = [4,1]  (matrix × vector)
	constexpr OaU32 M = 4, K = 8, N = 1;

	std::vector<float> a_data(M * K);
	std::vector<float> b_data(N * K);
	for (OaU32 i = 0; i < M * K; ++i) a_data[i] = static_cast<float>(i) * 0.1f;
	for (OaU32 i = 0; i < N * K; ++i) b_data[i] = static_cast<float>(i + 1);

	auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(a_data.data()), M * K * sizeof(float)),
		OaMatrixShape{M, K});
	auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(b_data.data()), N * K * sizeof(float)),
		OaMatrixShape{N, K});

	OaContext::Scope ctx_scope(OaContext::GetDefault());
	// Force Fp32: the Auto router otherwise picks a bf16 path that fails the
	// 1e-3 correctness tolerance on non-integer data.
	auto c = OaFnMatrix::MatMulNt(a, b, OaContextMatMulPrecision::Fp32);

	std::vector<float> c_want(M * N);
	MatMulNt(a_data, b_data, c_want, M, K, N);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, c_got.data(), M * N * sizeof(float)).IsOk());

	for (OaU32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-3f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Large) {
	// [M=64,K=128] @ [N=32,K=128]ᵀ = [64,32]
	constexpr OaU32 M = 64, K = 128, N = 32;

	auto a = OaFnMatrix::Rand(OaMatrixShape{M, K});
	auto b = OaFnMatrix::Rand(OaMatrixShape{N, K});

	std::vector<float> a_data(M * K);
	std::vector<float> b_data(N * K);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(a, a_data.data(), M * K * sizeof(float)).IsOk());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(b, b_data.data(), N * K * sizeof(float)).IsOk());

	OaContext::Scope ctx_scope(OaContext::GetDefault());
	// Force Fp32: the Auto router otherwise picks a bf16 path that fails the
	// 1e-3 correctness tolerance on non-integer data.
	auto c = OaFnMatrix::MatMulNt(a, b, OaContextMatMulPrecision::Fp32);

	std::vector<float> c_want(M * N);
	MatMulNt(a_data, b_data, c_want, M, K, N);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, c_got.data(), M * N * sizeof(float)).IsOk());

	for (OaU32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-2f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Identity) {
	// A @ Iᵀ = A (identity is symmetric, so the [N,K] vs [K,N] distinction vanishes)
	constexpr OaU32 N = 8;

	std::vector<float> a_data(N * N);
	std::vector<float> identity(N * N, 0.0f);
	for (OaU32 i = 0; i < N; ++i) {
		for (OaU32 j = 0; j < N; ++j) a_data[i * N + j] = static_cast<float>(i * N + j);
		identity[i * N + i] = 1.0f;
	}

	auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(a_data.data()), N * N * sizeof(float)),
		OaMatrixShape{N, N});
	auto I = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(identity.data()), N * N * sizeof(float)),
		OaMatrixShape{N, N});

	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto c = OaFnMatrix::MatMulNt(a, I);

	std::vector<float> c_got(N * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, c_got.data(), N * N * sizeof(float)).IsOk());

	for (OaU32 i = 0; i < N * N; ++i) EXPECT_NEAR(c_got[i], a_data[i], 1e-4f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, MatMul_Zeros) {
	// A @ 0ᵀ = 0
	constexpr OaU32 M = 4, K = 6, N = 5;

	auto a = OaFnMatrix::Rand(OaMatrixShape{M, K});
	auto zeros = OaFnMatrix::Zeros(OaMatrixShape{N, K});

	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto c = OaFnMatrix::MatMulNt(a, zeros);

	std::vector<float> c_got(M * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, c_got.data(), M * N * sizeof(float)).IsOk());

	for (OaU32 i = 0; i < M * N; ++i) EXPECT_NEAR(c_got[i], 0.0f, 1e-6f) << "i=" << i;
}

TEST_VK(TestFnMatrixBlas, Bmm_Batch) {
	// Batched standard matmul is OaFnMatrix::Bmm: A[B,M,K] @ B[B,K,N] = [B,M,N].
	// (OaFnMatrix::MatMulNt takes a single shared 2D B in [N,K] layout — it does
	// not do per-batch products, so Bmm is the right op here.)
	constexpr OaU32 BATCH = 2, M = 3, K = 4, N = 5;

	std::vector<float> a_data(BATCH * M * K);
	std::vector<float> b_data(BATCH * K * N);
	for (OaU32 i = 0; i < BATCH * M * K; ++i) a_data[i] = static_cast<float>(i) * 0.1f;
	for (OaU32 i = 0; i < BATCH * K * N; ++i) b_data[i] = static_cast<float>(i) * 0.2f;

	auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(a_data.data()), BATCH * M * K * sizeof(float)),
		OaMatrixShape{BATCH, M, K});
	auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(b_data.data()), BATCH * K * N * sizeof(float)),
		OaMatrixShape{BATCH, K, N});

	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto c = OaFnMatrix::Bmm(a, b);

	std::vector<float> c_want(BATCH * M * N);
	for (OaU32 batch = 0; batch < BATCH; ++batch) {
		std::vector<float> a_batch(a_data.begin() + batch * M * K, a_data.begin() + (batch + 1) * M * K);
		std::vector<float> b_batch(b_data.begin() + batch * K * N, b_data.begin() + (batch + 1) * K * N);
		std::vector<float> c_batch(M * N);
		MatMulStd(a_batch, b_batch, c_batch, M, K, N);
		std::copy(c_batch.begin(), c_batch.end(), c_want.begin() + batch * M * N);
	}

	std::vector<float> c_got(BATCH * M * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, c_got.data(), BATCH * M * N * sizeof(float)).IsOk());

	for (OaU32 i = 0; i < BATCH * M * N; ++i) EXPECT_NEAR(c_got[i], c_want[i], 1e-3f) << "i=" << i;
}

