// Minimal MatMul test to debug GEMM kernel issues

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <vector>
#include <iostream>

static OaComputeEngine* GRt = nullptr;

class TestFnMatrixBlasSimple : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixBlasSimple";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
	}
};

TEST_VK(TestFnMatrixBlasSimple, MatMul_2x2_Manual) {
	// OA contract: OaFnMatrix::MatMulNt(A, B) takes B in [N,K] layout and computes
	// C = A @ Bᵀ  (the weight convention shared with Linear / attention).
	// To verify a known *standard* product A_std @ B_std = [[19,22],[43,50]] with
	//   A_std = [[1,2],[3,4]], B_std = [[5,6],[7,8]]
	// we pass B = (B_std)ᵀ = [[5,7],[6,8]] so that A @ Bᵀ == A_std @ B_std.
	std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};  // A_std row-major: [1,2; 3,4]
	std::vector<float> b_data = {5.0f, 7.0f, 6.0f, 8.0f};  // B = (B_std)ᵀ, [N,K] layout
	
	auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(a_data.data()), 4 * sizeof(float)),
		OaMatrixShape{2, 2});
	auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(b_data.data()), 4 * sizeof(float)),
		OaMatrixShape{2, 2});
	
	std::cout << "Input A (row-major): [" << a_data[0] << ", " << a_data[1] << "; "
	          << a_data[2] << ", " << a_data[3] << "]\n";
	std::cout << "Input B (row-major): [" << b_data[0] << ", " << b_data[1] << "; "
	          << b_data[2] << ", " << b_data[3] << "]\n";
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto c = OaFnMatrix::MatMulNt(a, b);
	
	std::vector<float> c_got(4);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, c_got.data(), 4 * sizeof(float)).IsOk());
	
	std::cout << "Output C (row-major): [" << c_got[0] << ", " << c_got[1] << "; "
	          << c_got[2] << ", " << c_got[3] << "]\n";
	std::cout << "Expected: [19, 22; 43, 50]\n";
	
	// Expected: C[0,0] = 1*5 + 2*7 = 19
	EXPECT_NEAR(c_got[0], 19.0f, 1e-4f) << "C[0,0]";
	// Expected: C[0,1] = 1*6 + 2*8 = 22
	EXPECT_NEAR(c_got[1], 22.0f, 1e-4f) << "C[0,1]";
	// Expected: C[1,0] = 3*5 + 4*7 = 43
	EXPECT_NEAR(c_got[2], 43.0f, 1e-4f) << "C[1,0]";
	// Expected: C[1,1] = 3*6 + 4*8 = 50
	EXPECT_NEAR(c_got[3], 50.0f, 1e-4f) << "C[1,1]";
}

TEST_VK(TestFnMatrixBlasSimple, MatMul_2x3_3x2) {
	// Standard product to verify: A_std[2,3] @ B_std[3,2] = [[58,64],[139,154]]
	//   A_std = [[1,2,3],[4,5,6]],  B_std = [[7,8],[9,10],[11,12]]
	// OA MatMul takes B in [N,K] layout and computes A @ Bᵀ, so we pass
	//   B = (B_std)ᵀ = [[7,9,11],[8,10,12]]  (shape [N=2, K=3]).
	std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	std::vector<float> b_data = {7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f};  // (B_std)ᵀ, [N,K]

	auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(a_data.data()), 6 * sizeof(float)),
		OaMatrixShape{2, 3});
	auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(b_data.data()), 6 * sizeof(float)),
		OaMatrixShape{2, 3});
	
	std::cout << "A[2,3]: [" << a_data[0] << "," << a_data[1] << "," << a_data[2] << "; "
	          << a_data[3] << "," << a_data[4] << "," << a_data[5] << "]\n";
	std::cout << "B[N=2,K=3] (B_std transposed): [" << b_data[0] << "," << b_data[1] << "," << b_data[2] << "; "
	          << b_data[3] << "," << b_data[4] << "," << b_data[5] << "]\n";
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto c = OaFnMatrix::MatMulNt(a, b);
	
	std::vector<float> c_got(4);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, c_got.data(), 4 * sizeof(float)).IsOk());
	
	std::cout << "Output C[2,2]: [" << c_got[0] << ", " << c_got[1] << "; "
	          << c_got[2] << ", " << c_got[3] << "]\n";
	std::cout << "Expected: [58, 64; 139, 154]\n";
	
	EXPECT_NEAR(c_got[0], 58.0f, 1e-4f) << "C[0,0] = 1*7+2*9+3*11";
	EXPECT_NEAR(c_got[1], 64.0f, 1e-4f) << "C[0,1] = 1*8+2*10+3*12";
	EXPECT_NEAR(c_got[2], 139.0f, 1e-4f) << "C[1,0] = 4*7+5*9+6*11";
	EXPECT_NEAR(c_got[3], 154.0f, 1e-4f) << "C[1,1] = 4*8+5*10+6*12";
}

