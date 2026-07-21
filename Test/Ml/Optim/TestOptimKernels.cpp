// Optimizer kernel correctness tests
// CPU reference implementations vs GPU kernels

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "../../OaTest.h"
#include <Oa/Ml/MuonRef.h>


// ============================================================================
// CPU Reference Implementations
// ============================================================================

void CpuSgd(std::vector<float> &weights, const std::vector<float> &grads, 
            OaU32 count, float lr, float weight_decay) {
	for (OaU32 i = 0; i < count; ++i) {
		float g = grads[i];
		if (weight_decay > 0.0f) {
			g += weight_decay * weights[i];
		}
		weights[i] -= lr * g;
	}
}

void CpuAdam(std::vector<float> &weights, const std::vector<float> &grads,
             std::vector<float> &m, std::vector<float> &v,
             OaU32 count, float lr, float beta1, float beta2, float eps, OaU32 step) {
	for (OaU32 i = 0; i < count; ++i) {
		float g = grads[i];
		
		// Update moments
		m[i] = beta1 * m[i] + (1.0f - beta1) * g;
		v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
		
		// Bias correction
		float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
		float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
		float m_hat = m[i] / bc1;
		float v_hat = v[i] / bc2;
		
		// Update weights
		weights[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
	}
}

void CpuAdamW(std::vector<float> &weights, const std::vector<float> &grads,
              std::vector<float> &m, std::vector<float> &v,
              OaU32 count, float lr, float beta1, float beta2, float eps, 
              float weight_decay, OaU32 step) {
	for (OaU32 i = 0; i < count; ++i) {
		float g = grads[i];
		
		// Decoupled weight decay
		weights[i] -= lr * weight_decay * weights[i];
		
		// Update moments
		m[i] = beta1 * m[i] + (1.0f - beta1) * g;
		v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
		
		// Bias correction
		float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
		float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
		float m_hat = m[i] / bc1;
		float v_hat = v[i] / bc2;
		
		// Update weights
		weights[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
	}
}

// ============================================================================
// Muon helpers (OaMuonRef CPU + MuonVector GPU kernel)
// ============================================================================

static double MuonOrthogonalityError(const std::vector<float>& M, OaU32 rows, OaU32 cols) {
	if (rows != cols) return 0.0;
	double err = 0.0;
	for (OaU32 i = 0; i < rows; ++i) {
		for (OaU32 j = 0; j < rows; ++j) {
			double dot = 0.0;
			for (OaU32 k = 0; k < cols; ++k) {
				dot += static_cast<double>(M[static_cast<size_t>(i) * cols + k])
					* static_cast<double>(M[static_cast<size_t>(j) * cols + k]);
			}
			const double target = (i == j) ? 1.0 : 0.0;
			const double d = dot - target;
			err += d * d;
		}
	}
	return std::sqrt(err / static_cast<double>(rows));
}

// ============================================================================
// Test Fixture
// ============================================================================

class OptimKernels : public ::testing::Test {
protected:
	// Pinned engine: default-constructed empty, initialized in place in SetUp.
	// Rt_ is a stable reference so all test bodies use `Rt_.` / `&Rt_` unchanged.
	OaUniquePtr<OaEngine> RtStorage_ = OaMakeUniquePtr<OaEngine>();
	OaEngine&             Rt_        = *RtStorage_;
	OaContext* SavedContext_ = nullptr;

	void SetUp() override {
		SavedContext_ = OaContext::GetDefaultPtr();
		ASSERT_TRUE(Rt_.InitInPlace({
			.AppName = "OptimKernels",
			.RegisterAsGlobal = false,
		}).IsOk());
		OaContext::SetDefault(&Rt_.GetContext());

		// Load embedded shaders (release mode) or from disk (debug mode)
		auto status = Rt_.EnsureAllEmbeddedLiboaPipelines();
		if (!status.IsOk()) {
			// Fallback to disk loading if embed failed
			Rt_.AddShaderSearchPath("spirv");
			Rt_.AddShaderSearchPath("../../spirv");
			Rt_.AddShaderSearchPath("Build/Release/spirv");
			OaTestLoadShaders(Rt_);
		}

		OA_LOG_INFO(OaLogComponent::Core, "GPU: %s", Rt_.Device.Info.Hardware.DeviceName.c_str());
	}

	void TearDown() override {
		OaContext::GetDefault().Clear();
		OaContext::SetDefault(SavedContext_);
		Rt_.Destroy();
	}

	// Helper: compute max relative error
	double ComputeMaxRelativeError(const std::vector<float> &ref, const std::vector<float> &gpu, OaU32 count) {
		double maxError = 0.0;
		for (OaU32 i = 0; i < count; ++i) {
			float r = ref[i];
			float g = gpu[i];
			double absError = std::abs(r - g);
			double relError = absError / std::max(std::abs(r), 1e-6f);
			maxError = std::max(maxError, relError);
		}
		return maxError;
	}

	// Helper: compare results with detailed logging
	void CompareResults(const std::vector<float> &ref, const std::vector<float> &gpu,
	                    OaU32 count, float tolerance, const char *testName = "") {
		double maxError = ComputeMaxRelativeError(ref, gpu, count);
		if (maxError >= tolerance) {
			// Find first mismatch for debugging
			for (OaU32 i = 0; i < std::min(count, 10u); ++i) {
				if (std::abs(ref[i] - gpu[i]) / std::max(std::abs(ref[i]), 1e-6f) > tolerance) {
					OA_LOG_ERROR(OaLogComponent::Core, "%s: First mismatch at [%u]: CPU=%.6f GPU=%.6f (rel_err=%.2e)",
					             testName, i, ref[i], gpu[i],
					             std::abs(ref[i] - gpu[i]) / std::max(std::abs(ref[i]), 1e-6f));
					break;
				}
			}
		}
		EXPECT_LT(maxError, tolerance) << testName << ": Max relative error: " << maxError;
	}
};

// ============================================================================
// SGD Tests
// ============================================================================

TEST_VK(OptimKernels, Sgd) {
	auto testSgd = [this](OaU32 count, float lr, float weight_decay) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		
		std::vector<float> weights(count), grads(count);
		for (auto &v : weights) v = dist(rng);
		for (auto &v : grads) v = dist(rng);
		
		std::vector<float> weights_ref = weights;
		CpuSgd(weights_ref, grads, count, lr, weight_decay);
		
		auto resultWeights = Rt_.AllocBufferBar(count * 4);
		auto resultGrads = Rt_.AllocBufferBar(count * 4);
		ASSERT_TRUE(resultWeights.IsOk() && resultGrads.IsOk());
		
		auto bufWeights = std::move(resultWeights).GetValue();
		auto bufGrads = std::move(resultGrads).GetValue();
		
		std::memcpy(bufWeights.MappedPtr, weights.data(), count * 4);
		std::memcpy(bufGrads.MappedPtr, grads.data(), count * 4);
		
		struct { OaU32 count; float lr; float weight_decay; } pc = { count, lr, weight_decay };
		OaVkBuffer bufs[] = {bufWeights, bufGrads};
		OaU32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Sgd", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> weights_gpu(count);
		std::memcpy(weights_gpu.data(), bufWeights.MappedPtr, count * 4);
		
		CompareResults(weights_ref, weights_gpu, count, 1e-6f, "Sgd");
		
		Rt_.FreeBuffer(bufWeights);
		Rt_.FreeBuffer(bufGrads);
	};
	
	testSgd(1024, 0.01f, 0.0f);      // No weight decay
	testSgd(4096, 0.001f, 0.0001f);  // With weight decay
	testSgd(16384, 0.1f, 0.01f);     // Large step
}

// ============================================================================
// Adam Tests
// ============================================================================

TEST_VK(OptimKernels, Adam) {
	auto testAdam = [this](OaU32 count, float lr, OaU32 step) {
		constexpr float beta1 = 0.9f;
		constexpr float beta2 = 0.999f;
		constexpr float eps = 1e-8f;
		
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		
		std::vector<float> weights(count), grads(count), m(count), v(count);
		for (auto &val : weights) val = dist(rng);
		for (auto &val : grads) val = dist(rng);
		for (auto &val : m) val = dist(rng) * 0.1f;  // Small initial moments
		for (auto &val : v) val = dist(rng) * 0.1f;
		
		std::vector<float> weights_ref = weights;
		std::vector<float> m_ref = m;
		std::vector<float> v_ref = v;
		CpuAdam(weights_ref, grads, m_ref, v_ref, count, lr, beta1, beta2, eps, step);
		
		auto resultWeights = Rt_.AllocBufferBar(count * 4);
		auto resultGrads = Rt_.AllocBufferBar(count * 4);
		auto resultM = Rt_.AllocBufferBar(count * 4);
		auto resultV = Rt_.AllocBufferBar(count * 4);
		ASSERT_TRUE(resultWeights.IsOk() && resultGrads.IsOk() && resultM.IsOk() && resultV.IsOk());
		
		auto bufWeights = std::move(resultWeights).GetValue();
		auto bufGrads = std::move(resultGrads).GetValue();
		auto bufM = std::move(resultM).GetValue();
		auto bufV = std::move(resultV).GetValue();
		
		std::memcpy(bufWeights.MappedPtr, weights.data(), count * 4);
		std::memcpy(bufGrads.MappedPtr, grads.data(), count * 4);
		std::memcpy(bufM.MappedPtr, m.data(), count * 4);
		std::memcpy(bufV.MappedPtr, v.data(), count * 4);
		
		struct { OaU32 count; float lr; float beta1; float beta2; float eps; OaU32 step; } pc = 
			{ count, lr, beta1, beta2, eps, step };
		OaVkBuffer bufs[] = {bufWeights, bufGrads, bufM, bufV};
		OaU32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Adam", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> weights_gpu(count), m_gpu(count), v_gpu(count);
		std::memcpy(weights_gpu.data(), bufWeights.MappedPtr, count * 4);
		std::memcpy(m_gpu.data(), bufM.MappedPtr, count * 4);
		std::memcpy(v_gpu.data(), bufV.MappedPtr, count * 4);
		
		// Adam uses pow/sqrt - relaxed tolerance for transcendental error accumulation
		CompareResults(weights_ref, weights_gpu, count, 1e-3f, "Adam_weights");
		CompareResults(m_ref, m_gpu, count, 1e-3f, "Adam_m");
		CompareResults(v_ref, v_gpu, count, 1e-3f, "Adam_v");
		
		Rt_.FreeBuffer(bufWeights);
		Rt_.FreeBuffer(bufGrads);
		Rt_.FreeBuffer(bufM);
		Rt_.FreeBuffer(bufV);
	};
	
	testAdam(1024, 0.001f, 1);    // First step (bias correction matters)
	testAdam(4096, 0.001f, 10);   // Later step
	testAdam(16384, 0.0001f, 100); // Many steps
}

// ============================================================================
// AdamW Tests
// ============================================================================

TEST_VK(OptimKernels, Adamw) {
	auto testAdamW = [this](OaU32 count, float lr, float weight_decay, OaU32 step) {
		constexpr float beta1 = 0.9f;
		constexpr float beta2 = 0.999f;
		constexpr float eps = 1e-8f;
		
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		
		std::vector<float> weights(count), grads(count), m(count), v(count);
		for (auto &val : weights) val = dist(rng);
		for (auto &val : grads) val = dist(rng);
		for (auto &val : m) val = dist(rng) * 0.1f;
		for (auto &val : v) val = dist(rng) * 0.1f;
		
		std::vector<float> weights_ref = weights;
		std::vector<float> m_ref = m;
		std::vector<float> v_ref = v;
		CpuAdamW(weights_ref, grads, m_ref, v_ref, count, lr, beta1, beta2, eps, weight_decay, step);
		
		auto resultWeights = Rt_.AllocBufferBar(count * 4);
		auto resultGrads = Rt_.AllocBufferBar(count * 4);
		auto resultM = Rt_.AllocBufferBar(count * 4);
		auto resultV = Rt_.AllocBufferBar(count * 4);
		ASSERT_TRUE(resultWeights.IsOk() && resultGrads.IsOk() && resultM.IsOk() && resultV.IsOk());
		
		auto bufWeights = std::move(resultWeights).GetValue();
		auto bufGrads = std::move(resultGrads).GetValue();
		auto bufM = std::move(resultM).GetValue();
		auto bufV = std::move(resultV).GetValue();
		
		std::memcpy(bufWeights.MappedPtr, weights.data(), count * 4);
		std::memcpy(bufGrads.MappedPtr, grads.data(), count * 4);
		std::memcpy(bufM.MappedPtr, m.data(), count * 4);
		std::memcpy(bufV.MappedPtr, v.data(), count * 4);
		
		struct { OaU32 count; float lr; float beta1; float beta2; float eps; float weight_decay; OaU32 step; } pc = 
			{ count, lr, beta1, beta2, eps, weight_decay, step };
		OaVkBuffer bufs[] = {bufWeights, bufGrads, bufM, bufV};
		OaU32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Adamw", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> weights_gpu(count), m_gpu(count), v_gpu(count);
		std::memcpy(weights_gpu.data(), bufWeights.MappedPtr, count * 4);
		std::memcpy(m_gpu.data(), bufM.MappedPtr, count * 4);
		std::memcpy(v_gpu.data(), bufV.MappedPtr, count * 4);
		
		// AdamW uses pow/sqrt - relaxed tolerance for transcendental error accumulation
		CompareResults(weights_ref, weights_gpu, count, 1e-3f, "AdamW_weights");
		CompareResults(m_ref, m_gpu, count, 1e-3f, "AdamW_m");
		CompareResults(v_ref, v_gpu, count, 1e-3f, "AdamW_v");

		// Replay-safe variant: mutable hyperparameters and step live in a
		// device-visible state buffer, leaving the recorded push payload stable.
		std::memcpy(bufWeights.MappedPtr, weights.data(), count * 4);
		std::memcpy(bufM.MappedPtr, m.data(), count * 4);
		std::memcpy(bufV.MappedPtr, v.data(), count * 4);
		auto resultState = Rt_.AllocBufferBar(6 * sizeof(OaU32));
		ASSERT_TRUE(resultState.IsOk());
		auto bufState = std::move(resultState).GetValue();
		OaU32 state[6] = {step, 0, 0, 0, 0, 0};
		const OaF32 scalars[] = {lr, beta1, beta2, eps, weight_decay};
		std::memcpy(state + 1, scalars, sizeof(scalars));
		std::memcpy(bufState.MappedPtr, state, sizeof(state));
		struct { OaU32 count; } graphPc = {count};
		OaVkBuffer graphBufs[] = {bufWeights, bufGrads, bufM, bufV, bufState};
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "AdamwGraph", graphBufs,
			&graphPc, sizeof(graphPc), groups).IsOk());

		std::memcpy(weights_gpu.data(), bufWeights.MappedPtr, count * 4);
		std::memcpy(m_gpu.data(), bufM.MappedPtr, count * 4);
		std::memcpy(v_gpu.data(), bufV.MappedPtr, count * 4);
		CompareResults(weights_ref, weights_gpu, count, 1e-3f, "AdamWGraph_weights");
		CompareResults(m_ref, m_gpu, count, 1e-3f, "AdamWGraph_m");
		CompareResults(v_ref, v_gpu, count, 1e-3f, "AdamWGraph_v");
		
		Rt_.FreeBuffer(bufWeights);
		Rt_.FreeBuffer(bufGrads);
		Rt_.FreeBuffer(bufM);
		Rt_.FreeBuffer(bufV);
		Rt_.FreeBuffer(bufState);
	};
	
	testAdamW(1024, 0.001f, 0.01f, 1);     // First step with weight decay
	testAdamW(4096, 0.001f, 0.01f, 10);    // Later step
	testAdamW(16384, 0.0001f, 0.001f, 100); // Many steps, small decay
}

// ============================================================================
// Muon Tests
// ============================================================================

TEST(MuonRef, NewtonSchulz5SquareOrthogonality) {
	std::mt19937 rng(7);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	constexpr OaU32 kDim = 16;
	std::vector<float> input(static_cast<size_t>(kDim) * kDim);
	for (float& v : input) v = dist(rng);

	std::vector<float> ortho(input.size());
	OaMuonRef::NewtonSchulz5(ortho.data(), input.data(), kDim, kDim, 5, 1e-7f);

	EXPECT_LT(MuonOrthogonalityError(ortho, kDim, kDim), 0.35)
		<< "NS5 should approximate orthogonality on square matrices";
}

TEST(MuonRef, NewtonSchulz5TallMatchesWide) {
	std::mt19937 rng(11);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	constexpr OaU32 kRows = 8;
	constexpr OaU32 kCols = 32;
	std::vector<float> tall(static_cast<size_t>(kRows) * kCols);
	std::vector<float> wide(static_cast<size_t>(kCols) * kRows);
	for (OaU32 r = 0; r < kRows; ++r) {
		for (OaU32 c = 0; c < kCols; ++c) {
			const float v = dist(rng);
			tall[static_cast<size_t>(r) * kCols + c] = v;
			wide[static_cast<size_t>(c) * kRows + r] = v;
		}
	}

	std::vector<float> outTall(tall.size());
	std::vector<float> outWide(wide.size());
	OaMuonRef::NewtonSchulz5(outTall.data(), tall.data(), kRows, kCols, 5, 1e-7f);
	OaMuonRef::NewtonSchulz5(outWide.data(), wide.data(), kCols, kRows, 5, 1e-7f);

	for (OaU32 r = 0; r < kRows; ++r) {
		for (OaU32 c = 0; c < kCols; ++c) {
			const float a = outTall[static_cast<size_t>(r) * kCols + c];
			const float b = outWide[static_cast<size_t>(c) * kRows + r];
			EXPECT_NEAR(a, b, 1e-4f) << "r=" << r << " c=" << c;
		}
	}
}

TEST(MuonRef, RoutingSegments) {
	OaParameter matrixW;
	matrixW.RequiresGrad = true;
	matrixW.Data = OaMatrix(OaMatrixShape{8, 4}, 0.0f);

	EXPECT_FALSE(OaMuonRef::IsMuonMatrixParam("embed.weight", matrixW));
	EXPECT_FALSE(OaMuonRef::IsMuonMatrixParam("head.0.weight", matrixW));
	EXPECT_FALSE(OaMuonRef::IsMuonMatrixParam("blocks.0.gate.bias", matrixW));
	EXPECT_TRUE(OaMuonRef::IsMuonMatrixParam("blocks.0.gate.weight", matrixW));
	EXPECT_TRUE(OaMuonRef::IsMuonMatrixParam("blocks.1.down.weight", matrixW));

	OaParameter vectorB;
	vectorB.RequiresGrad = true;
	vectorB.Data = OaMatrix(OaMatrixShape{8}, 0.0f);
	EXPECT_FALSE(OaMuonRef::IsMuonMatrixParam("blocks.0.gate.bias", vectorB));

	OaNamedParameter named[] = {
		{"embed.weight", &matrixW},
		{"blocks.0.gate.weight", &matrixW},
		{"head.0.weight", &matrixW},
	};
	const OaMuonRef::OaOfficialMuonSplit split = OaMuonRef::SplitOfficialRouting(named);
	EXPECT_EQ(split.Muon.Size(), 1u);
	EXPECT_EQ(split.AdamW.Size(), 2u);
}

TEST(MuonRef, MatrixStepDeterministic) {
	constexpr OaU32 kRows = 4;
	constexpr OaU32 kCols = 8;
	const OaU32 count = kRows * kCols;
	std::vector<float> w(count), g(count), m(count, 0.0f);
	for (OaU32 i = 0; i < count; ++i) {
		w[i] = 0.01f * static_cast<float>(i);
		g[i] = 0.001f * static_cast<float>((i + 3) % 7);
	}

	std::vector<float> wRef = w;
	std::vector<float> mRef = m;
	OaMuonRef::MatrixStep(wRef.data(), mRef.data(), g.data(), kRows, kCols,
		0.01f, 0.95f, 0.1f, 1e-7f, 5);

	EXPECT_NE(wRef, w);
	EXPECT_NE(mRef, m);
	for (float v : wRef) {
		EXPECT_TRUE(std::isfinite(v));
	}
}

TEST_VK(OptimKernels, MuonVector) {
	auto testMuonVector = [this](OaU32 count, float lr, float beta, float weight_decay) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

		std::vector<float> weights(count), grads(count), momentum(count, 0.0f);
		for (auto& v : weights) v = dist(rng);
		for (auto& v : grads) v = dist(rng);

		std::vector<float> weightsRef = weights;
		std::vector<float> momentumRef = momentum;
		OaMuonRef::VectorStep(weightsRef.data(), momentumRef.data(), grads.data(),
			count, lr, beta, weight_decay);

		auto resultWeights = Rt_.AllocBufferBar(count * 4);
		auto resultGrads = Rt_.AllocBufferBar(count * 4);
		auto resultMomentum = Rt_.AllocBufferBar(count * 4);
		ASSERT_TRUE(resultWeights.IsOk() && resultGrads.IsOk() && resultMomentum.IsOk());

		auto bufWeights = std::move(resultWeights).GetValue();
		auto bufGrads = std::move(resultGrads).GetValue();
		auto bufMomentum = std::move(resultMomentum).GetValue();

		std::memcpy(bufWeights.MappedPtr, weights.data(), count * 4);
		std::memcpy(bufGrads.MappedPtr, grads.data(), count * 4);
		std::memcpy(bufMomentum.MappedPtr, momentum.data(), count * 4);

		struct { OaU32 count; float lr; float beta; float weight_decay; } pc =
			{ count, lr, beta, weight_decay };
		OaVkBuffer bufs[] = { bufWeights, bufGrads, bufMomentum };
		OaU32 groups = (count + 255) / 256;

		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "MuonVector", bufs, &pc, sizeof(pc), groups).IsOk());

		std::vector<float> weightsGpu(count), momentumGpu(count);
		std::memcpy(weightsGpu.data(), bufWeights.MappedPtr, count * 4);
		std::memcpy(momentumGpu.data(), bufMomentum.MappedPtr, count * 4);

		CompareResults(weightsRef, weightsGpu, count, 1e-5f, "MuonVector_weights");
		CompareResults(momentumRef, momentumGpu, count, 1e-5f, "MuonVector_momentum");

		Rt_.FreeBuffer(bufWeights);
		Rt_.FreeBuffer(bufGrads);
		Rt_.FreeBuffer(bufMomentum);
	};

	testMuonVector(1024, 0.01f, 0.95f, 0.1f);
	testMuonVector(4097, 0.001f, 0.95f, 0.01f);
}

TEST_VK(OptimKernels, MuonMatrixPipelineGpu) {
	// GPU kernel chain (Nesterov → Normalize → Apply) with CPU NS5 reference.
	// Production FnOptim::MuonStep uses OaFnMatrix MatMul for NS5; this test
	// validates the dispatch kernels against OaMuonRef::MatrixStep.
	constexpr OaU32 kRows = 4;
	constexpr OaU32 kCols = 8;
	const OaU32 count = kRows * kCols;
	constexpr float kLr = 0.01f;
	constexpr float kBeta = 0.95f;
	constexpr float kWd = 0.1f;
	constexpr float kEps = 1e-7f;
	constexpr OaI32 kNs5 = 5;

	std::mt19937 rng(23);
	std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

	std::vector<float> weights(count), grads(count), momentum(count, 0.0f);
	for (float& v : weights) v = dist(rng);
	for (float& v : grads) v = dist(rng);

	std::vector<float> weightsRef = weights;
	std::vector<float> momentumRef = momentum;
	OaMuonRef::MatrixStep(weightsRef.data(), momentumRef.data(), grads.data(),
		kRows, kCols, kLr, kBeta, kWd, kEps, kNs5);

	auto rw = Rt_.AllocBufferBar(count * sizeof(float));
	auto rg = Rt_.AllocBufferBar(count * sizeof(float));
	auto rm = Rt_.AllocBufferBar(count * sizeof(float));
	auto rUpdate = Rt_.AllocBufferBar(count * sizeof(float));
	auto rNorm = Rt_.AllocBufferBar(sizeof(float));
	auto rOrtho = Rt_.AllocBufferBar(count * sizeof(float));
	ASSERT_TRUE(rw.IsOk() && rg.IsOk() && rm.IsOk() && rUpdate.IsOk() && rNorm.IsOk() && rOrtho.IsOk());

	auto bufW = std::move(rw).GetValue();
	auto bufG = std::move(rg).GetValue();
	auto bufM = std::move(rm).GetValue();
	auto bufUpdate = std::move(rUpdate).GetValue();
	auto bufNorm = std::move(rNorm).GetValue();
	auto bufOrtho = std::move(rOrtho).GetValue();

	std::memcpy(bufW.MappedPtr, weights.data(), count * sizeof(float));
	std::memcpy(bufG.MappedPtr, grads.data(), count * sizeof(float));
	std::memset(bufM.MappedPtr, 0, count * sizeof(float));

	const OaU32 groups = (count + 255) / 256;
	{
		struct MuonNesterovPush { OaU32 Count; OaF32 Beta; } push{count, kBeta};
		OaVkBuffer bufs[] = {bufG, bufM, bufUpdate};
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "MuonNesterov", bufs, &push, sizeof(push), groups).IsOk());
	}

	std::vector<float> nesterov(count);
	std::memcpy(nesterov.data(), bufUpdate.MappedPtr, count * sizeof(float));
	std::vector<float> momentumGpu(count);
	std::memcpy(momentumGpu.data(), bufM.MappedPtr, count * sizeof(float));

	{
		struct MuonNormalizePush { OaU32 Rows; OaU32 Cols; OaF32 Eps; } push{kRows, kCols, kEps};
		OaVkBuffer bufs[] = {bufUpdate, bufUpdate, bufNorm};
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "MuonNormalize", bufs, &push, sizeof(push), 1).IsOk());
	}

	std::vector<float> ortho(count);
	OaMuonRef::NewtonSchulz5(ortho.data(), nesterov.data(), kRows, kCols, kNs5, kEps);
	std::memcpy(bufOrtho.MappedPtr, ortho.data(), count * sizeof(float));

	const OaF32 moonshotScale = OaMuonRef::MoonshotScale(kRows, kCols);
	{
		struct MuonApplyPush {
			OaU32 Count;
			OaF32 Lr;
			OaF32 WeightDecay;
			OaF32 MoonshotScale;
		} push{count, kLr, kWd, moonshotScale};
		OaVkBuffer bufs[] = {bufW, bufOrtho};
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "MuonApply", bufs, &push, sizeof(push), groups).IsOk());
	}

	std::vector<float> weightsGpu(count);
	std::memcpy(weightsGpu.data(), bufW.MappedPtr, count * sizeof(float));

	CompareResults(weightsRef, weightsGpu, count, 5e-2f, "MuonMatrixPipelineGpu_weights");
	CompareResults(momentumRef, momentumGpu, count, 1e-4f, "MuonMatrixPipelineGpu_momentum");

	Rt_.FreeBuffer(bufW);
	Rt_.FreeBuffer(bufG);
	Rt_.FreeBuffer(bufM);
	Rt_.FreeBuffer(bufUpdate);
	Rt_.FreeBuffer(bufNorm);
	Rt_.FreeBuffer(bufOrtho);
}

TEST_VK(OptimKernels, MuonNesterov) {
	constexpr OaU32 count = 512;
	constexpr float beta = 0.95f;

	std::mt19937 rng(19);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<float> grads(count), momentum(count, 0.0f);
	for (float& v : grads) v = dist(rng);

	std::vector<float> updateRef(count), momentumRef = momentum;
	for (OaU32 i = 0; i < count; ++i) {
		const float g = grads[i];
		const float mNew = beta * momentumRef[i] + (1.0f - beta) * g;
		updateRef[i] = (1.0f - beta) * g + beta * mNew;
		momentumRef[i] = mNew;
	}

	auto resultGrad = Rt_.AllocBufferBar(count * 4);
	auto resultMom = Rt_.AllocBufferBar(count * 4);
	auto resultUpdate = Rt_.AllocBufferBar(count * 4);
	ASSERT_TRUE(resultGrad.IsOk() && resultMom.IsOk() && resultUpdate.IsOk());

	auto bufGrad = std::move(resultGrad).GetValue();
	auto bufMom = std::move(resultMom).GetValue();
	auto bufUpdate = std::move(resultUpdate).GetValue();

	std::memcpy(bufGrad.MappedPtr, grads.data(), count * 4);
	std::memset(bufMom.MappedPtr, 0, count * 4);

	struct { OaU32 count; float beta; } pc = { count, beta };
	OaVkBuffer bufs[] = { bufGrad, bufMom, bufUpdate };
	OaU32 groups = (count + 255) / 256;

	ASSERT_TRUE(OaVkDispatch::Run(Rt_, "MuonNesterov", bufs, &pc, sizeof(pc), groups).IsOk());

	std::vector<float> updateGpu(count), momentumGpu(count);
	std::memcpy(updateGpu.data(), bufUpdate.MappedPtr, count * 4);
	std::memcpy(momentumGpu.data(), bufMom.MappedPtr, count * 4);

	CompareResults(updateRef, updateGpu, count, 1e-5f, "MuonNesterov_update");
	CompareResults(momentumRef, momentumGpu, count, 1e-5f, "MuonNesterov_momentum");

	Rt_.FreeBuffer(bufGrad);
	Rt_.FreeBuffer(bufMom);
	Rt_.FreeBuffer(bufUpdate);
}

TEST_VK(OptimKernels, AdamwMany4) {
	constexpr float lr = 0.001f;
	constexpr float beta1 = 0.9f;
	constexpr float beta2 = 0.999f;
	constexpr float eps = 1e-8f;
	constexpr float weightDecay = 0.01f;
	constexpr OaU32 step = 7;

	const OaU32 counts[] = {64 * 32, 32 * 16, 64, 32};
	constexpr OaU32 kParams = 4;

	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<float> weights[kParams], grads[kParams], m[kParams], v[kParams];
	std::vector<float> weightsRef[kParams], mRef[kParams], vRef[kParams];
	for (OaU32 p = 0; p < kParams; ++p) {
		const OaU32 count = counts[p];
		weights[p].resize(count);
		grads[p].resize(count);
		m[p].resize(count);
		v[p].resize(count);
		for (auto& val : weights[p]) val = dist(rng);
		for (auto& val : grads[p]) val = dist(rng);
		for (auto& val : m[p]) val = dist(rng) * 0.1f;
		for (auto& val : v[p]) val = std::abs(dist(rng)) * 0.1f;
		weightsRef[p] = weights[p];
		mRef[p] = m[p];
		vRef[p] = v[p];
		CpuAdamW(weightsRef[p], grads[p], mRef[p], vRef[p], count,
			lr, beta1, beta2, eps, weightDecay, step);
	}

	OaVkBuffer wBuf[kParams], gBuf[kParams], mBuf[kParams], vBuf[kParams];
	for (OaU32 p = 0; p < kParams; ++p) {
		const OaU32 count = counts[p];
		auto rw = Rt_.AllocBufferBar(count * sizeof(float));
		auto rg = Rt_.AllocBufferBar(count * sizeof(float));
		auto rm = Rt_.AllocBufferBar(count * sizeof(float));
		auto rv = Rt_.AllocBufferBar(count * sizeof(float));
		ASSERT_TRUE(rw.IsOk() && rg.IsOk() && rm.IsOk() && rv.IsOk());
		wBuf[p] = std::move(rw).GetValue();
		gBuf[p] = std::move(rg).GetValue();
		mBuf[p] = std::move(rm).GetValue();
		vBuf[p] = std::move(rv).GetValue();
		std::memcpy(wBuf[p].MappedPtr, weights[p].data(), count * sizeof(float));
		std::memcpy(gBuf[p].MappedPtr, grads[p].data(), count * sizeof(float));
		std::memcpy(mBuf[p].MappedPtr, m[p].data(), count * sizeof(float));
		std::memcpy(vBuf[p].MappedPtr, v[p].data(), count * sizeof(float));
	}

	struct Push {
		OaU32 Count0;
		OaU32 Count1;
		OaU32 Count2;
		OaU32 Count3;
		OaF32 Lr;
		OaF32 Beta1;
		OaF32 Beta2;
		OaF32 Eps;
		OaF32 WeightDecay;
		OaU32 Step;
	};
	const OaU32 maxCount = *std::max_element(counts, counts + kParams);
	Push pc{
		.Count0 = counts[0], .Count1 = counts[1], .Count2 = counts[2], .Count3 = counts[3],
		.Lr = lr, .Beta1 = beta1, .Beta2 = beta2, .Eps = eps, .WeightDecay = weightDecay, .Step = step
	};
	OaVkBuffer bufs[] = {
		wBuf[0], gBuf[0], mBuf[0], vBuf[0],
		wBuf[1], gBuf[1], mBuf[1], vBuf[1],
		wBuf[2], gBuf[2], mBuf[2], vBuf[2],
		wBuf[3], gBuf[3], mBuf[3], vBuf[3],
	};
	ASSERT_TRUE(OaVkDispatch::Run(Rt_, "AdamwMany4", bufs, &pc, sizeof(pc),
		(maxCount + 255U) / 256U).IsOk());

	for (OaU32 p = 0; p < kParams; ++p) {
		const OaU32 count = counts[p];
		std::vector<float> wGpu(count), mGpu(count), vGpu(count);
		std::memcpy(wGpu.data(), wBuf[p].MappedPtr, count * sizeof(float));
		std::memcpy(mGpu.data(), mBuf[p].MappedPtr, count * sizeof(float));
		std::memcpy(vGpu.data(), vBuf[p].MappedPtr, count * sizeof(float));
		CompareResults(weightsRef[p], wGpu, count, 2e-3f, "AdamwMany4_weights");
		CompareResults(mRef[p], mGpu, count, 2e-3f, "AdamwMany4_m");
		CompareResults(vRef[p], vGpu, count, 2e-3f, "AdamwMany4_v");
	}

	// Reset inputs and verify the replay-safe fused-four variant against the
	// same CPU oracle.
	for (OaU32 p = 0; p < kParams; ++p) {
		const OaU32 count = counts[p];
		std::memcpy(wBuf[p].MappedPtr, weights[p].data(), count * sizeof(float));
		std::memcpy(mBuf[p].MappedPtr, m[p].data(), count * sizeof(float));
		std::memcpy(vBuf[p].MappedPtr, v[p].data(), count * sizeof(float));
	}
	auto stateResult = Rt_.AllocBufferBar(6 * sizeof(OaU32));
	ASSERT_TRUE(stateResult.IsOk());
	auto stateBuf = std::move(stateResult).GetValue();
	OaU32 state[6] = {step, 0, 0, 0, 0, 0};
	const OaF32 stateScalars[] = {lr, beta1, beta2, eps, weightDecay};
	std::memcpy(state + 1, stateScalars, sizeof(stateScalars));
	std::memcpy(stateBuf.MappedPtr, state, sizeof(state));
	struct GraphPush {
		OaU32 Count0;
		OaU32 Count1;
		OaU32 Count2;
		OaU32 Count3;
	} graphPc{counts[0], counts[1], counts[2], counts[3]};
	OaVkBuffer graphBufs[] = {
		wBuf[0], gBuf[0], mBuf[0], vBuf[0],
		wBuf[1], gBuf[1], mBuf[1], vBuf[1],
		wBuf[2], gBuf[2], mBuf[2], vBuf[2],
		wBuf[3], gBuf[3], mBuf[3], vBuf[3],
		stateBuf,
	};
	ASSERT_TRUE(OaVkDispatch::Run(Rt_, "AdamwMany4Graph", graphBufs,
		&graphPc, sizeof(graphPc), (maxCount + 255U) / 256U).IsOk());
	for (OaU32 p = 0; p < kParams; ++p) {
		const OaU32 count = counts[p];
		std::vector<float> wGpu(count), mGpu(count), vGpu(count);
		std::memcpy(wGpu.data(), wBuf[p].MappedPtr, count * sizeof(float));
		std::memcpy(mGpu.data(), mBuf[p].MappedPtr, count * sizeof(float));
		std::memcpy(vGpu.data(), vBuf[p].MappedPtr, count * sizeof(float));
		CompareResults(weightsRef[p], wGpu, count, 2e-3f, "AdamwMany4Graph_weights");
		CompareResults(mRef[p], mGpu, count, 2e-3f, "AdamwMany4Graph_m");
		CompareResults(vRef[p], vGpu, count, 2e-3f, "AdamwMany4Graph_v");
	}

	for (OaU32 p = 0; p < kParams; ++p) {
		Rt_.FreeBuffer(wBuf[p]);
		Rt_.FreeBuffer(gBuf[p]);
		Rt_.FreeBuffer(mBuf[p]);
		Rt_.FreeBuffer(vBuf[p]);
	}
	Rt_.FreeBuffer(stateBuf);
}
