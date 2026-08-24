// Optimizer kernel correctness tests
// CPU reference implementations vs GPU kernels

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "../../oaTest.h"
#include <oa/ml/fnOptim.h>


// ============================================================================
// CPU Reference Implementations
// ============================================================================

void cpuSgd(std::vector<float> &weights, const std::vector<float> &grads, 
            oa::U32 count, float lr, float weight_decay) {
	for (oa::U32 i = 0; i < count; ++i) {
		float g = grads[i];
		if (weight_decay > 0.0f) {
			g += weight_decay * weights[i];
		}
		weights[i] -= lr * g;
	}
}

void cpuAdam(std::vector<float> &weights, const std::vector<float> &grads,
             std::vector<float> &m, std::vector<float> &v,
             oa::U32 count, float lr, float beta1, float beta2, float eps, oa::U32 step) {
	for (oa::U32 i = 0; i < count; ++i) {
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

void cpuAdamW(std::vector<float> &weights, const std::vector<float> &grads,
              std::vector<float> &m, std::vector<float> &v,
              oa::U32 count, float lr, float beta1, float beta2, float eps, 
              float weight_decay, oa::U32 step) {
	for (oa::U32 i = 0; i < count; ++i) {
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
// Test Fixture
// ============================================================================

class OptimKernels : public ::testing::Test {
protected:
	oa::UniquePtr<oa::Engine> rtStorage_;
	[[nodiscard]] oa::Engine& rt() noexcept { return *rtStorage_; }
	oa::ExecutionSession* savedContext_ = nullptr;

	void SetUp() override {
		savedContext_ = oa::ExecutionSession::getActivePtr();
		auto result = oa::Engine::create({
			.appName = "OptimKernels",
			.selectForThread = false,
		});
		ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
		rtStorage_ = oa::move(*result);
		oa::ExecutionSession::setActive(&oa::ExecutionSession::forEngine(rt()));

		OaLogInfo(oa::LogComponent::Ml, "GPU: %s", oa::EngineDeviceAccess::get(rt()).info.hardware.deviceName.cStr());
	}

	void TearDown() override {
		if (not rtStorage_) {
			oa::ExecutionSession::setActive(savedContext_);
			return;
		}
		oa::ExecutionSession::getActive().clear();
		oa::ExecutionSession::setActive(savedContext_);
		const auto closeStatus = rt().close();
		EXPECT_TRUE(closeStatus.isOk()) << closeStatus.toString();
	}

	// Helper: compare results with detailed logging
	void compareResults(const std::vector<float> &ref, const std::vector<float> &gpu,
	                    oa::U32 count, float relativeTolerance, const char *testName = "",
	                    float absoluteTolerance = 1e-7f) {
		double worstRatio = 0.0;
		oa::U32 worstIndex = 0;
		for (oa::U32 i = 0; i < count; ++i) {
			const double absError = std::abs(
				static_cast<double>(ref[i]) - static_cast<double>(gpu[i]));
			const double allowed = static_cast<double>(absoluteTolerance)
				+ static_cast<double>(relativeTolerance)
					* std::abs(static_cast<double>(ref[i]));
			const double ratio = absError / allowed;
			if (ratio > worstRatio) {
				worstRatio = ratio;
				worstIndex = i;
			}
		}
		if (worstRatio > 1.0) {
			const double absError = std::abs(
				static_cast<double>(ref[worstIndex])
				- static_cast<double>(gpu[worstIndex]));
			const double allowed = static_cast<double>(absoluteTolerance)
				+ static_cast<double>(relativeTolerance)
					* std::abs(static_cast<double>(ref[worstIndex]));
			OaLogError(
				oa::LogComponent::Ml,
				"%s: Worst mismatch at [%u]: CPU=%.9g GPU=%.9g "
				"(abs_err=%.3e allowed=%.3e)",
				testName,
				worstIndex,
				ref[worstIndex],
				gpu[worstIndex],
				absError,
				allowed);
		}
		EXPECT_LE(worstRatio, 1.0)
			<< testName << ": worst absolute-plus-relative error ratio: "
			<< worstRatio;
	}
};

// ============================================================================
// SGD Tests
// ============================================================================

TEST_VK(OptimKernels, Sgd) {
	auto testSgd = [this](oa::U32 count, float lr, float weight_decay) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		
		std::vector<float> weights(count), grads(count);
		for (auto &v : weights) v = dist(rng);
		for (auto &v : grads) v = dist(rng);
		
		std::vector<float> weights_ref = weights;
		cpuSgd(weights_ref, grads, count, lr, weight_decay);
		
		auto resultWeights = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		auto resultGrads = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		ASSERT_TRUE(resultWeights.isOk() && resultGrads.isOk());
		
		auto bufWeights = std::move(resultWeights).getValue();
		auto bufGrads = std::move(resultGrads).getValue();
		
		std::memcpy(bufWeights.mappedPtr, weights.data(), count * 4);
		std::memcpy(bufGrads.mappedPtr, grads.data(), count * 4);
		
		struct { oa::U32 count; float lr; float weight_decay; } pc = { count, lr, weight_decay };
		oavk::Buffer bufs[] = {bufWeights, bufGrads};
		oa::U32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(oavk::Dispatch::run(rt(), "Sgd", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());
		
		std::vector<float> weights_gpu(count);
		std::memcpy(weights_gpu.data(), bufWeights.mappedPtr, count * 4);
		
		compareResults(weights_ref, weights_gpu, count, 1e-6f, "Sgd");
		
		oa::EngineResourceAccess::freeBuffer(rt(), bufWeights);
		oa::EngineResourceAccess::freeBuffer(rt(), bufGrads);
	};
	
	testSgd(1024, 0.01f, 0.0f);      // No weight decay
	testSgd(4096, 0.001f, 0.0001f);  // With weight decay
	testSgd(16384, 0.1f, 0.01f);     // Large step
}

// ============================================================================
// Adam Tests
// ============================================================================

TEST_VK(OptimKernels, Adam) {
	auto testAdam = [this](oa::U32 count, float lr, oa::U32 step) {
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
		cpuAdam(weights_ref, grads, m_ref, v_ref, count, lr, beta1, beta2, eps, step);
		
		auto resultWeights = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		auto resultGrads = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		auto resultM = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		auto resultV = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		ASSERT_TRUE(resultWeights.isOk() && resultGrads.isOk() && resultM.isOk() && resultV.isOk());
		
		auto bufWeights = std::move(resultWeights).getValue();
		auto bufGrads = std::move(resultGrads).getValue();
		auto bufM = std::move(resultM).getValue();
		auto bufV = std::move(resultV).getValue();
		
		std::memcpy(bufWeights.mappedPtr, weights.data(), count * 4);
		std::memcpy(bufGrads.mappedPtr, grads.data(), count * 4);
		std::memcpy(bufM.mappedPtr, m.data(), count * 4);
		std::memcpy(bufV.mappedPtr, v.data(), count * 4);
		
		struct { oa::U32 count; float lr; float beta1; float beta2; float eps; oa::U32 step; } pc = 
			{ count, lr, beta1, beta2, eps, step };
		oavk::Buffer bufs[] = {bufWeights, bufGrads, bufM, bufV};
		oa::U32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(oavk::Dispatch::run(rt(), "Adam", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());
		
		std::vector<float> weights_gpu(count), m_gpu(count), v_gpu(count);
		std::memcpy(weights_gpu.data(), bufWeights.mappedPtr, count * 4);
		std::memcpy(m_gpu.data(), bufM.mappedPtr, count * 4);
		std::memcpy(v_gpu.data(), bufV.mappedPtr, count * 4);
		
		// Adam uses pow/sqrt - relaxed tolerance for transcendental error accumulation
		compareResults(weights_ref, weights_gpu, count, 1e-3f, "Adam_weights");
		compareResults(m_ref, m_gpu, count, 1e-3f, "Adam_m");
		compareResults(v_ref, v_gpu, count, 1e-3f, "Adam_v");
		
		oa::EngineResourceAccess::freeBuffer(rt(), bufWeights);
		oa::EngineResourceAccess::freeBuffer(rt(), bufGrads);
		oa::EngineResourceAccess::freeBuffer(rt(), bufM);
		oa::EngineResourceAccess::freeBuffer(rt(), bufV);
	};
	
	testAdam(1024, 0.001f, 1);    // first step (bias correction matters)
	testAdam(4096, 0.001f, 10);   // Later step
	testAdam(16384, 0.0001f, 100); // Many steps
}

// ============================================================================
// AdamW Tests
// ============================================================================

TEST_VK(OptimKernels, Adamw) {
	auto testAdamW = [this](oa::U32 count, float lr, float weight_decay, oa::U32 step) {
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
		cpuAdamW(weights_ref, grads, m_ref, v_ref, count, lr, beta1, beta2, eps, weight_decay, step);
		
		auto resultWeights = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		auto resultGrads = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		auto resultM = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		auto resultV = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		ASSERT_TRUE(resultWeights.isOk() && resultGrads.isOk() && resultM.isOk() && resultV.isOk());
		
		auto bufWeights = std::move(resultWeights).getValue();
		auto bufGrads = std::move(resultGrads).getValue();
		auto bufM = std::move(resultM).getValue();
		auto bufV = std::move(resultV).getValue();
		
		std::memcpy(bufWeights.mappedPtr, weights.data(), count * 4);
		std::memcpy(bufGrads.mappedPtr, grads.data(), count * 4);
		std::memcpy(bufM.mappedPtr, m.data(), count * 4);
		std::memcpy(bufV.mappedPtr, v.data(), count * 4);
		
		struct { oa::U32 count; float lr; float beta1; float beta2; float eps; float weight_decay; oa::U32 step; } pc = 
			{ count, lr, beta1, beta2, eps, weight_decay, step };
		oavk::Buffer bufs[] = {bufWeights, bufGrads, bufM, bufV};
		oa::U32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(oavk::Dispatch::run(rt(), "Adamw", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());
		
		std::vector<float> weights_gpu(count), m_gpu(count), v_gpu(count);
		std::memcpy(weights_gpu.data(), bufWeights.mappedPtr, count * 4);
		std::memcpy(m_gpu.data(), bufM.mappedPtr, count * 4);
		std::memcpy(v_gpu.data(), bufV.mappedPtr, count * 4);
		
		// AdamW uses pow/sqrt - relaxed tolerance for transcendental error accumulation
		compareResults(weights_ref, weights_gpu, count, 1e-3f, "AdamW_weights");
		compareResults(m_ref, m_gpu, count, 1e-3f, "AdamW_m");
		compareResults(v_ref, v_gpu, count, 1e-3f, "AdamW_v");

		// replay-safe variant: mutable hyperparameters and step live in a
		// device-visible state buffer, leaving the recorded push payload stable.
		std::memcpy(bufWeights.mappedPtr, weights.data(), count * 4);
		std::memcpy(bufM.mappedPtr, m.data(), count * 4);
		std::memcpy(bufV.mappedPtr, v.data(), count * 4);
		auto resultState = oa::EngineResourceAccess::allocBufferBar(rt(), 6 * sizeof(oa::U32));
		ASSERT_TRUE(resultState.isOk());
		auto bufState = std::move(resultState).getValue();
		oa::U32 state[6] = {step, 0, 0, 0, 0, 0};
		const oa::F32 scalars[] = {lr, beta1, beta2, eps, weight_decay};
		std::memcpy(state + 1, scalars, sizeof(scalars));
		std::memcpy(bufState.mappedPtr, state, sizeof(state));
		struct { oa::U32 count; } graphPc = {count};
		oavk::Buffer graphBufs[] = {bufWeights, bufGrads, bufM, bufV, bufState};
		ASSERT_TRUE(oavk::Dispatch::run(rt(), "AdamwGraph", graphBufs,
			&graphPc, sizeof(graphPc), oa::ScalarType::Float32, groups).isOk());

		std::memcpy(weights_gpu.data(), bufWeights.mappedPtr, count * 4);
		std::memcpy(m_gpu.data(), bufM.mappedPtr, count * 4);
		std::memcpy(v_gpu.data(), bufV.mappedPtr, count * 4);
		compareResults(weights_ref, weights_gpu, count, 1e-3f, "AdamWGraph_weights");
		compareResults(m_ref, m_gpu, count, 1e-3f, "AdamWGraph_m");
		compareResults(v_ref, v_gpu, count, 1e-3f, "AdamWGraph_v");
		
		oa::EngineResourceAccess::freeBuffer(rt(), bufWeights);
		oa::EngineResourceAccess::freeBuffer(rt(), bufGrads);
		oa::EngineResourceAccess::freeBuffer(rt(), bufM);
		oa::EngineResourceAccess::freeBuffer(rt(), bufV);
		oa::EngineResourceAccess::freeBuffer(rt(), bufState);
	};
	
	testAdamW(1024, 0.001f, 0.01f, 1);     // first step with weight decay
	testAdamW(4096, 0.001f, 0.01f, 10);    // Later step
	testAdamW(16384, 0.0001f, 0.001f, 100); // Many steps, small decay
}

TEST_VK(OptimKernels, AdamwMany4) {
	constexpr float lr = 0.001f;
	constexpr float beta1 = 0.9f;
	constexpr float beta2 = 0.999f;
	constexpr float eps = 1e-8f;
	constexpr float weightDecay = 0.01f;
	constexpr oa::U32 step = 7;

	const oa::U32 counts[] = {64 * 32, 32 * 16, 64, 32};
	constexpr oa::U32 kParams = 4;

	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<float> weights[kParams], grads[kParams], m[kParams], v[kParams];
	std::vector<float> weightsRef[kParams], mRef[kParams], vRef[kParams];
	for (oa::U32 p = 0; p < kParams; ++p) {
		const oa::U32 count = counts[p];
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
		cpuAdamW(weightsRef[p], grads[p], mRef[p], vRef[p], count,
			lr, beta1, beta2, eps, weightDecay, step);
	}

	oavk::Buffer wBuf[kParams], gBuf[kParams], mBuf[kParams], vBuf[kParams];
	for (oa::U32 p = 0; p < kParams; ++p) {
		const oa::U32 count = counts[p];
		auto rw = oa::EngineResourceAccess::allocBufferBar(rt(), count * sizeof(float));
		auto rg = oa::EngineResourceAccess::allocBufferBar(rt(), count * sizeof(float));
		auto rm = oa::EngineResourceAccess::allocBufferBar(rt(), count * sizeof(float));
		auto rv = oa::EngineResourceAccess::allocBufferBar(rt(), count * sizeof(float));
		ASSERT_TRUE(rw.isOk() && rg.isOk() && rm.isOk() && rv.isOk());
		wBuf[p] = std::move(rw).getValue();
		gBuf[p] = std::move(rg).getValue();
		mBuf[p] = std::move(rm).getValue();
		vBuf[p] = std::move(rv).getValue();
		std::memcpy(wBuf[p].mappedPtr, weights[p].data(), count * sizeof(float));
		std::memcpy(gBuf[p].mappedPtr, grads[p].data(), count * sizeof(float));
		std::memcpy(mBuf[p].mappedPtr, m[p].data(), count * sizeof(float));
		std::memcpy(vBuf[p].mappedPtr, v[p].data(), count * sizeof(float));
	}

	struct Push {
		oa::U32 count0;
		oa::U32 count1;
		oa::U32 count2;
		oa::U32 count3;
		oa::F32 lr;
		oa::F32 beta1;
		oa::F32 beta2;
		oa::F32 eps;
		oa::F32 weightDecay;
		oa::U32 step;
	};
	const oa::U32 maxCount = *std::max_element(counts, counts + kParams);
	Push pc{
		.count0 = counts[0], .count1 = counts[1], .count2 = counts[2], .count3 = counts[3],
		.lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .weightDecay = weightDecay, .step = step
	};
	oavk::Buffer bufs[] = {
		wBuf[0], gBuf[0], mBuf[0], vBuf[0],
		wBuf[1], gBuf[1], mBuf[1], vBuf[1],
		wBuf[2], gBuf[2], mBuf[2], vBuf[2],
		wBuf[3], gBuf[3], mBuf[3], vBuf[3],
	};
	ASSERT_TRUE(oavk::Dispatch::run(rt(), "AdamwMany4", bufs, &pc, sizeof(pc),
		oa::ScalarType::Float32, (maxCount + 255U) / 256U).isOk());

	for (oa::U32 p = 0; p < kParams; ++p) {
		const oa::U32 count = counts[p];
		std::vector<float> wGpu(count), mGpu(count), vGpu(count);
		std::memcpy(wGpu.data(), wBuf[p].mappedPtr, count * sizeof(float));
		std::memcpy(mGpu.data(), mBuf[p].mappedPtr, count * sizeof(float));
		std::memcpy(vGpu.data(), vBuf[p].mappedPtr, count * sizeof(float));
		compareResults(weightsRef[p], wGpu, count, 2e-3f, "AdamwMany4_weights");
		compareResults(mRef[p], mGpu, count, 2e-3f, "AdamwMany4_m");
		compareResults(vRef[p], vGpu, count, 2e-3f, "AdamwMany4_v");
	}

	// reset inputs and verify the replay-safe fused-four variant against the
	// same CPU oracle.
	for (oa::U32 p = 0; p < kParams; ++p) {
		const oa::U32 count = counts[p];
		std::memcpy(wBuf[p].mappedPtr, weights[p].data(), count * sizeof(float));
		std::memcpy(mBuf[p].mappedPtr, m[p].data(), count * sizeof(float));
		std::memcpy(vBuf[p].mappedPtr, v[p].data(), count * sizeof(float));
	}
	auto stateResult = oa::EngineResourceAccess::allocBufferBar(rt(), 6 * sizeof(oa::U32));
	ASSERT_TRUE(stateResult.isOk());
	auto stateBuf = std::move(stateResult).getValue();
	oa::U32 state[6] = {step, 0, 0, 0, 0, 0};
	const oa::F32 stateScalars[] = {lr, beta1, beta2, eps, weightDecay};
	std::memcpy(state + 1, stateScalars, sizeof(stateScalars));
	std::memcpy(stateBuf.mappedPtr, state, sizeof(state));
	struct GraphPush {
		oa::U32 count0;
		oa::U32 count1;
		oa::U32 count2;
		oa::U32 count3;
	} graphPc{counts[0], counts[1], counts[2], counts[3]};
	oavk::Buffer graphBufs[] = {
		wBuf[0], gBuf[0], mBuf[0], vBuf[0],
		wBuf[1], gBuf[1], mBuf[1], vBuf[1],
		wBuf[2], gBuf[2], mBuf[2], vBuf[2],
		wBuf[3], gBuf[3], mBuf[3], vBuf[3],
		stateBuf,
	};
	ASSERT_TRUE(oavk::Dispatch::run(rt(), "AdamwMany4Graph", graphBufs,
		&graphPc, sizeof(graphPc), oa::ScalarType::Float32,
		(maxCount + 255U) / 256U).isOk());
	for (oa::U32 p = 0; p < kParams; ++p) {
		const oa::U32 count = counts[p];
		std::vector<float> wGpu(count), mGpu(count), vGpu(count);
		std::memcpy(wGpu.data(), wBuf[p].mappedPtr, count * sizeof(float));
		std::memcpy(mGpu.data(), mBuf[p].mappedPtr, count * sizeof(float));
		std::memcpy(vGpu.data(), vBuf[p].mappedPtr, count * sizeof(float));
		compareResults(weightsRef[p], wGpu, count, 2e-3f, "AdamwMany4Graph_weights");
		compareResults(mRef[p], mGpu, count, 2e-3f, "AdamwMany4Graph_m");
		compareResults(vRef[p], vGpu, count, 2e-3f, "AdamwMany4Graph_v");
	}

	for (oa::U32 p = 0; p < kParams; ++p) {
		oa::EngineResourceAccess::freeBuffer(rt(), wBuf[p]);
		oa::EngineResourceAccess::freeBuffer(rt(), gBuf[p]);
		oa::EngineResourceAccess::freeBuffer(rt(), mBuf[p]);
		oa::EngineResourceAccess::freeBuffer(rt(), vBuf[p]);
	}
	oa::EngineResourceAccess::freeBuffer(rt(), stateBuf);
}
