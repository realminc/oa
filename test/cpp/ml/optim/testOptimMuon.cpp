// Muon optimizer correctness tests.
//
// The CPU code below is an independent oracle. Production Muon execution is
// GPU-only and is validated here through both the public optimizer and its
// internal Vulkan kernels.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "../../oaTest.h"
#include <oa/ml/fnOptim.h>

namespace {

namespace muonReference {

namespace {

constexpr oa::F32 kNsA = 3.4445f;
constexpr oa::F32 kNsB = -4.7750f;
constexpr oa::F32 kNsC = 2.0315f;

oa::F32 frobeniusNorm(const float* inData, oa::U32 inCount, oa::F32 inEps) {
	double sumsq = 0.0;
	for (oa::U32 i = 0; i < inCount; ++i) {
		const double v = static_cast<double>(inData[i]);
		sumsq += v * v;
	}
	return std::sqrt(static_cast<oa::F32>(sumsq) + inEps);
}

void transpose(
	float* out,
	const float* in,
	oa::U32 inRows,
	oa::U32 inCols) {
	for (oa::U32 r = 0; r < inRows; ++r) {
		for (oa::U32 col = 0; col < inCols; ++col) {
			out[static_cast<size_t>(col) * inRows + r] = in[static_cast<size_t>(r) * inCols + col];
		}
	}
}

} // namespace

void newtonSchulz5(
	float* outOrtho,
	const float* inUpdate,
	oa::U32 inRows,
	oa::U32 inCols,
	oa::I32 inNS5Steps,
	oa::F32 inEps) {
	const oa::U32 count = inRows * inCols;
	const bool transposed = inRows > inCols;
	const oa::U32 operRows = transposed ? inCols : inRows;
	const oa::U32 operCols = transposed ? inRows : inCols;

	std::vector<float> z(static_cast<size_t>(operRows) * operCols);
	if (transposed) {
		transpose(z.data(), inUpdate, inRows, inCols);
	} else {
		std::memcpy(z.data(), inUpdate, static_cast<size_t>(count) * sizeof(float));
	}

	const oa::F32 norm = frobeniusNorm(z.data(), operRows * operCols, inEps);
	for (float& v : z) {
		v /= norm;
	}

	auto zidx = [operCols](oa::U32 r, oa::U32 c) -> size_t {
		return static_cast<size_t>(r) * operCols + c;
	};

	for (oa::I32 step = 0; step < inNS5Steps; ++step) {
		std::vector<float> a(static_cast<size_t>(operRows) * operRows, 0.0f);
		for (oa::U32 i = 0; i < operRows; ++i) {
			for (oa::U32 k = 0; k < operCols; ++k) {
				const float zik = z[zidx(i, k)];
				for (oa::U32 j = 0; j < operRows; ++j) {
					a[static_cast<size_t>(i) * operRows + j] += zik * z[zidx(j, k)];
				}
			}
		}

		std::vector<float> aa(static_cast<size_t>(operRows) * operRows, 0.0f);
		for (oa::U32 i = 0; i < operRows; ++i) {
			for (oa::U32 k = 0; k < operRows; ++k) {
				const float aik = a[static_cast<size_t>(i) * operRows + k];
				for (oa::U32 j = 0; j < operRows; ++j) {
					aa[static_cast<size_t>(i) * operRows + j] += aik * a[static_cast<size_t>(k) * operRows + j];
				}
			}
		}

		std::vector<float> b(static_cast<size_t>(operRows) * operRows, 0.0f);
		for (oa::U32 i = 0; i < operRows * operRows; ++i) {
			b[i] = kNsB * a[i] + kNsC * aa[i];
		}

		std::vector<float> newZ(static_cast<size_t>(operRows) * operCols, 0.0f);
		for (oa::U32 i = 0; i < operRows; ++i) {
			for (oa::U32 k = 0; k < operCols; ++k) {
				newZ[zidx(i, k)] = kNsA * z[zidx(i, k)];
			}
		}
		for (oa::U32 i = 0; i < operRows; ++i) {
			for (oa::U32 k = 0; k < operRows; ++k) {
				const float bik = b[static_cast<size_t>(i) * operRows + k];
				for (oa::U32 j = 0; j < operCols; ++j) {
					newZ[zidx(i, j)] += bik * z[zidx(k, j)];
				}
			}
		}
		z.swap(newZ);
	}

	if (transposed) {
		transpose(outOrtho, z.data(), operRows, operCols);
	} else {
		std::memcpy(outOrtho, z.data(), static_cast<size_t>(count) * sizeof(float));
	}
}

void matrixStep(
	float* inOutWeights,
	float* inOutMomentum,
	const float* inGrads,
	oa::U32 inRows,
	oa::U32 inCols,
	oa::F32 inLr,
	oa::F32 inBeta,
	oa::F32 inWeightDecay,
	oa::F32 inEps,
	oa::I32 inNS5Steps,
	oa::F32 inRmsMatch = 0.2f) {
	const oa::U32 count = inRows * inCols;

	std::vector<float> nesterov(count);
	for (oa::U32 i = 0; i < count; ++i) {
		const float g = inGrads[i];
		const float mNew = inBeta * inOutMomentum[i] + (1.0f - inBeta) * g;
		nesterov[i] = (1.0f - inBeta) * g + inBeta * mNew;
		inOutMomentum[i] = mNew;
	}

	std::vector<float> ortho(count);
	newtonSchulz5(ortho.data(), nesterov.data(), inRows, inCols, inNS5Steps, inEps);

	const oa::U32 maxDimension = inRows > inCols ? inRows : inCols;
	const oa::F32 scale = inRmsMatch
		* std::sqrt(static_cast<oa::F32>(maxDimension));
	for (oa::U32 i = 0; i < count; ++i) {
		inOutWeights[i] = (1.0f - inLr * inWeightDecay) * inOutWeights[i]
			- inLr * scale * ortho[i];
	}
}

void vectorStep(
	float* inOutWeights,
	float* inOutMomentum,
	const float* inGrads,
	oa::U32 inCount,
	oa::F32 inLr,
	oa::F32 inBeta,
	oa::F32 inWeightDecay) {
	for (oa::U32 i = 0; i < inCount; ++i) {
		const float g = inGrads[i];
		const float mNew = inBeta * inOutMomentum[i] + (1.0f - inBeta) * g;
		const float update = (1.0f - inBeta) * g + inBeta * mNew;
		inOutMomentum[i] = mNew;
		inOutWeights[i] = (1.0f - inLr * inWeightDecay) * inOutWeights[i] - inLr * update;
	}
}

} // namespace muonReference

} // namespace


namespace {

static double muonOrthogonalityError(const std::vector<float>& M, oa::U32 rows, oa::U32 cols) {
	if (rows != cols) return 0.0;
	double err = 0.0;
	for (oa::U32 i = 0; i < rows; ++i) {
		for (oa::U32 j = 0; j < rows; ++j) {
			double dot = 0.0;
			for (oa::U32 k = 0; k < cols; ++k) {
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

} // namespace

class OptimMuon : public ::testing::Test {
protected:
	oa::UniquePtr<oa::Engine> rtStorage_;
	[[nodiscard]] oa::Engine& rt() noexcept { return *rtStorage_; }
	oa::ExecutionSession* savedContext_ = nullptr;

	void SetUp() override {
		savedContext_ = oa::ExecutionSession::getActivePtr();
		auto result = oa::Engine::create({
			.appName = "OptimMuon",
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

TEST(OptimMuonReference, NewtonSchulz5SquareOrthogonality) {
	std::mt19937 rng(7);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	constexpr oa::U32 kDim = 16;
	std::vector<float> input(static_cast<size_t>(kDim) * kDim);
	for (float& v : input) v = dist(rng);

	std::vector<float> ortho(input.size());
		muonReference::newtonSchulz5(
		ortho.data(), input.data(), kDim, kDim, 5, 1e-7f);

	EXPECT_LT(muonOrthogonalityError(ortho, kDim, kDim), 0.35)
		<< "NS5 should approximate orthogonality on square matrices";
}

TEST(OptimMuonReference, NewtonSchulz5TallMatchesWide) {
	std::mt19937 rng(11);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	constexpr oa::U32 kRows = 8;
	constexpr oa::U32 kCols = 32;
	std::vector<float> tall(static_cast<size_t>(kRows) * kCols);
	std::vector<float> wide(static_cast<size_t>(kCols) * kRows);
	for (oa::U32 r = 0; r < kRows; ++r) {
		for (oa::U32 c = 0; c < kCols; ++c) {
			const float v = dist(rng);
			tall[static_cast<size_t>(r) * kCols + c] = v;
			wide[static_cast<size_t>(c) * kRows + r] = v;
		}
	}

	std::vector<float> outTall(tall.size());
	std::vector<float> outWide(wide.size());
	muonReference::newtonSchulz5(
		outTall.data(), tall.data(), kRows, kCols, 5, 1e-7f);
	muonReference::newtonSchulz5(
		outWide.data(), wide.data(), kCols, kRows, 5, 1e-7f);

	for (oa::U32 r = 0; r < kRows; ++r) {
		for (oa::U32 c = 0; c < kCols; ++c) {
			const float a = outTall[static_cast<size_t>(r) * kCols + c];
			const float b = outWide[static_cast<size_t>(c) * kRows + r];
			EXPECT_NEAR(a, b, 1e-4f) << "r=" << r << " c=" << c;
		}
	}
}

TEST(OptimMuonReference, MatrixStepDeterministic) {
	constexpr oa::U32 kRows = 4;
	constexpr oa::U32 kCols = 8;
	const oa::U32 count = kRows * kCols;
	std::vector<float> w(count), g(count), m(count, 0.0f);
	for (oa::U32 i = 0; i < count; ++i) {
		w[i] = 0.01f * static_cast<float>(i);
		g[i] = 0.001f * static_cast<float>((i + 3) % 7);
	}

	std::vector<float> wRef = w;
	std::vector<float> mRef = m;
	muonReference::matrixStep(
		wRef.data(), mRef.data(), g.data(), kRows, kCols,
		0.01f, 0.95f, 0.1f, 1e-7f, 5);

	EXPECT_NE(wRef, w);
	EXPECT_NE(mRef, m);
	for (float v : wRef) {
		EXPECT_TRUE(std::isfinite(v));
	}
}

TEST_VK(OptimMuon, MuonStepMatrixMatchesReference) {
	auto runCase = [this](oa::U32 inRows, oa::U32 inColumns, oa::U32 inSeed) {
		const oa::U32 count = inRows * inColumns;
		constexpr oa::F32 kLearningRate = 0.01F;
		constexpr oa::F32 kBeta = 0.95F;
		constexpr oa::F32 kWeightDecay = 0.1F;
		constexpr oa::F32 kEpsilon = 1.0e-7F;
		constexpr oa::I32 kIterations = 5;

		std::mt19937 rng(inSeed);
		std::uniform_real_distribution<oa::F32> distribution(-0.5F, 0.5F);
		std::vector<oa::F32> weights(count);
		std::vector<oa::F32> gradients(count);
		std::vector<oa::F32> momentum(count);
		for (oa::U32 index = 0; index < count; ++index) {
			weights[index] = distribution(rng);
			gradients[index] = distribution(rng);
			momentum[index] = 0.1F * distribution(rng);
		}

		std::vector<oa::F32> expectedWeights = weights;
		std::vector<oa::F32> expectedMomentum = momentum;
		muonReference::matrixStep(
			expectedWeights.data(),
			expectedMomentum.data(),
			gradients.data(),
			inRows,
			inColumns,
			kLearningRate,
			kBeta,
			kWeightDecay,
			kEpsilon,
			kIterations
		);

		const auto byteSpan = [](const std::vector<oa::F32>& inValues) {
			return oa::Span<const oa::U8>(
				reinterpret_cast<const oa::U8*>(inValues.data()),
				inValues.size() * sizeof(oa::F32));
		};
		const oa::MatrixShape shape{
			static_cast<oa::I64>(inRows),
			static_cast<oa::I64>(inColumns),
		};
		oa::Matrix actualWeights = oa::FnMatrix::fromBytes(
			byteSpan(weights), shape, oa::ScalarType::Float32);
		oa::Matrix actualGradients = oa::FnMatrix::fromBytes(
			byteSpan(gradients), shape, oa::ScalarType::Float32);
		oa::Matrix actualMomentum = oa::FnMatrix::fromBytes(
			byteSpan(momentum), shape, oa::ScalarType::Float32);

		oa::FnOptim::muonStep(
			actualWeights,
			actualMomentum,
			actualGradients,
			kLearningRate,
			kBeta,
			kWeightDecay,
			kEpsilon,
			kIterations
		);

		std::vector<oa::F32> resultWeights(count);
		std::vector<oa::F32> resultMomentum(count);
		ASSERT_TRUE(oa::FnMatrix::copyToHost(
			actualWeights, resultWeights.data(), count * sizeof(oa::F32)).isOk());
		ASSERT_TRUE(oa::FnMatrix::copyToHost(
			actualMomentum, resultMomentum.data(), count * sizeof(oa::F32)).isOk());
		compareResults(
			expectedWeights, resultWeights, count, 2.0e-3F,
			"MuonStep_weights", 2.0e-5F);
		compareResults(
			expectedMomentum, resultMomentum, count, 1.0e-5F,
			"MuonStep_momentum", 1.0e-6F);
	};

	runCase(4, 8, 41);
	runCase(8, 4, 43);
	runCase(3, 5, 47);
	runCase(5, 3, 53);
	runCase(32, 320, 59);
	runCase(320, 32, 61);
}

TEST_VK(OptimMuon, MuonStepVectorMatchesReference) {
	constexpr oa::U32 kCount = 37;
	constexpr oa::F32 kLearningRate = 0.015F;
	constexpr oa::F32 kBeta = 0.9F;
	constexpr oa::F32 kWeightDecay = 0.03F;
	std::vector<oa::F32> weights(kCount);
	std::vector<oa::F32> gradients(kCount);
	std::vector<oa::F32> momentum(kCount);
	for (oa::U32 index = 0; index < kCount; ++index) {
		weights[index] = 0.02F * static_cast<oa::F32>(index) - 0.3F;
		gradients[index] = 0.01F * static_cast<oa::F32>((index * 7U) % 13U) - 0.04F;
		momentum[index] = 0.005F * static_cast<oa::F32>(index % 5U);
	}

	std::vector<oa::F32> expectedWeights = weights;
	std::vector<oa::F32> expectedMomentum = momentum;
	muonReference::vectorStep(
		expectedWeights.data(),
		expectedMomentum.data(),
		gradients.data(),
		kCount,
		kLearningRate,
		kBeta,
		kWeightDecay
	);

	const auto byteSpan = [](const std::vector<oa::F32>& inValues) {
		return oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::F32));
	};
	const oa::MatrixShape shape{static_cast<oa::I64>(kCount)};
	oa::Matrix actualWeights = oa::FnMatrix::fromBytes(
		byteSpan(weights), shape, oa::ScalarType::Float32);
	oa::Matrix actualGradients = oa::FnMatrix::fromBytes(
		byteSpan(gradients), shape, oa::ScalarType::Float32);
	oa::Matrix actualMomentum = oa::FnMatrix::fromBytes(
		byteSpan(momentum), shape, oa::ScalarType::Float32);

	oa::FnOptim::muonStep(
		actualWeights,
		actualMomentum,
		actualGradients,
		kLearningRate,
		kBeta,
		kWeightDecay,
		1.0e-7F,
		5
	);

	std::vector<oa::F32> resultWeights(kCount);
	std::vector<oa::F32> resultMomentum(kCount);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		actualWeights, resultWeights.data(), kCount * sizeof(oa::F32)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		actualMomentum, resultMomentum.data(), kCount * sizeof(oa::F32)).isOk());
	compareResults(
		expectedWeights, resultWeights, kCount, 1.0e-5F,
		"MuonVectorStep_weights", 1.0e-6F);
	compareResults(
		expectedMomentum, resultMomentum, kCount, 1.0e-5F,
		"MuonVectorStep_momentum", 1.0e-6F);
}

TEST_VK(OptimMuon, MuonOptimizerStepMatchesReference) {
	constexpr oa::U32 kRows = 4;
	constexpr oa::U32 kColumns = 8;
	constexpr oa::U32 kCount = kRows * kColumns;
	constexpr oa::F32 kLearningRate = 0.01F;
	constexpr oa::F32 kBeta = 0.95F;
	constexpr oa::F32 kWeightDecay = 0.1F;
	constexpr oa::F32 kEpsilon = 1.0e-7F;
	constexpr oa::I32 kIterations = 5;
	std::vector<oa::F32> weights(kCount);
	std::vector<oa::F32> gradients(kCount);
	for (oa::U32 index = 0; index < kCount; ++index) {
		weights[index] = 0.01F * static_cast<oa::F32>(index) - 0.15F;
		gradients[index] = 0.02F
			* static_cast<oa::F32>((index * 5U) % 11U) - 0.08F;
	}

	std::vector<oa::F32> expectedWeights = weights;
	std::vector<oa::F32> expectedMomentum(kCount, 0.0F);
	muonReference::matrixStep(
		expectedWeights.data(),
		expectedMomentum.data(),
		gradients.data(),
		kRows,
		kColumns,
		kLearningRate,
		kBeta,
		kWeightDecay,
		kEpsilon,
		kIterations
	);

	const auto byteSpan = [](const std::vector<oa::F32>& inValues) {
		return oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::F32));
	};
	const oa::MatrixShape shape{kRows, kColumns};
	oa::Parameter parameter;
	parameter.data = oa::FnMatrix::fromBytes(
		byteSpan(weights), shape, oa::ScalarType::Float32);
	parameter.data.setRequiresGrad(true);
	parameter.grad() = oa::FnMatrix::fromBytes(
		byteSpan(gradients), shape, oa::ScalarType::Float32);

	oa::Vector<oa::Parameter*> parameters{&parameter};
	oa::Muon optimizer(
		parameters,
		kLearningRate,
		kBeta,
		kWeightDecay,
		kEpsilon,
		kIterations
	);
	optimizer.step();

	std::vector<oa::F32> resultWeights(kCount);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		parameter.data, resultWeights.data(), kCount * sizeof(oa::F32)).isOk());
	compareResults(
		expectedWeights, resultWeights, kCount, 2.0e-3F,
		"MuonOptimizer_weights", 2.0e-5F);
	EXPECT_EQ(optimizer.getStep(), 1U);
}

TEST_VK(OptimMuon, MuonVector) {
	auto testMuonVector = [this](oa::U32 count, float lr, float beta, float weight_decay) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

		std::vector<float> weights(count), grads(count), momentum(count, 0.0f);
		for (auto& v : weights) v = dist(rng);
		for (auto& v : grads) v = dist(rng);

		std::vector<float> weightsRef = weights;
		std::vector<float> momentumRef = momentum;
		muonReference::vectorStep(
			weightsRef.data(), momentumRef.data(), grads.data(),
			count, lr, beta, weight_decay);

		auto resultWeights = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		auto resultGrads = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		auto resultMomentum = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
		ASSERT_TRUE(resultWeights.isOk() && resultGrads.isOk() && resultMomentum.isOk());

		auto bufWeights = std::move(resultWeights).getValue();
		auto bufGrads = std::move(resultGrads).getValue();
		auto bufMomentum = std::move(resultMomentum).getValue();

		std::memcpy(bufWeights.mappedPtr, weights.data(), count * 4);
		std::memcpy(bufGrads.mappedPtr, grads.data(), count * 4);
		std::memcpy(bufMomentum.mappedPtr, momentum.data(), count * 4);

		struct { oa::U32 count; float lr; float beta; float weight_decay; } pc =
			{ count, lr, beta, weight_decay };
		oavk::Buffer bufs[] = { bufWeights, bufGrads, bufMomentum };
		oa::U32 groups = (count + 255) / 256;

		ASSERT_TRUE(oavk::Dispatch::run(rt(), "MuonVector", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());

		std::vector<float> weightsGpu(count), momentumGpu(count);
		std::memcpy(weightsGpu.data(), bufWeights.mappedPtr, count * 4);
		std::memcpy(momentumGpu.data(), bufMomentum.mappedPtr, count * 4);

		compareResults(weightsRef, weightsGpu, count, 1e-5f, "MuonVector_weights");
		compareResults(momentumRef, momentumGpu, count, 1e-5f, "MuonVector_momentum");

		oa::EngineResourceAccess::freeBuffer(rt(), bufWeights);
		oa::EngineResourceAccess::freeBuffer(rt(), bufGrads);
		oa::EngineResourceAccess::freeBuffer(rt(), bufMomentum);
	};

	testMuonVector(1024, 0.01f, 0.95f, 0.1f);
	testMuonVector(4097, 0.001f, 0.95f, 0.01f);
}

TEST_VK(OptimMuon, MuonMatrixPipeline) {
	// GPU kernel chain (Nesterov → normalize → apply) with CPU NS5 reference.
	// Production FnOptim::muonStep uses oa::FnMatrix matmul for NS5; this test
	// validates the dispatch kernels against the independent CPU oracle.
	constexpr oa::U32 kRows = 4;
	constexpr oa::U32 kCols = 8;
	const oa::U32 count = kRows * kCols;
	constexpr float kLr = 0.01f;
	constexpr float kBeta = 0.95f;
	constexpr float kWd = 0.1f;
	constexpr float kEps = 1e-7f;
	constexpr oa::I32 kNs5 = 5;

	std::mt19937 rng(23);
	std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

	std::vector<float> weights(count), grads(count), momentum(count, 0.0f);
	for (float& v : weights) v = dist(rng);
	for (float& v : grads) v = dist(rng);

	std::vector<float> weightsRef = weights;
	std::vector<float> momentumRef = momentum;
	muonReference::matrixStep(
		weightsRef.data(), momentumRef.data(), grads.data(),
		kRows, kCols, kLr, kBeta, kWd, kEps, kNs5);

	auto rw = oa::EngineResourceAccess::allocBufferBar(rt(), count * sizeof(float));
	auto rg = oa::EngineResourceAccess::allocBufferBar(rt(), count * sizeof(float));
	auto rm = oa::EngineResourceAccess::allocBufferBar(rt(), count * sizeof(float));
	auto rUpdate = oa::EngineResourceAccess::allocBufferBar(rt(), count * sizeof(float));
	auto rNorm = oa::EngineResourceAccess::allocBufferBar(rt(), sizeof(float));
	auto rOrtho = oa::EngineResourceAccess::allocBufferBar(rt(), count * sizeof(float));
	ASSERT_TRUE(rw.isOk() && rg.isOk() && rm.isOk() && rUpdate.isOk() && rNorm.isOk() && rOrtho.isOk());

	auto bufW = std::move(rw).getValue();
	auto bufG = std::move(rg).getValue();
	auto bufM = std::move(rm).getValue();
	auto bufUpdate = std::move(rUpdate).getValue();
	auto bufNorm = std::move(rNorm).getValue();
	auto bufOrtho = std::move(rOrtho).getValue();

	std::memcpy(bufW.mappedPtr, weights.data(), count * sizeof(float));
	std::memcpy(bufG.mappedPtr, grads.data(), count * sizeof(float));
	std::memset(bufM.mappedPtr, 0, count * sizeof(float));

	const oa::U32 groups = (count + 255) / 256;
	{
		struct MuonNesterovPush { oa::U32 Count; oa::F32 Beta; } push{count, kBeta};
		oavk::Buffer bufs[] = {bufG, bufM, bufUpdate};
		ASSERT_TRUE(oavk::Dispatch::run(rt(), "MuonNesterov", bufs, &push, sizeof(push), oa::ScalarType::Float32, groups).isOk());
	}

	std::vector<float> nesterov(count);
	std::memcpy(nesterov.data(), bufUpdate.mappedPtr, count * sizeof(float));
	std::vector<float> momentumGpu(count);
	std::memcpy(momentumGpu.data(), bufM.mappedPtr, count * sizeof(float));

	{
		struct MuonNormalizePush { oa::U32 rows; oa::U32 cols; oa::F32 eps; } push{kRows, kCols, kEps};
		oavk::Buffer bufs[] = {bufUpdate, bufUpdate, bufNorm};
		ASSERT_TRUE(oavk::Dispatch::run(rt(), "MuonNormalize", bufs, &push, sizeof(push), oa::ScalarType::Float32, 1).isOk());
	}

	std::vector<float> ortho(count);
	muonReference::newtonSchulz5(
		ortho.data(), nesterov.data(), kRows, kCols, kNs5, kEps);
	std::memcpy(bufOrtho.mappedPtr, ortho.data(), count * sizeof(float));

	const oa::F32 moonshotScale = 0.2F
		* std::sqrt(static_cast<oa::F32>(std::max(kRows, kCols)));
	{
		struct MuonApplyPush {
			oa::U32 Count;
			oa::F32 lr;
			oa::F32 weightDecay;
			oa::F32 MoonshotScale;
		} push{count, kLr, kWd, moonshotScale};
		oavk::Buffer bufs[] = {bufW, bufOrtho};
		ASSERT_TRUE(oavk::Dispatch::run(rt(), "MuonApply", bufs, &push, sizeof(push), oa::ScalarType::Float32, groups).isOk());
	}

	std::vector<float> weightsGpu(count);
	std::memcpy(weightsGpu.data(), bufW.mappedPtr, count * sizeof(float));

	compareResults(weightsRef, weightsGpu, count, 5e-2f, "MuonMatrixPipeline_weights");
	compareResults(momentumRef, momentumGpu, count, 1e-4f, "MuonMatrixPipeline_momentum");

	oa::EngineResourceAccess::freeBuffer(rt(), bufW);
	oa::EngineResourceAccess::freeBuffer(rt(), bufG);
	oa::EngineResourceAccess::freeBuffer(rt(), bufM);
	oa::EngineResourceAccess::freeBuffer(rt(), bufUpdate);
	oa::EngineResourceAccess::freeBuffer(rt(), bufNorm);
	oa::EngineResourceAccess::freeBuffer(rt(), bufOrtho);
}

TEST_VK(OptimMuon, MuonNesterov) {
	constexpr oa::U32 count = 512;
	constexpr float beta = 0.95f;

	std::mt19937 rng(19);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<float> grads(count), momentum(count, 0.0f);
	for (float& v : grads) v = dist(rng);

	std::vector<float> updateRef(count), momentumRef = momentum;
	for (oa::U32 i = 0; i < count; ++i) {
		const float g = grads[i];
		const float mNew = beta * momentumRef[i] + (1.0f - beta) * g;
		updateRef[i] = (1.0f - beta) * g + beta * mNew;
		momentumRef[i] = mNew;
	}

	auto resultGrad = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
	auto resultMom = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
	auto resultUpdate = oa::EngineResourceAccess::allocBufferBar(rt(), count * 4);
	ASSERT_TRUE(resultGrad.isOk() && resultMom.isOk() && resultUpdate.isOk());

	auto bufGrad = std::move(resultGrad).getValue();
	auto bufMom = std::move(resultMom).getValue();
	auto bufUpdate = std::move(resultUpdate).getValue();

	std::memcpy(bufGrad.mappedPtr, grads.data(), count * 4);
	std::memset(bufMom.mappedPtr, 0, count * 4);

	struct { oa::U32 count; float beta; } pc = { count, beta };
	oavk::Buffer bufs[] = { bufGrad, bufMom, bufUpdate };
	oa::U32 groups = (count + 255) / 256;

	ASSERT_TRUE(oavk::Dispatch::run(rt(), "MuonNesterov", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());

	std::vector<float> updateGpu(count), momentumGpu(count);
	std::memcpy(updateGpu.data(), bufUpdate.mappedPtr, count * 4);
	std::memcpy(momentumGpu.data(), bufMom.mappedPtr, count * 4);

	compareResults(updateRef, updateGpu, count, 1e-5f, "MuonNesterov_update");
	compareResults(momentumRef, momentumGpu, count, 1e-5f, "MuonNesterov_momentum");

	oa::EngineResourceAccess::freeBuffer(rt(), bufGrad);
	oa::EngineResourceAccess::freeBuffer(rt(), bufMom);
	oa::EngineResourceAccess::freeBuffer(rt(), bufUpdate);
}
