// Comprehensive ML kernel Correctness Tests
// Validates numerical accuracy of ML kernels against CPU reference
//
// priority: production activation, normalization, BLAS and optimizer kernels.
//
// usage:
//   ./oa-test --gtest_filter=MlKernels.*

#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/ml/type.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/dispatch.h>
#include <oa/core/log.h>
#include <gtest/gtest.h>
#include "../../oaTest.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

static_assert(static_cast<oa::U8>(oa::Activation::None) == 0);
static_assert(static_cast<oa::U8>(oa::Activation::Relu) == 1);
static_assert(static_cast<oa::U8>(oa::Activation::Gelu) == 2);
static_assert(static_cast<oa::U8>(oa::Activation::Silu) == 3);
static_assert(static_cast<oa::U8>(oa::UpsampleMode::Nearest) == 0);
static_assert(static_cast<oa::U8>(oa::UpsampleMode::Bilinear) == 1);

// ============================================================================
// CPU Reference Implementations
// ============================================================================

static void cpuGelu(const std::vector<float> &x, std::vector<float> &out, oa::U32 count) {
	// GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of the standard normal
	// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
	constexpr float kSqrt2OverPi = 0.7978845608f;
	constexpr float kCoeff = 0.044715f;
	for (oa::U32 i = 0; i < count; ++i) {
		float x_val = x[i];
		float x3 = x_val * x_val * x_val;
		float inner = kSqrt2OverPi * (x_val + kCoeff * x3);
		out[i] = 0.5f * x_val * (1.0f + std::tanh(inner));
	}
}

static void cpuRelu(const std::vector<float> &x, std::vector<float> &out, oa::U32 count) {
	for (oa::U32 i = 0; i < count; ++i) {
		out[i] = std::max(0.0f, x[i]);
	}
}

static void cpuSilu(const std::vector<float> &x, std::vector<float> &out, oa::U32 count) {
	// siLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
	for (oa::U32 i = 0; i < count; ++i) {
		float x_val = x[i];
		out[i] = x_val / (1.0f + std::exp(-x_val));
	}
}

static void cpuBiasAdd(std::vector<float> &biased, const std::vector<float> &bias,
                       oa::U32 rows, oa::U32 cols) {
	for (oa::U32 row = 0; row < rows; ++row) {
		for (oa::U32 col = 0; col < cols; ++col) {
			biased[row * cols + col] += bias[col];
		}
	}
}

static void cpuLayerNorm(const std::vector<float> &x, const std::vector<float> &weight,
                         const std::vector<float> &bias, std::vector<float> &out,
                         oa::U32 rows, oa::U32 cols, float eps) {
	for (oa::U32 row = 0; row < rows; ++row) {
		oa::U32 base = row * cols;
		
		// Compute mean
		float sum = 0.0f;
		for (oa::U32 i = 0; i < cols; ++i) {
			sum += x[base + i];
		}
		float mean = sum / static_cast<float>(cols);
		
		// Compute variance
		float var_sum = 0.0f;
		for (oa::U32 i = 0; i < cols; ++i) {
			float diff = x[base + i] - mean;
			var_sum += diff * diff;
		}
		float inv_std = 1.0f / std::sqrt(var_sum / static_cast<float>(cols) + eps);
		
		// normalize and scale
		for (oa::U32 i = 0; i < cols; ++i) {
			float normed = (x[base + i] - mean) * inv_std;
			out[base + i] = normed * weight[i] + bias[i];
		}
	}
}

static void cpuRmsNorm(const std::vector<float> &x, const std::vector<float> &weight,
                       std::vector<float> &out, oa::U32 rows, oa::U32 cols, float eps) {
	for (oa::U32 row = 0; row < rows; ++row) {
		oa::U32 base = row * cols;
		
		// Compute RMS
		float sum_sq = 0.0f;
		for (oa::U32 i = 0; i < cols; ++i) {
			float val = x[base + i];
			sum_sq += val * val;
		}
		float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
		
		// scale
		for (oa::U32 i = 0; i < cols; ++i) {
			out[base + i] = x[base + i] * inv_rms * weight[i];
		}
	}
}

static void cpuSoftmax(const std::vector<float> &x, std::vector<float> &out,
                       oa::U32 rows, oa::U32 cols) {
	for (oa::U32 row = 0; row < rows; ++row) {
		oa::U32 base = row * cols;
		
		// find max for numerical stability
		float row_max = x[base];
		for (oa::U32 i = 1; i < cols; ++i) {
			row_max = std::max(row_max, x[base + i]);
		}
		
		// Compute exp and sum
		float sum = 0.0f;
		for (oa::U32 i = 0; i < cols; ++i) {
			float exp_val = std::exp(x[base + i] - row_max);
			out[base + i] = exp_val;
			sum += exp_val;
		}
		
		// normalize
		float inv_sum = 1.0f / sum;
		for (oa::U32 i = 0; i < cols; ++i) {
			out[base + i] *= inv_sum;
		}
	}
}

static void cpuBmm(const std::vector<float>& a, const std::vector<float>& b,
		std::vector<float>& out, oa::U32 batch, oa::U32 m, oa::U32 k, oa::U32 p) {
	for (oa::U32 n = 0; n < batch; ++n) {
		for (oa::U32 row = 0; row < m; ++row) {
			for (oa::U32 col = 0; col < p; ++col) {
				float accum = 0.0F;
				for (oa::U32 inner = 0; inner < k; ++inner) {
					accum += a[(n * m + row) * k + inner]
						* b[(n * k + inner) * p + col];
				}
				out[(n * m + row) * p + col] = accum;
			}
		}
	}
}

static void cpuBmmNt(const std::vector<float>& a, const std::vector<float>& b,
		std::vector<float>& out, oa::U32 batch, oa::U32 m, oa::U32 k, oa::U32 p) {
	for (oa::U32 n = 0; n < batch; ++n) {
		for (oa::U32 row = 0; row < m; ++row) {
			for (oa::U32 col = 0; col < p; ++col) {
				float accum = 0.0F;
				for (oa::U32 inner = 0; inner < k; ++inner) {
					accum += a[(n * m + row) * k + inner]
						* b[(n * p + col) * k + inner];
				}
				out[(n * m + row) * p + col] = accum;
			}
		}
	}
}

static void cpuLinearWeightBiasBwd(
	const std::vector<float>& input, const std::vector<float>& gradOutput,
	std::vector<float>& gradWeight, std::vector<float>& gradBias,
	oa::U32 m, oa::U32 n, oa::U32 k) {
	for (oa::U32 row = 0; row < n; ++row) {
		float bias = 0.0F;
		for (oa::U32 sample = 0; sample < m; ++sample) {
			bias += gradOutput[sample * n + row];
		}
		gradBias[row] = bias;
		for (oa::U32 col = 0; col < k; ++col) {
			float weight = 0.0F;
			for (oa::U32 sample = 0; sample < m; ++sample) {
				weight += gradOutput[sample * n + row]
					* input[sample * k + col];
			}
			gradWeight[row * k + col] = weight;
		}
	}
}

static void cpuSwiglu(const std::vector<float> &gate, const std::vector<float> &up,
                      std::vector<float> &out, oa::U32 count) {
	// swiGLU(gate, up) = Silu(gate) * up = (gate / (1 + exp(-gate))) * up
	for (oa::U32 i = 0; i < count; ++i) {
		float g = gate[i];
		float silu_g = g / (1.0f + std::exp(-g));
		out[i] = silu_g * up[i];
	}
}

// CPU reference for oa::Ffn::forward. Mirrors the module exactly:
//   normed = rMSNorm(x) * norm_w
//   swiglu = silu(normed @ Wgateᵀ) * (normed @ Wupᵀ)
//   out    = x + swiglu @ Wdownᵀ          (pre-norm residual)
// Linear weights are [out, in] row-major (out[j] = Σ_k in[k]·W[j·in + k]); biases
// are left at zero by the caller, matching oa::Linear's default-initialized bias.
static void cpuFfnForward(const std::vector<float> &x, const std::vector<float> &norm_w,
                          const std::vector<float> &gate_w, const std::vector<float> &up_w,
                          const std::vector<float> &down_w, std::vector<float> &out,
                          oa::U32 rows, oa::U32 d_model, oa::U32 d_ff, float eps) {
	for (oa::U32 row = 0; row < rows; ++row) {
		const oa::U32 x_base = row * d_model;

		// RMSNorm over the d_model axis, scaled by the per-channel weight.
		float sum_sq = 0.0f;
		for (oa::U32 i = 0; i < d_model; ++i) {
			sum_sq += x[x_base + i] * x[x_base + i];
		}
		const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(d_model) + eps);
		std::vector<float> normed(d_model);
		for (oa::U32 i = 0; i < d_model; ++i) {
			normed[i] = x[x_base + i] * inv_rms * norm_w[i];
		}

		// gate / Up projections → swiGLU (silu(gate) * up).
		std::vector<float> swiglu(d_ff);
		for (oa::U32 j = 0; j < d_ff; ++j) {
			float gate = 0.0f;
			float up = 0.0f;
			for (oa::U32 k = 0; k < d_model; ++k) {
				gate += normed[k] * gate_w[j * d_model + k];
				up   += normed[k] * up_w[j * d_model + k];
			}
			const float silu_g = gate / (1.0f + std::exp(-gate));
			swiglu[j] = silu_g * up;
		}

		// Down projection + residual back onto the raw input.
		for (oa::U32 i = 0; i < d_model; ++i) {
			float down = 0.0f;
			for (oa::U32 j = 0; j < d_ff; ++j) {
				down += swiglu[j] * down_w[i * d_ff + j];
			}
			out[x_base + i] = x[x_base + i] + down;
		}
	}
}

static void cpuCrossEntropyFwd(const std::vector<float> &logits, const std::vector<oa::U32> &targets,
                                std::vector<float> &loss, oa::U32 batch, oa::U32 classes) {
	for (oa::U32 row = 0; row < batch; ++row) {
		oa::U32 base = row * classes;
		
		// find max for numerical stability
		float row_max = logits[base];
		for (oa::U32 i = 1; i < classes; ++i) {
			row_max = std::max(row_max, logits[base + i]);
		}
		
		// Compute log_sum_exp
		float sum_exp = 0.0f;
		for (oa::U32 i = 0; i < classes; ++i) {
			sum_exp += std::exp(logits[base + i] - row_max);
		}
		float log_sum_exp = std::log(sum_exp) + row_max;
		
		// CrossEntropy = log_sum_exp - logits[target]
		oa::U32 target = targets[row];
		loss[row] = log_sum_exp - logits[base + target];
	}
}

static void cpuCrossEntropyBwd(const std::vector<float> &logits, const std::vector<oa::U32> &targets,
                                std::vector<float> &d_logits, oa::U32 batch, oa::U32 classes) {
	for (oa::U32 row = 0; row < batch; ++row) {
		oa::U32 base = row * classes;
		
		// Compute softmax (same as forward)
		float row_max = logits[base];
		for (oa::U32 i = 1; i < classes; ++i) {
			row_max = std::max(row_max, logits[base + i]);
		}
		
		float sum_exp = 0.0f;
		for (oa::U32 i = 0; i < classes; ++i) {
			sum_exp += std::exp(logits[base + i] - row_max);
		}
		float inv_sum = 1.0f / sum_exp;
		
		// Gradient: (softmax - onehot) / batch
		oa::U32 target = targets[row];
		float inv_batch = 1.0f / static_cast<float>(batch);
		for (oa::U32 i = 0; i < classes; ++i) {
			float prob = std::exp(logits[base + i] - row_max) * inv_sum;
			float grad = prob;
			if (i == target) grad -= 1.0f;
			d_logits[base + i] = grad * inv_batch;
		}
	}
}

// ============================================================================
// Test Fixture
// ============================================================================

class MlKernels : public VkEngineTestFixture {
protected:
	oa::Engine& rt_ = rt();

	void SetUp() override {
		VkEngineTestFixture::SetUp();  // Asserts suite engine exists
		OaLogInfo(oa::LogComponent::Ml, "GPU: {}",
			oa::EngineDeviceAccess::get(rt_).info.hardware.deviceName.cStr());
	}

	// Helper: compare results with detailed absolute/relative error logging.
	void compareResults(const std::vector<float> &ref, const std::vector<float> &gpu,
	                    oa::U32 count, float relativeTolerance, const char *testName = "",
	                    float absoluteTolerance = 0.0F) {
		bool matches = true;
		oa::U32 worstIndex = 0;
		double worstRatio = 0.0;
		double maxAbsError = 0.0;
		double maxRelError = 0.0;
		for (oa::U32 i = 0; i < count; ++i) {
			const double absError = std::abs(ref[i] - gpu[i]);
			const double relError = absError / std::max(std::abs(ref[i]), 1e-6f);
			const double relativeScale = std::max(std::abs(ref[i]), 1e-6F);
			const double allowedError = absoluteTolerance + relativeTolerance * relativeScale;
			maxAbsError = std::max(maxAbsError, absError);
			maxRelError = std::max(maxRelError, relError);
			if (absError > allowedError) {
				matches = false;
				const double ratio = allowedError > 0.0 ? absError / allowedError : absError;
				if (ratio >= worstRatio) {
					worstIndex = i;
					worstRatio = ratio;
				}
			}
		}

		if (!matches) {
			OaLogError(oa::LogComponent::Ml,
			             "{}: Worst mismatch at [{}]: CPU={:.9g} GPU={:.9g} (error/limit={:.2e})",
			             testName, worstIndex, ref[worstIndex], gpu[worstIndex], worstRatio);
		}
		EXPECT_TRUE(matches) << testName << ": Max absolute error: " << maxAbsError
		                     << ", max relative error: " << maxRelError;
	}
};

// ============================================================================
// Activation Tests
// ============================================================================

TEST_VK(MlKernels, Gelu) {
	auto testGelu = [this](oa::U32 count) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		
		std::vector<float> x(count), out_ref(count);
		for (auto &v : x) v = dist(rng);
		cpuGelu(x, out_ref, count);
		
		auto resultX = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		auto resultOut = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		ASSERT_TRUE(resultX.isOk() && resultOut.isOk());
		
		auto bufX = std::move(resultX).getValue();
		auto bufOut = std::move(resultOut).getValue();
		
		std::memcpy(bufX.mappedPtr, x.data(), count * 4);
		
		struct { oa::U32 count; } pc = { count };
		oavk::Buffer bufs[] = {bufX, bufOut};
		oa::U32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "Gelu", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.mappedPtr, count * 4);
		
		// GELU uses tanh approximation - slightly relaxed tolerance
		compareResults(out_ref, out_gpu, count, 1e-4f, "Gelu");
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufX);
		oa::EngineResourceAccess::freeBuffer(rt_, bufOut);
	};
	
	testGelu(1024);
	testGelu(4096);
}

TEST_VK(MlKernels, Relu) {
	auto testRelu = [this](oa::U32 count) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
		
		std::vector<float> x(count), out_ref(count);
		for (auto &v : x) v = dist(rng);
		cpuRelu(x, out_ref, count);
		
		auto resultX = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		auto resultOut = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		ASSERT_TRUE(resultX.isOk() && resultOut.isOk());
		
		auto bufX = std::move(resultX).getValue();
		auto bufOut = std::move(resultOut).getValue();
		
		std::memcpy(bufX.mappedPtr, x.data(), count * 4);
		
		struct { oa::U32 count; } pc = { count };
		oavk::Buffer bufs[] = {bufX, bufOut};
		oa::U32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "Relu", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.mappedPtr, count * 4);
		
		compareResults(out_ref, out_gpu, count, 1e-6f, "Relu");
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufX);
		oa::EngineResourceAccess::freeBuffer(rt_, bufOut);
	};
	
	testRelu(1024);
	testRelu(4096);
}

TEST_VK(MlKernels, Silu) {
	auto testSilu = [this](oa::U32 count) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		
		std::vector<float> x(count), out_ref(count);
		for (auto &v : x) v = dist(rng);
		cpuSilu(x, out_ref, count);
		
		auto resultX = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		auto resultOut = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		ASSERT_TRUE(resultX.isOk() && resultOut.isOk());
		
		auto bufX = std::move(resultX).getValue();
		auto bufOut = std::move(resultOut).getValue();
		
		std::memcpy(bufX.mappedPtr, x.data(), count * 4);
		
		struct { oa::U32 count; } pc = { count };
		oavk::Buffer bufs[] = {bufX, bufOut};
		oa::U32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "Silu", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.mappedPtr, count * 4);
		
		// SiLU uses exp - slightly relaxed tolerance
		compareResults(out_ref, out_gpu, count, 1e-5f, "Silu");
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufX);
		oa::EngineResourceAccess::freeBuffer(rt_, bufOut);
	};
	
	testSilu(1024);
	testSilu(4096);
}

// ============================================================================
// ops Tests
// ============================================================================

TEST_VK(MlKernels, BiasAdd) {
	auto testBiasAdd = [this](oa::U32 rows, oa::U32 cols) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		
		std::vector<float> biased(rows * cols), bias(cols);
		for (auto &v : biased) v = dist(rng);
		for (auto &v : bias) v = dist(rng);
		
		std::vector<float> biased_ref = biased;
		cpuBiasAdd(biased_ref, bias, rows, cols);
		
		auto resultBiased = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * 4);
		auto resultBias = oa::EngineResourceAccess::allocBufferBar(rt_, cols * 4);
		ASSERT_TRUE(resultBiased.isOk() && resultBias.isOk());
		
		auto bufBiased = std::move(resultBiased).getValue();
		auto bufBias = std::move(resultBias).getValue();
		
		std::memcpy(bufBiased.mappedPtr, biased.data(), rows * cols * 4);
		std::memcpy(bufBias.mappedPtr, bias.data(), cols * 4);
		
		struct { oa::U32 rows; oa::U32 cols; } pc = { rows, cols };
		oavk::Buffer bufs[] = {bufBiased, bufBias};
		oa::U32 groups = (rows * cols + 255) / 256;
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "BiasAdd", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());
		
		std::vector<float> biased_gpu(rows * cols);
		std::memcpy(biased_gpu.data(), bufBiased.mappedPtr, rows * cols * 4);
		
		compareResults(biased_ref, biased_gpu, rows * cols, 1e-6f, "BiasAdd");
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufBiased);
		oa::EngineResourceAccess::freeBuffer(rt_, bufBias);
	};
	
	testBiasAdd(32, 128);   // Small
	testBiasAdd(256, 512);  // Medium
	testBiasAdd(1024, 768); // large (typical transformer)
}

// ============================================================================
// Normalization Tests
// ============================================================================

TEST_VK(MlKernels, LayerNorm) {
	auto testLayerNorm = [this](oa::U32 rows, oa::U32 cols) {
		constexpr float eps = 1e-5f;
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
		
		std::vector<float> x(rows * cols), weight(cols), bias(cols), out_ref(rows * cols);
		for (auto &v : x) v = dist(rng);
		for (auto &v : weight) v = dist(rng);
		for (auto &v : bias) v = dist(rng);
		
		cpuLayerNorm(x, weight, bias, out_ref, rows, cols, eps);
		
		auto resultX = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * 4);
		auto resultWeight = oa::EngineResourceAccess::allocBufferBar(rt_, cols * 4);
		auto resultBias = oa::EngineResourceAccess::allocBufferBar(rt_, cols * 4);
		auto resultOut = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * 4);
		ASSERT_TRUE(resultX.isOk() && resultWeight.isOk() && resultBias.isOk() && resultOut.isOk());
		
		auto bufX = std::move(resultX).getValue();
		auto bufWeight = std::move(resultWeight).getValue();
		auto bufBias = std::move(resultBias).getValue();
		auto bufOut = std::move(resultOut).getValue();
		
		std::memcpy(bufX.mappedPtr, x.data(), rows * cols * 4);
		std::memcpy(bufWeight.mappedPtr, weight.data(), cols * 4);
		std::memcpy(bufBias.mappedPtr, bias.data(), cols * 4);
		
		struct { oa::U32 rows; oa::U32 cols; float eps; } pc = { rows, cols, eps };
		oavk::Buffer bufs[] = {bufX, bufWeight, bufBias, bufOut};
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "LayerNorm", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, rows).isOk());
		
		std::vector<float> out_gpu(rows * cols);
		std::memcpy(out_gpu.data(), bufOut.mappedPtr, rows * cols * 4);
		
		// near-zero outputs need an absolute floor in addition to the relative
		// bound; otherwise sub-micro-unit FP32 differences dominate the ratio.
		compareResults(out_ref, out_gpu, rows * cols, 2e-2f, "LayerNorm", 1e-5f);
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufX);
		oa::EngineResourceAccess::freeBuffer(rt_, bufWeight);
		oa::EngineResourceAccess::freeBuffer(rt_, bufBias);
		oa::EngineResourceAccess::freeBuffer(rt_, bufOut);
	};
	
	testLayerNorm(8, 128);    // Small
	testLayerNorm(32, 512);   // Medium
	testLayerNorm(128, 768);  // large (typical transformer)
}

TEST_VK(MlKernels, RmsNorm) {
	auto testRmsNorm = [this](oa::U32 rows, oa::U32 cols) {
		constexpr float eps = 1e-6f;
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
		
		std::vector<float> x(rows * cols), weight(cols), out_ref(rows * cols);
		for (auto &v : x) v = dist(rng);
		for (auto &v : weight) v = dist(rng);
		
		cpuRmsNorm(x, weight, out_ref, rows, cols, eps);
		
		auto resultX = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * 4);
		auto resultWeight = oa::EngineResourceAccess::allocBufferBar(rt_, cols * 4);
		auto resultOut = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * 4);
		ASSERT_TRUE(resultX.isOk() && resultWeight.isOk() && resultOut.isOk());
		
		auto bufX = std::move(resultX).getValue();
		auto bufWeight = std::move(resultWeight).getValue();
		auto bufOut = std::move(resultOut).getValue();
		
		std::memcpy(bufX.mappedPtr, x.data(), rows * cols * 4);
		std::memcpy(bufWeight.mappedPtr, weight.data(), cols * 4);
		
		struct { oa::U32 rows; oa::U32 cols; float eps; } pc = { rows, cols, eps };
		oavk::Buffer bufs[] = {bufX, bufWeight, bufOut};
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "RmsNorm", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, rows).isOk());
		
		std::vector<float> out_gpu(rows * cols);
		std::memcpy(out_gpu.data(), bufOut.mappedPtr, rows * cols * 4);
		
		compareResults(out_ref, out_gpu, rows * cols, 1e-4f, "RmsNorm");
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufX);
		oa::EngineResourceAccess::freeBuffer(rt_, bufWeight);
		oa::EngineResourceAccess::freeBuffer(rt_, bufOut);
	};
	
	testRmsNorm(8, 128);
	testRmsNorm(32, 512);
	testRmsNorm(128, 768);
}

TEST_VK(MlKernels, Softmax) {
	auto testSoftmax = [this](oa::U32 rows, oa::U32 cols) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
		
		std::vector<float> x(rows * cols), out_ref(rows * cols);
		for (auto &v : x) v = dist(rng);
		
		cpuSoftmax(x, out_ref, rows, cols);
		
		auto resultX = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * 4);
		auto resultOut = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * 4);
		ASSERT_TRUE(resultX.isOk() && resultOut.isOk());
		
		auto bufX = std::move(resultX).getValue();
		auto bufOut = std::move(resultOut).getValue();
		
		std::memcpy(bufX.mappedPtr, x.data(), rows * cols * 4);
		
		struct { oa::U32 outerSize; oa::U32 dimSize; oa::U32 innerSize; } pc = {
			rows, cols, 1};
		oavk::Buffer bufs[] = {bufX, bufOut};
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "Softmax", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, rows).isOk());
		
		std::vector<float> out_gpu(rows * cols);
		std::memcpy(out_gpu.data(), bufOut.mappedPtr, rows * cols * 4);
		
		// Softmax uses exp - slightly relaxed tolerance
		compareResults(out_ref, out_gpu, rows * cols, 1e-5f, "Softmax");
		
		// verify sum to 1.0 for each row
		for (oa::U32 row = 0; row < rows; ++row) {
			float sum = 0.0f;
			for (oa::U32 col = 0; col < cols; ++col) {
				sum += out_gpu[row * cols + col];
			}
			EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax row " << row << " doesn't sum to 1.0";
		}
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufX);
		oa::EngineResourceAccess::freeBuffer(rt_, bufOut);
	};
	
	testSoftmax(8, 64);
	testSoftmax(32, 128);
	testSoftmax(128, 512);
}

TEST_VK(MlKernels, NarrowRowNormalizationSchedules) {
	constexpr oa::U32 rows = 7U;
	constexpr float eps = 1e-5F;
	constexpr float scale = 0.375F;
	std::mt19937 rng(1776);
	std::uniform_real_distribution<float> dist(-2.0F, 2.0F);

	for (const oa::U32 cols : {1U, 3U, 16U, 17U, 31U, 32U}) {
		std::vector<float> x(rows * cols), weight(cols), bias(cols);
		std::vector<float> layerRef(rows * cols), layerGpu(rows * cols);
		std::vector<float> scores(rows * cols), mask(rows * cols);
		std::vector<float> transformed(rows * cols), softmaxRef(rows * cols), softmaxGpu(rows * cols);
		for (auto& value : x) value = dist(rng);
		for (auto& value : weight) value = dist(rng);
		for (auto& value : bias) value = dist(rng);
		for (oa::U32 i = 0; i < rows * cols; ++i) {
			scores[i] = dist(rng);
			mask[i] = (i % 5U) == 0U ? -3.0F : 0.0F;
			transformed[i] = scores[i] * scale + mask[i];
		}
		cpuLayerNorm(x, weight, bias, layerRef, rows, cols, eps);
		cpuSoftmax(transformed, softmaxRef, rows, cols);

		auto xResult = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * sizeof(float));
		auto weightResult = oa::EngineResourceAccess::allocBufferBar(rt_, cols * sizeof(float));
		auto biasResult = oa::EngineResourceAccess::allocBufferBar(rt_, cols * sizeof(float));
		auto layerOutResult = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * sizeof(float));
		auto scoresResult = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * sizeof(float));
		auto maskResult = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * sizeof(float));
		auto softmaxOutResult = oa::EngineResourceAccess::allocBufferBar(rt_, rows * cols * sizeof(float));
		ASSERT_TRUE(xResult.isOk() and weightResult.isOk() and biasResult.isOk()
			and layerOutResult.isOk() and scoresResult.isOk() and maskResult.isOk()
			and softmaxOutResult.isOk());
		auto xBuffer = std::move(xResult).getValue();
		auto weightBuffer = std::move(weightResult).getValue();
		auto biasBuffer = std::move(biasResult).getValue();
		auto layerOutBuffer = std::move(layerOutResult).getValue();
		auto scoresBuffer = std::move(scoresResult).getValue();
		auto maskBuffer = std::move(maskResult).getValue();
		auto softmaxOutBuffer = std::move(softmaxOutResult).getValue();
		std::memcpy(xBuffer.mappedPtr, x.data(), rows * cols * sizeof(float));
		std::memcpy(weightBuffer.mappedPtr, weight.data(), cols * sizeof(float));
		std::memcpy(biasBuffer.mappedPtr, bias.data(), cols * sizeof(float));
		std::memcpy(scoresBuffer.mappedPtr, scores.data(), rows * cols * sizeof(float));
		std::memcpy(maskBuffer.mappedPtr, mask.data(), rows * cols * sizeof(float));

		struct { oa::U32 rows; oa::U32 cols; float eps; } layerPush{rows, cols, eps};
		oavk::Buffer layerBuffers[] = {
			xBuffer, weightBuffer, biasBuffer, layerOutBuffer};
		ASSERT_TRUE(oavk::Dispatch::run(
			rt_, "LayerNormN32", layerBuffers, &layerPush, sizeof(layerPush),
			oa::ScalarType::Float32, rows).isOk());

		struct { oa::U32 rows; oa::U32 cols; float scale; } softmaxPush{rows, cols, scale};
		oavk::Buffer softmaxBuffers[] = {
			scoresBuffer, maskBuffer, softmaxOutBuffer};
		ASSERT_TRUE(oavk::Dispatch::run(
			rt_, "SoftmaxScaledMaskedN32", softmaxBuffers,
			&softmaxPush, sizeof(softmaxPush), oa::ScalarType::Float32, rows).isOk());

		std::memcpy(layerGpu.data(), layerOutBuffer.mappedPtr,
			rows * cols * sizeof(float));
		std::memcpy(softmaxGpu.data(), softmaxOutBuffer.mappedPtr,
			rows * cols * sizeof(float));
		compareResults(layerRef, layerGpu, rows * cols, 2e-2F,
			"LayerNormN32", 1e-5F);
		compareResults(softmaxRef, softmaxGpu, rows * cols, 1e-5F,
			"SoftmaxScaledMaskedN32", 1e-6F);

		oa::EngineResourceAccess::freeBuffer(rt_, xBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, weightBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, biasBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, layerOutBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, scoresBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, maskBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, softmaxOutBuffer);
	}
}

TEST_VK(MlKernels, BmmTiled16Tails) {
	struct Shape { oa::U32 batch, m, k, p; };
	constexpr Shape shapes[] = {
		{1U, 8U, 8U, 8U},
		{2U, 15U, 17U, 9U},
		{3U, 16U, 32U, 16U},
		{2U, 17U, 19U, 31U},
		{2U, 16U, 16U, 32U},
	};
	std::mt19937 rng(1777);
	std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

	for (const auto& shape : shapes) {
		const oa::U32 aCount = shape.batch * shape.m * shape.k;
		const oa::U32 bCount = shape.batch * shape.k * shape.p;
		const oa::U32 outCount = shape.batch * shape.m * shape.p;
		std::vector<float> a(aCount), b(bCount), ref(outCount), gpu(outCount);
		for (auto& value : a) value = dist(rng);
		for (auto& value : b) value = dist(rng);
		cpuBmm(a, b, ref, shape.batch, shape.m, shape.k, shape.p);

		auto aResult = oa::EngineResourceAccess::allocBufferBar(rt_, aCount * sizeof(float));
		auto bResult = oa::EngineResourceAccess::allocBufferBar(rt_, bCount * sizeof(float));
		auto outResult = oa::EngineResourceAccess::allocBufferBar(rt_, outCount * sizeof(float));
		ASSERT_TRUE(aResult.isOk() and bResult.isOk() and outResult.isOk());
		auto aBuffer = std::move(aResult).getValue();
		auto bBuffer = std::move(bResult).getValue();
		auto outBuffer = std::move(outResult).getValue();
		std::memcpy(aBuffer.mappedPtr, a.data(), aCount * sizeof(float));
		std::memcpy(bBuffer.mappedPtr, b.data(), bCount * sizeof(float));

		struct { oa::U32 batch, m, k, p; } push{
			shape.batch, shape.m, shape.k, shape.p};
		oavk::Buffer buffers[] = {aBuffer, bBuffer, outBuffer};
		ASSERT_TRUE(oavk::Dispatch::run(
			rt_, "BmmTiled16", buffers, &push, sizeof(push),
			oa::ScalarType::Float32, (shape.p + 15U) / 16U,
			(shape.m + 15U) / 16U, shape.batch).isOk());
		std::memcpy(gpu.data(), outBuffer.mappedPtr,
			outCount * sizeof(float));
		compareResults(ref, gpu, outCount, 2e-4F, "BmmTiled16", 2e-5F);

		oa::EngineResourceAccess::freeBuffer(rt_, aBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, bBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, outBuffer);
	}
}

TEST_VK(MlKernels, BmmNtTiled16Tails) {
	struct Shape { oa::U32 batch, m, k, p; };
	constexpr Shape shapes[] = {
		{1U, 8U, 8U, 8U},
		{2U, 15U, 17U, 9U},
		{3U, 16U, 32U, 16U},
		{2U, 17U, 19U, 31U},
		{2U, 16U, 16U, 32U},
	};
	std::mt19937 rng(1778);
	std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

	for (const auto& shape : shapes) {
		const oa::U32 aCount = shape.batch * shape.m * shape.k;
		const oa::U32 bCount = shape.batch * shape.p * shape.k;
		const oa::U32 outCount = shape.batch * shape.m * shape.p;
		std::vector<float> a(aCount), b(bCount), ref(outCount), gpu(outCount);
		for (auto& value : a) value = dist(rng);
		for (auto& value : b) value = dist(rng);
		cpuBmmNt(a, b, ref, shape.batch, shape.m, shape.k, shape.p);

		auto aResult = oa::EngineResourceAccess::allocBufferBar(rt_, aCount * sizeof(float));
		auto bResult = oa::EngineResourceAccess::allocBufferBar(rt_, bCount * sizeof(float));
		auto outResult = oa::EngineResourceAccess::allocBufferBar(rt_, outCount * sizeof(float));
		ASSERT_TRUE(aResult.isOk() and bResult.isOk() and outResult.isOk());
		auto aBuffer = std::move(aResult).getValue();
		auto bBuffer = std::move(bResult).getValue();
		auto outBuffer = std::move(outResult).getValue();
		std::memcpy(aBuffer.mappedPtr, a.data(), aCount * sizeof(float));
		std::memcpy(bBuffer.mappedPtr, b.data(), bCount * sizeof(float));

		struct { oa::U32 batch, m, k, p; } push{
			shape.batch, shape.m, shape.k, shape.p};
		oavk::Buffer buffers[] = {aBuffer, bBuffer, outBuffer};
		ASSERT_TRUE(oavk::Dispatch::run(
			rt_, "BmmNtTiled16", buffers, &push, sizeof(push),
			oa::ScalarType::Float32, (shape.p + 15U) / 16U,
			(shape.m + 15U) / 16U, shape.batch).isOk());
		std::memcpy(gpu.data(), outBuffer.mappedPtr,
			outCount * sizeof(float));
		compareResults(ref, gpu, outCount, 2e-4F, "BmmNtTiled16", 2e-5F);

		oa::EngineResourceAccess::freeBuffer(rt_, aBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, bBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, outBuffer);
	}
}

TEST_VK(MlKernels, LinearWeightBiasBwdRows32Tails) {
	struct Shape { oa::U32 m, n, k; };
	constexpr Shape shapes[] = {
		{1U, 1U, 1U},
		{7U, 17U, 9U},
		{65U, 33U, 31U},
		{1024U, 32U, 32U},
		{1031U, 37U, 35U},
	};
	std::mt19937 rng(1779);
	std::uniform_real_distribution<float> dist(-0.25F, 0.25F);

	for (const auto& shape : shapes) {
		std::vector<float> input(shape.m * shape.k);
		std::vector<float> gradOutput(shape.m * shape.n);
		std::vector<float> weightRef(shape.n * shape.k);
		std::vector<float> biasRef(shape.n);
		for (auto& value : input) value = dist(rng);
		for (auto& value : gradOutput) value = dist(rng);
		cpuLinearWeightBiasBwd(
			input, gradOutput, weightRef, biasRef, shape.m, shape.n, shape.k);

		auto inputResult = oa::EngineResourceAccess::allocBufferBar(rt_, input.size() * sizeof(float));
		auto gradOutputResult = oa::EngineResourceAccess::allocBufferBar(rt_, gradOutput.size() * sizeof(float));
		auto weightResult = oa::EngineResourceAccess::allocBufferBar(rt_, weightRef.size() * sizeof(float));
		auto biasResult = oa::EngineResourceAccess::allocBufferBar(rt_, biasRef.size() * sizeof(float));
		ASSERT_TRUE(inputResult.isOk() and gradOutputResult.isOk()
			and weightResult.isOk() and biasResult.isOk());
		auto inputBuffer = std::move(inputResult).getValue();
		auto gradOutputBuffer = std::move(gradOutputResult).getValue();
		auto weightBuffer = std::move(weightResult).getValue();
		auto biasBuffer = std::move(biasResult).getValue();
		std::memcpy(inputBuffer.mappedPtr, input.data(), input.size() * sizeof(float));
		std::memcpy(gradOutputBuffer.mappedPtr, gradOutput.data(),
			gradOutput.size() * sizeof(float));

		struct { oa::U32 m, n, k; } push{shape.m, shape.n, shape.k};
		oavk::Buffer buffers[] = {
			gradOutputBuffer, inputBuffer, weightBuffer, biasBuffer};
		ASSERT_TRUE(oavk::Dispatch::run(
			rt_, "LinearWeightBiasBwdRows32", buffers, &push, sizeof(push),
			oa::ScalarType::Float32, shape.n, (shape.k + 31U) / 32U, 1U).isOk());

		std::vector<float> weightGpu(weightRef.size());
		std::vector<float> biasGpu(biasRef.size());
		std::memcpy(weightGpu.data(), weightBuffer.mappedPtr,
			weightGpu.size() * sizeof(float));
		std::memcpy(biasGpu.data(), biasBuffer.mappedPtr,
			biasGpu.size() * sizeof(float));
		compareResults(weightRef, weightGpu, static_cast<oa::U32>(weightGpu.size()), 2e-3F,
			"LinearWeightBiasBwdRows32 weight", 2e-4F);
		compareResults(biasRef, biasGpu, static_cast<oa::U32>(biasGpu.size()), 2e-3F,
			"LinearWeightBiasBwdRows32 bias", 2e-4F);

		oa::EngineResourceAccess::freeBuffer(rt_, inputBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, gradOutputBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, weightBuffer);
		oa::EngineResourceAccess::freeBuffer(rt_, biasBuffer);
	}
}

TEST_VK(MlKernels, Swiglu) {
	auto testSwiglu = [this](oa::U32 count) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		
		std::vector<float> gate(count), up(count), out_ref(count);
		for (auto &v : gate) v = dist(rng);
		for (auto &v : up) v = dist(rng);
		
		cpuSwiglu(gate, up, out_ref, count);
		
		auto resultGate = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		auto resultUp = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		auto resultOut = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		ASSERT_TRUE(resultGate.isOk() && resultUp.isOk() && resultOut.isOk());
		
		auto bufGate = std::move(resultGate).getValue();
		auto bufUp = std::move(resultUp).getValue();
		auto bufOut = std::move(resultOut).getValue();
		
		std::memcpy(bufGate.mappedPtr, gate.data(), count * 4);
		std::memcpy(bufUp.mappedPtr, up.data(), count * 4);
		
		struct { oa::U32 count; } pc = { count };
		oavk::Buffer bufs[] = {bufGate, bufUp, bufOut};
		oa::U32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "Swiglu", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.mappedPtr, count * 4);
		
		// SwiGLU uses exp - slightly relaxed tolerance
		compareResults(out_ref, out_gpu, count, 1e-5f, "Swiglu");
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufGate);
		oa::EngineResourceAccess::freeBuffer(rt_, bufUp);
		oa::EngineResourceAccess::freeBuffer(rt_, bufOut);
	};
	
	testSwiglu(1024);
	testSwiglu(4096);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_VK(MlKernels, EdgeCases) {
	// Test single element
	{
		std::vector<float> x = {2.5f}, out_ref(1);
		cpuGelu(x, out_ref, 1);
		
		auto resultX = oa::EngineResourceAccess::allocBufferBar(rt_, 4);
		auto resultOut = oa::EngineResourceAccess::allocBufferBar(rt_, 4);
		ASSERT_TRUE(resultX.isOk() && resultOut.isOk());
		
		auto bufX = std::move(resultX).getValue();
		auto bufOut = std::move(resultOut).getValue();
		
		std::memcpy(bufX.mappedPtr, x.data(), 4);
		
		struct { oa::U32 count; } pc = { 1 };
		oavk::Buffer bufs[] = {bufX, bufOut};
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "Gelu", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, 1).isOk());
		
		float out_gpu;
		std::memcpy(&out_gpu, bufOut.mappedPtr, 4);
		
		EXPECT_NEAR(out_ref[0], out_gpu, 1e-4f) << "Gelu single element";
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufX);
		oa::EngineResourceAccess::freeBuffer(rt_, bufOut);
	}
	
	// Test non-aligned size
	{
		constexpr oa::U32 count = 257;
		std::vector<float> x(count), out_ref(count);
		for (oa::U32 i = 0; i < count; ++i) x[i] = static_cast<float>(i) * 0.01f - 1.0f;
		cpuRelu(x, out_ref, count);
		
		auto resultX = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		auto resultOut = oa::EngineResourceAccess::allocBufferBar(rt_, count * 4);
		ASSERT_TRUE(resultX.isOk() && resultOut.isOk());
		
		auto bufX = std::move(resultX).getValue();
		auto bufOut = std::move(resultOut).getValue();
		
		std::memcpy(bufX.mappedPtr, x.data(), count * 4);
		
		struct { oa::U32 count; } pc = { count };
		oavk::Buffer bufs[] = {bufX, bufOut};
		oa::U32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "Relu", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, groups).isOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.mappedPtr, count * 4);
		
		compareResults(out_ref, out_gpu, count, 1e-6f, "Relu non-aligned");
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufX);
		oa::EngineResourceAccess::freeBuffer(rt_, bufOut);
	}
}

// ============================================================================
// CrossEntropy tests (CRITICAL - gradient bug fixed)
// ============================================================================

TEST_VK(MlKernels, CrossEntropyFwd) {
	auto testCrossEntropy = [this](oa::U32 batch, oa::U32 classes, float tolerance = 1e-5f) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		std::uniform_int_distribution<oa::U32> targetDist(0, classes - 1);
		
		std::vector<float> logits(batch * classes), loss_ref(batch);
		std::vector<oa::U32> targets_u32(batch);
		std::vector<oa::U8> targets(batch);  // GPU kernel expects uint8_t
		
		for (auto &v : logits) v = dist(rng);
		for (auto &t : targets_u32) t = targetDist(rng);
		for (oa::U32 i = 0; i < batch; ++i) targets[i] = static_cast<oa::U8>(targets_u32[i]);
		
		cpuCrossEntropyFwd(logits, targets_u32, loss_ref, batch, classes);
		
		auto resultLogits = oa::EngineResourceAccess::allocBufferBar(rt_, batch * classes * 4);
		auto resultTargets = oa::EngineResourceAccess::allocBufferBar(rt_, batch);  // 1 byte per target
		auto resultLoss = oa::EngineResourceAccess::allocBufferBar(rt_, batch * 4);
		ASSERT_TRUE(resultLogits.isOk() && resultTargets.isOk() && resultLoss.isOk());
		
		auto bufLogits = std::move(resultLogits).getValue();
		auto bufTargets = std::move(resultTargets).getValue();
		auto bufLoss = std::move(resultLoss).getValue();
		
		std::memcpy(bufLogits.mappedPtr, logits.data(), batch * classes * 4);
		std::memcpy(bufTargets.mappedPtr, targets.data(), batch);  // Copy uint8_t targets
		
		struct { oa::U32 batch; oa::U32 classes; oa::U32 target_dtype; } pc = { batch, classes, 0 };  // 0=UInt8
		oavk::Buffer bufs[] = {bufLogits, bufTargets, bufLoss};
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "CrossEntropy", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, batch).isOk());
		
		std::vector<float> loss_gpu(batch);
		std::memcpy(loss_gpu.data(), bufLoss.mappedPtr, batch * 4);
		
		compareResults(loss_ref, loss_gpu, batch, tolerance, "CrossEntropyFwd");
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufLogits);
		oa::EngineResourceAccess::freeBuffer(rt_, bufTargets);
		oa::EngineResourceAccess::freeBuffer(rt_, bufLoss);
	};
	
	testCrossEntropy(8, 10);    // Small: 10 classes
	testCrossEntropy(32, 100);  // Medium: 100 classes
	testCrossEntropy(128, 256); // Large: 256 classes (max for uint8_t)
}


TEST_VK(MlKernels, CrossEntropyBwd) {
	auto testCrossEntropyBwd = [this](oa::U32 batch, oa::U32 classes) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		std::uniform_int_distribution<oa::U32> targetDist(0, classes - 1);
		
		std::vector<float> logits(batch * classes), d_logits_ref(batch * classes);
		std::vector<oa::U32> targets_u32(batch);
		std::vector<oa::U8> targets(batch);  // GPU kernel expects uint8_t
		
		for (auto &v : logits) v = dist(rng);
		for (auto &t : targets_u32) t = targetDist(rng);
		for (oa::U32 i = 0; i < batch; ++i) targets[i] = static_cast<oa::U8>(targets_u32[i]);
		
		cpuCrossEntropyBwd(logits, targets_u32, d_logits_ref, batch, classes);
		
		auto resultLogits = oa::EngineResourceAccess::allocBufferBar(rt_, batch * classes * 4);
		auto resultTargets = oa::EngineResourceAccess::allocBufferBar(rt_, batch * 4);  // 4 bytes per uint32 target
		auto resultDLogits = oa::EngineResourceAccess::allocBufferBar(rt_, batch * classes * 4);
		ASSERT_TRUE(resultLogits.isOk() && resultTargets.isOk() && resultDLogits.isOk());
		
		auto bufLogits = std::move(resultLogits).getValue();
		auto bufTargets = std::move(resultTargets).getValue();
		auto bufDLogits = std::move(resultDLogits).getValue();
		
		std::memcpy(bufLogits.mappedPtr, logits.data(), batch * classes * 4);
		std::memcpy(bufTargets.mappedPtr, targets_u32.data(), batch * 4);  // Copy uint32 targets
		
		struct { oa::U32 batch; oa::U32 classes; oa::U32 target_dtype; } pc = { batch, classes, 1 };  // 1=UInt32
		oavk::Buffer bufs[] = {bufLogits, bufTargets, bufDLogits};
		
		ASSERT_TRUE(oavk::Dispatch::run(rt_, "CrossEntropyBwd", bufs, &pc, sizeof(pc), oa::ScalarType::Float32, batch).isOk());
		
		std::vector<float> d_logits_gpu(batch * classes);
		std::memcpy(d_logits_gpu.data(), bufDLogits.mappedPtr, batch * classes * 4);
		
		compareResults(d_logits_ref, d_logits_gpu, batch * classes, 1e-5f, "CrossEntropyBwd");
		
		// verify gradient sum per row ≈ 0 (softmax property)
		for (oa::U32 row = 0; row < batch; ++row) {
			float row_sum = 0.0f;
			for (oa::U32 col = 0; col < classes; ++col) {
				row_sum += d_logits_gpu[row * classes + col];
			}
			EXPECT_NEAR(row_sum, 0.0f, 1e-5f) << "CrossEntropyBwd row " << row << " gradients don't sum to 0";
		}
		
		oa::EngineResourceAccess::freeBuffer(rt_, bufLogits);
		oa::EngineResourceAccess::freeBuffer(rt_, bufTargets);
		oa::EngineResourceAccess::freeBuffer(rt_, bufDLogits);
	};
	
	testCrossEntropyBwd(8, 10);    // Small: 10 classes
	testCrossEntropyBwd(32, 100);  // Medium: 100 classes
	testCrossEntropyBwd(128, 256); // Large: 256 classes (max for uint8_t)
}

TEST_VK(MlKernels, CrossEntropyFusedLossGradUsesPortableByteTargets) {
	constexpr oa::U32 batch = 5;
	constexpr oa::U32 classes = 4;
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F, 1.0F,
		-0.5F, 1.25F, 0.75F, -2.0F,
		0.1F, 0.2F, 0.3F, 0.4F,
		3.0F, 2.0F, 1.0F, 0.0F,
		-1.0F, -0.5F, 0.5F, 1.5F,
	};
	const std::vector<oa::U32> targets = {0U, 1U, 2U, 3U, 1U};
	const std::vector<oa::U8> byteTargets = {0U, 1U, 2U, 3U, 1U};
	std::vector<float> expectedLoss(batch);
	std::vector<float> expectedGradient(batch * classes);
	cpuCrossEntropyFwd(
		logits, targets, expectedLoss, batch, classes);
	cpuCrossEntropyBwd(
		logits, targets, expectedGradient, batch, classes);

	auto logitsResult = oa::EngineResourceAccess::allocBufferBar(rt_, logits.size() * sizeof(float));
	auto targetsResult = oa::EngineResourceAccess::allocBufferBar(rt_, byteTargets.size());
	auto lossResult = oa::EngineResourceAccess::allocBufferBar(rt_, batch * sizeof(float));
	auto gradientResult = oa::EngineResourceAccess::allocBufferBar(rt_,
		expectedGradient.size() * sizeof(float));
	ASSERT_TRUE(logitsResult.isOk() and targetsResult.isOk()
		and lossResult.isOk() and gradientResult.isOk());
	auto logitsBuffer = std::move(logitsResult).getValue();
	auto targetsBuffer = std::move(targetsResult).getValue();
	auto lossBuffer = std::move(lossResult).getValue();
	auto gradientBuffer = std::move(gradientResult).getValue();
	std::memcpy(
		logitsBuffer.mappedPtr, logits.data(), logits.size() * sizeof(float));
	std::memcpy(
		targetsBuffer.mappedPtr, byteTargets.data(), byteTargets.size());

	struct { oa::U32 batch, classes, targetDtype; }
		push{batch, classes, 0U};
	oavk::Buffer buffers[] = {
		logitsBuffer, targetsBuffer, lossBuffer, gradientBuffer};
	ASSERT_TRUE(oavk::Dispatch::run(
		rt_, "CrossEntropyLossGradBwd", buffers,
		&push, sizeof(push), oa::ScalarType::Float32, batch).isOk());

	std::vector<float> actualLoss(batch);
	std::vector<float> actualGradient(batch * classes);
	std::memcpy(
		actualLoss.data(), lossBuffer.mappedPtr, actualLoss.size() * sizeof(float));
	std::memcpy(
		actualGradient.data(), gradientBuffer.mappedPtr,
		actualGradient.size() * sizeof(float));
	compareResults(
		expectedLoss, actualLoss, batch, 1.0e-5F,
		"CrossEntropyFusedLoss");
	compareResults(
		expectedGradient, actualGradient, batch * classes, 1.0e-5F,
		"CrossEntropyFusedGradient");

	oa::EngineResourceAccess::freeBuffer(rt_, logitsBuffer);
	oa::EngineResourceAccess::freeBuffer(rt_, targetsBuffer);
	oa::EngineResourceAccess::freeBuffer(rt_, lossBuffer);
	oa::EngineResourceAccess::freeBuffer(rt_, gradientBuffer);
}

TEST_VK(MlKernels, MaskedCrossEntropyFwdBwd) {
	constexpr oa::U32 rows = 4;
	constexpr oa::U32 classes = 3;
	constexpr oa::U32 valid = 2;
	const std::vector<float> logits = {
		1.0f, 2.0f, 3.0f,  3.0f, 1.0f, 0.0f,
		9.0f, 8.0f, 7.0f, -2.0f, 0.0f, 2.0f};
	const std::vector<oa::U32> targets = {2, 0, 0, 1};
	const std::vector<oa::U8> byteTargets = {2, 0, 0, 1};
	const std::vector<float> mask = {1.0f, 1.0f, 0.0f, 0.0f};
	std::vector<float> refLoss(rows), refGrad(rows * classes);
	cpuCrossEntropyFwd(logits, targets, refLoss, rows, classes);
	cpuCrossEntropyBwd(logits, targets, refGrad, rows, classes);
	for (oa::U32 r = 0; r < rows; ++r) {
		if (mask[r] == 0.0f) {
			refLoss[r] = 0.0f;
			for (oa::U32 c = 0; c < classes; ++c) refGrad[r * classes + c] = 0.0f;
		} else {
			// CPU helper normalized by rows; masked CE normalizes by valid rows.
			for (oa::U32 c = 0; c < classes; ++c)
				refGrad[r * classes + c] *= static_cast<float>(rows) / valid;
		}
	}

	auto logitsResult = oa::EngineResourceAccess::allocBufferBar(rt_, logits.size() * sizeof(float));
	auto targetsResult = oa::EngineResourceAccess::allocBufferBar(rt_, byteTargets.size());
	auto maskResult = oa::EngineResourceAccess::allocBufferBar(rt_, mask.size() * sizeof(float));
	auto lossResult = oa::EngineResourceAccess::allocBufferBar(rt_, rows * sizeof(float));
	auto gradResult = oa::EngineResourceAccess::allocBufferBar(rt_, logits.size() * sizeof(float));
	ASSERT_TRUE(logitsResult.isOk() && targetsResult.isOk() && maskResult.isOk()
		&& lossResult.isOk() && gradResult.isOk());
	auto logitsBuf = std::move(logitsResult).getValue();
	auto targetsBuf = std::move(targetsResult).getValue();
	auto maskBuf = std::move(maskResult).getValue();
	auto lossBuf = std::move(lossResult).getValue();
	auto gradBuf = std::move(gradResult).getValue();
	std::memcpy(logitsBuf.mappedPtr, logits.data(), logits.size() * sizeof(float));
	std::memcpy(targetsBuf.mappedPtr, byteTargets.data(), byteTargets.size());
	std::memcpy(maskBuf.mappedPtr, mask.data(), mask.size() * sizeof(float));

	struct { oa::U32 rows, classes, targetDtype; } fwd{rows, classes, 0};
	oavk::Buffer fwdBufs[] = {logitsBuf, targetsBuf, maskBuf, lossBuf};
	ASSERT_TRUE(oavk::Dispatch::run(rt_, "MaskedCrossEntropy", fwdBufs,
		&fwd, sizeof(fwd), oa::ScalarType::Float32, rows).isOk());
	std::vector<float> gpuLoss(rows);
	std::memcpy(gpuLoss.data(), lossBuf.mappedPtr, rows * sizeof(float));
	compareResults(refLoss, gpuLoss, rows, 1e-5f, "MaskedCrossEntropyFwd");

	struct { oa::U32 rows, classes, targetDtype, ValidCount; } bwd{rows, classes, 0, valid};
	oavk::Buffer bwdBufs[] = {logitsBuf, targetsBuf, maskBuf, gradBuf};
	ASSERT_TRUE(oavk::Dispatch::run(rt_, "MaskedCrossEntropyBwd", bwdBufs,
		&bwd, sizeof(bwd), oa::ScalarType::Float32, rows).isOk());
	std::vector<float> gpuGrad(logits.size());
	std::memcpy(gpuGrad.data(), gradBuf.mappedPtr, gpuGrad.size() * sizeof(float));
	compareResults(refGrad, gpuGrad, static_cast<oa::U32>(gpuGrad.size()), 1e-5f,
		"MaskedCrossEntropyBwd");

	oa::EngineResourceAccess::freeBuffer(rt_, logitsBuf); oa::EngineResourceAccess::freeBuffer(rt_, targetsBuf); oa::EngineResourceAccess::freeBuffer(rt_, maskBuf);
	oa::EngineResourceAccess::freeBuffer(rt_, lossBuf); oa::EngineResourceAccess::freeBuffer(rt_, gradBuf);
}



// ============================================================================
// Module Integration tests (high-level API)
// ============================================================================

TEST(NN, Linear2D) {
	oa::Linear linear(4, 3);
	auto out = linear.forward(oa::FnMatrix::ones(oa::MatrixShape{2, 4}));
	expectShape(out, {2, 3});
	expectFinite(out);
}

TEST(NN, Linear3D) {
	oa::Linear linear(8, 4);
	auto out = linear.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 5, 8}));
	expectShape(out, {2, 5, 4});
	expectFinite(out);
}

TEST(NN, Embedding) {
	oa::Embedding emb(10, 4);
	auto out = emb.forward(makeByteIndices({1, 5, 9}));
	expectShape(out, {3, 4});
}

TEST(NN, LayerNorm) {
	oa::LayerNorm ln(4);
	auto out = ln.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 4}));
	expectShape(out, {2, 4});
	expectFinite(out);
}

TEST(NN, RMSNorm) {
	oa::RmsNorm rn(8);
	auto out = rn.forward(oa::FnMatrix::rand(oa::MatrixShape{3, 8}));
	expectShape(out, {3, 8});
	expectFinite(out);
}

TEST(NN, Activations) {
	auto input = oa::FnMatrix::rand(oa::MatrixShape{2, 4});
	oa::Relu relu; oa::Gelu gelu; oa::Silu silu;
	expectFinite(relu.forward(input));
	expectFinite(gelu.forward(input));
	expectFinite(silu.forward(input));
}

TEST(NN, SoftmaxModule) {
	oa::Softmax sm(-1);
	auto out = sm.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 4}));
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	expectValidProbability(out);
}

TEST(NN, Conv1d) {
	oa::Conv1d conv(1, 2, 3, 1, 0);
	auto out = conv.forward(oa::FnMatrix::rand(oa::MatrixShape{1, 1, 10}));
	expectShape(out, {1, 2, 8});
}

TEST(NN, Conv1dGradCheck) {
	// Finite-difference check of conv1dBwdData (dX) and conv1dBwdWeight (dW, dB)
	// against the forward kernel. loss = sum(out), so dOut = ones and the analytic
	// grads come straight from the Bwd kernels.
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I32 B = 1, cin = 2, L = 5, cout = 3, K = 3;
	const oa::U32 stride = 1, pad = 1;

	std::vector<float> xh(static_cast<size_t>(B) * cin * L);
	std::vector<float> wh(static_cast<size_t>(cout) * cin * K);
	std::vector<float> bh(static_cast<size_t>(cout));
	for (size_t i = 0; i < xh.size(); ++i) xh[i] = std::sin(0.7 * static_cast<double>(i) + 1.0);
	for (size_t i = 0; i < wh.size(); ++i) wh[i] = std::cos(0.5 * static_cast<double>(i) + 0.3);
	for (size_t i = 0; i < bh.size(); ++i) bh[i] = 0.1 * static_cast<double>(i);

	auto mk = [](const std::vector<float>& v, const oa::MatrixShape& s) {
		return oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(v.data()), v.size() * sizeof(float)), s);
	};
	auto fwdLoss = [&](const std::vector<float>& xv, const std::vector<float>& wv, const std::vector<float>& bv) -> float {
		ctx.clear();
		auto o = oa::FnMatrix::conv1dGemm(mk(xv, oa::MatrixShape{B, cin, L}), mk(wv, oa::MatrixShape{cout, cin, K}),
			mk(bv, oa::MatrixShape{cout}), stride, pad);
		auto s = oa::FnMatrix::sum(o);
		(void)testSubmitAndWait(ctx);
		return s.at(0);
	};

	// Analytic grads (dOut = ones).
	ctx.clear();
	auto X = mk(xh, oa::MatrixShape{B, cin, L});
	auto W = mk(wh, oa::MatrixShape{cout, cin, K});
	auto o = oa::FnMatrix::conv1dGemm(X, W, mk(bh, oa::MatrixShape{cout}), stride, pad);
	auto dOut = oa::FnMatrix::ones(o.getShape());
	auto dX = oa::FnMatrix::conv1dBwdData(dOut, W, stride, pad, 1, X.getShape());
	auto dwb = oa::FnMatrix::conv1dBwdWeight(X, dOut, W, stride, pad, 1);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const float eps = 1e-3f;
	int badX = 0, badW = 0, badB = 0;
	for (size_t i = 0; i < xh.size(); ++i) {
		auto xp = xh; xp[i] += eps; auto xm = xh; xm[i] -= eps;
		const float num = (fwdLoss(xp, wh, bh) - fwdLoss(xm, wh, bh)) / (2 * eps);
		const float ana = dX.at(static_cast<oa::I64>(i));
		if (std::fabs(num - ana) > 1e-2f) { ++badX; printf("  dX[%zu] num=%.4f ana=%.4f\n", i, num, ana); }
	}
	for (size_t i = 0; i < wh.size(); ++i) {
		auto wp = wh; wp[i] += eps; auto wm = wh; wm[i] -= eps;
		const float num = (fwdLoss(xh, wp, bh) - fwdLoss(xh, wm, bh)) / (2 * eps);
		const float ana = dwb.gradWeight.at(static_cast<oa::I64>(i));
		if (std::fabs(num - ana) > 1e-2f) { ++badW; printf("  dW[%zu] num=%.4f ana=%.4f\n", i, num, ana); }
	}
	for (size_t i = 0; i < bh.size(); ++i) {
		auto bp = bh; bp[i] += eps; auto bm = bh; bm[i] -= eps;
		const float num = (fwdLoss(xh, wh, bp) - fwdLoss(xh, wh, bm)) / (2 * eps);
		const float ana = dwb.gradBias.at(static_cast<oa::I64>(i));
		if (std::fabs(num - ana) > 1e-2f) { ++badB; printf("  dB[%zu] num=%.4f ana=%.4f\n", i, num, ana); }
	}
	printf("Conv1dGradCheck: badX=%d/%zu badW=%d/%zu badB=%d/%zu\n",
		badX, xh.size(), badW, wh.size(), badB, bh.size());
	EXPECT_EQ(badX, 0); EXPECT_EQ(badW, 0); EXPECT_EQ(badB, 0);
}

// forward correctness of the 1-D conv GEMM path (Conv1dGemm / Conv1dReluGemm) —
// THE conv path now that the scalar direct kernel is retired — against an
// independent CPU nested-loop reference, across every conv shape the tokenizers
// use: the bare K3/S1/P1 encode/decode convs, the strided K4/S2/P1 downsample, and
// a channel-mixing case (inC != outC).
TEST(NN, conv1dGemmMatchesCpuReference) {
	auto& ctx = oa::ExecutionSession::getActive();

	struct Cfg { oa::I32 b, cin, l, cout, k, s, p; const char* name; };
	const Cfg cfgs[] = {
		{2, 8, 16, 8, 3, 1, 1, "K3/S1/P1 (encode/decode)"},
		{2, 8, 16, 8, 4, 2, 1, "K4/S2/P1 (downsample)"},
		{1, 6, 12, 10, 3, 1, 1, "K3/S1/P1 inC!=outC"},
	};

	auto mk = [](const std::vector<float>& v, const oa::MatrixShape& s) {
		return oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(v.data()), v.size() * sizeof(float)), s);
	};

	for (const auto& c : cfgs) {
		const oa::I32 outL = ((c.l + 2 * c.p) - (c.k - 1) - 1) / c.s + 1;
		std::vector<float> xh(static_cast<size_t>(c.b) * c.cin * c.l);
		std::vector<float> wh(static_cast<size_t>(c.cout) * c.cin * c.k);
		std::vector<float> bh(static_cast<size_t>(c.cout));
		for (size_t i = 0; i < xh.size(); ++i) xh[i] = static_cast<float>(std::sin(0.7 * static_cast<double>(i) + 1.0));
		for (size_t i = 0; i < wh.size(); ++i) wh[i] = static_cast<float>(std::cos(0.5 * static_cast<double>(i) + 0.3));
		for (size_t i = 0; i < bh.size(); ++i) bh[i] = static_cast<float>(0.1 * static_cast<double>(i) - 0.2);

		// CPU reference conv: out[n,oc,ol] = bias[oc] + sum_{ic,k} x[n,ic,ol*S-P+k] * w[oc,ic,k]
		std::vector<float> ref(static_cast<size_t>(c.b) * c.cout * outL);
		auto xAt = [&](oa::I32 n, oa::I32 ic, oa::I32 l) {
			return xh[((static_cast<size_t>(n) * c.cin) + ic) * c.l + l];
		};
		for (oa::I32 n = 0; n < c.b; ++n)
		for (oa::I32 oc = 0; oc < c.cout; ++oc)
		for (oa::I32 ol = 0; ol < outL; ++ol) {
			double acc = bh[static_cast<size_t>(oc)];
			for (oa::I32 ic = 0; ic < c.cin; ++ic)
			for (oa::I32 k = 0; k < c.k; ++k) {
				const oa::I32 l = ol * c.s - c.p + k;
				if (l >= 0 && l < c.l) acc += static_cast<double>(xAt(n, ic, l)) * wh[(static_cast<size_t>(oc) * c.cin + ic) * c.k + k];
			}
			ref[(static_cast<size_t>(n) * c.cout + oc) * outL + ol] = static_cast<float>(acc);
		}

		ctx.clear();
		auto xMatrix = mk(xh, oa::MatrixShape{c.b, c.cin, c.l});
		auto weight = mk(wh, oa::MatrixShape{c.cout, c.cin, c.k});
		auto bias = mk(bh, oa::MatrixShape{c.cout});
		auto gemm = oa::FnMatrix::conv1dGemm(
			xMatrix, weight, bias, c.s, c.p, 1);
		auto gemmRelu = oa::FnMatrix::conv1dReluGemm(
			xMatrix, weight, bias, c.s, c.p, 1);
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk()) << c.name;

		ASSERT_EQ(gemm.numElements(), static_cast<oa::I64>(ref.size())) << c.name;
		float maxBare = 0.0f, maxRelu = 0.0f;
		for (oa::I64 i = 0; i < gemm.numElements(); ++i) {
			const float r = ref[static_cast<size_t>(i)];
			maxBare = std::max(maxBare, std::fabs(gemm.at(i) - r));
			maxRelu = std::max(maxRelu, std::fabs(gemmRelu.at(i) - std::max(0.0f, r)));
		}
		printf("conv1dGemmMatchesCpuReference[%s]: maxBareDiff=%.2e maxReluDiff=%.2e\n",
			c.name, maxBare, maxRelu);
		EXPECT_LT(maxBare, 2e-3f) << c.name << ": Conv1dGemm disagrees with CPU reference";
		EXPECT_LT(maxRelu, 2e-3f) << c.name << ": Conv1dReluGemm disagrees with CPU reference";
	}
}

// Finite-difference gradcheck of the GEMM-composed autograd backward (Im2Col1d +
// MatMulNt + BiasAdd), the backward that oa::Conv1d gets once its hand-written
// GradConv1d node is dropped. Confirms the composed graph produces correct
// dX / dW / dB, not just that it runs. CrossEntropy over the conv output as
// [rows, cout] logits is the differentiable scalar root (Sum has no autograd node).
TEST(NN, conv1dGemmGradCheck) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I32 B = 1, cin = 2, L = 5, cout = 3, K = 3;
	const oa::I32 stride = 1, pad = 1;
	const oa::I32 outL = ((L + 2 * pad) - (K - 1) - 1) / stride + 1;
	const oa::I64 rows = static_cast<oa::I64>(B) * outL;

	std::vector<float> xh(static_cast<size_t>(B) * cin * L);
	std::vector<float> wh(static_cast<size_t>(cout) * cin * K);
	std::vector<float> bh(static_cast<size_t>(cout));
	std::vector<oa::U32> th(static_cast<size_t>(rows));
	for (size_t i = 0; i < xh.size(); ++i) xh[i] = std::sin(0.7 * static_cast<double>(i) + 1.0);
	for (size_t i = 0; i < wh.size(); ++i) wh[i] = std::cos(0.5 * static_cast<double>(i) + 0.3);
	for (size_t i = 0; i < bh.size(); ++i) bh[i] = 0.1 * static_cast<double>(i);
	for (size_t i = 0; i < th.size(); ++i) th[i] = static_cast<oa::U32>(i % cout);

	auto mk = [](const std::vector<float>& v, const oa::MatrixShape& s) {
		return oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(v.data()), v.size() * sizeof(float)), s);
	};
	auto mkTargets = [&] {
		auto t = oa::FnMatrix::empty(oa::MatrixShape{rows}, oa::ScalarType::UInt32);
		auto* p = t.dataAs<oa::U32>();
		for (size_t i = 0; i < th.size(); ++i) p[i] = th[i];
		return t;
	};
	auto logits = [&](const oa::Matrix& conv) {
		// [B, cout, outL] -> [B, outL, cout] -> [rows, cout]
		return oa::FnMatrix::reshape(oa::FnMatrix::transpose(conv, 1, 2), oa::MatrixShape{rows, cout});
	};
	auto readGrad = [](const oa::Matrix& g) {
		std::vector<float> h(static_cast<size_t>(g.numElements()));
		(void)oa::FnMatrix::copyToHost(g, h.data(), h.size() * sizeof(float));
		return h;
	};
	auto fwdLoss = [&](const std::vector<float>& xv, const std::vector<float>& wv, const std::vector<float>& bv) -> float {
		ctx.clear();
		auto o = oa::FnMatrix::conv1dGemm(mk(xv, oa::MatrixShape{B, cin, L}), mk(wv, oa::MatrixShape{cout, cin, K}),
			mk(bv, oa::MatrixShape{cout}), stride, pad, 1);
		auto loss = oa::FnLoss::crossEntropy(logits(o), mkTargets());
		(void)testSubmitAndWait(ctx);
		return loss.at(0);
	};

	// Analytic grads via the composed autograd backward.
	ctx.clear();
	auto X = mk(xh, oa::MatrixShape{B, cin, L}); X.setRequiresGrad(true);
	auto W = mk(wh, oa::MatrixShape{cout, cin, K}); W.setRequiresGrad(true);
	auto Bs = mk(bh, oa::MatrixShape{cout}); Bs.setRequiresGrad(true);
	oa::GradientTape tape;
	auto o = oa::FnMatrix::conv1dGemm(X, W, Bs, stride, pad, 1);
	auto loss = oa::FnLoss::crossEntropy(logits(o), mkTargets());
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	auto dX = readGrad(X.gradMatrix());
	auto dW = readGrad(W.gradMatrix());
	auto dB = readGrad(Bs.gradMatrix());

	const float eps = 1e-3f;
	int badX = 0, badW = 0, badB = 0;
	for (size_t i = 0; i < xh.size(); ++i) {
		auto xp = xh; xp[i] += eps; auto xm = xh; xm[i] -= eps;
		const float num = (fwdLoss(xp, wh, bh) - fwdLoss(xm, wh, bh)) / (2 * eps);
		if (std::fabs(num - dX[i]) > 2e-2f) { ++badX; printf("  dX[%zu] num=%.4f ana=%.4f\n", i, num, dX[i]); }
	}
	for (size_t i = 0; i < wh.size(); ++i) {
		auto wp = wh; wp[i] += eps; auto wm = wh; wm[i] -= eps;
		const float num = (fwdLoss(xh, wp, bh) - fwdLoss(xh, wm, bh)) / (2 * eps);
		if (std::fabs(num - dW[i]) > 2e-2f) { ++badW; printf("  dW[%zu] num=%.4f ana=%.4f\n", i, num, dW[i]); }
	}
	for (size_t i = 0; i < bh.size(); ++i) {
		auto bp = bh; bp[i] += eps; auto bm = bh; bm[i] -= eps;
		const float num = (fwdLoss(xh, wh, bp) - fwdLoss(xh, wh, bm)) / (2 * eps);
		if (std::fabs(num - dB[i]) > 2e-2f) { ++badB; printf("  dB[%zu] num=%.4f ana=%.4f\n", i, num, dB[i]); }
	}
	printf("conv1dGemmGradCheck: badX=%d/%zu badW=%d/%zu badB=%d/%zu\n",
		badX, xh.size(), badW, wh.size(), badB, bh.size());
	EXPECT_EQ(badX, 0); EXPECT_EQ(badW, 0); EXPECT_EQ(badB, 0);
}

TEST(NN, Conv2d) {
	oa::Conv2d conv(1, 4, 3, 1, 0);
	auto out = conv.forward(oa::FnMatrix::rand(oa::MatrixShape{1, 1, 8, 8}));
	expectShape(out, {1, 4, 6, 6});
}

TEST(NN, Conv2dDepthwiseGroups) {
	const std::array<oa::F32, 8> inputValues{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
	};
	const std::array<oa::F32, 2> weightValues{2.0f, 3.0f};
	const std::array<oa::F32, 2> biasValues{10.0f, 20.0f};

	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inputValues.data()),
			inputValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, 2, 2, 2});
	auto weight = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(weightValues.data()),
			weightValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{2, 1, 1, 1});
	auto bias = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(biasValues.data()),
			biasValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{2});

	auto out = oa::FnMatrix::conv2d(input, weight, bias, 1, 0, 2);
	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	expectShape(out, {1, 2, 2, 2});

	const std::array<oa::F32, 8> expected{
		12.0f, 14.0f, 16.0f, 18.0f,
		35.0f, 38.0f, 41.0f, 44.0f,
	};
	for (oa::I64 i = 0; i < static_cast<oa::I64>(expected.size()); ++i) {
		EXPECT_FLOAT_EQ(out.at(i), expected[static_cast<oa::Usize>(i)]);
	}
}

TEST(NN, Conv2dDepthwiseGroupsBackward) {
	const std::array<oa::F32, 8> inputValues{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
	};
	const std::array<oa::F32, 2> weightValues{2.0f, 3.0f};
	const std::array<oa::F32, 8> gradValues{
		1.0f, 1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f, 1.0f,
	};

	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inputValues.data()),
			inputValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, 2, 2, 2});
	auto weight = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(weightValues.data()),
			weightValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{2, 1, 1, 1});
	auto gradOut = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(gradValues.data()),
			gradValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, 2, 2, 2});

	auto gradInput = oa::FnMatrix::conv2dBwdData(
		gradOut, weight, 1, 0, input.getShape(), 2);
	auto gradWeightBias = oa::FnMatrix::conv2dBwdWeight(
		input, gradOut, weight, 1, 0, 2);
	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());

	const std::array<oa::F32, 8> expectedGradInput{
		2.0f, 2.0f, 2.0f, 2.0f,
		3.0f, 3.0f, 3.0f, 3.0f,
	};
	for (oa::I64 i = 0; i < static_cast<oa::I64>(expectedGradInput.size()); ++i) {
		EXPECT_FLOAT_EQ(
			gradInput.at(i), expectedGradInput[static_cast<oa::Usize>(i)]);
	}
	EXPECT_FLOAT_EQ(gradWeightBias.gradWeight.at(0), 10.0f);
	EXPECT_FLOAT_EQ(gradWeightBias.gradWeight.at(1), 26.0f);
	EXPECT_FLOAT_EQ(gradWeightBias.gradBias.at(0), 4.0f);
	EXPECT_FLOAT_EQ(gradWeightBias.gradBias.at(1), 4.0f);
}

TEST(NN, MatrixSliceConcatRecordedGraph) {
	const std::array<oa::F32, 8> inputValues{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
	};
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inputValues.data()),
			inputValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, 2, 2, 2});

	// Keep producer, channel split, and merge in one graph. This is the C3k2 path.
	auto produced = oa::FnMatrix::add(input, input);
	auto first = oa::FnMatrix::slice(produced, 1, 0, 1);
	auto second = oa::FnMatrix::slice(produced, 1, 1, 2);
	oa::Matrix parts[] = {second, first};
	auto merged = oa::FnMatrix::concat(oa::Span<oa::Matrix>(parts, 2), 1);

	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	expectShape(merged, {1, 2, 2, 2});

	const std::array<oa::F32, 8> expected{
		10.0f, 12.0f, 14.0f, 16.0f,
		2.0f, 4.0f, 6.0f, 8.0f,
	};
	for (oa::I64 i = 0; i < static_cast<oa::I64>(expected.size()); ++i) {
		EXPECT_FLOAT_EQ(merged.at(i), expected[static_cast<oa::Usize>(i)]);
	}
}

TEST(NN, MatrixConcatDetectHeadDimension) {
	const std::array<oa::F32, 4> firstValues{1.0f, 2.0f, 3.0f, 4.0f};
	const std::array<oa::F32, 2> secondValues{5.0f, 6.0f};
	auto first = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(firstValues.data()),
			firstValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, 2, 2});
	auto second = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(secondValues.data()),
			secondValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, 2, 1});
	oa::Matrix parts[] = {first, second};
	auto merged = oa::FnMatrix::concat(oa::Span<oa::Matrix>(parts, 2), 2);

	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	expectShape(merged, {1, 2, 3});

	const std::array<oa::F32, 6> expected{1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f};
	for (oa::I64 i = 0; i < static_cast<oa::I64>(expected.size()); ++i) {
		EXPECT_FLOAT_EQ(merged.at(i), expected[static_cast<oa::Usize>(i)]);
	}
}

TEST(NN, BatchNormUpsampleRecordedGraph) {
	const std::array<oa::F32, 4> inputValues{1.0f, 2.0f, 3.0f, 4.0f};
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inputValues.data()),
			inputValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, 1, 2, 2});

	oa::BatchNorm2d batchNorm(1);
	oa::Upsample upsample(2, oa::UpsampleMode::Nearest);
	oa::Module::ScopedEval eval(batchNorm);
	auto produced = oa::FnMatrix::add(input, input);
	auto normalized = batchNorm.forward(produced);
	auto activated = oa::FnMatrix::silu(normalized);
	auto output = upsample.forward(activated);

	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	expectShape(output, {1, 1, 4, 4});
	expectFinite(output);
	EXPECT_FLOAT_EQ(output.at(0), output.at(1));
	EXPECT_FLOAT_EQ(output.at(0), output.at(4));
	EXPECT_GT(output.at(15), output.at(0));
}

TEST(NN, BatchNormConstantLargeMagnitudeRemainsFinite) {
	// E[x^2] - E[x]^2 is cancellation-prone in FP32. The statistics kernel
	// must never feed a small negative variance into rsqrt.
	auto input = oa::FnMatrix::full(oa::MatrixShape{1, 1, 1, 513}, 10000.0F);
	oa::BatchNorm2d batchNorm(1);
	auto output = batchNorm.forward(input);

	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	expectFinite(output);
	ASSERT_EQ(batchNorm.buffers().size(), 2u);
	EXPECT_TRUE(std::isfinite(batchNorm.buffers()[1].data.at(0)));
	EXPECT_GE(batchNorm.buffers()[1].data.at(0), 0.0F);
}

class ModuleBufferFixture final : public oa::Module {
public:
	ModuleBufferFixture() {
		registerParameter("weight", oa::FnMatrix::ones(oa::MatrixShape{2}));
		registerBuffer("persistent", oa::FnMatrix::zeros(oa::MatrixShape{2}));
		registerBuffer("scratch", oa::FnMatrix::zeros(oa::MatrixShape{1}), false);
	}
};

class ModuleModeFixture final : public oa::Module {
public:
	void addChild(oa::SharedPtr<oa::Module> inChild) {
		registerModule("child", oa::move(inChild));
	}
};

TEST(NN, ModuleModeIsRecursiveAndMlOwned) {
	ModuleModeFixture model;
	auto child = oa::makeShared<ModuleModeFixture>();
	model.addChild(child);

	EXPECT_TRUE(model.isTraining());
	EXPECT_TRUE(child->isTraining());
	model.eval();
	EXPECT_FALSE(model.isTraining());
	EXPECT_FALSE(child->isTraining());

	auto lateChild = oa::makeShared<ModuleModeFixture>();
	model.addChild(lateChild);
	EXPECT_FALSE(lateChild->isTraining());

	model.train();
	{
		oa::Module::ScopedEval eval(model);
		EXPECT_FALSE(model.isTraining());
		EXPECT_FALSE(child->isTraining());
		EXPECT_FALSE(lateChild->isTraining());
	}
	EXPECT_TRUE(model.isTraining());
	EXPECT_TRUE(child->isTraining());
	EXPECT_TRUE(lateChild->isTraining());
}

TEST(NN, ModuleBuffersAreStateNotParameters) {
	ModuleBufferFixture module;
	EXPECT_EQ(module.parameters().size(), 1u);
	EXPECT_EQ(module.allParameterPtrs().size(), 1u);
	EXPECT_EQ(module.numParameters(), 2);
	EXPECT_EQ(module.buffers().size(), 2u);
	EXPECT_EQ(module.allBufferPtrs().size(), 2u);
	EXPECT_EQ(module.allBufferPtrs(true).size(), 1u);
	EXPECT_FALSE(module.buffers()[0].data.requiresGrad());
}

TEST(NN, ModulePersistentBuffersRoundTripThroughModelFileState) {
	ModuleBufferFixture source;
	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	source.parameters()[0].data.set(0, 3.0f);
	source.parameters()[0].data.set(1, 4.0f);
	source.buffers()[0].data.set(0, 7.0f);
	source.buffers()[0].data.set(1, 8.0f);
	source.buffers()[1].data.set(0, 9.0f);

	oa::ModelFile checkpoint;
	ASSERT_TRUE(source.saveTo(testEngine(), checkpoint).isOk());
	ASSERT_NE(checkpoint.findWeight("weight"), nullptr);
	ASSERT_NE(checkpoint.findState("persistent"), nullptr);
	EXPECT_EQ(checkpoint.findState("scratch"), nullptr);

	ModuleBufferFixture restored;
	ASSERT_TRUE(restored.loadFrom(testEngine(), checkpoint).isOk());
	EXPECT_FLOAT_EQ(restored.parameters()[0].data.at(0), 3.0f);
	EXPECT_FLOAT_EQ(restored.parameters()[0].data.at(1), 4.0f);
	EXPECT_FLOAT_EQ(restored.buffers()[0].data.at(0), 7.0f);
	EXPECT_FLOAT_EQ(restored.buffers()[0].data.at(1), 8.0f);
	EXPECT_FLOAT_EQ(restored.buffers()[1].data.at(0), 0.0f);
}

TEST(NN, BatchNormUpdatesPersistentRunningState) {
	const std::array<oa::F32, 4> inputValues{1.0f, 2.0f, 3.0f, 4.0f};
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inputValues.data()),
			inputValues.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, 1, 2, 2});

	oa::BatchNorm2d batchNorm(1, 1e-5f, 0.1f);
	(void)batchNorm.forward(input);
	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());

	ASSERT_EQ(batchNorm.parameters().size(), 2u);
	ASSERT_EQ(batchNorm.buffers().size(), 2u);
	EXPECT_NEAR(batchNorm.buffers()[0].data.at(0), 0.25f, 1e-5f);
	EXPECT_NEAR(batchNorm.buffers()[1].data.at(0), 1.025f, 1e-5f);
}

TEST(NN, Sequential) {
	oa::Sequential model;
	model.add(oa::makeShared<oa::Linear>(4, 8));
	model.add(oa::makeShared<oa::Gelu>());
	model.add(oa::makeShared<oa::Linear>(8, 2));
	EXPECT_GT(model.numParameters(), 0);
	auto out = model.forward(oa::FnMatrix::rand(oa::MatrixShape{3, 4}));
	expectShape(out, {3, 2});
}

TEST(NN, SequentialVariadicAndFlatten) {
	oa::Sequential model(
		oa::makeShared<oa::Flatten>(),
		oa::makeShared<oa::Linear>(4, 2),
		oa::makeShared<oa::Identity>()
	);
	EXPECT_EQ(model.children().size(), 3u);
	EXPECT_GT(model.numParameters(), 0);
	auto out = model.forward(oa::FnMatrix::rand(oa::MatrixShape{3, 2, 2}));
	expectShape(out, {3, 2});
}

TEST(NN, DropoutEvalPassthrough) {
	oa::Dropout drop(0.5f);
	oa::Module::ScopedEval guard(drop);
	auto input = oa::FnMatrix::ones(oa::MatrixShape{10});
	auto out = drop.forward(input);
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	EXPECT_FLOAT_EQ(out.at(0), 1.0f);
}

TEST(NN, DropoutTrainingIsGpuNativeAndInverted) {
	constexpr oa::I64 N = 4096;
	constexpr oa::F32 P = 0.25f;
	oa::Dropout drop(P);
	oa::FnMatrix::setRngSeed(12345);
	auto input = oa::FnMatrix::ones(oa::MatrixShape{N});
	auto out = drop.forward(input);
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());

	oa::I64 kept = 0;
	for (oa::I64 i = 0; i < N; ++i) {
		const oa::F32 value = out.at(i);
		if (value != 0.0f) {
			++kept;
			EXPECT_NEAR(value, 1.0f / (1.0f - P), 1e-6f);
		}
	}
	const oa::F32 keepFraction = static_cast<oa::F32>(kept) / static_cast<oa::F32>(N);
	EXPECT_NEAR(keepFraction, 1.0f - P, 0.03f);
}

TEST(NN, FfnForwardCorrectness) {
	constexpr oa::U32 kRows = 4;
	constexpr oa::U32 kDModel = 8;
	constexpr oa::U32 kDFF = 16;
	constexpr float kEps = 1e-5f;

	// varied (not uniform) input + weights. Uniform weights make every output
	// column identical, which would hide transpose/indexing bugs in the GEMMs —
	// distinct per-element values make this a real correctness check.
	std::vector<float> x_cpu(kRows * kDModel);
	for (oa::U32 i = 0; i < x_cpu.size(); ++i) {
		x_cpu[i] = std::sin(static_cast<float>(i) * 0.3f) * 0.5f;
	}
	std::vector<float> norm_w(kDModel);
	std::vector<float> gate_w(kDFF * kDModel);
	std::vector<float> up_w(kDFF * kDModel);
	std::vector<float> down_w(kDModel * kDFF);
	for (oa::U32 i = 0; i < norm_w.size(); ++i) norm_w[i] = 0.8f + 0.05f * static_cast<float>(i % 5);
	for (oa::U32 i = 0; i < gate_w.size(); ++i) gate_w[i] = std::cos(static_cast<float>(i) * 0.2f) * 0.25f;
	for (oa::U32 i = 0; i < up_w.size(); ++i)   up_w[i]   = std::sin(static_cast<float>(i) * 0.17f) * 0.25f;
	for (oa::U32 i = 0; i < down_w.size(); ++i) down_w[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.2f;

	// CPU reference
	std::vector<float> expected(kRows * kDModel);
	cpuFfnForward(x_cpu, norm_w, gate_w, up_w, down_w, expected, kRows, kDModel, kDFF, kEps);

	// GPU computation. oa::Ffn composes child modules (norm/gate/up/down), so its
	// weights live in allParameterPtrs(), in registration order:
	//   [0] norm.weight  [1] gate.weight [2] gate.bias
	//   [3] up.weight    [4] up.bias     [5] down.weight [6] down.bias
	oa::Ffn ffn(kDModel, kDFF, kEps);
	auto ptrs = ffn.allParameterPtrs();
	ASSERT_EQ(ptrs.size(), static_cast<oa::Usize>(7));

	// Overwrite each weight with our deterministic values (shape-preserving).
	// Biases stay zero (oa::Linear default) — the CPU reference assumes the same.
	auto setWeights = [](oa::Parameter* p, const std::vector<float>& w) {
		p->data = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(w.data()), w.size() * sizeof(float)),
			p->data.getShape());
	};
	setWeights(ptrs[0], norm_w);
	setWeights(ptrs[1], gate_w);
	setWeights(ptrs[3], up_w);
	setWeights(ptrs[5], down_w);

	auto x_gpu = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(x_cpu.data()), x_cpu.size() * sizeof(float)),
		oa::MatrixShape{kRows, kDModel});

	auto out_gpu = ffn.forward(x_gpu);
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());

	// Combined abs+rel tolerance for the engine's bf16 tensor-core gEMMs (3
	// matmuls feed each output). The pre-residual signal is small here, so bf16
	// noise is significant in relative terms — but an absolute floor covers it.
	// A structural/indexing bug produces errors far larger than this envelope.
	const float* gpu_data = out_gpu.dataAs<const float>();
	for (oa::U32 i = 0; i < expected.size(); ++i) {
		const float tol = 1e-2f + 4e-2f * std::abs(expected[i]);
		EXPECT_LE(std::abs(gpu_data[i] - expected[i]), tol) << "Mismatch at index " << i
			<< " gpu=" << gpu_data[i] << " expected=" << expected[i];
	}
}

TEST(NN, FfnModuleParameterRegistration) {
	constexpr oa::I32 kDModel = 32;
	constexpr oa::I32 kDFF = 64;
	constexpr float kEps = 1e-5f;

	oa::Ffn ffn(kDModel, kDFF, kEps);

	// FFN registers norm/gate/up/down as child modules, so all weights surface
	// through allParameterPtrs() (7 total) — the FFN's own direct parameters()
	// and buffers() vectors are empty.
	EXPECT_EQ(ffn.parameters().size(), static_cast<oa::Usize>(0));
	EXPECT_EQ(ffn.buffers().size(), static_cast<oa::Usize>(0));
	EXPECT_GT(ffn.numParameters(), 0);

	auto ptrs = ffn.allParameterPtrs();
	ASSERT_EQ(ptrs.size(), static_cast<oa::Usize>(7));
	// order: norm.weight, gate.{weight,bias}, up.{weight,bias}, down.{weight,bias}
	EXPECT_EQ(ptrs[0]->data.getShape(), oa::MatrixShape{kDModel});         // RMSNorm weight
	EXPECT_EQ(ptrs[1]->data.getShape(), (oa::MatrixShape{kDFF, kDModel}));   // gate weight
	EXPECT_EQ(ptrs[2]->data.getShape(), oa::MatrixShape{kDFF});            // gate bias
	EXPECT_EQ(ptrs[3]->data.getShape(), (oa::MatrixShape{kDFF, kDModel}));   // Up weight
	EXPECT_EQ(ptrs[4]->data.getShape(), oa::MatrixShape{kDFF});            // Up bias
	EXPECT_EQ(ptrs[5]->data.getShape(), (oa::MatrixShape{kDModel, kDFF}));   // Down weight
	EXPECT_EQ(ptrs[6]->data.getShape(), oa::MatrixShape{kDModel});         // Down bias
}

TEST(NN, FfnModuleForwardShape) {
	constexpr oa::I32 kDModel = 16;
	constexpr oa::I32 kDFF = 32;
	constexpr oa::I32 kBatch = 4;

	oa::Ffn ffn(kDModel, kDFF);
	auto x = oa::FnMatrix::rand(oa::MatrixShape{kBatch, kDModel});

	auto out = ffn.forward(x);
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());

	expectShape(out, {kBatch, kDModel});
	expectFinite(out);
}

TEST(NN, FfnModuleAutograd) {
	constexpr oa::I32 kDModel = 8;
	constexpr oa::I32 kDFF = 16;
	constexpr oa::I32 kBatch = 2;

	oa::Ffn ffn(kDModel, kDFF);
	auto x = oa::FnMatrix::rand(oa::MatrixShape{kBatch, kDModel});

	// Differentiable scalar loss root. oa::FnMatrix::sum has no autograd node, so
	// treat the FFN output as logits over kDModel classes and use CrossEntropy
	// (the same differentiable root the MNIST tutorial uses).
	auto targets = oa::FnMatrix::empty(oa::MatrixShape{kBatch}, oa::ScalarType::UInt32);
	auto* tgt = targets.dataAs<oa::U32>();
	for (oa::I32 i = 0; i < kBatch; ++i) tgt[i] = static_cast<oa::U32>(i % kDModel);

	oa::GradientTape tape;
	auto out = ffn.forward(x);
	auto loss = oa::FnLoss::crossEntropy(out, targets);
	tape.backward(loss);
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());

	// Every parameter must carry a gradient of matching shape.
	auto ptrs = ffn.allParameterPtrs();
	ASSERT_EQ(ptrs.size(), static_cast<oa::Usize>(7));
	for (oa::Usize i = 0; i < ptrs.size(); ++i) {
		EXPECT_TRUE(ptrs[i]->grad().hasStorage()) << "Parameter " << i << " has no gradient storage";
		EXPECT_EQ(ptrs[i]->grad().getShape(), ptrs[i]->data.getShape())
			<< "Parameter " << i << " grad shape mismatch";
	}

	// Gradient must actually flow: the down-projection weight (index 5) feeds the
	// loss directly, so its accumulated gradient is non-zero.
	const oa::Matrix& downGrad = ptrs[5]->grad();
	std::vector<float> hostGrad(static_cast<size_t>(downGrad.numElements()));
	(void)oa::FnMatrix::copyToHost(downGrad, hostGrad.data(), hostGrad.size() * sizeof(float));
	float gradMag = 0.0f;
	for (float g : hostGrad) gradMag += std::abs(g);
	EXPECT_GT(gradMag, 0.0f) << "down weight gradient did not flow";
}

TEST(NN, FfnModuleSaveLoadRoundtrip) {
	constexpr oa::I32 kDModel = 8;
	constexpr oa::I32 kDFF = 16;

	oa::Ffn ffn(kDModel, kDFF);
	auto x = oa::FnMatrix::rand(oa::MatrixShape{2, kDModel});

	// forward before save
	auto out1 = ffn.forward(x);
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());

	// save
	const oa::String ckptPath = "/tmp/ffn_test.oam";
	auto saveStatus = ffn.save(testEngine(), ckptPath);
	ASSERT_TRUE(saveStatus.isOk()) << "save failed: " << saveStatus.getMessage();

	// load into new instance
	oa::Ffn ffn2(kDModel, kDFF);
	auto loadStatus = ffn2.load(testEngine(), ckptPath);
	ASSERT_TRUE(loadStatus.isOk()) << "load failed: " << loadStatus.getMessage();

	// forward after load
	auto out2 = ffn2.forward(x);
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());

	// compare outputs
	const float* data1 = out1.dataAs<const float>();
	const float* data2 = out2.dataAs<const float>();
	for (oa::I64 i = 0; i < out1.numElements(); ++i) {
		EXPECT_FLOAT_EQ(data1[i], data2[i]) << "Mismatch at index " << i;
	}
}
