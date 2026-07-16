// Comprehensive ML Kernel Correctness Tests
// Validates numerical accuracy of ML kernels against CPU reference
//
// Priority: ByteEmbed (caused production bugs), activations, BiasAdd
//
// Usage:
//   ./oa-test --gtest_filter=MlKernels.*

#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Core/Log.h>
#include <gtest/gtest.h>
#include "../../OaTest.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// ============================================================================
// CPU Reference Implementations
// ============================================================================

static void CpuByteEmbed(const std::vector<OaU32> &indices, const std::vector<float> &table,
                         std::vector<float> &out, OaU32 count, OaU32 dim) {
	for (OaU32 token = 0; token < count; ++token) {
		OaU32 byte_val = indices[token];
		for (OaU32 d = 0; d < dim; ++d) {
			out[token * dim + d] = table[byte_val * dim + d];
		}
	}
}

static void CpuGelu(const std::vector<float> &x, std::vector<float> &out, OaU32 count) {
	// GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of the standard normal
	// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
	constexpr float kSqrt2OverPi = 0.7978845608f;
	constexpr float kCoeff = 0.044715f;
	for (OaU32 i = 0; i < count; ++i) {
		float x_val = x[i];
		float x3 = x_val * x_val * x_val;
		float inner = kSqrt2OverPi * (x_val + kCoeff * x3);
		out[i] = 0.5f * x_val * (1.0f + std::tanh(inner));
	}
}

static void CpuRelu(const std::vector<float> &x, std::vector<float> &out, OaU32 count) {
	for (OaU32 i = 0; i < count; ++i) {
		out[i] = std::max(0.0f, x[i]);
	}
}

static void CpuSilu(const std::vector<float> &x, std::vector<float> &out, OaU32 count) {
	// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
	for (OaU32 i = 0; i < count; ++i) {
		float x_val = x[i];
		out[i] = x_val / (1.0f + std::exp(-x_val));
	}
}

static void CpuBiasAdd(std::vector<float> &biased, const std::vector<float> &bias,
                       OaU32 rows, OaU32 cols) {
	for (OaU32 row = 0; row < rows; ++row) {
		for (OaU32 col = 0; col < cols; ++col) {
			biased[row * cols + col] += bias[col];
		}
	}
}

static void CpuLayerNorm(const std::vector<float> &x, const std::vector<float> &weight,
                         const std::vector<float> &bias, std::vector<float> &out,
                         OaU32 rows, OaU32 cols, float eps) {
	for (OaU32 row = 0; row < rows; ++row) {
		OaU32 base = row * cols;
		
		// Compute mean
		float sum = 0.0f;
		for (OaU32 i = 0; i < cols; ++i) {
			sum += x[base + i];
		}
		float mean = sum / static_cast<float>(cols);
		
		// Compute variance
		float var_sum = 0.0f;
		for (OaU32 i = 0; i < cols; ++i) {
			float diff = x[base + i] - mean;
			var_sum += diff * diff;
		}
		float inv_std = 1.0f / std::sqrt(var_sum / static_cast<float>(cols) + eps);
		
		// Normalize and scale
		for (OaU32 i = 0; i < cols; ++i) {
			float normed = (x[base + i] - mean) * inv_std;
			out[base + i] = normed * weight[i] + bias[i];
		}
	}
}

static void CpuRmsNorm(const std::vector<float> &x, const std::vector<float> &weight,
                       std::vector<float> &out, OaU32 rows, OaU32 cols, float eps) {
	for (OaU32 row = 0; row < rows; ++row) {
		OaU32 base = row * cols;
		
		// Compute RMS
		float sum_sq = 0.0f;
		for (OaU32 i = 0; i < cols; ++i) {
			float val = x[base + i];
			sum_sq += val * val;
		}
		float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
		
		// Scale
		for (OaU32 i = 0; i < cols; ++i) {
			out[base + i] = x[base + i] * inv_rms * weight[i];
		}
	}
}

static void CpuSoftmax(const std::vector<float> &x, std::vector<float> &out,
                       OaU32 rows, OaU32 cols) {
	for (OaU32 row = 0; row < rows; ++row) {
		OaU32 base = row * cols;
		
		// Find max for numerical stability
		float row_max = x[base];
		for (OaU32 i = 1; i < cols; ++i) {
			row_max = std::max(row_max, x[base + i]);
		}
		
		// Compute exp and sum
		float sum = 0.0f;
		for (OaU32 i = 0; i < cols; ++i) {
			float exp_val = std::exp(x[base + i] - row_max);
			out[base + i] = exp_val;
			sum += exp_val;
		}
		
		// Normalize
		float inv_sum = 1.0f / sum;
		for (OaU32 i = 0; i < cols; ++i) {
			out[base + i] *= inv_sum;
		}
	}
}

static void CpuSwiglu(const std::vector<float> &gate, const std::vector<float> &up,
                      std::vector<float> &out, OaU32 count) {
	// SwiGLU(gate, up) = Silu(gate) * up = (gate / (1 + exp(-gate))) * up
	for (OaU32 i = 0; i < count; ++i) {
		float g = gate[i];
		float silu_g = g / (1.0f + std::exp(-g));
		out[i] = silu_g * up[i];
	}
}

// CPU reference for OaFfn::Forward. Mirrors the module exactly:
//   normed = RMSNorm(x) * norm_w
//   swiglu = silu(normed @ Wgateᵀ) * (normed @ Wupᵀ)
//   out    = x + swiglu @ Wdownᵀ          (pre-norm residual)
// Linear weights are [out, in] row-major (out[j] = Σ_k in[k]·W[j·in + k]); biases
// are left at zero by the caller, matching OaLinear's default-initialized bias.
static void CpuFfnForward(const std::vector<float> &x, const std::vector<float> &norm_w,
                          const std::vector<float> &gate_w, const std::vector<float> &up_w,
                          const std::vector<float> &down_w, std::vector<float> &out,
                          OaU32 rows, OaU32 d_model, OaU32 d_ff, float eps) {
	for (OaU32 row = 0; row < rows; ++row) {
		const OaU32 x_base = row * d_model;

		// RMSNorm over the d_model axis, scaled by the per-channel weight.
		float sum_sq = 0.0f;
		for (OaU32 i = 0; i < d_model; ++i) {
			sum_sq += x[x_base + i] * x[x_base + i];
		}
		const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(d_model) + eps);
		std::vector<float> normed(d_model);
		for (OaU32 i = 0; i < d_model; ++i) {
			normed[i] = x[x_base + i] * inv_rms * norm_w[i];
		}

		// Gate / Up projections → SwiGLU (silu(gate) * up).
		std::vector<float> swiglu(d_ff);
		for (OaU32 j = 0; j < d_ff; ++j) {
			float gate = 0.0f;
			float up = 0.0f;
			for (OaU32 k = 0; k < d_model; ++k) {
				gate += normed[k] * gate_w[j * d_model + k];
				up   += normed[k] * up_w[j * d_model + k];
			}
			const float silu_g = gate / (1.0f + std::exp(-gate));
			swiglu[j] = silu_g * up;
		}

		// Down projection + residual back onto the raw input.
		for (OaU32 i = 0; i < d_model; ++i) {
			float down = 0.0f;
			for (OaU32 j = 0; j < d_ff; ++j) {
				down += swiglu[j] * down_w[i * d_ff + j];
			}
			out[x_base + i] = x[x_base + i] + down;
		}
	}
}

static void CpuEmbedding(const std::vector<OaU32> &indices, const std::vector<float> &table,
                         std::vector<float> &out, OaU32 numIndices, OaU32 dim) {
	for (OaU32 i = 0; i < numIndices; ++i) {
		OaU32 tableRow = indices[i];
		for (OaU32 d = 0; d < dim; ++d) {
			out[i * dim + d] = table[tableRow * dim + d];
		}
	}
}

static void CpuCrossEntropyFwd(const std::vector<float> &logits, const std::vector<OaU32> &targets,
                                std::vector<float> &loss, OaU32 batch, OaU32 classes) {
	for (OaU32 row = 0; row < batch; ++row) {
		OaU32 base = row * classes;
		
		// Find max for numerical stability
		float row_max = logits[base];
		for (OaU32 i = 1; i < classes; ++i) {
			row_max = std::max(row_max, logits[base + i]);
		}
		
		// Compute log_sum_exp
		float sum_exp = 0.0f;
		for (OaU32 i = 0; i < classes; ++i) {
			sum_exp += std::exp(logits[base + i] - row_max);
		}
		float log_sum_exp = std::log(sum_exp) + row_max;
		
		// CrossEntropy = log_sum_exp - logits[target]
		OaU32 target = targets[row];
		loss[row] = log_sum_exp - logits[base + target];
	}
}

static void CpuCrossEntropyBwd(const std::vector<float> &logits, const std::vector<OaU32> &targets,
                                std::vector<float> &d_logits, OaU32 batch, OaU32 classes) {
	for (OaU32 row = 0; row < batch; ++row) {
		OaU32 base = row * classes;
		
		// Compute softmax (same as forward)
		float row_max = logits[base];
		for (OaU32 i = 1; i < classes; ++i) {
			row_max = std::max(row_max, logits[base + i]);
		}
		
		float sum_exp = 0.0f;
		for (OaU32 i = 0; i < classes; ++i) {
			sum_exp += std::exp(logits[base + i] - row_max);
		}
		float inv_sum = 1.0f / sum_exp;
		
		// Gradient: (softmax - onehot) / batch
		OaU32 target = targets[row];
		float inv_batch = 1.0f / static_cast<float>(batch);
		for (OaU32 i = 0; i < classes; ++i) {
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

class MlKernels : public OaVkEngineTestFixture {
protected:
	OaComputeEngine& Rt_ = Rt();

	void SetUp() override {
		OaVkEngineTestFixture::SetUp();  // Asserts global engine exists
		OA_LOG_INFO(OaLogComponent::Core, "GPU: %s", Rt_.Device.Info.Hardware.DeviceName.c_str());
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
// ByteEmbed Tests (CRITICAL - caused production bugs)
// ============================================================================

TEST_VK(MlKernels, ByteEmbed_Basic) {
	constexpr OaU32 vocab = 256;
	constexpr OaU32 dim = 64;
	constexpr OaU32 count = 16;
	
	// Create embedding table (256 x 64)
	std::vector<float> table(vocab * dim);
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (auto &v : table) v = dist(rng);
	
	// Create indices (16 tokens)
	std::vector<OaU32> indices(count);
	for (OaU32 i = 0; i < count; ++i) {
		indices[i] = (i * 17) % vocab;  // Varied indices
	}
	
	// CPU reference
	std::vector<float> out_ref(count * dim);
	CpuByteEmbed(indices, table, out_ref, count, dim);
	
	// GPU dispatch
	auto resultIndices = Rt_.AllocBufferBar(count * 4);
	auto resultTable = Rt_.AllocBufferBar(vocab * dim * 4);
	auto resultOut = Rt_.AllocBufferBar(count * dim * 4);
	ASSERT_TRUE(resultIndices.IsOk() && resultTable.IsOk() && resultOut.IsOk());
	
	auto bufIndices = std::move(resultIndices).GetValue();
	auto bufTable = std::move(resultTable).GetValue();
	auto bufOut = std::move(resultOut).GetValue();
	
	std::memcpy(bufIndices.MappedPtr, indices.data(), count * 4);
	std::memcpy(bufTable.MappedPtr, table.data(), vocab * dim * 4);
	
	struct { OaU32 count; OaU32 dim; } pc = { count, dim };
	OaVkBuffer bufs[] = {bufIndices, bufTable, bufOut};
	OaU32 groups = (count * dim + 255) / 256;
	
	ASSERT_TRUE(OaVkDispatch::Run(Rt_, "ByteEmbed", bufs, &pc, sizeof(pc), groups).IsOk());
	
	std::vector<float> out_gpu(count * dim);
	std::memcpy(out_gpu.data(), bufOut.MappedPtr, count * dim * 4);
	
	CompareResults(out_ref, out_gpu, count * dim, 1e-6f, "ByteEmbed_Basic");
	
	Rt_.FreeBuffer(bufIndices);
	Rt_.FreeBuffer(bufTable);
	Rt_.FreeBuffer(bufOut);
}

TEST_VK(MlKernels, ByteEmbed_EdgeCases) {
	constexpr OaU32 vocab = 256;
	constexpr OaU32 dim = 128;
	
	auto testCase = [this](OaU32 count, const char *name) {
		std::vector<float> table(vocab * dim);
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		for (auto &v : table) v = dist(rng);
		
		std::vector<OaU32> indices(count);
		for (OaU32 i = 0; i < count; ++i) {
			indices[i] = (i * 13) % vocab;
		}
		
		std::vector<float> out_ref(count * dim);
		CpuByteEmbed(indices, table, out_ref, count, dim);
		
		auto resultIndices = Rt_.AllocBufferBar(count * 4);
		auto resultTable = Rt_.AllocBufferBar(vocab * dim * 4);
		auto resultOut = Rt_.AllocBufferBar(count * dim * 4);
		ASSERT_TRUE(resultIndices.IsOk() && resultTable.IsOk() && resultOut.IsOk());
		
		auto bufIndices = std::move(resultIndices).GetValue();
		auto bufTable = std::move(resultTable).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufIndices.MappedPtr, indices.data(), count * 4);
		std::memcpy(bufTable.MappedPtr, table.data(), vocab * dim * 4);
		
		struct { OaU32 count; OaU32 dim; } pc = { count, dim };
		OaVkBuffer bufs[] = {bufIndices, bufTable, bufOut};
		OaU32 groups = (count * dim + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "ByteEmbed", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> out_gpu(count * dim);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, count * dim * 4);
		
		CompareResults(out_ref, out_gpu, count * dim, 1e-6f, name);
		
		Rt_.FreeBuffer(bufIndices);
		Rt_.FreeBuffer(bufTable);
		Rt_.FreeBuffer(bufOut);
	};
	
	testCase(1, "ByteEmbed_SingleToken");
	testCase(257, "ByteEmbed_NonAligned");
	testCase(1024, "ByteEmbed_LargeSeq");
}

// ============================================================================
// Activation Tests
// ============================================================================

TEST_VK(MlKernels, Gelu) {
	auto testGelu = [this](OaU32 count) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		
		std::vector<float> x(count), out_ref(count);
		for (auto &v : x) v = dist(rng);
		CpuGelu(x, out_ref, count);
		
		auto resultX = Rt_.AllocBufferBar(count * 4);
		auto resultOut = Rt_.AllocBufferBar(count * 4);
		ASSERT_TRUE(resultX.IsOk() && resultOut.IsOk());
		
		auto bufX = std::move(resultX).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufX.MappedPtr, x.data(), count * 4);
		
		struct { OaU32 count; } pc = { count };
		OaVkBuffer bufs[] = {bufX, bufOut};
		OaU32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Gelu", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, count * 4);
		
		// GELU uses tanh approximation - slightly relaxed tolerance
		CompareResults(out_ref, out_gpu, count, 1e-4f, "Gelu");
		
		Rt_.FreeBuffer(bufX);
		Rt_.FreeBuffer(bufOut);
	};
	
	testGelu(1024);
	testGelu(4096);
}

TEST_VK(MlKernels, Relu) {
	auto testRelu = [this](OaU32 count) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
		
		std::vector<float> x(count), out_ref(count);
		for (auto &v : x) v = dist(rng);
		CpuRelu(x, out_ref, count);
		
		auto resultX = Rt_.AllocBufferBar(count * 4);
		auto resultOut = Rt_.AllocBufferBar(count * 4);
		ASSERT_TRUE(resultX.IsOk() && resultOut.IsOk());
		
		auto bufX = std::move(resultX).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufX.MappedPtr, x.data(), count * 4);
		
		struct { OaU32 count; } pc = { count };
		OaVkBuffer bufs[] = {bufX, bufOut};
		OaU32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Relu", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, count * 4);
		
		CompareResults(out_ref, out_gpu, count, 1e-6f, "Relu");
		
		Rt_.FreeBuffer(bufX);
		Rt_.FreeBuffer(bufOut);
	};
	
	testRelu(1024);
	testRelu(4096);
}

TEST_VK(MlKernels, Silu) {
	auto testSilu = [this](OaU32 count) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		
		std::vector<float> x(count), out_ref(count);
		for (auto &v : x) v = dist(rng);
		CpuSilu(x, out_ref, count);
		
		auto resultX = Rt_.AllocBufferBar(count * 4);
		auto resultOut = Rt_.AllocBufferBar(count * 4);
		ASSERT_TRUE(resultX.IsOk() && resultOut.IsOk());
		
		auto bufX = std::move(resultX).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufX.MappedPtr, x.data(), count * 4);
		
		struct { OaU32 count; } pc = { count };
		OaVkBuffer bufs[] = {bufX, bufOut};
		OaU32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Silu", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, count * 4);
		
		// SiLU uses exp - slightly relaxed tolerance
		CompareResults(out_ref, out_gpu, count, 1e-5f, "Silu");
		
		Rt_.FreeBuffer(bufX);
		Rt_.FreeBuffer(bufOut);
	};
	
	testSilu(1024);
	testSilu(4096);
}

// ============================================================================
// Ops Tests
// ============================================================================

TEST_VK(MlKernels, BiasAdd) {
	auto testBiasAdd = [this](OaU32 rows, OaU32 cols) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		
		std::vector<float> biased(rows * cols), bias(cols);
		for (auto &v : biased) v = dist(rng);
		for (auto &v : bias) v = dist(rng);
		
		std::vector<float> biased_ref = biased;
		CpuBiasAdd(biased_ref, bias, rows, cols);
		
		auto resultBiased = Rt_.AllocBufferBar(rows * cols * 4);
		auto resultBias = Rt_.AllocBufferBar(cols * 4);
		ASSERT_TRUE(resultBiased.IsOk() && resultBias.IsOk());
		
		auto bufBiased = std::move(resultBiased).GetValue();
		auto bufBias = std::move(resultBias).GetValue();
		
		std::memcpy(bufBiased.MappedPtr, biased.data(), rows * cols * 4);
		std::memcpy(bufBias.MappedPtr, bias.data(), cols * 4);
		
		struct { OaU32 rows; OaU32 cols; } pc = { rows, cols };
		OaVkBuffer bufs[] = {bufBiased, bufBias};
		OaU32 groups = (rows * cols + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "BiasAdd", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> biased_gpu(rows * cols);
		std::memcpy(biased_gpu.data(), bufBiased.MappedPtr, rows * cols * 4);
		
		CompareResults(biased_ref, biased_gpu, rows * cols, 1e-6f, "BiasAdd");
		
		Rt_.FreeBuffer(bufBiased);
		Rt_.FreeBuffer(bufBias);
	};
	
	testBiasAdd(32, 128);   // Small
	testBiasAdd(256, 512);  // Medium
	testBiasAdd(1024, 768); // Large (typical transformer)
}

// ============================================================================
// Normalization Tests
// ============================================================================

TEST_VK(MlKernels, LayerNorm) {
	auto testLayerNorm = [this](OaU32 rows, OaU32 cols) {
		constexpr float eps = 1e-5f;
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
		
		std::vector<float> x(rows * cols), weight(cols), bias(cols), out_ref(rows * cols);
		for (auto &v : x) v = dist(rng);
		for (auto &v : weight) v = dist(rng);
		for (auto &v : bias) v = dist(rng);
		
		CpuLayerNorm(x, weight, bias, out_ref, rows, cols, eps);
		
		auto resultX = Rt_.AllocBufferBar(rows * cols * 4);
		auto resultWeight = Rt_.AllocBufferBar(cols * 4);
		auto resultBias = Rt_.AllocBufferBar(cols * 4);
		auto resultOut = Rt_.AllocBufferBar(rows * cols * 4);
		ASSERT_TRUE(resultX.IsOk() && resultWeight.IsOk() && resultBias.IsOk() && resultOut.IsOk());
		
		auto bufX = std::move(resultX).GetValue();
		auto bufWeight = std::move(resultWeight).GetValue();
		auto bufBias = std::move(resultBias).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufX.MappedPtr, x.data(), rows * cols * 4);
		std::memcpy(bufWeight.MappedPtr, weight.data(), cols * 4);
		std::memcpy(bufBias.MappedPtr, bias.data(), cols * 4);
		
		struct { OaU32 rows; OaU32 cols; float eps; } pc = { rows, cols, eps };
		OaVkBuffer bufs[] = {bufX, bufWeight, bufBias, bufOut};
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "LayerNorm", bufs, &pc, sizeof(pc), rows).IsOk());
		
		std::vector<float> out_gpu(rows * cols);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, rows * cols * 4);
		
		// LayerNorm uses rsqrt + shared memory reductions - relaxed tolerance for numerical error
		CompareResults(out_ref, out_gpu, rows * cols, 2e-2f, "LayerNorm");
		
		Rt_.FreeBuffer(bufX);
		Rt_.FreeBuffer(bufWeight);
		Rt_.FreeBuffer(bufBias);
		Rt_.FreeBuffer(bufOut);
	};
	
	testLayerNorm(8, 128);    // Small
	testLayerNorm(32, 512);   // Medium
	testLayerNorm(128, 768);  // Large (typical transformer)
}

TEST_VK(MlKernels, RmsNorm) {
	auto testRmsNorm = [this](OaU32 rows, OaU32 cols) {
		constexpr float eps = 1e-6f;
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
		
		std::vector<float> x(rows * cols), weight(cols), out_ref(rows * cols);
		for (auto &v : x) v = dist(rng);
		for (auto &v : weight) v = dist(rng);
		
		CpuRmsNorm(x, weight, out_ref, rows, cols, eps);
		
		auto resultX = Rt_.AllocBufferBar(rows * cols * 4);
		auto resultWeight = Rt_.AllocBufferBar(cols * 4);
		auto resultOut = Rt_.AllocBufferBar(rows * cols * 4);
		ASSERT_TRUE(resultX.IsOk() && resultWeight.IsOk() && resultOut.IsOk());
		
		auto bufX = std::move(resultX).GetValue();
		auto bufWeight = std::move(resultWeight).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufX.MappedPtr, x.data(), rows * cols * 4);
		std::memcpy(bufWeight.MappedPtr, weight.data(), cols * 4);
		
		struct { OaU32 rows; OaU32 cols; float eps; } pc = { rows, cols, eps };
		OaVkBuffer bufs[] = {bufX, bufWeight, bufOut};
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "RmsNorm", bufs, &pc, sizeof(pc), rows).IsOk());
		
		std::vector<float> out_gpu(rows * cols);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, rows * cols * 4);
		
		CompareResults(out_ref, out_gpu, rows * cols, 1e-4f, "RmsNorm");
		
		Rt_.FreeBuffer(bufX);
		Rt_.FreeBuffer(bufWeight);
		Rt_.FreeBuffer(bufOut);
	};
	
	testRmsNorm(8, 128);
	testRmsNorm(32, 512);
	testRmsNorm(128, 768);
}

TEST_VK(MlKernels, Softmax) {
	auto testSoftmax = [this](OaU32 rows, OaU32 cols) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
		
		std::vector<float> x(rows * cols), out_ref(rows * cols);
		for (auto &v : x) v = dist(rng);
		
		CpuSoftmax(x, out_ref, rows, cols);
		
		auto resultX = Rt_.AllocBufferBar(rows * cols * 4);
		auto resultOut = Rt_.AllocBufferBar(rows * cols * 4);
		ASSERT_TRUE(resultX.IsOk() && resultOut.IsOk());
		
		auto bufX = std::move(resultX).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufX.MappedPtr, x.data(), rows * cols * 4);
		
		struct { OaU32 rows; OaU32 cols; } pc = { rows, cols };
		OaVkBuffer bufs[] = {bufX, bufOut};
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Softmax", bufs, &pc, sizeof(pc), rows).IsOk());
		
		std::vector<float> out_gpu(rows * cols);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, rows * cols * 4);
		
		// Softmax uses exp - slightly relaxed tolerance
		CompareResults(out_ref, out_gpu, rows * cols, 1e-5f, "Softmax");
		
		// Verify sum to 1.0 for each row
		for (OaU32 row = 0; row < rows; ++row) {
			float sum = 0.0f;
			for (OaU32 col = 0; col < cols; ++col) {
				sum += out_gpu[row * cols + col];
			}
			EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax row " << row << " doesn't sum to 1.0";
		}
		
		Rt_.FreeBuffer(bufX);
		Rt_.FreeBuffer(bufOut);
	};
	
	testSoftmax(8, 64);
	testSoftmax(32, 128);
	testSoftmax(128, 512);
}

TEST_VK(MlKernels, Swiglu) {
	auto testSwiglu = [this](OaU32 count) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		
		std::vector<float> gate(count), up(count), out_ref(count);
		for (auto &v : gate) v = dist(rng);
		for (auto &v : up) v = dist(rng);
		
		CpuSwiglu(gate, up, out_ref, count);
		
		auto resultGate = Rt_.AllocBufferBar(count * 4);
		auto resultUp = Rt_.AllocBufferBar(count * 4);
		auto resultOut = Rt_.AllocBufferBar(count * 4);
		ASSERT_TRUE(resultGate.IsOk() && resultUp.IsOk() && resultOut.IsOk());
		
		auto bufGate = std::move(resultGate).GetValue();
		auto bufUp = std::move(resultUp).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufGate.MappedPtr, gate.data(), count * 4);
		std::memcpy(bufUp.MappedPtr, up.data(), count * 4);
		
		struct { OaU32 count; } pc = { count };
		OaVkBuffer bufs[] = {bufGate, bufUp, bufOut};
		OaU32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Swiglu", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, count * 4);
		
		// SwiGLU uses exp - slightly relaxed tolerance
		CompareResults(out_ref, out_gpu, count, 1e-5f, "Swiglu");
		
		Rt_.FreeBuffer(bufGate);
		Rt_.FreeBuffer(bufUp);
		Rt_.FreeBuffer(bufOut);
	};
	
	testSwiglu(1024);
	testSwiglu(4096);
}

TEST_VK(MlKernels, Embedding) {
	auto testEmbedding = [this](OaU32 vocabSize, OaU32 dim, OaU32 numIndices) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		std::uniform_int_distribution<OaU32> idxDist(0, vocabSize - 1);
		
		std::vector<float> table(vocabSize * dim), out_ref(numIndices * dim);
		std::vector<OaU32> indices(numIndices);
		
		for (auto &v : table) v = dist(rng);
		for (auto &idx : indices) idx = idxDist(rng);
		
		CpuEmbedding(indices, table, out_ref, numIndices, dim);
		
		auto resultTable = Rt_.AllocBufferBar(vocabSize * dim * 4);
		auto resultIndices = Rt_.AllocBufferBar(numIndices * 4);
		auto resultOut = Rt_.AllocBufferBar(numIndices * dim * 4);
		ASSERT_TRUE(resultTable.IsOk() && resultIndices.IsOk() && resultOut.IsOk());
		
		auto bufTable = std::move(resultTable).GetValue();
		auto bufIndices = std::move(resultIndices).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufTable.MappedPtr, table.data(), vocabSize * dim * 4);
		std::memcpy(bufIndices.MappedPtr, indices.data(), numIndices * 4);
		
		struct { OaU32 numIndices; OaU32 dim; } pc = { numIndices, dim };
		OaVkBuffer bufs[] = {bufTable, bufIndices, bufOut};
		OaU32 groups = (numIndices * dim + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Embedding", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> out_gpu(numIndices * dim);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, numIndices * dim * 4);
		
		CompareResults(out_ref, out_gpu, numIndices * dim, 1e-6f, "Embedding");
		
		Rt_.FreeBuffer(bufTable);
		Rt_.FreeBuffer(bufIndices);
		Rt_.FreeBuffer(bufOut);
	};
	
	testEmbedding(1000, 64, 16);    // Small vocab
	testEmbedding(10000, 128, 32);  // Medium vocab
	testEmbedding(50000, 256, 64);  // Large vocab (typical)
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_VK(MlKernels, EdgeCases) {
	// Test single element
	{
		std::vector<float> x = {2.5f}, out_ref(1);
		CpuGelu(x, out_ref, 1);
		
		auto resultX = Rt_.AllocBufferBar(4);
		auto resultOut = Rt_.AllocBufferBar(4);
		ASSERT_TRUE(resultX.IsOk() && resultOut.IsOk());
		
		auto bufX = std::move(resultX).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufX.MappedPtr, x.data(), 4);
		
		struct { OaU32 count; } pc = { 1 };
		OaVkBuffer bufs[] = {bufX, bufOut};
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Gelu", bufs, &pc, sizeof(pc), 1).IsOk());
		
		float out_gpu;
		std::memcpy(&out_gpu, bufOut.MappedPtr, 4);
		
		EXPECT_NEAR(out_ref[0], out_gpu, 1e-4f) << "Gelu single element";
		
		Rt_.FreeBuffer(bufX);
		Rt_.FreeBuffer(bufOut);
	}
	
	// Test non-aligned size
	{
		constexpr OaU32 count = 257;
		std::vector<float> x(count), out_ref(count);
		for (OaU32 i = 0; i < count; ++i) x[i] = static_cast<float>(i) * 0.01f - 1.0f;
		CpuRelu(x, out_ref, count);
		
		auto resultX = Rt_.AllocBufferBar(count * 4);
		auto resultOut = Rt_.AllocBufferBar(count * 4);
		ASSERT_TRUE(resultX.IsOk() && resultOut.IsOk());
		
		auto bufX = std::move(resultX).GetValue();
		auto bufOut = std::move(resultOut).GetValue();
		
		std::memcpy(bufX.MappedPtr, x.data(), count * 4);
		
		struct { OaU32 count; } pc = { count };
		OaVkBuffer bufs[] = {bufX, bufOut};
		OaU32 groups = (count + 255) / 256;
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "Relu", bufs, &pc, sizeof(pc), groups).IsOk());
		
		std::vector<float> out_gpu(count);
		std::memcpy(out_gpu.data(), bufOut.MappedPtr, count * 4);
		
		CompareResults(out_ref, out_gpu, count, 1e-6f, "Relu non-aligned");
		
		Rt_.FreeBuffer(bufX);
		Rt_.FreeBuffer(bufOut);
	}
}

// ============================================================================
// CrossEntropy Tests (CRITICAL - gradient bug fixed)
// ============================================================================

TEST_VK(MlKernels, CrossEntropyFwd) {
	auto testCrossEntropy = [this](OaU32 batch, OaU32 classes, float tolerance = 1e-5f) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		std::uniform_int_distribution<OaU32> targetDist(0, classes - 1);
		
		std::vector<float> logits(batch * classes), loss_ref(batch);
		std::vector<OaU32> targets_u32(batch);
		std::vector<OaU8> targets(batch);  // GPU kernel expects uint8_t
		
		for (auto &v : logits) v = dist(rng);
		for (auto &t : targets_u32) t = targetDist(rng);
		for (OaU32 i = 0; i < batch; ++i) targets[i] = static_cast<OaU8>(targets_u32[i]);
		
		CpuCrossEntropyFwd(logits, targets_u32, loss_ref, batch, classes);
		
		auto resultLogits = Rt_.AllocBufferBar(batch * classes * 4);
		auto resultTargets = Rt_.AllocBufferBar(batch);  // 1 byte per target
		auto resultLoss = Rt_.AllocBufferBar(batch * 4);
		ASSERT_TRUE(resultLogits.IsOk() && resultTargets.IsOk() && resultLoss.IsOk());
		
		auto bufLogits = std::move(resultLogits).GetValue();
		auto bufTargets = std::move(resultTargets).GetValue();
		auto bufLoss = std::move(resultLoss).GetValue();
		
		std::memcpy(bufLogits.MappedPtr, logits.data(), batch * classes * 4);
		std::memcpy(bufTargets.MappedPtr, targets.data(), batch);  // Copy uint8_t targets
		
		struct { OaU32 batch; OaU32 classes; OaU32 target_dtype; } pc = { batch, classes, 0 };  // 0=UInt8
		OaVkBuffer bufs[] = {bufLogits, bufTargets, bufLoss};
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "CrossEntropy", bufs, &pc, sizeof(pc), batch).IsOk());
		
		std::vector<float> loss_gpu(batch);
		std::memcpy(loss_gpu.data(), bufLoss.MappedPtr, batch * 4);
		
		CompareResults(loss_ref, loss_gpu, batch, tolerance, "CrossEntropyFwd");
		
		Rt_.FreeBuffer(bufLogits);
		Rt_.FreeBuffer(bufTargets);
		Rt_.FreeBuffer(bufLoss);
	};
	
	testCrossEntropy(8, 10);    // Small: 10 classes
	testCrossEntropy(32, 100);  // Medium: 100 classes
	testCrossEntropy(128, 256); // Large: 256 classes (max for uint8_t)
}


TEST_VK(MlKernels, CrossEntropyBwd) {
	auto testCrossEntropyBwd = [this](OaU32 batch, OaU32 classes) {
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		std::uniform_int_distribution<OaU32> targetDist(0, classes - 1);
		
		std::vector<float> logits(batch * classes), d_logits_ref(batch * classes);
		std::vector<OaU32> targets_u32(batch);
		std::vector<OaU8> targets(batch);  // GPU kernel expects uint8_t
		
		for (auto &v : logits) v = dist(rng);
		for (auto &t : targets_u32) t = targetDist(rng);
		for (OaU32 i = 0; i < batch; ++i) targets[i] = static_cast<OaU8>(targets_u32[i]);
		
		CpuCrossEntropyBwd(logits, targets_u32, d_logits_ref, batch, classes);
		
		auto resultLogits = Rt_.AllocBufferBar(batch * classes * 4);
		auto resultTargets = Rt_.AllocBufferBar(batch * 4);  // 4 bytes per uint32 target
		auto resultDLogits = Rt_.AllocBufferBar(batch * classes * 4);
		ASSERT_TRUE(resultLogits.IsOk() && resultTargets.IsOk() && resultDLogits.IsOk());
		
		auto bufLogits = std::move(resultLogits).GetValue();
		auto bufTargets = std::move(resultTargets).GetValue();
		auto bufDLogits = std::move(resultDLogits).GetValue();
		
		std::memcpy(bufLogits.MappedPtr, logits.data(), batch * classes * 4);
		std::memcpy(bufTargets.MappedPtr, targets_u32.data(), batch * 4);  // Copy uint32 targets
		
		struct { OaU32 batch; OaU32 classes; OaU32 target_dtype; } pc = { batch, classes, 1 };  // 1=UInt32
		OaVkBuffer bufs[] = {bufLogits, bufTargets, bufDLogits};
		
		ASSERT_TRUE(OaVkDispatch::Run(Rt_, "CrossEntropyBwd", bufs, &pc, sizeof(pc), batch).IsOk());
		
		std::vector<float> d_logits_gpu(batch * classes);
		std::memcpy(d_logits_gpu.data(), bufDLogits.MappedPtr, batch * classes * 4);
		
		CompareResults(d_logits_ref, d_logits_gpu, batch * classes, 1e-5f, "CrossEntropyBwd");
		
		// Verify gradient sum per row ≈ 0 (softmax property)
		for (OaU32 row = 0; row < batch; ++row) {
			float row_sum = 0.0f;
			for (OaU32 col = 0; col < classes; ++col) {
				row_sum += d_logits_gpu[row * classes + col];
			}
			EXPECT_NEAR(row_sum, 0.0f, 1e-5f) << "CrossEntropyBwd row " << row << " gradients don't sum to 0";
		}
		
		Rt_.FreeBuffer(bufLogits);
		Rt_.FreeBuffer(bufTargets);
		Rt_.FreeBuffer(bufDLogits);
	};
	
	testCrossEntropyBwd(8, 10);    // Small: 10 classes
	testCrossEntropyBwd(32, 100);  // Medium: 100 classes
	testCrossEntropyBwd(128, 256); // Large: 256 classes (max for uint8_t)
}

TEST_VK(MlKernels, MaskedCrossEntropyFwdBwd) {
	constexpr OaU32 rows = 4;
	constexpr OaU32 classes = 3;
	constexpr OaU32 valid = 2;
	const std::vector<float> logits = {
		1.0f, 2.0f, 3.0f,  3.0f, 1.0f, 0.0f,
		9.0f, 8.0f, 7.0f, -2.0f, 0.0f, 2.0f};
	const std::vector<OaU32> targets = {2, 0, 0, 1};
	const std::vector<float> mask = {1.0f, 1.0f, 0.0f, 0.0f};
	std::vector<float> refLoss(rows), refGrad(rows * classes);
	CpuCrossEntropyFwd(logits, targets, refLoss, rows, classes);
	CpuCrossEntropyBwd(logits, targets, refGrad, rows, classes);
	for (OaU32 r = 0; r < rows; ++r) {
		if (mask[r] == 0.0f) {
			refLoss[r] = 0.0f;
			for (OaU32 c = 0; c < classes; ++c) refGrad[r * classes + c] = 0.0f;
		} else {
			// CPU helper normalized by rows; masked CE normalizes by valid rows.
			for (OaU32 c = 0; c < classes; ++c)
				refGrad[r * classes + c] *= static_cast<float>(rows) / valid;
		}
	}

	auto logitsResult = Rt_.AllocBufferBar(logits.size() * sizeof(float));
	auto targetsResult = Rt_.AllocBufferBar(targets.size() * sizeof(OaU32));
	auto maskResult = Rt_.AllocBufferBar(mask.size() * sizeof(float));
	auto lossResult = Rt_.AllocBufferBar(rows * sizeof(float));
	auto gradResult = Rt_.AllocBufferBar(logits.size() * sizeof(float));
	ASSERT_TRUE(logitsResult.IsOk() && targetsResult.IsOk() && maskResult.IsOk()
		&& lossResult.IsOk() && gradResult.IsOk());
	auto logitsBuf = std::move(logitsResult).GetValue();
	auto targetsBuf = std::move(targetsResult).GetValue();
	auto maskBuf = std::move(maskResult).GetValue();
	auto lossBuf = std::move(lossResult).GetValue();
	auto gradBuf = std::move(gradResult).GetValue();
	std::memcpy(logitsBuf.MappedPtr, logits.data(), logits.size() * sizeof(float));
	std::memcpy(targetsBuf.MappedPtr, targets.data(), targets.size() * sizeof(OaU32));
	std::memcpy(maskBuf.MappedPtr, mask.data(), mask.size() * sizeof(float));

	struct { OaU32 Rows, Classes, TargetDtype; } fwd{rows, classes, 1};
	OaVkBuffer fwdBufs[] = {logitsBuf, targetsBuf, maskBuf, lossBuf};
	ASSERT_TRUE(OaVkDispatch::Run(Rt_, "MaskedCrossEntropy", fwdBufs,
		&fwd, sizeof(fwd), rows).IsOk());
	std::vector<float> gpuLoss(rows);
	std::memcpy(gpuLoss.data(), lossBuf.MappedPtr, rows * sizeof(float));
	CompareResults(refLoss, gpuLoss, rows, 1e-5f, "MaskedCrossEntropyFwd");

	struct { OaU32 Rows, Classes, TargetDtype, ValidCount; } bwd{rows, classes, 1, valid};
	OaVkBuffer bwdBufs[] = {logitsBuf, targetsBuf, maskBuf, gradBuf};
	ASSERT_TRUE(OaVkDispatch::Run(Rt_, "MaskedCrossEntropyBwd", bwdBufs,
		&bwd, sizeof(bwd), rows).IsOk());
	std::vector<float> gpuGrad(logits.size());
	std::memcpy(gpuGrad.data(), gradBuf.MappedPtr, gpuGrad.size() * sizeof(float));
	CompareResults(refGrad, gpuGrad, static_cast<OaU32>(gpuGrad.size()), 1e-5f,
		"MaskedCrossEntropyBwd");

	Rt_.FreeBuffer(logitsBuf); Rt_.FreeBuffer(targetsBuf); Rt_.FreeBuffer(maskBuf);
	Rt_.FreeBuffer(lossBuf); Rt_.FreeBuffer(gradBuf);
}



// ============================================================================
// Module Integration Tests (High-Level API)
// ============================================================================

TEST(NN, Linear2D) {
	OaLinear linear(4, 3);
	auto out = linear.Forward(OaFnMatrix::Ones(OaMatrixShape{2, 4}));
	OaExpectShape(out, {2, 3});
	OaExpectFinite(out);
}

TEST(NN, Linear3D) {
	OaLinear linear(8, 4);
	auto out = linear.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 5, 8}));
	OaExpectShape(out, {2, 5, 4});
	OaExpectFinite(out);
}

TEST(NN, PackedLinearThreeWayMatchesIndependentProjections) {
	auto x = OaFnMatrix::Rand(OaMatrixShape{7, 11});
	auto w0 = OaFnMatrix::Rand(OaMatrixShape{5, 11});
	auto w1 = OaFnMatrix::Rand(OaMatrixShape{3, 11});
	auto w2 = OaFnMatrix::Rand(OaMatrixShape{9, 11});
	auto b0 = OaFnMatrix::Rand(OaMatrixShape{5});
	auto b1 = OaFnMatrix::Rand(OaMatrixShape{3});
	auto b2 = OaFnMatrix::Rand(OaMatrixShape{9});
	auto packed = OaFnMatrix::PackedLinear3(x, w0, w1, w2, b0, b1, b2);
	OaI64 widths[] = {5, 3, 9};
	auto split = OaFnMatrix::Split(packed, OaSpan<OaI64>(widths, 3), 1);
	auto ref0 = OaFnMatrix::Linear(x, w0, b0);
	auto ref1 = OaFnMatrix::Linear(x, w1, b1);
	auto ref2 = OaFnMatrix::Linear(x, w2, b2);
	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());
	OaExpectMatrixNear(split[0], ref0, 2e-5F);
	OaExpectMatrixNear(split[1], ref1, 2e-5F);
	OaExpectMatrixNear(split[2], ref2, 2e-5F);
}

TEST(NN, Embedding) {
	OaEmbedding emb(10, 4);
	auto out = emb.Forward(OaMakeByteIndices({1, 5, 9}));
	OaExpectShape(out, {3, 4});
}

TEST(NN, LayerNorm) {
	OaLayerNorm ln(4);
	auto out = ln.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 4}));
	OaExpectShape(out, {2, 4});
	OaExpectFinite(out);
}

TEST(NN, RMSNorm) {
	OaRmsNorm rn(8);
	auto out = rn.Forward(OaFnMatrix::Rand(OaMatrixShape{3, 8}));
	OaExpectShape(out, {3, 8});
	OaExpectFinite(out);
}

TEST(NN, Activations) {
	auto input = OaFnMatrix::Rand(OaMatrixShape{2, 4});
	OaRelu relu; OaGelu gelu; OaSilu silu;
	OaExpectFinite(relu.Forward(input));
	OaExpectFinite(gelu.Forward(input));
	OaExpectFinite(silu.Forward(input));
}

TEST(NN, SoftmaxModule) {
	OaSoftmax sm(-1);
	auto out = sm.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 4}));
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	OaExpectValidProbability(out);
}

TEST(NN, Conv1d) {
	OaConv1d conv(1, 2, 3, 1, 0);
	auto out = conv.Forward(OaFnMatrix::Rand(OaMatrixShape{1, 1, 10}));
	OaExpectShape(out, {1, 2, 8});
}

TEST(NN, Conv1dGradCheck) {
	// Finite-difference check of Conv1dBwdData (dX) and Conv1dBwdWeight (dW, dB)
	// against the forward kernel. Loss = sum(out), so dOut = ones and the analytic
	// grads come straight from the Bwd kernels.
	auto& ctx = OaContext::GetDefault();
	const OaI32 B = 1, Cin = 2, L = 5, Cout = 3, K = 3;
	const OaU32 stride = 1, pad = 1;

	std::vector<float> xh(static_cast<size_t>(B) * Cin * L);
	std::vector<float> wh(static_cast<size_t>(Cout) * Cin * K);
	std::vector<float> bh(static_cast<size_t>(Cout));
	for (size_t i = 0; i < xh.size(); ++i) xh[i] = std::sin(0.7 * static_cast<double>(i) + 1.0);
	for (size_t i = 0; i < wh.size(); ++i) wh[i] = std::cos(0.5 * static_cast<double>(i) + 0.3);
	for (size_t i = 0; i < bh.size(); ++i) bh[i] = 0.1 * static_cast<double>(i);

	auto mk = [](const std::vector<float>& v, const OaMatrixShape& s) {
		return OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(v.data()), v.size() * sizeof(float)), s);
	};
	auto fwdLoss = [&](const std::vector<float>& xv, const std::vector<float>& wv, const std::vector<float>& bv) -> float {
		ctx.Clear();
		auto o = OaFnMatrix::Conv1dGemm(mk(xv, OaMatrixShape{B, Cin, L}), mk(wv, OaMatrixShape{Cout, Cin, K}),
			mk(bv, OaMatrixShape{Cout}), stride, pad);
		auto s = OaFnMatrix::Sum(o);
		(void)ctx.Execute(); (void)ctx.Sync();
		return s.At(0);
	};

	// Analytic grads (dOut = ones).
	ctx.Clear();
	auto X = mk(xh, OaMatrixShape{B, Cin, L});
	auto W = mk(wh, OaMatrixShape{Cout, Cin, K});
	auto o = OaFnMatrix::Conv1dGemm(X, W, mk(bh, OaMatrixShape{Cout}), stride, pad);
	auto dOut = OaFnMatrix::Ones(o.GetShape());
	auto dX = OaFnMatrix::Conv1dBwdData(dOut, W, stride, pad, 1, X.GetShape());
	auto dwb = OaFnMatrix::Conv1dBwdWeight(X, dOut, W, stride, pad, 1);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	const float eps = 1e-3f;
	int badX = 0, badW = 0, badB = 0;
	for (size_t i = 0; i < xh.size(); ++i) {
		auto xp = xh; xp[i] += eps; auto xm = xh; xm[i] -= eps;
		const float num = (fwdLoss(xp, wh, bh) - fwdLoss(xm, wh, bh)) / (2 * eps);
		const float ana = dX.At(static_cast<OaI64>(i));
		if (std::fabs(num - ana) > 1e-2f) { ++badX; printf("  dX[%zu] num=%.4f ana=%.4f\n", i, num, ana); }
	}
	for (size_t i = 0; i < wh.size(); ++i) {
		auto wp = wh; wp[i] += eps; auto wm = wh; wm[i] -= eps;
		const float num = (fwdLoss(xh, wp, bh) - fwdLoss(xh, wm, bh)) / (2 * eps);
		const float ana = dwb.GradWeight.At(static_cast<OaI64>(i));
		if (std::fabs(num - ana) > 1e-2f) { ++badW; printf("  dW[%zu] num=%.4f ana=%.4f\n", i, num, ana); }
	}
	for (size_t i = 0; i < bh.size(); ++i) {
		auto bp = bh; bp[i] += eps; auto bm = bh; bm[i] -= eps;
		const float num = (fwdLoss(xh, wh, bp) - fwdLoss(xh, wh, bm)) / (2 * eps);
		const float ana = dwb.GradBias.At(static_cast<OaI64>(i));
		if (std::fabs(num - ana) > 1e-2f) { ++badB; printf("  dB[%zu] num=%.4f ana=%.4f\n", i, num, ana); }
	}
	printf("Conv1dGradCheck: badX=%d/%zu badW=%d/%zu badB=%d/%zu\n",
		badX, xh.size(), badW, wh.size(), badB, bh.size());
	EXPECT_EQ(badX, 0); EXPECT_EQ(badW, 0); EXPECT_EQ(badB, 0);
}

// Forward correctness of the 1-D conv GEMM path (Conv1dGemm / Conv1dReluGemm) —
// THE conv path now that the scalar direct kernel is retired — against an
// independent CPU nested-loop reference, across every conv shape the tokenizers
// use: the bare K3/S1/P1 encode/decode convs, the strided K4/S2/P1 downsample, and
// a channel-mixing case (InC != OutC).
TEST(NN, Conv1dGemmMatchesCpuReference) {
	auto& ctx = OaContext::GetDefault();

	struct Cfg { OaI32 B, Cin, L, Cout, K, S, P; const char* name; };
	const Cfg cfgs[] = {
		{2, 8, 16, 8, 3, 1, 1, "K3/S1/P1 (encode/decode)"},
		{2, 8, 16, 8, 4, 2, 1, "K4/S2/P1 (downsample)"},
		{1, 6, 12, 10, 3, 1, 1, "K3/S1/P1 InC!=OutC"},
	};

	auto mk = [](const std::vector<float>& v, const OaMatrixShape& s) {
		return OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(v.data()), v.size() * sizeof(float)), s);
	};

	for (const auto& c : cfgs) {
		const OaI32 outL = ((c.L + 2 * c.P) - (c.K - 1) - 1) / c.S + 1;
		std::vector<float> xh(static_cast<size_t>(c.B) * c.Cin * c.L);
		std::vector<float> wh(static_cast<size_t>(c.Cout) * c.Cin * c.K);
		std::vector<float> bh(static_cast<size_t>(c.Cout));
		for (size_t i = 0; i < xh.size(); ++i) xh[i] = static_cast<float>(std::sin(0.7 * static_cast<double>(i) + 1.0));
		for (size_t i = 0; i < wh.size(); ++i) wh[i] = static_cast<float>(std::cos(0.5 * static_cast<double>(i) + 0.3));
		for (size_t i = 0; i < bh.size(); ++i) bh[i] = static_cast<float>(0.1 * static_cast<double>(i) - 0.2);

		// CPU reference conv: out[n,oc,ol] = bias[oc] + sum_{ic,k} x[n,ic,ol*S-P+k] * w[oc,ic,k]
		std::vector<float> ref(static_cast<size_t>(c.B) * c.Cout * outL);
		auto xAt = [&](OaI32 n, OaI32 ic, OaI32 l) {
			return xh[((static_cast<size_t>(n) * c.Cin) + ic) * c.L + l];
		};
		for (OaI32 n = 0; n < c.B; ++n)
		for (OaI32 oc = 0; oc < c.Cout; ++oc)
		for (OaI32 ol = 0; ol < outL; ++ol) {
			double acc = bh[static_cast<size_t>(oc)];
			for (OaI32 ic = 0; ic < c.Cin; ++ic)
			for (OaI32 k = 0; k < c.K; ++k) {
				const OaI32 l = ol * c.S - c.P + k;
				if (l >= 0 && l < c.L) acc += static_cast<double>(xAt(n, ic, l)) * wh[(static_cast<size_t>(oc) * c.Cin + ic) * c.K + k];
			}
			ref[(static_cast<size_t>(n) * c.Cout + oc) * outL + ol] = static_cast<float>(acc);
		}

		ctx.Clear();
		auto X = mk(xh, OaMatrixShape{c.B, c.Cin, c.L});
		auto W = mk(wh, OaMatrixShape{c.Cout, c.Cin, c.K});
		auto B = mk(bh, OaMatrixShape{c.Cout});
		auto gemm     = OaFnMatrix::Conv1dGemm(X, W, B, c.S, c.P, 1);
		auto gemmRelu = OaFnMatrix::Conv1dReluGemm(X, W, B, c.S, c.P, 1);
		ASSERT_TRUE(ctx.Execute().IsOk()) << c.name;
		ASSERT_TRUE(ctx.Sync().IsOk()) << c.name;

		ASSERT_EQ(gemm.NumElements(), static_cast<OaI64>(ref.size())) << c.name;
		float maxBare = 0.0f, maxRelu = 0.0f;
		for (OaI64 i = 0; i < gemm.NumElements(); ++i) {
			const float r = ref[static_cast<size_t>(i)];
			maxBare = std::max(maxBare, std::fabs(gemm.At(i) - r));
			maxRelu = std::max(maxRelu, std::fabs(gemmRelu.At(i) - std::max(0.0f, r)));
		}
		printf("Conv1dGemmMatchesCpuReference[%s]: maxBareDiff=%.2e maxReluDiff=%.2e\n",
			c.name, maxBare, maxRelu);
		EXPECT_LT(maxBare, 2e-3f) << c.name << ": Conv1dGemm disagrees with CPU reference";
		EXPECT_LT(maxRelu, 2e-3f) << c.name << ": Conv1dReluGemm disagrees with CPU reference";
	}
}

// Finite-difference gradcheck of the GEMM-composed autograd backward (Im2Col1d +
// MatMulNt + BiasAdd), the backward that OaConv1d gets once its hand-written
// OaGradConv1d node is dropped. Confirms the composed graph produces correct
// dX / dW / dB, not just that it runs. CrossEntropy over the conv output as
// [rows, Cout] logits is the differentiable scalar root (Sum has no autograd node).
TEST(NN, Conv1dGemmGradCheck) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 B = 1, Cin = 2, L = 5, Cout = 3, K = 3;
	const OaI32 stride = 1, pad = 1;
	const OaI32 outL = ((L + 2 * pad) - (K - 1) - 1) / stride + 1;
	const OaI64 rows = static_cast<OaI64>(B) * outL;

	std::vector<float> xh(static_cast<size_t>(B) * Cin * L);
	std::vector<float> wh(static_cast<size_t>(Cout) * Cin * K);
	std::vector<float> bh(static_cast<size_t>(Cout));
	std::vector<OaU32> th(static_cast<size_t>(rows));
	for (size_t i = 0; i < xh.size(); ++i) xh[i] = std::sin(0.7 * static_cast<double>(i) + 1.0);
	for (size_t i = 0; i < wh.size(); ++i) wh[i] = std::cos(0.5 * static_cast<double>(i) + 0.3);
	for (size_t i = 0; i < bh.size(); ++i) bh[i] = 0.1 * static_cast<double>(i);
	for (size_t i = 0; i < th.size(); ++i) th[i] = static_cast<OaU32>(i % Cout);

	auto mk = [](const std::vector<float>& v, const OaMatrixShape& s) {
		return OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(v.data()), v.size() * sizeof(float)), s);
	};
	auto mkTargets = [&] {
		auto t = OaFnMatrix::Empty(OaMatrixShape{rows}, OaScalarType::UInt32);
		auto* p = t.DataAs<OaU32>();
		for (size_t i = 0; i < th.size(); ++i) p[i] = th[i];
		return t;
	};
	auto logits = [&](const OaMatrix& conv) {
		// [B, Cout, outL] -> [B, outL, Cout] -> [rows, Cout]
		return OaFnMatrix::Reshape(OaFnMatrix::Transpose(conv, 1, 2), OaMatrixShape{rows, Cout});
	};
	auto readGrad = [](const OaMatrix& g) {
		std::vector<float> h(static_cast<size_t>(g.NumElements()));
		(void)OaFnMatrix::CopyToHost(g, h.data(), h.size() * sizeof(float));
		return h;
	};
	auto fwdLoss = [&](const std::vector<float>& xv, const std::vector<float>& wv, const std::vector<float>& bv) -> float {
		ctx.Clear();
		auto o = OaFnMatrix::Conv1dGemm(mk(xv, OaMatrixShape{B, Cin, L}), mk(wv, OaMatrixShape{Cout, Cin, K}),
			mk(bv, OaMatrixShape{Cout}), stride, pad, 1);
		auto loss = OaFnLoss::CrossEntropy(logits(o), mkTargets());
		(void)ctx.Execute(); (void)ctx.Sync();
		return loss.At(0);
	};

	// Analytic grads via the composed autograd backward.
	ctx.Clear();
	auto X = mk(xh, OaMatrixShape{B, Cin, L}); X.SetRequiresGrad(true);
	auto W = mk(wh, OaMatrixShape{Cout, Cin, K}); W.SetRequiresGrad(true);
	auto Bs = mk(bh, OaMatrixShape{Cout}); Bs.SetRequiresGrad(true);
	OaGradientTape tape;
	auto o = OaFnMatrix::Conv1dGemm(X, W, Bs, stride, pad, 1);
	auto loss = OaFnLoss::CrossEntropy(logits(o), mkTargets());
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	auto dX = readGrad(X.GradMatrix());
	auto dW = readGrad(W.GradMatrix());
	auto dB = readGrad(Bs.GradMatrix());

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
	printf("Conv1dGemmGradCheck: badX=%d/%zu badW=%d/%zu badB=%d/%zu\n",
		badX, xh.size(), badW, wh.size(), badB, bh.size());
	EXPECT_EQ(badX, 0); EXPECT_EQ(badW, 0); EXPECT_EQ(badB, 0);
}

TEST(NN, Conv2d) {
	OaConv2d conv(1, 4, 3, 1, 0);
	auto out = conv.Forward(OaFnMatrix::Rand(OaMatrixShape{1, 1, 8, 8}));
	OaExpectShape(out, {1, 4, 6, 6});
}

TEST(NN, Conv2dDepthwiseGroups) {
	const std::array<OaF32, 8> inputValues{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
	};
	const std::array<OaF32, 2> weightValues{2.0f, 3.0f};
	const std::array<OaF32, 2> biasValues{10.0f, 20.0f};

	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(inputValues.data()),
			inputValues.size() * sizeof(OaF32)),
		OaMatrixShape{1, 2, 2, 2});
	auto weight = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(weightValues.data()),
			weightValues.size() * sizeof(OaF32)),
		OaMatrixShape{2, 1, 1, 1});
	auto bias = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(biasValues.data()),
			biasValues.size() * sizeof(OaF32)),
		OaMatrixShape{2});

	auto out = OaFnMatrix::Conv2d(input, weight, bias, 1, 0, 2);
	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());
	OaExpectShape(out, {1, 2, 2, 2});

	const std::array<OaF32, 8> expected{
		12.0f, 14.0f, 16.0f, 18.0f,
		35.0f, 38.0f, 41.0f, 44.0f,
	};
	for (OaI64 i = 0; i < static_cast<OaI64>(expected.size()); ++i) {
		EXPECT_FLOAT_EQ(out.At(i), expected[static_cast<OaUsize>(i)]);
	}
}

TEST(NN, Conv2dDepthwiseGroupsBackward) {
	const std::array<OaF32, 8> inputValues{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
	};
	const std::array<OaF32, 2> weightValues{2.0f, 3.0f};
	const std::array<OaF32, 8> gradValues{
		1.0f, 1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f, 1.0f,
	};

	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(inputValues.data()),
			inputValues.size() * sizeof(OaF32)),
		OaMatrixShape{1, 2, 2, 2});
	auto weight = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(weightValues.data()),
			weightValues.size() * sizeof(OaF32)),
		OaMatrixShape{2, 1, 1, 1});
	auto gradOut = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(gradValues.data()),
			gradValues.size() * sizeof(OaF32)),
		OaMatrixShape{1, 2, 2, 2});

	auto gradInput = OaFnMatrix::Conv2dBwdData(
		gradOut, weight, 1, 0, input.GetShape(), 2);
	auto gradWeightBias = OaFnMatrix::Conv2dBwdWeight(
		input, gradOut, weight, 1, 0, 2);
	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());

	const std::array<OaF32, 8> expectedGradInput{
		2.0f, 2.0f, 2.0f, 2.0f,
		3.0f, 3.0f, 3.0f, 3.0f,
	};
	for (OaI64 i = 0; i < static_cast<OaI64>(expectedGradInput.size()); ++i) {
		EXPECT_FLOAT_EQ(
			gradInput.At(i), expectedGradInput[static_cast<OaUsize>(i)]);
	}
	EXPECT_FLOAT_EQ(gradWeightBias.GradWeight.At(0), 10.0f);
	EXPECT_FLOAT_EQ(gradWeightBias.GradWeight.At(1), 26.0f);
	EXPECT_FLOAT_EQ(gradWeightBias.GradBias.At(0), 4.0f);
	EXPECT_FLOAT_EQ(gradWeightBias.GradBias.At(1), 4.0f);
}

TEST(NN, MatrixSliceConcatRecordedGraph) {
	const std::array<OaF32, 8> inputValues{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
	};
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(inputValues.data()),
			inputValues.size() * sizeof(OaF32)),
		OaMatrixShape{1, 2, 2, 2});

	// Keep producer, channel split, and merge in one graph. This is the C3k2 path.
	auto produced = OaFnMatrix::Add(input, input);
	auto first = OaFnMatrix::Slice(produced, 1, 0, 1);
	auto second = OaFnMatrix::Slice(produced, 1, 1, 2);
	OaMatrix parts[] = {second, first};
	auto merged = OaFnMatrix::Concat(OaSpan<OaMatrix>(parts, 2), 1);

	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());
	OaExpectShape(merged, {1, 2, 2, 2});

	const std::array<OaF32, 8> expected{
		10.0f, 12.0f, 14.0f, 16.0f,
		2.0f, 4.0f, 6.0f, 8.0f,
	};
	for (OaI64 i = 0; i < static_cast<OaI64>(expected.size()); ++i) {
		EXPECT_FLOAT_EQ(merged.At(i), expected[static_cast<OaUsize>(i)]);
	}
}

TEST(NN, MatrixConcatDetectHeadDimension) {
	const std::array<OaF32, 4> firstValues{1.0f, 2.0f, 3.0f, 4.0f};
	const std::array<OaF32, 2> secondValues{5.0f, 6.0f};
	auto first = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(firstValues.data()),
			firstValues.size() * sizeof(OaF32)),
		OaMatrixShape{1, 2, 2});
	auto second = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(secondValues.data()),
			secondValues.size() * sizeof(OaF32)),
		OaMatrixShape{1, 2, 1});
	OaMatrix parts[] = {first, second};
	auto merged = OaFnMatrix::Concat(OaSpan<OaMatrix>(parts, 2), 2);

	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());
	OaExpectShape(merged, {1, 2, 3});

	const std::array<OaF32, 6> expected{1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f};
	for (OaI64 i = 0; i < static_cast<OaI64>(expected.size()); ++i) {
		EXPECT_FLOAT_EQ(merged.At(i), expected[static_cast<OaUsize>(i)]);
	}
}

TEST(NN, BatchNormUpsampleRecordedGraph) {
	const std::array<OaF32, 4> inputValues{1.0f, 2.0f, 3.0f, 4.0f};
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(inputValues.data()),
			inputValues.size() * sizeof(OaF32)),
		OaMatrixShape{1, 1, 2, 2});

	OaBatchNorm2d batchNorm(1);
	OaUpsample upsample(2, OaUpsampleMode::Nearest);
	OaContext::ScopedEval eval(OaContext::GetDefault());
	auto produced = OaFnMatrix::Add(input, input);
	auto normalized = batchNorm.Forward(produced);
	auto activated = OaFnMatrix::Silu(normalized);
	auto output = upsample.Forward(activated);

	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());
	OaExpectShape(output, {1, 1, 4, 4});
	OaExpectFinite(output);
	EXPECT_FLOAT_EQ(output.At(0), output.At(1));
	EXPECT_FLOAT_EQ(output.At(0), output.At(4));
	EXPECT_GT(output.At(15), output.At(0));
}

TEST(NN, BatchNormConstantLargeMagnitudeRemainsFinite) {
	// E[x^2] - E[x]^2 is cancellation-prone in FP32. The statistics kernel
	// must never feed a small negative variance into rsqrt.
	auto input = OaFnMatrix::Full(OaMatrixShape{1, 1, 1, 513}, 10000.0F);
	OaBatchNorm2d batchNorm(1);
	auto output = batchNorm.Forward(input);

	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());
	OaExpectFinite(output);
	ASSERT_EQ(batchNorm.Buffers().Size(), 2u);
	EXPECT_TRUE(std::isfinite(batchNorm.Buffers()[1].Data.At(0)));
	EXPECT_GE(batchNorm.Buffers()[1].Data.At(0), 0.0F);
}

class OaModuleBufferFixture final : public OaModule {
public:
	OaModuleBufferFixture() {
		RegisterParameter("weight", OaFnMatrix::Ones(OaMatrixShape{2}));
		RegisterBuffer("persistent", OaFnMatrix::Zeros(OaMatrixShape{2}));
		RegisterBuffer("scratch", OaFnMatrix::Zeros(OaMatrixShape{1}), false);
	}
};

TEST(NN, ModuleBuffersAreStateNotParameters) {
	OaModuleBufferFixture module;
	EXPECT_EQ(module.Parameters().Size(), 1u);
	EXPECT_EQ(module.AllParameterPtrs().Size(), 1u);
	EXPECT_EQ(module.NumParameters(), 2);
	EXPECT_EQ(module.Buffers().Size(), 2u);
	EXPECT_EQ(module.AllBufferPtrs().Size(), 2u);
	EXPECT_EQ(module.AllBufferPtrs(true).Size(), 1u);
	EXPECT_FALSE(module.Buffers()[0].Data.RequiresGrad());
}

TEST(NN, ModulePersistentBuffersRoundTripThroughOamState) {
	OaModuleBufferFixture source;
	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());
	source.Parameters()[0].Data.Set(0, 3.0f);
	source.Parameters()[0].Data.Set(1, 4.0f);
	source.Buffers()[0].Data.Set(0, 7.0f);
	source.Buffers()[0].Data.Set(1, 8.0f);
	source.Buffers()[1].Data.Set(0, 9.0f);

	OamModel checkpoint;
	source.SaveTo(checkpoint);
	ASSERT_NE(checkpoint.FindWeight("weight"), nullptr);
	ASSERT_NE(checkpoint.FindState("persistent"), nullptr);
	EXPECT_EQ(checkpoint.FindState("scratch"), nullptr);

	OaModuleBufferFixture restored;
	restored.LoadFrom(checkpoint);
	EXPECT_FLOAT_EQ(restored.Parameters()[0].Data.At(0), 3.0f);
	EXPECT_FLOAT_EQ(restored.Parameters()[0].Data.At(1), 4.0f);
	EXPECT_FLOAT_EQ(restored.Buffers()[0].Data.At(0), 7.0f);
	EXPECT_FLOAT_EQ(restored.Buffers()[0].Data.At(1), 8.0f);
	EXPECT_FLOAT_EQ(restored.Buffers()[1].Data.At(0), 0.0f);
}

TEST(NN, BatchNormUpdatesPersistentRunningState) {
	const std::array<OaF32, 4> inputValues{1.0f, 2.0f, 3.0f, 4.0f};
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(inputValues.data()),
			inputValues.size() * sizeof(OaF32)),
		OaMatrixShape{1, 1, 2, 2});

	OaBatchNorm2d batchNorm(1, 1e-5f, 0.1f);
	(void)batchNorm.Forward(input);
	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());

	ASSERT_EQ(batchNorm.Parameters().Size(), 2u);
	ASSERT_EQ(batchNorm.Buffers().Size(), 2u);
	EXPECT_NEAR(batchNorm.Buffers()[0].Data.At(0), 0.25f, 1e-5f);
	EXPECT_NEAR(batchNorm.Buffers()[1].Data.At(0), 1.025f, 1e-5f);
}

TEST(NN, Sequential) {
	OaSequential model;
	model.Add(OaMakeSharedPtr<OaLinear>(4, 8));
	model.Add(OaMakeSharedPtr<OaGelu>());
	model.Add(OaMakeSharedPtr<OaLinear>(8, 2));
	EXPECT_GT(model.NumParameters(), 0);
	auto out = model.Forward(OaFnMatrix::Rand(OaMatrixShape{3, 4}));
	OaExpectShape(out, {3, 2});
}

TEST(NN, SequentialInitializerListAndFlatten) {
	OaSequential model({
		OaMakeSharedPtr<OaFlatten>(),
		OaMakeSharedPtr<OaLinear>(4, 2),
		OaMakeSharedPtr<OaIdentity>(),
	});
	EXPECT_EQ(model.Children().Size(), 3u);
	EXPECT_GT(model.NumParameters(), 0);
	auto out = model.Forward(OaFnMatrix::Rand(OaMatrixShape{3, 2, 2}));
	OaExpectShape(out, {3, 2});
}

TEST(NN, DropoutEvalPassthrough) {
	OaDropout drop(0.5f);
	OaContext::ScopedEval guard(OaContext::GetDefault());
	auto input = OaFnMatrix::Ones(OaMatrixShape{10});
	auto out = drop.Forward(input);
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	EXPECT_FLOAT_EQ(out.At(0), 1.0f);
}

TEST(NN, DropoutTrainingIsGpuNativeAndInverted) {
	constexpr OaI64 N = 4096;
	constexpr OaF32 P = 0.25f;
	OaDropout drop(P);
	OaFnMatrix::SetRngSeed(12345);
	auto input = OaFnMatrix::Ones(OaMatrixShape{N});
	auto out = drop.Forward(input);
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();

	OaI64 kept = 0;
	for (OaI64 i = 0; i < N; ++i) {
		const OaF32 value = out.At(i);
		if (value != 0.0f) {
			++kept;
			EXPECT_NEAR(value, 1.0f / (1.0f - P), 1e-6f);
		}
	}
	const OaF32 keepFraction = static_cast<OaF32>(kept) / static_cast<OaF32>(N);
	EXPECT_NEAR(keepFraction, 1.0f - P, 0.03f);
}

TEST(NN, FfnForwardCorrectness) {
	constexpr OaU32 kRows = 4;
	constexpr OaU32 kDModel = 8;
	constexpr OaU32 kDFF = 16;
	constexpr float kEps = 1e-5f;

	// Varied (not uniform) input + weights. Uniform weights make every output
	// column identical, which would hide transpose/indexing bugs in the GEMMs —
	// distinct per-element values make this a real correctness check.
	std::vector<float> x_cpu(kRows * kDModel);
	for (OaU32 i = 0; i < x_cpu.size(); ++i) {
		x_cpu[i] = std::sin(static_cast<float>(i) * 0.3f) * 0.5f;
	}
	std::vector<float> norm_w(kDModel);
	std::vector<float> gate_w(kDFF * kDModel);
	std::vector<float> up_w(kDFF * kDModel);
	std::vector<float> down_w(kDModel * kDFF);
	for (OaU32 i = 0; i < norm_w.size(); ++i) norm_w[i] = 0.8f + 0.05f * static_cast<float>(i % 5);
	for (OaU32 i = 0; i < gate_w.size(); ++i) gate_w[i] = std::cos(static_cast<float>(i) * 0.2f) * 0.25f;
	for (OaU32 i = 0; i < up_w.size(); ++i)   up_w[i]   = std::sin(static_cast<float>(i) * 0.17f) * 0.25f;
	for (OaU32 i = 0; i < down_w.size(); ++i) down_w[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.2f;

	// CPU reference
	std::vector<float> expected(kRows * kDModel);
	CpuFfnForward(x_cpu, norm_w, gate_w, up_w, down_w, expected, kRows, kDModel, kDFF, kEps);

	// GPU computation. OaFfn composes child modules (norm/gate/up/down), so its
	// weights live in AllParameterPtrs(), in registration order:
	//   [0] norm.weight  [1] gate.weight [2] gate.bias
	//   [3] up.weight    [4] up.bias     [5] down.weight [6] down.bias
	OaFfn ffn(kDModel, kDFF, kEps);
	auto ptrs = ffn.AllParameterPtrs();
	ASSERT_EQ(ptrs.Size(), static_cast<OaUsize>(7));

	// Overwrite each weight with our deterministic values (shape-preserving).
	// Biases stay zero (OaLinear default) — the CPU reference assumes the same.
	auto setWeights = [](OaParameter* p, const std::vector<float>& w) {
		p->Data = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(w.data()), w.size() * sizeof(float)),
			p->Data.GetShape());
	};
	setWeights(ptrs[0], norm_w);
	setWeights(ptrs[1], gate_w);
	setWeights(ptrs[3], up_w);
	setWeights(ptrs[5], down_w);

	auto x_gpu = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(x_cpu.data()), x_cpu.size() * sizeof(float)),
		OaMatrixShape{kRows, kDModel});

	auto out_gpu = ffn.Forward(x_gpu);
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();

	// Combined abs+rel tolerance for the engine's bf16 tensor-core GEMMs (3
	// matmuls feed each output). The pre-residual signal is small here, so bf16
	// noise is significant in relative terms — but an absolute floor covers it.
	// A structural/indexing bug produces errors far larger than this envelope.
	const float* gpu_data = out_gpu.DataAs<const float>();
	for (OaU32 i = 0; i < expected.size(); ++i) {
		const float tol = 1e-2f + 4e-2f * std::abs(expected[i]);
		EXPECT_LE(std::abs(gpu_data[i] - expected[i]), tol) << "Mismatch at index " << i
			<< " gpu=" << gpu_data[i] << " expected=" << expected[i];
	}
}

TEST(NN, FfnModuleParameterRegistration) {
	constexpr OaI32 kDModel = 32;
	constexpr OaI32 kDFF = 64;
	constexpr float kEps = 1e-5f;

	OaFfn ffn(kDModel, kDFF, kEps);

	// FFN registers norm/gate/up/down as child modules, so all weights surface
	// through AllParameterPtrs() (7 total) — the FFN's own direct Parameters()
	// and Buffers() vectors are empty.
	EXPECT_EQ(ffn.Parameters().Size(), static_cast<OaUsize>(0));
	EXPECT_EQ(ffn.Buffers().Size(), static_cast<OaUsize>(0));
	EXPECT_GT(ffn.NumParameters(), 0);

	auto ptrs = ffn.AllParameterPtrs();
	ASSERT_EQ(ptrs.Size(), static_cast<OaUsize>(7));
	// Order: norm.weight, gate.{weight,bias}, up.{weight,bias}, down.{weight,bias}
	EXPECT_EQ(ptrs[0]->Data.GetShape(), OaMatrixShape{kDModel});         // RMSNorm weight
	EXPECT_EQ(ptrs[1]->Data.GetShape(), (OaMatrixShape{kDFF, kDModel}));   // Gate weight
	EXPECT_EQ(ptrs[2]->Data.GetShape(), OaMatrixShape{kDFF});            // Gate bias
	EXPECT_EQ(ptrs[3]->Data.GetShape(), (OaMatrixShape{kDFF, kDModel}));   // Up weight
	EXPECT_EQ(ptrs[4]->Data.GetShape(), OaMatrixShape{kDFF});            // Up bias
	EXPECT_EQ(ptrs[5]->Data.GetShape(), (OaMatrixShape{kDModel, kDFF}));   // Down weight
	EXPECT_EQ(ptrs[6]->Data.GetShape(), OaMatrixShape{kDModel});         // Down bias
}

TEST(NN, FfnModuleForwardShape) {
	constexpr OaI32 kDModel = 16;
	constexpr OaI32 kDFF = 32;
	constexpr OaI32 kBatch = 4;

	OaFfn ffn(kDModel, kDFF);
	auto x = OaFnMatrix::Rand(OaMatrixShape{kBatch, kDModel});

	auto out = ffn.Forward(x);
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();

	OaExpectShape(out, {kBatch, kDModel});
	OaExpectFinite(out);
}

TEST(NN, FfnModuleAutograd) {
	constexpr OaI32 kDModel = 8;
	constexpr OaI32 kDFF = 16;
	constexpr OaI32 kBatch = 2;

	OaFfn ffn(kDModel, kDFF);
	auto x = OaFnMatrix::Rand(OaMatrixShape{kBatch, kDModel});

	// Differentiable scalar loss root. OaFnMatrix::Sum has no autograd node, so
	// treat the FFN output as logits over kDModel classes and use CrossEntropy
	// (the same differentiable root the MNIST tutorial uses).
	auto targets = OaFnMatrix::Empty(OaMatrixShape{kBatch}, OaScalarType::UInt32);
	auto* tgt = targets.DataAs<OaU32>();
	for (OaI32 i = 0; i < kBatch; ++i) tgt[i] = static_cast<OaU32>(i % kDModel);

	OaGradientTape tape;
	auto out = ffn.Forward(x);
	auto loss = OaFnLoss::CrossEntropy(out, targets);
	tape.Backward(loss);
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();

	// Every parameter must carry a gradient of matching shape.
	auto ptrs = ffn.AllParameterPtrs();
	ASSERT_EQ(ptrs.Size(), static_cast<OaUsize>(7));
	for (OaUsize i = 0; i < ptrs.Size(); ++i) {
		EXPECT_TRUE(ptrs[i]->Grad().HasStorage()) << "Parameter " << i << " has no gradient storage";
		EXPECT_EQ(ptrs[i]->Grad().GetShape(), ptrs[i]->Data.GetShape())
			<< "Parameter " << i << " grad shape mismatch";
	}

	// Gradient must actually flow: the down-projection weight (index 5) feeds the
	// loss directly, so its accumulated gradient is non-zero.
	const OaMatrix& downGrad = ptrs[5]->Grad();
	std::vector<float> hostGrad(static_cast<size_t>(downGrad.NumElements()));
	(void)OaFnMatrix::CopyToHost(downGrad, hostGrad.data(), hostGrad.size() * sizeof(float));
	float gradMag = 0.0f;
	for (float g : hostGrad) gradMag += std::abs(g);
	EXPECT_GT(gradMag, 0.0f) << "down weight gradient did not flow";
}

TEST(NN, FfnModuleSaveLoadRoundtrip) {
	constexpr OaI32 kDModel = 8;
	constexpr OaI32 kDFF = 16;

	OaFfn ffn(kDModel, kDFF);
	auto x = OaFnMatrix::Rand(OaMatrixShape{2, kDModel});

	// Forward before save
	auto out1 = ffn.Forward(x);
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();

	// Save
	const OaString ckptPath = "/tmp/ffn_test.oam";
	auto saveStatus = ffn.Save(ckptPath);
	ASSERT_TRUE(saveStatus.IsOk()) << "Save failed: " << saveStatus.GetMessage();

	// Load into new instance
	OaFfn ffn2(kDModel, kDFF);
	auto loadStatus = ffn2.Load(ckptPath);
	ASSERT_TRUE(loadStatus.IsOk()) << "Load failed: " << loadStatus.GetMessage();

	// Forward after load
	auto out2 = ffn2.Forward(x);
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();

	// Compare outputs
	const float* data1 = out1.DataAs<const float>();
	const float* data2 = out2.DataAs<const float>();
	for (OaI64 i = 0; i < out1.NumElements(); ++i) {
		EXPECT_FLOAT_EQ(data1[i], data2[i]) << "Mismatch at index " << i;
	}
}
