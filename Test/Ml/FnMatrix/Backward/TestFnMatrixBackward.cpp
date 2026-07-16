// Test suite for OaFnMatrix backward operations
// Tests gradient computation for neural network operations

#include <Oa/Oa.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Spirv.h>
#include <OaTest.h>
#include <cmath>
#include <cstdlib>
#include <numeric>

// ─── Helper Functions ───────────────────────────────────────────────────────

static float Relu(float x) { return x > 0.0f ? x : 0.0f; }
static float ReluGrad(float x) { return x > 0.0f ? 1.0f : 0.0f; }

static float Tanh(float x) { return std::tanh(x); }
static float TanhGrad(float y) { return 1.0f - y * y; }

static float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float SigmoidGrad(float y) { return y * (1.0f - y); }

static float LeakyRelu(float x, float alpha = 0.01f) { return x > 0.0f ? x : alpha * x; }
static float LeakyReluGrad(float x, float alpha = 0.01f) { return x > 0.0f ? 1.0f : alpha; }

static float Elu(float x, float alpha = 1.0f) { return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f); }
static float EluGrad(float y, float alpha = 1.0f) { return y > 0.0f ? 1.0f : y + alpha; }

// ─── Simple Activation Backward Tests ──────────────────────────────────────

TEST(OaFnMatrixBackward, ReluBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::Relu(input);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});
	
	auto grad_input = OaFnMatrix::ReluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_1 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	// CPU reference
	std::vector<float> expected(5);
	for (size_t i = 0; i < 5; i++) {
		expected[i] = ReluGrad(input_data[i]) * grad_output_data[i];
	}
	
	for (size_t i = 0; i < 5; i++) {
		EXPECT_NEAR(result[i], expected[i], 1e-5f);
	}
}

TEST(OaFnMatrixBackward, TanhBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::Tanh(input);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});
	
	auto grad_input = OaFnMatrix::TanhBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_2 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	std::vector<float> forward_host(5);
	[[maybe_unused]] auto copy_result_3 = OaFnMatrix::CopyToHost(forward_output, forward_host.data(), forward_host.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		float expected = TanhGrad(forward_host[i]) * grad_output_data[i];
		EXPECT_NEAR(result[i], expected, 1e-5f);
	}
}

TEST(OaFnMatrixBackward, SigmoidBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::Sigmoid(input);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});
	
	auto grad_input = OaFnMatrix::SigmoidBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_4 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	std::vector<float> forward_host(5);
	[[maybe_unused]] auto copy_result_5 = OaFnMatrix::CopyToHost(forward_output, forward_host.data(), forward_host.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		float expected = SigmoidGrad(forward_host[i]) * grad_output_data[i];
		EXPECT_NEAR(result[i], expected, 1e-5f);
	}
}

TEST(OaFnMatrixBackward, GeluBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::Gelu(input);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});
	
	auto grad_input = OaFnMatrix::GeluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_6 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(OaFnMatrixBackward, SiluBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::Silu(input);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});
	
	auto grad_input = OaFnMatrix::SiluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_7 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(OaFnMatrixBackward, SoftplusBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::Softplus(input);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});
	
	auto grad_input = OaFnMatrix::SoftplusBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_8 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(OaFnMatrixBackward, LeakyReluBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::LeakyRelu(input, 0.01f);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});
	
	auto grad_input = OaFnMatrix::LeakyReluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_9 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		float expected = LeakyReluGrad(input_data[i]) * grad_output_data[i];
		EXPECT_NEAR(result[i], expected, 1e-5f);
	}
}

TEST(OaFnMatrixBackward, EluBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::Elu(input, 1.0f);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});
	
	auto grad_input = OaFnMatrix::EluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_10 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	std::vector<float> forward_host(5);
	[[maybe_unused]] auto copy_result_11 = OaFnMatrix::CopyToHost(forward_output, forward_host.data(), forward_host.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		float expected = EluGrad(forward_host[i]) * grad_output_data[i];
		EXPECT_NEAR(result[i], expected, 1e-4f);
	}
}

// ─── Non-default-alpha autograd regression ─────────────────────────────────
// The value-level *Bwd above always ran with the default alpha. The bug these
// guard: the generated OaGrad{LeakyRelu,Elu} node dropped alpha, so the *tape*
// always backpropagated with the defaulted slope (0.01 / 1.0). With a NON-default
// alpha the negative-side analytic grad diverged from finite differences. Now alpha
// is threaded through the autograd node (ctor_args=["InAlpha"]); these must match FD.

// Central-difference gradcheck of a leaf input through the autograd tape.
static void AlphaAutogradGradcheck(
	OaContext& InCtx, OaMatrix& InInput, const std::function<OaMatrix()>& InForward,
	const OaMatrix& InTarget, const char* InName)
{
	OaGradientTape tape;
	auto out  = InForward();
	auto loss = OaFnLoss::Mse(out, InTarget);
	tape.Backward(loss);
	(void)InCtx.Execute(); (void)InCtx.Sync();

	auto analytic = InInput.GradMatrix();
	const OaF32* ana = analytic.DataAs<const OaF32>();

	auto lossFunc = [&]() -> OaF32 {
		OaGradNo noGrad;
		auto o = InForward();
		auto l = OaFnLoss::Mse(o, InTarget);
		(void)InCtx.Execute(); (void)InCtx.Sync();
		return l.DataAs<const OaF32>()[0];
	};

	OaF32* data = InInput.DataAs<OaF32>();
	const OaI64 n = InInput.NumElements();
	const OaF32 eps = 1e-3f;
	int failed = 0, nonTrivial = 0;
	for (OaI64 i = 0; i < n; ++i) {
		const OaF32 orig = data[i];
		data[i] = orig + eps; (void)InCtx.Sync(); const OaF32 lp = lossFunc();
		data[i] = orig - eps; (void)InCtx.Sync(); const OaF32 lm = lossFunc();
		data[i] = orig; (void)InCtx.Sync();
		const OaF32 num = (lp - lm) / (2.0f * eps);
		const OaF32 tol = 2e-3f + 2e-2f * std::abs(num);
		if (std::abs(num - ana[i]) > tol) {
			++failed;
			printf("  [%s] idx %lld: analytic=%.6f numerical=%.6f MISMATCH\n",
				InName, static_cast<long long>(i), ana[i], num);
		}
		if (std::abs(num) > 5e-4f) ++nonTrivial;
	}
	EXPECT_EQ(failed, 0) << InName << ": autograd grad disagrees with finite differences";
	EXPECT_GE(nonTrivial, 3) << InName << ": gradients all ~0 — vacuous check";
}

// ─── SPIR-V push-constant reflection (backs the bindless-contract assert) ──
// The Record debug assert is only as good as the reflected block size. Pin the
// exact reflected sizes for representative kernel shapes so the check can never
// silently go vacuous (e.g. reflection regressing to always-0). Each shader's
// PushConstants struct = [one uint buffer-index per bound buffer] ++ [host tail].
TEST(OaFnMatrixBackward, SpirvPushBlockReflection) {
	// binary {a_idx,b_idx,out_idx,count}                 = 4*4 = 16
	EXPECT_EQ(OaSpvPushConstantBlockSizeByName("Add"), 16u);
	// unary {x_idx,out_idx,count}                        = 3*4 = 12
	EXPECT_EQ(OaSpvPushConstantBlockSizeByName("Sigmoid"), 12u);
	// unary_scalar {x_idx,out_idx,count,alpha}           = 4*4 = 16
	EXPECT_EQ(OaSpvPushConstantBlockSizeByName("LeakyRelu"), 16u);
	EXPECT_EQ(OaSpvPushConstantBlockSizeByName("Elu"), 16u);
	// hand-written bwd {x_idx,grad_out_idx,grad_in_idx,count,alpha} = 5*4 = 20
	EXPECT_EQ(OaSpvPushConstantBlockSizeByName("LeakyReluBwd"), 20u);
	// unknown kernel → 0 (conservative; assert is skipped)
	EXPECT_EQ(OaSpvPushConstantBlockSizeByName("NoSuchKernel_xyz"), 0u);
}

TEST(OaFnMatrixBackward, LeakyReluAutogradNonDefaultAlpha) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	const OaF32 alpha = 0.25f;  // far from the 0.01 default → exposes a dropped alpha

	std::vector<float> x = {-2.0f, -1.3f, -0.4f, 0.5f, 1.7f, 2.5f};
	std::vector<float> t = { 0.3f, -0.7f,  0.9f, 0.1f, -1.1f, 0.6f};
	auto input  = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(x.data()), x.size() * sizeof(float)), OaMatrixShape{6});
	auto target = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(t.data()), t.size() * sizeof(float)), OaMatrixShape{6});
	input.SetRequiresGrad(true);

	AlphaAutogradGradcheck(ctx, input, [&]() { return OaFnMatrix::LeakyRelu(input, alpha); },
		target, "LeakyRelu(alpha=0.25)");
}

TEST(OaFnMatrixBackward, EluAutogradNonDefaultAlpha) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	const OaF32 alpha = 0.5f;  // != 1.0 default → exposes a dropped alpha on y<=0

	std::vector<float> x = {-2.0f, -1.3f, -0.4f, 0.5f, 1.7f, 2.5f};
	std::vector<float> t = { 0.3f, -0.7f,  0.9f, 0.1f, -1.1f, 0.6f};
	auto input  = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(x.data()), x.size() * sizeof(float)), OaMatrixShape{6});
	auto target = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(t.data()), t.size() * sizeof(float)), OaMatrixShape{6});
	input.SetRequiresGrad(true);

	AlphaAutogradGradcheck(ctx, input, [&]() { return OaFnMatrix::Elu(input, alpha); },
		target, "Elu(alpha=0.5)");
}

TEST(OaFnMatrixBackward, MishBwd) {
	auto& ctx = OaContext::GetDefault();

	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::Mish(input);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});

	auto grad_input = OaFnMatrix::MishBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();

	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_12 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));

	for (size_t i = 0; i < 5; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(OaFnMatrixBackward, SoftmaxBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	std::vector<float> grad_output_data = {0.1f, 0.2f, 0.3f, 0.2f, 0.2f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{5});
	auto forward_output = OaFnMatrix::Softmax(input);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{5});
	
	auto grad_input = OaFnMatrix::SoftmaxBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_13 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}



// ─── Pooling Backward ───────────────────────────────────────────────────────

TEST(OaFnMatrixBackward, MaxPool2dBwd) {
	// Input: [1, 1, 4, 4] with distinct values so each 2x2 window has a clear max.
	std::vector<float> input_data = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{1, 1, 4, 4});

	// kernel=2, stride=2 → output [1,1,2,2]; maxes are at 6,8,14,16 (flat 5,7,13,15).
	auto pool_result = OaFnMatrix::MaxPool2d(input, 2, 2, 0);
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{1, 1, 2, 2});

	auto grad_input = OaFnMatrix::MaxPool2dBwd(input, pool_result.Indices, grad_output, 2, 2, 0);
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute(); (void)ctx.Sync();

	const OaF32* g = grad_input.DataAs<const OaF32>();
	// Each grad_output value must land on exactly its window's argmax input position.
	const int maxFlat[4] = {5, 7, 13, 15};
	const float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	int nonZero = 0;
	for (int i = 0; i < 16; ++i) if (g[i] != 0.0f) ++nonZero;
	EXPECT_EQ(nonZero, 4) << "gradient must flow to exactly the 4 max positions";
	for (int k = 0; k < 4; ++k)
		EXPECT_NEAR(g[maxFlat[k]], expected[k], 1e-5f)
			<< "grad routed to wrong position / value at window " << k;
}

TEST(OaFnMatrixBackward, AvgPool2dBwd) {
	auto& ctx = OaContext::GetDefault();
	
	// Input: [1, 1, 4, 4]
	std::vector<float> input_data(16, 1.0f);
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{1, 1, 4, 4});
	
	// Grad output: [1, 1, 2, 2]
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f};
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{1, 1, 2, 2});
	
	// AvgPool2dBwd now requires pooling parameters (kernel=2, stride=2, padding=0)
	auto grad_input = OaFnMatrix::AvgPool2dBwd(input, grad_output, 2, 2, 0);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	std::vector<float> result(16);
	[[maybe_unused]] auto copy_result_15 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	// Each output gradient should be distributed equally to 4 input positions
	for (size_t i = 0; i < 16; i++) {
		EXPECT_NEAR(result[i], 0.25f, 1e-5f);
	}
}

// ─── Linear Backward ────────────────────────────────────────────────────────

TEST(OaFnMatrixBackward, LinearDataBwd) {
	auto& ctx = OaContext::GetDefault();
	
	// grad_output: [2, 3], weight: [3, 4]
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> weight_data = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f
	};
	
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{2, 3});
	auto weight = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(weight_data.data()), weight_data.size() * sizeof(float)), OaMatrixShape{3, 4});
	
	auto grad_input = OaFnMatrix::LinearDataBwd(grad_output, weight);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	EXPECT_EQ(grad_input.Shape_, (OaMatrixShape{2, 4}));
	
	std::vector<float> result(8);
	[[maybe_unused]] auto copy_result_16 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 8; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(OaFnMatrixBackward, LinearWeightBwd) {
	auto& ctx = OaContext::GetDefault();
	
	// input: [2, 4], grad_output: [2, 3]
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{2, 4});
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{2, 3});
	
	auto grad_weight = OaFnMatrix::LinearWeightBwd(input, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	EXPECT_EQ(grad_weight.Shape_, (OaMatrixShape{3, 4}));
	
	std::vector<float> result(12);
	[[maybe_unused]] auto copy_result_17 = OaFnMatrix::CopyToHost(grad_weight, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 12; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(OaFnMatrixBackward, LinearWeightBiasBwd) {
	auto& ctx = OaContext::GetDefault();
	
	// input: [2, 4], grad_output: [2, 3]
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{2, 4});
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{2, 3});
	
	auto [grad_weight, grad_bias] = OaFnMatrix::LinearWeightBiasBwd(input, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	EXPECT_EQ(grad_weight.Shape_, (OaMatrixShape{3, 4}));
	EXPECT_EQ(grad_bias.Shape_, OaMatrixShape{3});
	
	std::vector<float> weight_result(12);
	std::vector<float> bias_result(3);
	[[maybe_unused]] auto copy_result_18 = OaFnMatrix::CopyToHost(grad_weight, weight_result.data(), weight_result.size() * sizeof(float));
	[[maybe_unused]] auto copy_result_19 = OaFnMatrix::CopyToHost(grad_bias, bias_result.data(), bias_result.size() * sizeof(float));
	
	for (size_t i = 0; i < 12; i++) {
		EXPECT_TRUE(std::isfinite(weight_result[i]));
	}
	for (size_t i = 0; i < 3; i++) {
		EXPECT_TRUE(std::isfinite(bias_result[i]));
	}
}

TEST(OaFnMatrixBackward, LinearWeightBiasBwd_LargeShape) {
	auto& ctx = OaContext::GetDefault();

	// Large shape exercises the scalar LinearWeightBiasBwd kernel.
	constexpr OaU32 M = 64;
	constexpr OaU32 N = 64;
	constexpr OaU32 K = 64;
	
	std::vector<float> input_data(M * K);
	std::vector<float> grad_output_data(M * N);
	for (OaU32 i = 0; i < M * K; ++i) input_data[i] = static_cast<float>(i % 7) * 0.1F;
	for (OaU32 i = 0; i < M * N; ++i) grad_output_data[i] = static_cast<float>(i % 5) * 0.2F;
	
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{M, K});
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{M, N});
	
	auto [grad_weight, grad_bias] = OaFnMatrix::LinearWeightBiasBwd(input, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	EXPECT_EQ(grad_weight.Shape_, (OaMatrixShape{N, K}));
	EXPECT_EQ(grad_bias.Shape_, OaMatrixShape{N});
	
	std::vector<float> weight_result(N * K);
	std::vector<float> bias_result(N);
	[[maybe_unused]] auto copy_weight = OaFnMatrix::CopyToHost(grad_weight, weight_result.data(), weight_result.size() * sizeof(float));
	[[maybe_unused]] auto copy_bias = OaFnMatrix::CopyToHost(grad_bias, bias_result.data(), bias_result.size() * sizeof(float));
	
	// CPU reference: gradWeight[n, k] = sum_m gradOut[m, n] * input[m, k]
	std::vector<float> weight_ref(N * K, 0.0F);
	std::vector<float> bias_ref(N, 0.0F);
	for (OaU32 n = 0; n < N; ++n) {
		for (OaU32 k = 0; k < K; ++k) {
			float sum = 0.0F;
			for (OaU32 m = 0; m < M; ++m) {
				sum += grad_output_data[m * N + n] * input_data[m * K + k];
			}
			weight_ref[n * K + k] = sum;
		}
		float bias_sum = 0.0F;
		for (OaU32 m = 0; m < M; ++m) {
			bias_sum += grad_output_data[m * N + n];
		}
		bias_ref[n] = bias_sum;
	}
	
	for (OaU32 i = 0; i < N * K; ++i) {
		EXPECT_NEAR(weight_result[i], weight_ref[i], 0.1F) << "i=" << i;
		EXPECT_TRUE(std::isfinite(weight_result[i]));
	}
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_NEAR(bias_result[i], bias_ref[i], 0.1F) << "i=" << i;
		EXPECT_TRUE(std::isfinite(bias_result[i]));
	}
}

TEST(OaFnMatrixBackward, LinearWeightBiasBwd_TiledRouteOddShape) {
	auto& ctx = OaContext::GetDefault();
	constexpr OaU32 M = 67, N = 97, K = 95;
	std::vector<float> inputData(M * K);
	std::vector<float> gradOutputData(M * N);
	for (OaU32 i = 0; i < M * K; ++i)
		inputData[i] = (static_cast<float>(i % 17) - 8.0F) * 0.03125F;
	for (OaU32 i = 0; i < M * N; ++i)
		gradOutputData[i] = (static_cast<float>(i % 13) - 6.0F) * 0.0625F;

	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(inputData.data()),
			inputData.size() * sizeof(float)), {M, K});
	auto gradOutput = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(gradOutputData.data()),
			gradOutputData.size() * sizeof(float)), {M, N});
	auto [gradWeight, gradBias] = OaFnMatrix::LinearWeightBiasBwd(input, gradOutput);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	std::vector<float> weight(N * K), bias(N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		gradWeight, weight.data(), weight.size() * sizeof(float)).IsOk());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		gradBias, bias.data(), bias.size() * sizeof(float)).IsOk());
	for (OaU32 n = 0; n < N; ++n) {
		float biasRef = 0.0F;
		for (OaU32 m = 0; m < M; ++m)
			biasRef += gradOutputData[m * N + n];
		EXPECT_NEAR(bias[n], biasRef, 2e-4F) << "bias n=" << n;
		for (OaU32 k = 0; k < K; ++k) {
			float weightRef = 0.0F;
			for (OaU32 m = 0; m < M; ++m)
				weightRef += gradOutputData[m * N + n] * inputData[m * K + k];
			EXPECT_NEAR(weight[n * K + k], weightRef, 2e-4F)
				<< "weight n=" << n << " k=" << k;
		}
	}
}

// ─── Normalization Backward (numerical gradient verification) ────────────────
//
// These replace earlier empty stubs. They run the FULL autograd path
// (forward → MSE → tape.Backward) and compare analytical parameter gradients to
// central finite differences. Crucially they use rows>1 and rank-3 [B,T,C]
// inputs: LayerNormBwd/RmsNormBwd write a per-row dw_contrib buffer of size
// rows*cols and the host must column-sum it. An earlier bug allocated that
// buffer as just [cols] (heap overflow for rows>1) and skipped the reduction,
// so dWeight/dBias were wrong for any batch>1 — invisible to rows==1 tests.

static void NormForceFp32() { setenv("OA_GEMM_FORCE_FP32", "1", 1); }

static bool NormGradClose(OaF32 InA, OaF32 InN, OaF32 InAtol = 2e-3F, OaF32 InRtol = 3e-2F) {
	return std::abs(InA - InN) <= (InAtol + (InRtol * std::abs(InN)));
}

// Run a gradcheck over every element of InParam given a loss-producing closure.
// Returns {numChecked, numFailed, numNonTrivial}.
struct NormGradStats { int Checked = 0; int Failed = 0; int NonTrivial = 0; };
static NormGradStats NormGradCheck(
	OaContext& InCtx, const std::function<OaF32()>& InLossFunc,
	OaMatrix& InParam, const OaF32* InAnalytical, const char* InName)
{
	NormGradStats s;
	OaF32* data = InParam.DataAs<OaF32>();
	const OaI64 n = InParam.NumElements();
	const OaF32 eps = 1e-2f;
	printf("  [%s] %lld elements\n", InName, static_cast<long long>(n));
	for (OaI64 i = 0; i < n; ++i) {
		const OaF32 orig = data[i];
		data[i] = orig + eps; (void)InCtx.Sync();
		const OaF32 lp = InLossFunc();
		data[i] = orig - eps; (void)InCtx.Sync();
		const OaF32 lm = InLossFunc();
		data[i] = orig; (void)InCtx.Sync();
		const OaF32 numerical = (lp - lm) / (2.0f * eps);
		const OaF32 analytical = InAnalytical[i];
		const bool close = NormGradClose(analytical, numerical);
		++s.Checked;
		if (not close) ++s.Failed;
		if (std::abs(numerical) > 5e-4f) ++s.NonTrivial;  // > FD noise → real signal
		if (not close)
			printf("    idx %lld: analytical=%.6f numerical=%.6f  MISMATCH\n",
				static_cast<long long>(i), analytical, numerical);
	}
	return s;
}

TEST(OaFnMatrixBackward, LayerNormBwd) {
	NormForceFp32();
	OaFnMatrix::SetRngSeed(7);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	// rank-3 [B,T,C] = [2,3,4] → rows=6, cols=4. Exercises the rows>1 dw_contrib
	// path AND the dBias rank-3 reduction (must sum over B and T, not just B).
	constexpr OaI32 B = 2, T = 3, C = 4;

	auto x = OaFnMatrix::Scale(OaFnMatrix::RandN(OaMatrixShape{B, T, C}, OaScalarType::Float32), 1.5f);
	auto weight = OaFnMatrix::Scale(OaFnMatrix::RandN(OaMatrixShape{C}, OaScalarType::Float32), 0.5f);
	auto bias   = OaFnMatrix::Scale(OaFnMatrix::RandN(OaMatrixShape{C}, OaScalarType::Float32), 0.5f);
	for (auto* m : {&x, &weight, &bias}) { m->SetRequiresGrad(true); }

	auto target = OaFnMatrix::RandN(OaMatrixShape{B, T, C}, OaScalarType::Float32);

	OaGradientTape tape;
	auto out  = OaFnMatrix::LayerNorm(x, weight, bias, 1e-5f);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto dWeight = weight.GradMatrix();
	auto dBias   = bias.GradMatrix();
	// Shapes must match the params (not [T,C] from a partial reduction).
	EXPECT_EQ(dWeight.NumElements(), C);
	EXPECT_EQ(dBias.NumElements(), C);

	auto lossFunc = [&]() -> OaF32 {
		OaGradNo noGrad;
		auto o = OaFnMatrix::LayerNorm(x, weight, bias, 1e-5f);
		auto l = OaFnLoss::Mse(o, target);
		(void)ctx.Execute(); (void)ctx.Sync();
		return l.DataAs<const OaF32>()[0];
	};

	printf("\nLayerNorm [B=%d,T=%d,C=%d] gradcheck:\n", B, T, C);
	auto sw = NormGradCheck(ctx, lossFunc, weight, dWeight.DataAs<const OaF32>(), "weight");
	auto sb = NormGradCheck(ctx, lossFunc, bias,   dBias.DataAs<const OaF32>(),   "bias");
	const int failed = sw.Failed + sb.Failed;
	const int nonTrivial = sw.NonTrivial + sb.NonTrivial;
	printf("LayerNorm gradcheck: %d/%d pass, %d non-trivial\n",
		(sw.Checked + sb.Checked) - failed, sw.Checked + sb.Checked, nonTrivial);
	EXPECT_EQ(failed, 0) << "LayerNorm weight/bias gradient mismatch";
	EXPECT_GE(nonTrivial, 3) << "gradients all ~0 — vacuous check";
}

TEST(OaFnMatrixBackward, RmsNormBwd) {
	NormForceFp32();
	OaFnMatrix::SetRngSeed(11);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	// rank-3 [B,T,C] = [2,3,4] → rows=6, cols=4 (rows>1 dw_contrib path).
	constexpr OaI32 B = 2, T = 3, C = 4;

	auto x = OaFnMatrix::Scale(OaFnMatrix::RandN(OaMatrixShape{B, T, C}, OaScalarType::Float32), 1.5f);
	auto weight = OaFnMatrix::Scale(OaFnMatrix::RandN(OaMatrixShape{C}, OaScalarType::Float32), 0.5f);
	x.SetRequiresGrad(true);
	weight.SetRequiresGrad(true);

	auto target = OaFnMatrix::RandN(OaMatrixShape{B, T, C}, OaScalarType::Float32);

	OaGradientTape tape;
	auto out  = OaFnMatrix::RmsNorm(x, weight, 1e-5f);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto dWeight = weight.GradMatrix();
	EXPECT_EQ(dWeight.NumElements(), C);

	auto lossFunc = [&]() -> OaF32 {
		OaGradNo noGrad;
		auto o = OaFnMatrix::RmsNorm(x, weight, 1e-5f);
		auto l = OaFnLoss::Mse(o, target);
		(void)ctx.Execute(); (void)ctx.Sync();
		return l.DataAs<const OaF32>()[0];
	};

	printf("\nRmsNorm [B=%d,T=%d,C=%d] gradcheck:\n", B, T, C);
	auto sw = NormGradCheck(ctx, lossFunc, weight, dWeight.DataAs<const OaF32>(), "weight");
	printf("RmsNorm gradcheck: %d/%d pass, %d non-trivial\n",
		sw.Checked - sw.Failed, sw.Checked, sw.NonTrivial);
	EXPECT_EQ(sw.Failed, 0) << "RmsNorm weight gradient mismatch";
	EXPECT_GE(sw.NonTrivial, 3) << "gradients all ~0 — vacuous check";
}

// ─── Embedding Backward ─────────────────────────────────────────────────────

TEST(OaFnMatrixBackward, GatherBwd) {
	auto& ctx = OaContext::GetDefault();

	// Embedding backward = scatter-add of the upstream gradient rows into the
	// table rows named by the (integer) gather indices. indices {0,2,1} route
	// grad row 0 → table[0], grad row 1 → table[2], grad row 2 → table[1];
	// table rows 3,4 are never named and must stay zero.
	constexpr OaI32 kVocab = 5, kEmbed = 4;
	std::vector<OaU32> indices_data = {0, 2, 1};
	std::vector<float> grad_output_data = {
		1.0f, 1.0f, 1.0f, 1.0f,   // → table[0]
		2.0f, 2.0f, 2.0f, 2.0f,   // → table[2]
		3.0f, 3.0f, 3.0f, 3.0f,   // → table[1]
	};

	auto indices = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(indices_data.data()),
			indices_data.size() * sizeof(OaU32)),
		OaMatrixShape{3}, OaScalarType::UInt32);
	auto grad_output = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()),
			grad_output_data.size() * sizeof(float)),
		OaMatrixShape{3, 4});

	auto grad_table = OaFnMatrix::GatherBwd(indices, grad_output, kVocab, kEmbed);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	EXPECT_EQ(grad_table.GetShape()[0], kVocab);
	EXPECT_EQ(grad_table.GetShape()[1], kEmbed);

	std::vector<float> result(kVocab * kEmbed);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(grad_table, result.data(),
		result.size() * sizeof(float)).IsOk());

	const float expected[kVocab][kEmbed] = {
		{1, 1, 1, 1},   // table[0] ← grad row 0
		{3, 3, 3, 3},   // table[1] ← grad row 2
		{2, 2, 2, 2},   // table[2] ← grad row 1
		{0, 0, 0, 0},   // untouched
		{0, 0, 0, 0},   // untouched
	};
	for (int v = 0; v < kVocab; ++v)
		for (int e = 0; e < kEmbed; ++e)
			EXPECT_NEAR(result[v * kEmbed + e], expected[v][e], 1e-5f)
				<< "table[" << v << "][" << e << "]";
}

// ─── Complex Backward Operations ────────────────────────────────────────────

TEST(OaFnMatrixBackward, SwigluBwd) {
	auto& ctx = OaContext::GetDefault();
	
	std::vector<float> gate_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> up_data = {5.0f, 6.0f, 7.0f, 8.0f};
	std::vector<float> out_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f};
	
	auto gate = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(gate_data.data()), gate_data.size() * sizeof(float)), OaMatrixShape{4});
	auto up = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(up_data.data()), up_data.size() * sizeof(float)), OaMatrixShape{4});
	auto out = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(out_data.data()), out_data.size() * sizeof(float)), OaMatrixShape{4});
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{4});
	
	auto [gate_grad, up_grad] = OaFnMatrix::SwigluBwd(gate, up, out, grad_output);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	EXPECT_EQ(gate_grad.Shape_, OaMatrixShape{4});
	EXPECT_EQ(up_grad.Shape_, OaMatrixShape{4});
	
	std::vector<float> gate_result(4);
	std::vector<float> up_result(4);
	[[maybe_unused]] auto copy_result_24 = OaFnMatrix::CopyToHost(gate_grad, gate_result.data(), gate_result.size() * sizeof(float));
	[[maybe_unused]] auto copy_result_25 = OaFnMatrix::CopyToHost(up_grad, up_result.data(), up_result.size() * sizeof(float));
	
	for (size_t i = 0; i < 4; i++) {
		EXPECT_TRUE(std::isfinite(gate_result[i]));
		EXPECT_TRUE(std::isfinite(up_result[i]));
	}
}

// Real finite-difference gradcheck of SiluMulBwd against the forward SiluMul.
// The forward (flat split) computes y[i] = SiLU(x[i]) * x[i+N] for i in [0,N), with
// input x = [gate(N) ; up(N)]. The backward takes the INPUT (not the un-invertible
// output) and must match d/dx of loss = sum_i dY[i]*y[i]. Was a vacuous isfinite
// check; the kernel didn't even exist (not compiled / not registered) until now.
TEST(OaFnMatrixBackward, SiluMulBwd) {
	auto& ctx = OaContext::GetDefault();
	const OaU32 N = 8;            // intermediate size
	const OaU32 M = 2 * N;        // input length (gate||up)

	std::vector<float> x = {-1.5f, -0.5f, 0.25f, 1.0f, 2.0f, -2.0f, 0.75f, -0.1f,
	                         0.4f,  1.3f, -0.8f, 0.6f, -1.2f, 0.9f, 2.5f, -0.3f};  // M=16
	std::vector<float> dY(N);
	for (OaU32 i = 0; i < N; ++i) dY[i] = 0.5f + 0.1f * static_cast<float>(i);

	auto upload = [](const std::vector<float>& v) {
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(v.data()),
			v.size() * sizeof(float)), OaMatrixShape{static_cast<OaI64>(v.size())});
	};

	// loss(x) = sum_{i<N} dY[i] * SiluMul(x)[i]
	auto lossOf = [&](const std::vector<float>& xv) -> double {
		OaGradNo noGrad;
		auto xm = upload(xv);
		auto y = OaFnMatrix::SiluMul(xm, N);
		(void)ctx.Execute(); (void)ctx.Sync();
		std::vector<float> yh(N);
		EXPECT_TRUE(OaFnMatrix::CopyToHost(y, yh.data(), N * sizeof(float)).IsOk());
		double s = 0.0;
		for (OaU32 i = 0; i < N; ++i) s += static_cast<double>(dY[i]) * static_cast<double>(yh[i]);
		return s;
	};

	// Analytic gradient via the kernel under test.
	auto xm = upload(x);
	auto dYm = upload(dY);
	auto gradInput = OaFnMatrix::SiluMulBwd(xm, dYm);
	(void)ctx.Execute(); (void)ctx.Sync();
	std::vector<float> analytic(M);
	(void)OaFnMatrix::CopyToHost(gradInput, analytic.data(), M * sizeof(float));

	// Central differences vs analytic for every input element.
	const float eps = 1e-3f;
	for (OaU32 j = 0; j < M; ++j) {
		std::vector<float> xp = x, xn = x;
		xp[j] += eps; xn[j] -= eps;
		double num = (lossOf(xp) - lossOf(xn)) / (2.0 * static_cast<double>(eps));
		EXPECT_NEAR(static_cast<double>(analytic[j]), num,
			2e-2 + 2e-2 * std::abs(num)) << "grad mismatch at input index " << j;
	}
}

// Real finite-difference gradcheck of GegluBwd against the forward Geglu.
// Forward (flat split): y[i] = x[i] * GELU(x[i+N]) for i in [0,N), input x =
// [up(N) ; gate(N)]. Backward takes the INPUT and must match d/dx of
// loss = sum_i dY[i]*y[i]. Was vacuous isfinite; kernel didn't exist until now.
TEST(OaFnMatrixBackward, GegluBwd) {
	auto& ctx = OaContext::GetDefault();
	const OaU32 N = 8;
	const OaU32 M = 2 * N;

	std::vector<float> x = {-1.5f, -0.5f, 0.25f, 1.0f, 2.0f, -2.0f, 0.75f, -0.1f,
	                         0.4f,  1.3f, -0.8f, 0.6f, -1.2f, 0.9f, 2.5f, -0.3f};
	std::vector<float> dY(N);
	for (OaU32 i = 0; i < N; ++i) dY[i] = 0.5f + 0.1f * static_cast<float>(i);

	auto upload = [](const std::vector<float>& v) {
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(v.data()),
			v.size() * sizeof(float)), OaMatrixShape{static_cast<OaI64>(v.size())});
	};

	auto lossOf = [&](const std::vector<float>& xv) -> double {
		OaGradNo noGrad;
		auto xm = upload(xv);
		auto y = OaFnMatrix::Geglu(xm, N);
		(void)ctx.Execute(); (void)ctx.Sync();
		std::vector<float> yh(N);
		EXPECT_TRUE(OaFnMatrix::CopyToHost(y, yh.data(), N * sizeof(float)).IsOk());
		double s = 0.0;
		for (OaU32 i = 0; i < N; ++i) s += static_cast<double>(dY[i]) * static_cast<double>(yh[i]);
		return s;
	};

	auto xm = upload(x);
	auto dYm = upload(dY);
	auto gradInput = OaFnMatrix::GegluBwd(xm, dYm);
	(void)ctx.Execute(); (void)ctx.Sync();
	std::vector<float> analytic(M);
	(void)OaFnMatrix::CopyToHost(gradInput, analytic.data(), M * sizeof(float));

	const float eps = 1e-3f;
	for (OaU32 j = 0; j < M; ++j) {
		std::vector<float> xp = x, xn = x;
		xp[j] += eps; xn[j] -= eps;
		double num = (lossOf(xp) - lossOf(xn)) / (2.0 * static_cast<double>(eps));
		EXPECT_NEAR(static_cast<double>(analytic[j]), num,
			2e-2 + 2e-2 * std::abs(num)) << "grad mismatch at input index " << j;
	}
}

// Full-reduction Max backward: grad routes only to the element equal to the max.
// (Previously this test merely checked isfinite — vacuous; the kernel was in fact
// reading buffers/push fields the host never supplied. The SPIR-V push-block
// assert in Record now makes that class of mismatch impossible.)
//
// The earlier "fails in-suite, passes standalone" flakiness had TWO independent
// root causes, both fixed: (1) a real bug in the forward Max.slang cross-wave
// reduction (it folded only wave-leaders 0,1,2,4 and dropped 3,5,6,7 — global max
// lost when it landed in a dropped wave; rewritten as a subgroup-size-independent
// tree reduction mirroring Sum); and (2) the preceding SiluMulBwd test recorded
// an op with no shader, whose failed Compile permanently bricked the shared default
// context so every later Execute returned zeros. OaContext::Execute now resets the
// graph on compile/replay failure so one bad kernel can't poison the context.
TEST(OaFnMatrixBackward, MaxBwd) {
	auto& ctx = OaContext::GetDefault();

	std::vector<float> input_data = {1.0f, 4.0f, 3.0f, 2.0f};  // max is index 1
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{4});

	std::vector<float> grad_data = {2.5f};  // upstream scalar grad
	auto gradOut = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_data.data()), grad_data.size() * sizeof(float)), OaMatrixShape{1});

	auto maxVal = OaFnMatrix::Max(input);
	// Materialize the forward max before the value-API backward reads it (OA is
	// deferred; MaxBwd consumes maxVal's contents, so it must be flushed first).
	[[maybe_unused]] auto e0 = ctx.Execute();
	[[maybe_unused]] auto s0 = ctx.Sync();
	auto gradInput = OaFnMatrix::MaxBwd(input, maxVal, gradOut);
	[[maybe_unused]] auto e = ctx.Execute();
	[[maybe_unused]] auto s = ctx.Sync();

	std::vector<float> result(4);
	[[maybe_unused]] auto c = OaFnMatrix::CopyToHost(gradInput, result.data(), result.size() * sizeof(float));

	// Only the argmax (index 1) receives the gradient; everything else is 0.
	EXPECT_NEAR(result[0], 0.0f, 1e-6f);
	EXPECT_NEAR(result[1], 2.5f, 1e-6f);
	EXPECT_NEAR(result[2], 0.0f, 1e-6f);
	EXPECT_NEAR(result[3], 0.0f, 1e-6f);
}

// Max is now differentiable end-to-end: tape backward must match the analytic
// scatter (grad to the max element only). Re-enabled with the Max.slang
// reduction fix (see MaxBwd note above).
TEST(OaFnMatrixBackward, MaxAutogradTape) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	std::vector<float> x = {-1.0f, 0.5f, 3.0f, 2.0f, -4.0f};  // max at index 2
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(x.data()), x.size() * sizeof(float)), OaMatrixShape{5});
	input.SetRequiresGrad(true);

	OaGradientTape tape;
	auto m = OaFnMatrix::Max(input);
	auto loss = OaFnMatrix::Scale(m, 3.0f);  // d loss / d max = 3
	(void)ctx.Execute(); (void)ctx.Sync();   // materialize forward before backward
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();

	auto g = input.GradMatrix();
	std::vector<float> grad(5);
	[[maybe_unused]] auto c = OaFnMatrix::CopyToHost(g, grad.data(), grad.size() * sizeof(float));

	for (size_t i = 0; i < 5; ++i)
		EXPECT_NEAR(grad[i], (i == 2) ? 3.0f : 0.0f, 1e-6f) << "i=" << i;
}

// ─── Vision Backward ────────────────────────────────────────────────────────

TEST(OaFnMatrixBackward, UpsampleBwd_Nearest) {
	auto& ctx = OaContext::GetDefault();
	
	// Input: [1, 1, 2, 2]
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{1, 1, 2, 2});
	
	// Grad output: [1, 1, 4, 4] (upsampled 2x)
	std::vector<float> grad_output_data(16, 1.0f);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{1, 1, 4, 4});
	
	auto grad_input = OaFnMatrix::UpsampleBwd(input, grad_output, 2, false);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	EXPECT_EQ(grad_input.Shape_, (OaMatrixShape{1, 1, 2, 2}));
	
	std::vector<float> result(4);
	[[maybe_unused]] auto copy_result_29 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	// Each input pixel should accumulate gradients from 4 output pixels
	for (size_t i = 0; i < 4; i++) {
		EXPECT_NEAR(result[i], 4.0f, 1e-5f);
	}
}

TEST(OaFnMatrixBackward, UpsampleBwd_Bilinear) {
	auto& ctx = OaContext::GetDefault();
	
	// Input: [1, 1, 2, 2]
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto input = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)), OaMatrixShape{1, 1, 2, 2});
	
	// Grad output: [1, 1, 4, 4] (upsampled 2x)
	std::vector<float> grad_output_data(16, 1.0f);
	auto grad_output = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), OaMatrixShape{1, 1, 4, 4});
	
	auto grad_input = OaFnMatrix::UpsampleBwd(input, grad_output, 2, true);
	[[maybe_unused]] auto exec_result = ctx.Execute();
	[[maybe_unused]] auto sync_result = ctx.Sync();
	
	EXPECT_EQ(grad_input.Shape_, (OaMatrixShape{1, 1, 2, 2}));
	
	std::vector<float> result(4);
	[[maybe_unused]] auto copy_result_30 = OaFnMatrix::CopyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 4; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}
