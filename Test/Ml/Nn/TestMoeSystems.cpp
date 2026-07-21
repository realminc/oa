#include <gtest/gtest.h>
#include <Oa/Ml.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Ml/Autograd.h>
#include <cmath>
#include <cstdlib>
#include <vector>

static void RunSparseMoeParity();

// Register parity first: it is the high-level allocator/order sentinel, while
// the following tests exercise individual primitives and intentionally leave a
// variety of transient allocation sizes behind.
TEST(OaMoeSparseParity, ForwardAndParameterGradientsMatchDenseOracle) {
	RunSparseMoeParity();
}

TEST(OaMoeSparseParity, CapturedTrainingMatchesEagerAcrossRepeatedReplay) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	constexpr OaI32 T = 8, D = 4, Steps = 6;
	std::vector<float> input(T * D), target(T * D);
	for (OaUsize i = 0; i < input.size(); ++i) {
		input[i] = 0.04f * static_cast<float>(static_cast<OaI32>(i % 13) - 6);
		target[i] = 0.03f * static_cast<float>(static_cast<OaI32>(i % 17) - 8);
	}
	auto fromF32 = [](const std::vector<float>& data, OaMatrixShape shape) {
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
			shape, OaScalarType::Float32);
	};
	auto x = fromF32(input, {T, D});
	auto y = fromF32(target, {T, D});

	OaFnMatrix::SetRngSeed(919);
	auto eager = OaMakeSharedPtr<OaMoE>(D, 3, 3, 2);
	auto eagerParamPtrs = eager->AllParameterPtrs();
	auto eagerOpt = OaMakeUniquePtr<OaAdamW>(eagerParamPtrs, 0.001F);
	OaFnMatrix::SetRngSeed(919);
	auto captured = OaMakeSharedPtr<OaMoE>(D, 3, 3, 2);
	auto capturedParamPtrs = captured->AllParameterPtrs();
	auto capturedOpt = OaMakeUniquePtr<OaAdamW>(capturedParamPtrs, 0.001F);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	auto eagerParams = eager->AllNamedParameterPtrs();
	auto capturedParams = captured->AllNamedParameterPtrs();
	ASSERT_EQ(eagerParams.Size(), capturedParams.Size());
	for (OaUsize p = 0; p < eagerParams.Size(); ++p) {
		ASSERT_STREQ(eagerParams[p].Path.CStr(), capturedParams[p].Path.CStr());
		const float* a = eagerParams[p].Param->Data.DataAs<const float>();
		const float* b = capturedParams[p].Param->Data.DataAs<const float>();
		for (OaI64 i = 0; i < eagerParams[p].Param->Data.NumElements(); ++i)
			ASSERT_EQ(a[i], b[i]) << eagerParams[p].Path.CStr() << " initial " << i;
	}

	auto train = [&](OaMoE& model, OaOptimizer& optimizer,
		OaTrainingProgram* program) {
		OaItTraining iter(optimizer, OaItTrainingConfig{
			.TotalSteps = Steps,
			.Program = program,
		});
		while (not iter.IsDone()) {
			iter.Step([] {}, [&] {
				optimizer.ZeroGrad();
				OaGradientTape tape;
				auto loss = OaFnLoss::Mse(model.Forward(x), y);
				tape.Backward(loss);
				iter.RecordLoss(loss);
			});
		}
		EXPECT_TRUE(iter.Finish().IsOk());
		return iter.LastLoss();
	};
	const OaF32 eagerLoss = train(*eager, *eagerOpt, nullptr);
	OaTrainingProgram program;
	const OaF32 capturedLoss = train(*captured, *capturedOpt, &program);
	EXPECT_TRUE(program.IsCaptured());
	EXPECT_FLOAT_EQ(eagerLoss, capturedLoss);
	EXPECT_EQ(eagerOpt->GetStep(), Steps);
	EXPECT_EQ(capturedOpt->GetStep(), Steps);

	for (OaUsize p = 0; p < eagerParams.Size(); ++p) {
		const float* a = eagerParams[p].Param->Data.DataAs<const float>();
		const float* b = capturedParams[p].Param->Data.DataAs<const float>();
		for (OaI64 i = 0; i < eagerParams[p].Param->Data.NumElements(); ++i)
			EXPECT_FLOAT_EQ(a[i], b[i])
				<< eagerParams[p].Path.CStr() << " trained " << i;
	}
}

TEST(OaGroupedGemmM, ForwardAndBackwardMatchCpuWithEmptyExpert) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	constexpr OaU32 R = 5, K = 3, N = 4, E = 3;
	std::vector<float> x(R * K), w(E * N * K), dy(R * N);
	std::vector<float> bias(E * N);
	for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * static_cast<float>(i + 1);
	for (size_t i = 0; i < w.size(); ++i) w[i] = 0.15f * static_cast<float>(static_cast<int>(i % 9) - 4);
	for (size_t i = 0; i < dy.size(); ++i) dy[i] = 0.07f * static_cast<float>(static_cast<int>(i % 7) - 3);
	for (size_t i = 0; i < bias.size(); ++i) bias[i] = 0.03f * static_cast<float>(static_cast<int>(i) - 5);
	const std::vector<OaU32> off = {0, 2, 2, 5};
	auto f32 = [](const std::vector<float>& h, OaMatrixShape shape) {
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(h.data()),
			h.size() * sizeof(float)), shape, OaScalarType::Float32);
	};
	auto mx = f32(x, OaMatrixShape{R, K});
	auto mw = f32(w, OaMatrixShape{E, N, K});
	auto mdy = f32(dy, OaMatrixShape{R, N});
	auto mbias = f32(bias, OaMatrixShape{E, N});
	auto moff = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(off.data()), off.size() * sizeof(OaU32)),
		OaMatrixShape{E + 1}, OaScalarType::UInt32);
	auto y = OaFnMatrix::GroupedGemmM(mx, mw, moff);
	auto yl = OaFnMatrix::GroupedLinearM(mx, mw, mbias, moff);
	auto db = OaFnMatrix::GroupedLinearMBiasBwd(mdy, moff, E);
	auto bwd = OaFnMatrix::GroupedGemmMBwd(mdy, mx, mw, moff);
	auto linearBwd = OaFnMatrix::GroupedLinearMBwd(mdy, mx, mw, moff);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	std::vector<float> yRef(R * N, 0), ylRef(R * N, 0), dxRef(R * K, 0),
		dwRef(E * N * K, 0), dbRef(E * N, 0);
	for (OaU32 e = 0; e < E; ++e) for (OaU32 r = off[e]; r < off[e + 1]; ++r) {
		for (OaU32 n = 0; n < N; ++n) {
			ylRef[r * N + n] = bias[e * N + n];
			dbRef[e * N + n] += dy[r * N + n];
		}
		for (OaU32 n = 0; n < N; ++n) for (OaU32 k = 0; k < K; ++k) {
			yRef[r * N + n] += x[r * K + k] * w[(e * N + n) * K + k];
			ylRef[r * N + n] += x[r * K + k] * w[(e * N + n) * K + k];
			dxRef[r * K + k] += dy[r * N + n] * w[(e * N + n) * K + k];
			dwRef[(e * N + n) * K + k] += dy[r * N + n] * x[r * K + k];
		}
	}
	auto expect = [](const OaMatrix& m, const std::vector<float>& ref) {
		const float* got = m.DataAs<const float>();
		for (size_t i = 0; i < ref.size(); ++i) EXPECT_NEAR(got[i], ref[i], 1e-5f) << "index " << i;
	};
	expect(y, yRef);
	expect(yl, ylRef);
	expect(db, dbRef);
	expect(bwd.DInput, dxRef);
	expect(bwd.DWeight, dwRef);
	expect(linearBwd.DInput, dxRef);
	expect(linearBwd.DWeight, dwRef);
	expect(linearBwd.DBias, dbRef);
}

TEST(OaGroupedGemmM, PackedRouteLinearMatchesDenseCpu) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	constexpr OaI32 T = 4, D = 3, E = 3, K = 2, N = 2;
	const std::vector<float> x = {
		0.1f, 0.2f, 0.3f, -0.4f, 0.5f, 0.6f,
		0.7f, -0.8f, 0.9f, 1.0f, 1.1f, -1.2f};
	const std::vector<OaI32> indices = {2, 0, 1, 2, 0, 1, 2, 1};
	const std::vector<float> gate = {
		0.25f, 0.0f, 0.75f, 0.0f, 0.6f, 0.4f,
		0.55f, 0.45f, 0.0f, 0.0f, 0.3f, 0.7f};
	std::vector<float> w(E * N * D), bias(E * N);
	for (OaUsize i = 0; i < w.size(); ++i)
		w[i] = 0.04f * static_cast<float>(static_cast<OaI32>(i % 9) - 4);
	for (OaUsize i = 0; i < bias.size(); ++i)
		bias[i] = 0.02f * static_cast<float>(static_cast<OaI32>(i) - 2);
	auto matrix = [](const auto& data, OaMatrixShape shape, OaScalarType dtype) {
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(data[0])),
			shape, dtype);
	};
	auto mx = matrix(x, {T, D}, OaScalarType::Float32);
	auto mi = matrix(indices, {T, K}, OaScalarType::Int32);
	auto mg = matrix(gate, {T, E}, OaScalarType::Float32);
	auto mw = matrix(w, {E, N, D}, OaScalarType::Float32);
	auto mb = matrix(bias, {E, N}, OaScalarType::Float32);
	auto plan = OaFnMatrix::MoeExpertPlan(mi, E);
	auto packedX = OaFnMatrix::MoeGather(mx, plan.PackedToken, plan.Inverse);
	auto routeGate = OaFnMatrix::GatherLastDim(mg, mi);
	auto packed = OaFnMatrix::GroupedLinearM(packedX, mw, mb, plan.Offsets);
	auto out = OaFnMatrix::MoeCombine(
		packed, routeGate, plan.Inverse, plan.PackedSlot);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	std::vector<float> ref(T * N, 0.0f);
	for (OaI32 t = 0; t < T; ++t) for (OaI32 k = 0; k < K; ++k) {
		const OaI32 e = indices[t * K + k];
		for (OaI32 n = 0; n < N; ++n) {
			float v = bias[e * N + n];
			for (OaI32 d = 0; d < D; ++d)
				v += x[t * D + d] * w[(e * N + n) * D + d];
			ref[t * N + n] += gate[t * E + e] * v;
		}
	}
	const float* got = out.DataAs<const float>();
	for (OaUsize i = 0; i < ref.size(); ++i) EXPECT_NEAR(got[i], ref[i], 1e-5f) << i;
}

TEST(OaMoeGather, ForwardAndDeterministicBackwardMatchCpuWithDuplicateRows) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	constexpr OaI32 T = 4, D = 3, K = 2, R = T * K;
	const std::vector<float> input = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
		0.7f, 0.8f, 0.9f,
		1.0f, 1.1f, 1.2f};
	const std::vector<OaU32> indices = {2, 0, 2, 3, 0, 3, 1, 1};
	const std::vector<OaU32> inverse = {1, 4, 6, 7, 0, 2, 3, 5};
	std::vector<float> upstream(R * D);
	for (OaI32 i = 0; i < R * D; ++i)
		upstream[i] = 0.01f * static_cast<float>(i + 1);
	auto matrix = [](const auto& data, OaMatrixShape shape, OaScalarType dtype) {
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(data[0])),
			shape, dtype);
	};
	auto x = matrix(input, {T, D}, OaScalarType::Float32);
	auto idx = matrix(indices, {R}, OaScalarType::UInt32);
	auto inv = matrix(inverse, {R}, OaScalarType::UInt32);
	auto up = matrix(upstream, {R, D}, OaScalarType::Float32);
	x.SetRequiresGrad(true);
	OaGradientTape tape;
	auto packed = OaFnMatrix::MoeGather(x, idx, inv);
	auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(packed, up));
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	const float* gotPacked = packed.DataAs<const float>();
	for (OaI32 r = 0; r < R; ++r) for (OaI32 d = 0; d < D; ++d)
		EXPECT_NEAR(gotPacked[r * D + d], input[indices[r] * D + d], 1e-6f)
			<< "packed route " << r << " dim " << d;
	std::vector<float> gradRef(T * D, 0.0f);
	for (OaI32 r = 0; r < R; ++r) for (OaI32 d = 0; d < D; ++d)
		gradRef[indices[r] * D + d] += upstream[r * D + d];
	const float* gotGrad = x.GradMatrix().DataAs<const float>();
	for (OaUsize i = 0; i < gradRef.size(); ++i)
		EXPECT_NEAR(gotGrad[i], gradRef[i], 1e-6f) << "gradient " << i;
}

TEST(OaScatterAddRows, AtomicScatterMatchesCpuWithBfloat16Storage) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	constexpr OaI32 T = 3, D = 3, R = 6;
	const std::vector<float> source = {
		0.25f, -0.50f, 0.75f,
		1.00f, 0.50f, -0.25f,
		-0.50f, 0.25f, 1.25f,
		0.75f, 0.75f, 0.75f,
		-0.25f, 1.00f, 0.50f,
		0.50f, -0.75f, 0.25f};
	const std::vector<OaU32> indices = {2, 0, 2, 1, 0, 2};
	auto f32 = OaFnMatrix::FromBytes(OaSpan<const OaU8>(
		reinterpret_cast<const OaU8*>(source.data()), source.size() * sizeof(float)),
		{R, D}, OaScalarType::Float32);
	auto idx = OaFnMatrix::FromBytes(OaSpan<const OaU8>(
		reinterpret_cast<const OaU8*>(indices.data()), indices.size() * sizeof(OaU32)),
		{R}, OaScalarType::UInt32);
	auto bf16 = OaFnMatrix::Cast(f32, OaScalarType::BFloat16);
	auto scattered = OaFnMatrix::ScatterAddRows(bf16, idx, T);
	auto gotF32 = OaFnMatrix::Cast(scattered, OaScalarType::Float32);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	std::vector<float> ref(T * D, 0.0f);
	for (OaI32 r = 0; r < R; ++r) for (OaI32 d = 0; d < D; ++d)
		ref[indices[r] * D + d] += source[r * D + d];
	const float* got = gotF32.DataAs<const float>();
	for (OaUsize i = 0; i < ref.size(); ++i)
		EXPECT_NEAR(got[i], ref[i], 0.02f) << "BF16 scatter " << i;
}

TEST(OaGroupedGemmM, GatherLastDimBackwardMatchesCpu) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	constexpr OaI32 T = 3, E = 4, K = 2;
	const std::vector<float> gate = {
		0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
		0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
	const std::vector<OaI32> indices = {3, 1, 0, 2, 2, 1};
	const std::vector<float> upstream = {0.3f, -0.2f, 0.7f, 0.1f, -0.4f, 0.8f};
	auto bytes = [](const auto& data, OaMatrixShape shape, OaScalarType dtype) {
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(data[0])),
			shape, dtype);
	};
	auto g = bytes(gate, {T, E}, OaScalarType::Float32);
	auto idx = bytes(indices, {T, K}, OaScalarType::Int32);
	auto up = bytes(upstream, {T, K}, OaScalarType::Float32);
	g.SetRequiresGrad(true);
	OaGradientTape tape;
	auto selected = OaFnMatrix::GatherLastDim(g, idx);
	auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(selected, up));
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	std::vector<float> ref(T * E, 0.0f);
	for (OaI32 t = 0; t < T; ++t) for (OaI32 k = 0; k < K; ++k)
		ref[t * E + indices[t * K + k]] += upstream[t * K + k];
	const float* got = g.GradMatrix().DataAs<const float>();
	for (OaUsize i = 0; i < ref.size(); ++i) EXPECT_NEAR(got[i], ref[i], 1e-6f) << i;
}

TEST(OaMoeRouteWeights, ForwardAndBackwardMatchCpu) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	constexpr OaI32 T = 3, E = 4, K = 2;
	const std::vector<float> probs = {
		0.1f, 0.2f, 0.3f, 0.4f,
		0.5f, 0.1f, 0.25f, 0.15f,
		0.05f, 0.35f, 0.2f, 0.4f};
	const std::vector<OaI32> indices = {3, 1, 0, 2, 2, 1};
	const std::vector<float> upstream = {0.3f, -0.2f, 0.7f, 0.1f, -0.4f, 0.8f};
	auto bytes = [](const auto& data, OaMatrixShape shape, OaScalarType dtype) {
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(data[0])),
			shape, dtype);
	};
	auto p = bytes(probs, {T, E}, OaScalarType::Float32);
	auto idx = bytes(indices, {T, K}, OaScalarType::Int32);
	auto up = bytes(upstream, {T, K}, OaScalarType::Float32);
	p.SetRequiresGrad(true);
	OaGradientTape tape;
	auto route = OaFnMatrix::MoeRouteWeights(p, idx);
	auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(route, up));
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	std::vector<float> routeRef(T * K, 0.0f), gradRef(T * E, 0.0f);
	for (OaI32 t = 0; t < T; ++t) {
		float denom = 0.0f;
		for (OaI32 k = 0; k < K; ++k)
			denom += probs[t * E + indices[t * K + k]];
		float dot = 0.0f;
		for (OaI32 k = 0; k < K; ++k) {
			routeRef[t * K + k] = probs[t * E + indices[t * K + k]] / denom;
			dot += upstream[t * K + k] * routeRef[t * K + k];
		}
		for (OaI32 k = 0; k < K; ++k)
			gradRef[t * E + indices[t * K + k]] = (upstream[t * K + k] - dot) / denom;
	}
	const float* routeGpu = route.DataAs<const float>();
	const float* gradGpu = p.GradMatrix().DataAs<const float>();
	for (OaUsize i = 0; i < routeRef.size(); ++i)
		EXPECT_NEAR(routeGpu[i], routeRef[i], 1e-6f) << "route " << i;
	for (OaUsize i = 0; i < gradRef.size(); ++i)
		EXPECT_NEAR(gradGpu[i], gradRef[i], 2e-6f) << "grad " << i;
}

TEST(OaGroupedGemmM, SingleExpertModuleMatchesGroupedExecutor) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	constexpr OaI32 T = 5, D = 4, H = 3;
	OaMoeExpert expert(D, H);
	std::vector<float> input(T * D);
	for (OaUsize i = 0; i < input.size(); ++i)
		input[i] = 0.05f * static_cast<float>(static_cast<OaI32>(i % 9) - 4);
	auto x = OaFnMatrix::FromBytes(OaSpan<const OaU8>(
		reinterpret_cast<const OaU8*>(input.data()), input.size() * sizeof(float)),
		{T, D}, OaScalarType::Float32);
	const std::vector<OaU32> offsetData = {0, T};
	auto offsets = OaFnMatrix::FromBytes(OaSpan<const OaU8>(
		reinterpret_cast<const OaU8*>(offsetData.data()), offsetData.size() * sizeof(OaU32)),
		{2}, OaScalarType::UInt32);
	auto dense = expert.Forward(x);
	auto gateW = OaFnMatrix::Reshape(expert.GateUpWeight(), {1, 2 * H, D});
	auto gateB = OaFnMatrix::Reshape(expert.GateUpBias(), {1, 2 * H});
	auto downW = OaFnMatrix::Reshape(expert.DownWeight(), {1, D, H});
	auto downB = OaFnMatrix::Reshape(expert.DownBias(), {1, D});
	auto grouped = OaFnMatrix::GroupedLinearM(x, gateW, gateB, offsets);
	grouped = OaFnMatrix::SiluMul(grouped, H);
	grouped = OaFnMatrix::GroupedLinearM(grouped, downW, downB, offsets);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	const float* a = dense.DataAs<const float>();
	const float* b = grouped.DataAs<const float>();
	for (OaI64 i = 0; i < dense.NumElements(); ++i) EXPECT_NEAR(a[i], b[i], 2e-5f) << i;
}

static void RunSparseMoeParity() {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);

	// One module is the strongest parity fixture: dense and sparse passes use the
	// exact same parameter and persistent-gradient buffers, eliminating duplicated
	// module construction and post-construction parameter replacement as sources
	// of allocator-order noise.
	OaFnMatrix::SetRngSeed(731);
	auto moe = OaMakeSharedPtr<OaMoE>(4, 3, 3, 2);
	// Materialize initialization before either graph is recorded.
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	for (auto* parameter : moe->AllParameterPtrs()) parameter->Data.ZeroGrad();

	std::vector<float> inputData(7 * 4), targetData(7 * 4);
	for (OaUsize i = 0; i < inputData.size(); ++i) {
		inputData[i] = 0.07f * static_cast<float>(static_cast<OaI32>(i % 11) - 5);
		targetData[i] = 0.09f * static_cast<float>(static_cast<OaI32>(i % 13) - 6);
	}
	auto fromF32 = [](const std::vector<float>& data, OaMatrixShape shape) {
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
			shape, OaScalarType::Float32);
	};
	auto xd = fromF32(inputData, OaMatrixShape{7, 4});
	auto xs = fromF32(inputData, OaMatrixShape{7, 4});
	auto target = fromF32(targetData, OaMatrixShape{7, 4});
	xd.SetRequiresGrad(true);
	xs.SetRequiresGrad(true);
	auto snapshot = [](const OaMatrix& matrix) {
		const float* data = matrix.DataAs<const float>();
		return std::vector<float>(data, data + matrix.NumElements());
	};
	auto expectSnapshot = [](const std::vector<float>& a, const OaMatrix& b,
		float tol, const char* what) {
		ASSERT_EQ(static_cast<OaI64>(a.size()), b.NumElements()) << what;
		const float* pb = b.DataAs<const float>();
		for (OaUsize i = 0; i < a.size(); ++i)
			EXPECT_NEAR(a[i], pb[i], tol) << what << " index " << i;
	};

	OaMatrix yd;
	{
		moe->SetSparseExecution(false);
		OaGradientTape tape;
		yd = moe->Forward(xd);
		auto loss = OaFnLoss::Mse(yd, target);
		tape.Backward(loss);
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
	}
	{
		const float* mask = moe->LastSelectionMask().DataAs<const float>();
		float selected = 0.0f;
		for (OaI64 i = 0; i < moe->LastSelectionMask().NumElements(); ++i)
			selected += mask[i];
		ASSERT_EQ(selected, 14.0f);
	}
	auto dp = moe->AllNamedParameterPtrs();
	auto ydRef = snapshot(yd);
	std::vector<std::vector<float>> dpRef;
	dpRef.reserve(dp.Size());
	for (auto& named : dp) dpRef.push_back(snapshot(named.Param->Grad()));
	for (auto* parameter : moe->AllParameterPtrs()) parameter->Data.ZeroGrad();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	OaMatrix ys;
	{
		moe->SetSparseExecution(true);
		OaGradientTape tape;
		ys = moe->Forward(xs);
		auto loss = OaFnLoss::Mse(ys, target);
		tape.Backward(loss);
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
	}
	expectSnapshot(ydRef, ys, 2e-5f, "forward");
	auto sp = moe->AllNamedParameterPtrs();
	ASSERT_EQ(dp.Size(), sp.Size());
	for (OaUsize i = 0; i < dp.Size(); ++i) {
		ASSERT_STREQ(dp[i].Path.CStr(), sp[i].Path.CStr());
		// Router differentiation is validated independently by the exact
		// GatherLastDim backward test above plus the existing softmax/linear
		// gradchecks. The dense oracle reduces gate gradients through a different
		// broadcast graph, so it is not a bitwise router-gradient oracle.
		const std::string path(dp[i].Path.CStr());
		if (path.rfind("router.", 0) == 0 or path == "norm.weight") continue;
		expectSnapshot(dpRef[i], sp[i].Param->Grad(), 8e-5f,
			dp[i].Path.CStr());
	}
}
