// Controlled raw-GEMM variant benchmark.
//
// This is deliberately not registered with ctest. It uses GPU timestamps on
// the production OaFnMatrix::MatMulNt path and exists for kernel-selection
// experiments, not pass/fail CI. Run multiple fresh processes before changing
// routing policy; integrated-GPU clocks and thermals can move during a sweep.

#include "../../OaTest.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Gemm/Router.h>
#include <Oa/Runtime/Gemm/Tuner.h>

#include <cstdlib>

namespace {

OaU32 EnvIterations(const char* InName, OaU32 InDefault) {
	const char* text = std::getenv(InName);
	if (text == nullptr or *text == '\0') return InDefault;
	char* end = nullptr;
	const unsigned long value = std::strtoul(text, &end, 10);
	if (end == text or *end != '\0' or value == 0UL or value > 1000UL) return InDefault;
	return static_cast<OaU32>(value);
}

} // namespace

TEST(BenchGemmVariants, ProductionShapes) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& engine = *OaComputeEngine::GetGlobal();
	const OaU32 warmup = EnvIterations("OA_BENCH_GEMM_WARMUP", 5U);
	const OaU32 iterations = EnvIterations("OA_BENCH_GEMM_ITERS", 30U);

	const OaGemmTunerShape shapes[] = {
		{1024U, 32U, 32U, "nlp_qkv"},
		{1024U, 64U, 32U, "nlp_ffn1"},
		{1024U, 64U, 32U, "nlp_ffn1_bias_gelu", OaGemmEpilogue::BiasGelu},
		{4096U, 384U, 384U, "alm_qkv"},
		{4096U, 384U, 384U, "alm_qkv_bias", OaGemmEpilogue::Bias},
		{4096U, 1536U, 384U, "alm_ffn1"},
		{4096U, 1536U, 384U, "alm_ffn1_bias_gelu", OaGemmEpilogue::BiasGelu},
		{4096U, 384U, 1536U, "alm_ffn2"},
	};

	for (const auto& shape : shapes) {
		OaGemmTunerResult result{};
		const OaStatus status = OaGemmTuner::BenchmarkShape(
			engine, shape, warmup, iterations, result);
		ASSERT_TRUE(status.IsOk()) << status.GetMessage().Data();
		auto problem = OaGemmRouter::ProblemForRaw(
			shape.M, shape.N, shape.K,
			OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
		problem.Training = true;
		problem.PrecisionHint = OaGemmPrecision::Auto;
		problem.Epilogue = shape.Epilogue;
		EXPECT_EQ(OaGemmRouter::Select(engine, problem).Variant, result.BestVariant)
			<< "measured winner was not replayed for " << shape.Name;
	}
}
