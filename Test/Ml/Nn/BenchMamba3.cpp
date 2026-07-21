// Mamba-3 GPU microbenchmark.
//
// Default run: the exact NLP-suite corner (B=64, S=16, D=32).
// Full shape catalog (isolated process required): OA_BENCH_MAMBA_FULL=1
// One shape: OA_BENCH_MAMBA_FULL=1 OA_BENCH_MAMBA_SHAPE=p64_d512_s1024 BenchMamba3
// Machine-readable rows: OA_BENCH_MAMBA_TSV=1 BenchMamba3 | grep '^OAMAMBA'
// Stable Xe protocol: three fresh processes per shape, PREHEAT=3, WARMUP=1, ITERS=5.

#include "../../OaTest.h"

#include <Oa/Core/PerfStat.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

constexpr OaI32 kDState = 32;
constexpr OaI32 kExpand = 2;
constexpr OaI32 kRopeAngles = 8;

struct Shape {
	const char* Tag;
	OaI32 Batch;
	OaI32 Seq;
	OaI32 DModel;
	OaI32 HeadDim;
};

struct Inputs {
	OaFnMatrix::OaSsmConfig ScanConfig;
	OaFnMatrix::OaMamba3PreprocessConfig PreprocessConfig;
	OaMatrix C, B, X, Z, Adt, Dt, Trap, Angle, CBias, BBias, D, DOut, DAngleHead;
	OaMatrix Projected, DtBias;
	OaMatrix SsmState, AngleState, KState, VState;
};

struct Sample {
	OaF64 Mean;
	OaF64 P50;
	OaF64 P95;
	OaF64 Min;
	OaF64 Max;
	std::vector<OaF64> Values;
};

static int EnvInt(const char* InName, int InDefault) {
	const char* text = std::getenv(InName);
	if (text == nullptr || *text == '\0') return InDefault;
	char* end = nullptr;
	const long value = std::strtol(text, &end, 10);
	if (end == text || *end != '\0' || value < 1 || value > 10000) return InDefault;
	return static_cast<int>(value);
}

static bool EnvFlag(const char* InName) {
	const char* text = std::getenv(InName);
	return text != nullptr && text[0] == '1' && text[1] == '\0';
}

static const char* EnvText(const char* InName) {
	const char* text = std::getenv(InName);
	return (text == nullptr || *text == '\0') ? nullptr : text;
}

static Inputs MakeInputs(const Shape& InShape) {
	const OaI32 dInner = kExpand * InShape.DModel;
	const OaI32 nHeads = dInner / InShape.HeadDim;
	const OaI32 rows = InShape.Batch * InShape.Seq;
	const OaI32 dInProj = 2 * dInner + 2 * kDState + kRopeAngles + 3 * nHeads;

	Inputs in;
	in.ScanConfig = {
		.Batch = static_cast<OaU32>(InShape.Batch),
		.SeqLen = static_cast<OaU32>(InShape.Seq),
		.NHeads = static_cast<OaU32>(nHeads),
		.HeadDim = static_cast<OaU32>(InShape.HeadDim),
		.StateSize = static_cast<OaU32>(kDState),
		.NumRopeAngles = static_cast<OaU32>(kRopeAngles),
		.HasZ = 1,
		.HasD = 1,
	};
	in.PreprocessConfig = {
		.DInner = dInner,
		.DState = kDState,
		.NHeads = nHeads,
		.NumRopeAngles = kRopeAngles,
		.NGroups = 1,
		.MimoRank = 1,
		.Eps = 1.0e-5F,
		.DtMin = 0.001F,
		.DtMax = 0.1F,
		.AFloor = 1.0e-4F,
	};

	in.C = OaFnMatrix::Ones(OaMatrixShape{InShape.Batch, InShape.Seq, nHeads, kDState}, OaScalarType::Float32);
	in.B = OaFnMatrix::Ones(in.C.GetShape(), OaScalarType::Float32);
	in.X = OaFnMatrix::Ones(OaMatrixShape{InShape.Batch, InShape.Seq, nHeads, InShape.HeadDim}, OaScalarType::Float32);
	in.Z = OaFnMatrix::Ones(in.X.GetShape(), OaScalarType::Float32);
	in.Adt = OaFnMatrix::Ones(OaMatrixShape{InShape.Batch, InShape.Seq, nHeads}, OaScalarType::Float32) * -0.01F;
	in.Dt = OaFnMatrix::Ones(in.Adt.GetShape(), OaScalarType::Float32) * 0.01F;
	in.Trap = OaFnMatrix::Zeros(in.Adt.GetShape(), OaScalarType::Float32);
	in.Angle = OaFnMatrix::Ones(OaMatrixShape{InShape.Batch, InShape.Seq, kRopeAngles}, OaScalarType::Float32) * 0.01F;
	in.CBias = OaFnMatrix::Zeros(OaMatrixShape{nHeads, kDState}, OaScalarType::Float32);
	in.BBias = OaFnMatrix::Zeros(in.CBias.GetShape(), OaScalarType::Float32);
	in.D = OaFnMatrix::Zeros(OaMatrixShape{nHeads}, OaScalarType::Float32);
	in.DOut = OaFnMatrix::Ones(in.X.GetShape(), OaScalarType::Float32);
	in.DAngleHead = OaFnMatrix::Ones(
		OaMatrixShape{InShape.Batch, nHeads, InShape.Seq, kRopeAngles}, OaScalarType::Float32);
	in.Projected = OaFnMatrix::Ones(OaMatrixShape{rows, dInProj}, OaScalarType::Float32);
	in.DtBias = OaFnMatrix::Zeros(OaMatrixShape{1, nHeads}, OaScalarType::Float32);
	in.SsmState = OaFnMatrix::Zeros(OaMatrixShape{InShape.Batch, nHeads, InShape.HeadDim, kDState}, OaScalarType::Float32);
	in.AngleState = OaFnMatrix::Zeros(OaMatrixShape{InShape.Batch, nHeads, kRopeAngles}, OaScalarType::Float32);
	in.KState = OaFnMatrix::Zeros(OaMatrixShape{InShape.Batch, nHeads, kDState}, OaScalarType::Float32);
	in.VState = OaFnMatrix::Zeros(OaMatrixShape{InShape.Batch, nHeads, InShape.HeadDim}, OaScalarType::Float32);
	return in;
}

template <typename Enqueue>
static Sample Measure(
	OaEngine& InEngine,
	const char* InName,
	int InWarmup,
	int InIterations,
	Enqueue&& InEnqueue)
{
	OaGpuTimer timer;
	if (auto status = timer.Init(InEngine, InName); not status.IsOk()) {
		throw std::runtime_error("BenchMamba3: GPU timer initialization failed");
	}
	OaPerfStat stat(InName, static_cast<OaU32>(InIterations), static_cast<OaU32>(InWarmup));
	std::vector<OaF64> values;
	values.reserve(static_cast<size_t>(InIterations));
	auto& context = OaContext::GetDefault();
	for (int iteration = 0; iteration < InWarmup + InIterations; ++iteration) {
		InEnqueue();
		auto submitted = context.Submit(&timer);
		if (not submitted.IsOk()) {
			timer.Destroy(InEngine.Device);
			throw std::runtime_error("BenchMamba3: GPU execution failed");
		}
		if (auto status = context.Wait(submitted.GetValue()); not status.IsOk()) {
			timer.Destroy(InEngine.Device);
			throw std::runtime_error("BenchMamba3: GPU completion failed");
		}
		const OaF64 elapsed = timer.ReadbackMs(InEngine.Device);
		if (!(elapsed > 0.0)) {
			timer.Destroy(InEngine.Device);
			throw std::runtime_error("BenchMamba3: non-positive GPU timestamp");
		}
		stat.Push(elapsed);
		if (iteration >= InWarmup) values.push_back(elapsed);
	}
	timer.Destroy(InEngine.Device);
	if (not stat.IsReady()) throw std::runtime_error("BenchMamba3: empty statistics window");
	return {stat.Mean(), stat.P50(), stat.P95(), stat.Min(), stat.Max(), std::move(values)};
}

static void PrintSample(const Shape& InShape, const char* InOperation, const Sample& InSample) {
	const OaI32 heads = (kExpand * InShape.DModel) / InShape.HeadDim;
	std::printf("  %-14s B=%-3d S=%-4d D=%-4d H=%-3d P=%-3d mean=%8.4f ms  p50=%8.4f  p95=%8.4f\n",
		InOperation, InShape.Batch, InShape.Seq, InShape.DModel, heads, InShape.HeadDim,
		InSample.Mean, InSample.P50, InSample.P95);
	if (EnvFlag("OA_BENCH_MAMBA_TSV")) {
		std::printf("OAMAMBA\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
			InShape.Tag, InOperation, InShape.Batch, InShape.Seq, InShape.DModel, heads, InShape.HeadDim,
			InSample.Mean, InSample.P50, InSample.P95, InSample.Min, InSample.Max);
		for (size_t sample = 0; sample < InSample.Values.size(); ++sample) {
			std::printf("OAMAMBASAMPLE\t%s\t%s\t%zu\t%.6f\n",
				InShape.Tag, InOperation, sample, InSample.Values[sample]);
		}
	}
}

static void PreheatGpu(OaEngine&, int InIterations) {
	OaGradNo noGrad;
	auto inputs = MakeInputs({"preheat", 64, 16, 32, 16});
	auto& context = OaContext::GetDefault();
	if (auto status = context.Execute(); not status.IsOk()) throw std::runtime_error("BenchMamba3: preheat initialization failed");
	if (auto status = context.Sync(); not status.IsOk()) throw std::runtime_error("BenchMamba3: preheat initialization sync failed");
	OaVec<OaMatrix> keepAlive;
	for (int iteration = 0; iteration < InIterations; ++iteration) {
		auto out = OaFnMatrix::Mamba3SisoBwd(
			inputs.DOut, inputs.C, inputs.B, inputs.X, inputs.Z, inputs.Adt, inputs.Dt,
			inputs.Trap, inputs.Angle, inputs.CBias, inputs.BBias, inputs.D, inputs.ScanConfig);
		keepAlive = {out.DC, out.DB, out.DX, out.DZ, out.DAdt, out.DDt, out.DTrap,
			out.DAngle, out.DCBias, out.DBBias, out.DD};
		if (auto status = context.Execute(); not status.IsOk()) throw std::runtime_error("BenchMamba3: GPU preheat failed");
		if (auto status = context.Sync(); not status.IsOk()) throw std::runtime_error("BenchMamba3: GPU preheat sync failed");
	}
}

static void BenchmarkShape(OaEngine& InEngine, const Shape& InShape, int InWarmup, int InIterations) {
	OaGradNo noGrad;
	auto inputs = MakeInputs(InShape);
	auto& context = OaContext::GetDefault();
	if (auto status = context.Execute(); not status.IsOk()) throw std::runtime_error("BenchMamba3: input initialization failed");
	if (auto status = context.Sync(); not status.IsOk()) throw std::runtime_error("BenchMamba3: input initialization sync failed");

	OaVec<OaMatrix> keepAlive;
	auto pre = Measure(InEngine, "mamba_preprocess", InWarmup, InIterations, [&] {
		auto out = OaFnMatrix::Mamba3Preprocess(inputs.Projected, inputs.DtBias, inputs.PreprocessConfig);
		keepAlive = {out.X, out.Z, out.Bh, out.Ch, out.DT, out.ADT, out.Trap, out.Angle};
	});
	PrintSample(InShape, "preprocess", pre);

	auto fwd = Measure(InEngine, "mamba_siso_fwd", InWarmup, InIterations, [&] {
		keepAlive = {OaFnMatrix::Mamba3Siso(
			inputs.C, inputs.B, inputs.X, inputs.Z, inputs.Adt, inputs.Dt, inputs.Trap,
			inputs.Angle, inputs.CBias, inputs.BBias, inputs.D, inputs.ScanConfig)};
	});
	PrintSample(InShape, "siso_fwd", fwd);

	auto bwd = Measure(InEngine, "mamba_siso_bwd", InWarmup, InIterations, [&] {
		auto out = OaFnMatrix::Mamba3SisoBwd(
			inputs.DOut, inputs.C, inputs.B, inputs.X, inputs.Z, inputs.Adt, inputs.Dt,
			inputs.Trap, inputs.Angle, inputs.CBias, inputs.BBias, inputs.D, inputs.ScanConfig);
		keepAlive = {out.DC, out.DB, out.DX, out.DZ, out.DAdt, out.DDt, out.DTrap,
			out.DAngle, out.DCBias, out.DBBias, out.DD};
	});
	PrintSample(InShape, "siso_bwd", bwd);

	// The public backward result applies one sigmoid derivative and four bias/
	// head reductions after the core shader. Time the same postlude separately
	// so the monolithic recurrence is not blamed for unrelated reduction cost.
	auto bwdPost = Measure(InEngine, "mamba_bwd_post", InWarmup, InIterations, [&] {
		auto trapSigmoid = OaFnMatrix::Sigmoid(inputs.Trap);
		auto dTrap = OaFnMatrix::SigmoidBwd(trapSigmoid, inputs.Dt);
		auto dAngle = OaFnMatrix::Sum(inputs.DAngleHead, 1)
			.Reshape(OaMatrixShape{InShape.Batch, InShape.Seq, kRopeAngles});
		auto dCBias = OaFnMatrix::Sum(
			inputs.C.Reshape(OaMatrixShape{InShape.Batch * InShape.Seq,
				inputs.ScanConfig.NHeads * kDState}), 0);
		auto dBBias = OaFnMatrix::Sum(
			inputs.B.Reshape(OaMatrixShape{InShape.Batch * InShape.Seq,
				inputs.ScanConfig.NHeads * kDState}), 0);
		auto dD = OaFnMatrix::Sum(
			inputs.Dt.Reshape(OaMatrixShape{InShape.Batch * InShape.Seq,
				inputs.ScanConfig.NHeads}), 0);
		keepAlive = {dTrap, dAngle, dCBias, dBBias, dD};
	});
	PrintSample(InShape, "bwd_post", bwdPost);

	OaMamba3Module module(InShape.DModel, kDState, kExpand, InShape.HeadDim, 1, 0.5F, false, 4,
		0.001F, 0.1F, 1.0e-4F, 1.0e-4F, true);
	auto blockInput = OaFnMatrix::Ones(
		OaMatrixShape{InShape.Batch, InShape.Seq, InShape.DModel}, OaScalarType::Float32);
	if (auto status = context.Execute(); not status.IsOk()) throw std::runtime_error("BenchMamba3: module initialization failed");
	if (auto status = context.Sync(); not status.IsOk()) throw std::runtime_error("BenchMamba3: module initialization sync failed");
	auto prefill = Measure(InEngine, "mamba_prefill", InWarmup, InIterations, [&] {
		keepAlive = {module.Forward(blockInput)};
	});
	PrintSample(InShape, "prefill", prefill);

	if (InShape.Seq == 16) {
		Shape stepShape{InShape.Tag, InShape.Batch, 1, InShape.DModel, InShape.HeadDim};
		auto stepInputs = MakeInputs(stepShape);
		if (auto status = context.Execute(); not status.IsOk()) throw std::runtime_error("BenchMamba3: step input initialization failed");
		if (auto status = context.Sync(); not status.IsOk()) throw std::runtime_error("BenchMamba3: step input initialization sync failed");
		auto step = Measure(InEngine, "mamba_siso_step", InWarmup, InIterations, [&] {
			keepAlive = {OaFnMatrix::Mamba3SisoStep(
				stepInputs.C, stepInputs.B, stepInputs.X, stepInputs.Z, stepInputs.Adt,
				stepInputs.Dt, stepInputs.Trap, stepInputs.Angle, stepInputs.CBias,
				stepInputs.BBias, stepInputs.D, stepInputs.SsmState, stepInputs.AngleState,
				stepInputs.KState, stepInputs.VState, stepInputs.ScanConfig)};
		});
		PrintSample(stepShape, "siso_step", step);
	}
}

} // namespace

TEST(BenchMamba3, KernelBreakdown) {
	ASSERT_TRUE(OaVkTestEngineOk()) << "OaVkTestEnvironment did not create OaEngine";
	auto& engine = *OaEngine::GetGlobal();
	const int warmup = EnvInt("OA_BENCH_MAMBA_WARMUP", 3);
	const int iterations = EnvInt("OA_BENCH_MAMBA_ITERS", 10);
	const int preheat = EnvInt("OA_BENCH_MAMBA_PREHEAT", 3);
	const char* shapeFilter = EnvText("OA_BENCH_MAMBA_SHAPE");
	const bool full = EnvFlag("OA_BENCH_MAMBA_FULL");
	ASSERT_FALSE(full && shapeFilter == nullptr)
		<< "full sweep must run one OA_BENCH_MAMBA_SHAPE per fresh process";

	std::printf("\nMamba-3 GPU kernel breakdown (FP32)\n");
	std::printf("  preheat=%d suite-backward iterations; warmup=%d iterations=%d\n", preheat, warmup, iterations);
	std::printf("  timestamps bracket queued GPU work only\n");
	std::printf("  default is the NLP-suite corner; OA_BENCH_MAMBA_FULL=1 enables B=1 S x D x P sweep\n\n");
	ASSERT_NO_THROW(PreheatGpu(engine, preheat));

	if (shapeFilter == nullptr || std::strcmp(shapeFilter, "suite") == 0) {
		ASSERT_NO_THROW(BenchmarkShape(engine, {"suite", 64, 16, 32, 16}, warmup, iterations));
	}
	bool matched = shapeFilter == nullptr || std::strcmp(shapeFilter, "suite") == 0;
	if (full) {
		static constexpr OaI32 kSequences[] = {16, 64, 256, 1024};
		static constexpr OaI32 kWidths[] = {32, 128, 512};
		static constexpr OaI32 kHeadDims[] = {16, 32, 64};
		for (const OaI32 headDim : kHeadDims) {
			for (const OaI32 width : kWidths) {
				for (const OaI32 sequence : kSequences) {
					char tag[40];
					std::snprintf(tag, sizeof(tag), "p%d_d%d_s%d", headDim, width, sequence);
					if (shapeFilter != nullptr && std::strcmp(shapeFilter, tag) != 0) continue;
					matched = true;
					ASSERT_NO_THROW(BenchmarkShape(engine, {tag, 1, sequence, width, headDim}, warmup, iterations));
				}
			}
		}
	}
	ASSERT_TRUE(matched) << "OA_BENCH_MAMBA_SHAPE did not match suite or a full-sweep shape";
}
