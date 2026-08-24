// Mamba-3 GPU microbenchmark.
//
// Default run: the exact NLP-suite corner (B=64, S=16, D=32).
// Full shape catalog (isolated process required): OA_BENCH_MAMBA_FULL=1
// One shape: OA_BENCH_MAMBA_FULL=1 OA_BENCH_MAMBA_SHAPE=p64_d512_s1024 BenchMamba3
// Machine-readable rows: OA_BENCH_MAMBA_TSV=1 BenchMamba3 | grep '^OAMAMBA'
// Stable Xe protocol: three fresh processes per shape, PREHEAT=3, WARMUP=1, ITERS=5.

#include "../../oaTest.h"

#include <oa/core/perfStat.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/ml/nn.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/timer.h>
#include <oa/runtime/executableGraph.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

constexpr oa::I32 kDState = 32;
constexpr oa::I32 kExpand = 2;
constexpr oa::I32 kRopeAngles = 8;

struct Shape {
	const char* tag;
	oa::I32 batch;
	oa::I32 seq;
	oa::I32 dModel;
	oa::I32 headDim;
};

struct Inputs {
	oa::SsmConfig scanConfig;
	oa::Mamba3PreprocessConfig preprocessConfig;
	oa::Matrix c, b, x, z, adt, dt, trap, angle, cBias, bBias, d, dOut, dAngleHead;
	oa::Matrix projected, dtBias;
	oa::Matrix dZ, dX, dBh, dCh, dDT, dADT, dTrap, dAngle;
	oa::Matrix ssmState, angleState, kState, vState;
};

struct Sample {
	oa::F64 mean;
	oa::F64 p50;
	oa::F64 p95;
	oa::F64 min;
	oa::F64 max;
	std::vector<oa::F64> values;
};

struct FeaturePairSample {
	Sample baseline;
	Sample optimized;
	Sample gainPercent;
};

static int envInt(const char* inName, int inDefault) {
	const char* text = std::getenv(inName);
	if (text == nullptr || *text == '\0') return inDefault;
	char* end = nullptr;
	const long value = std::strtol(text, &end, 10);
	if (end == text || *end != '\0' || value < 1 || value > 10000) return inDefault;
	return static_cast<int>(value);
}

static bool envFlag(const char* inName) {
	const char* text = std::getenv(inName);
	return text != nullptr && text[0] == '1' && text[1] == '\0';
}

static const char* envText(const char* inName) {
	const char* text = std::getenv(inName);
	return (text == nullptr || *text == '\0') ? nullptr : text;
}

static void setEnvironmentFlag(const char* inName, bool inEnabled) {
#ifdef _WIN32
	_putenv_s(inName, inEnabled ? "1" : "");
#else
	if (inEnabled) setenv(inName, "1", 1);
	else unsetenv(inName);
#endif
}

static Inputs makeInputs(const Shape& inShape) {
	const oa::I32 dInner = kExpand * inShape.dModel;
	const oa::I32 nHeads = dInner / inShape.headDim;
	const oa::I32 rows = inShape.batch * inShape.seq;
	const oa::I32 dInProj = 2 * dInner + 2 * kDState + kRopeAngles + 3 * nHeads;

	Inputs in;
	in.scanConfig = {
		.batch = static_cast<oa::U32>(inShape.batch),
		.seqLen = static_cast<oa::U32>(inShape.seq),
		.nHeads = static_cast<oa::U32>(nHeads),
		.headDim = static_cast<oa::U32>(inShape.headDim),
		.stateSize = static_cast<oa::U32>(kDState),
		.numRopeAngles = static_cast<oa::U32>(kRopeAngles),
		.hasZ = 1,
		.hasD = 1,
	};
	in.preprocessConfig = {
		.dInner = dInner,
		.dState = kDState,
		.nHeads = nHeads,
		.numRopeAngles = kRopeAngles,
		.nGroups = 1,
		.mimoRank = 1,
		.eps = 1.0e-5F,
		.dtMin = 0.001F,
		.dtMax = 0.1F,
		.aFloor = 1.0e-4F,
	};

	in.c = oa::FnMatrix::ones(oa::MatrixShape{inShape.batch, inShape.seq, nHeads, kDState}, oa::ScalarType::Float32);
	in.b = oa::FnMatrix::ones(in.c.getShape(), oa::ScalarType::Float32);
	in.x = oa::FnMatrix::ones(oa::MatrixShape{inShape.batch, inShape.seq, nHeads, inShape.headDim}, oa::ScalarType::Float32);
	in.z = oa::FnMatrix::ones(in.x.getShape(), oa::ScalarType::Float32);
	in.adt = oa::FnMatrix::ones(oa::MatrixShape{inShape.batch, inShape.seq, nHeads}, oa::ScalarType::Float32) * -0.01F;
	in.dt = oa::FnMatrix::ones(in.adt.getShape(), oa::ScalarType::Float32) * 0.01F;
	in.trap = oa::FnMatrix::zeros(in.adt.getShape(), oa::ScalarType::Float32);
	in.angle = oa::FnMatrix::ones(oa::MatrixShape{inShape.batch, inShape.seq, kRopeAngles}, oa::ScalarType::Float32) * 0.01F;
	in.cBias = oa::FnMatrix::zeros(oa::MatrixShape{nHeads, kDState}, oa::ScalarType::Float32);
	in.bBias = oa::FnMatrix::zeros(in.cBias.getShape(), oa::ScalarType::Float32);
	in.d = oa::FnMatrix::zeros(oa::MatrixShape{nHeads}, oa::ScalarType::Float32);
	in.dOut = oa::FnMatrix::ones(in.x.getShape(), oa::ScalarType::Float32);
	in.dAngleHead = oa::FnMatrix::ones(
		oa::MatrixShape{inShape.batch, nHeads, inShape.seq, kRopeAngles}, oa::ScalarType::Float32);
	in.projected = oa::FnMatrix::ones(oa::MatrixShape{rows, dInProj}, oa::ScalarType::Float32);
	in.dtBias = oa::FnMatrix::zeros(oa::MatrixShape{1, nHeads}, oa::ScalarType::Float32);
	in.dZ = oa::FnMatrix::ones(oa::MatrixShape{rows, dInner}, oa::ScalarType::Float32);
	in.dX = oa::FnMatrix::ones(in.dZ.getShape(), oa::ScalarType::Float32);
	in.dBh = oa::FnMatrix::ones(oa::MatrixShape{rows, kDState}, oa::ScalarType::Float32);
	in.dCh = oa::FnMatrix::ones(in.dBh.getShape(), oa::ScalarType::Float32);
	in.dDT = oa::FnMatrix::ones(oa::MatrixShape{rows, nHeads}, oa::ScalarType::Float32);
	in.dADT = oa::FnMatrix::ones(in.dDT.getShape(), oa::ScalarType::Float32);
	in.dTrap = oa::FnMatrix::ones(in.dDT.getShape(), oa::ScalarType::Float32);
	in.dAngle = oa::FnMatrix::ones(oa::MatrixShape{rows, kRopeAngles}, oa::ScalarType::Float32);
	in.ssmState = oa::FnMatrix::zeros(oa::MatrixShape{inShape.batch, nHeads, inShape.headDim, kDState}, oa::ScalarType::Float32);
	in.angleState = oa::FnMatrix::zeros(oa::MatrixShape{inShape.batch, nHeads, kRopeAngles}, oa::ScalarType::Float32);
	in.kState = oa::FnMatrix::zeros(oa::MatrixShape{inShape.batch, nHeads, kDState}, oa::ScalarType::Float32);
	in.vState = oa::FnMatrix::zeros(oa::MatrixShape{inShape.batch, nHeads, inShape.headDim}, oa::ScalarType::Float32);
	return in;
}

template <typename Enqueue>
static Sample measure(
	oa::Engine& inEngine,
	const char* inName,
	int inWarmup,
	int inIterations,
	Enqueue&& inEnqueue)
{
	oa::Timer timer;
	if (auto status = timer.init(inEngine, inName); not status.isOk()) {
		throw std::runtime_error("BenchMamba3: GPU timer initialization failed");
	}
	oa::PerfStat stat(inName, static_cast<oa::U32>(inIterations), static_cast<oa::U32>(inWarmup));
	std::vector<oa::F64> values;
	values.reserve(static_cast<size_t>(inIterations));
	auto& context = oa::ExecutionSession::getActive();
	for (int iteration = 0; iteration < inWarmup + inIterations; ++iteration) {
		inEnqueue();
		auto submitted = context.submit(&timer);
		if (not submitted.isOk()) {
			throw std::runtime_error("BenchMamba3: GPU execution failed");
		}
		if (auto status = context.wait(submitted.getValue()); not status.isOk()) {
			throw std::runtime_error("BenchMamba3: GPU completion failed");
		}
		const oa::F64 elapsed = *timer.commit(inEngine);
		if (!(elapsed > 0.0)) {
			throw std::runtime_error("BenchMamba3: non-positive GPU timestamp");
		}
		stat.push(elapsed);
		if (iteration >= inWarmup) values.push_back(elapsed);
	}
	if (not stat.isReady()) throw std::runtime_error("BenchMamba3: empty statistics window");
	return {stat.mean(), stat.p50(), stat.p95(), stat.min(), stat.max(), std::move(values)};
}

template <typename Enqueue>
static FeaturePairSample measureFeaturePair(
	oa::Engine& inEngine,
	const char* inName,
	const char* inDisableFlag,
	int inWarmup,
	int inIterations,
	Enqueue&& inEnqueue)
{
	oa::Timer timer;
	if (auto status = timer.init(inEngine, inName); not status.isOk()) {
		throw std::runtime_error("BenchMamba3: feature-pair timer initialization failed");
	}
	oa::PerfStat baselineStat("mamba_feature_baseline",
		static_cast<oa::U32>(inIterations), static_cast<oa::U32>(inWarmup));
	oa::PerfStat optimizedStat("mamba_feature_optimized",
		static_cast<oa::U32>(inIterations), static_cast<oa::U32>(inWarmup));
	oa::PerfStat gainStat("mamba_feature_gain",
		static_cast<oa::U32>(inIterations), static_cast<oa::U32>(inWarmup));
	std::vector<oa::F64> baselineValues;
	std::vector<oa::F64> optimizedValues;
	std::vector<oa::F64> gainValues;
	baselineValues.reserve(static_cast<size_t>(inIterations));
	optimizedValues.reserve(static_cast<size_t>(inIterations));
	gainValues.reserve(static_cast<size_t>(inIterations));
	auto& context = oa::ExecutionSession::getActive();
	auto run = [&](bool inOptimized, oa::PerfStat& inStat) {
		setEnvironmentFlag(inDisableFlag, not inOptimized);
		inEnqueue();
		auto submitted = context.submit(&timer);
		if (not submitted.isOk()) {
			throw std::runtime_error("BenchMamba3: feature-pair execution failed");
		}
		if (auto status = context.wait(submitted.getValue()); not status.isOk()) {
			throw std::runtime_error("BenchMamba3: feature-pair completion failed");
		}
		const oa::F64 elapsed = *timer.commit(inEngine);
		inStat.push(elapsed);
		return elapsed;
	};
	for (int iteration = 0; iteration < inWarmup + inIterations; ++iteration) {
		oa::F64 baseline = 0.0;
		oa::F64 optimized = 0.0;
		if ((iteration & 1) == 0) {
			baseline = run(false, baselineStat);
			optimized = run(true, optimizedStat);
		} else {
			optimized = run(true, optimizedStat);
			baseline = run(false, baselineStat);
		}
		const oa::F64 gain = 100.0 * (baseline - optimized) / baseline;
		gainStat.push(gain);
		if (iteration >= inWarmup) {
			baselineValues.push_back(baseline);
			optimizedValues.push_back(optimized);
			gainValues.push_back(gain);
		}
	}
	setEnvironmentFlag(inDisableFlag, false);
	return {
		{baselineStat.mean(), baselineStat.p50(), baselineStat.p95(),
			baselineStat.min(), baselineStat.max(), std::move(baselineValues)},
		{optimizedStat.mean(), optimizedStat.p50(), optimizedStat.p95(),
			optimizedStat.min(), optimizedStat.max(), std::move(optimizedValues)},
		{gainStat.mean(), gainStat.p50(), gainStat.p95(),
			gainStat.min(), gainStat.max(), std::move(gainValues)},
	};
}

static void printSample(const Shape& inShape, const char* inOperation, const Sample& inSample) {
	const oa::I32 heads = (kExpand * inShape.dModel) / inShape.headDim;
	std::printf("  %-14s B=%-3d S=%-4d D=%-4d H=%-3d P=%-3d mean=%8.4f ms  p50=%8.4f  p95=%8.4f\n",
		inOperation, inShape.batch, inShape.seq, inShape.dModel, heads, inShape.headDim,
		inSample.mean, inSample.p50, inSample.p95);
	if (envFlag("OA_BENCH_MAMBA_TSV")) {
		std::printf("OAMAMBA\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
			inShape.tag, inOperation, inShape.batch, inShape.seq, inShape.dModel, heads, inShape.headDim,
			inSample.mean, inSample.p50, inSample.p95, inSample.min, inSample.max);
		for (size_t sample = 0; sample < inSample.values.size(); ++sample) {
			std::printf("OAMAMBASAMPLE\t%s\t%s\t%zu\t%.6f\n",
				inShape.tag, inOperation, sample, inSample.values[sample]);
		}
	}
}

static void preheatGpu(oa::Engine&, int inIterations) {
	oa::GradNo noGrad;
	auto inputs = makeInputs({"preheat", 64, 16, 32, 16});
	auto& context = oa::ExecutionSession::getActive();
	if (auto status = testSubmitAndWait(context); not status.isOk()) throw std::runtime_error("BenchMamba3: preheat initialization failed");
	oa::Vec<oa::Matrix> keepAlive;
	for (int iteration = 0; iteration < inIterations; ++iteration) {
		auto out = oa::FnMatrix::mamba3SisoBwd(
			inputs.dOut, inputs.c, inputs.b, inputs.x, inputs.z, inputs.adt, inputs.dt,
			inputs.trap, inputs.angle, inputs.cBias, inputs.bBias, inputs.d, inputs.scanConfig);
		keepAlive = {out.dC, out.dB, out.dX, out.dZ, out.dAdt, out.dDt, out.dTrap,
			out.dAngle, out.dCBias, out.dBBias, out.dD};
		if (auto status = testSubmitAndWait(context); not status.isOk()) throw std::runtime_error("BenchMamba3: GPU preheat failed");
	}
}

static void benchmarkShape(oa::Engine& inEngine, const Shape& inShape, int inWarmup, int inIterations) {
	oa::GradNo noGrad;
	const oa::I32 nHeads = (kExpand * inShape.dModel) / inShape.headDim;
	auto inputs = makeInputs(inShape);
	auto& context = oa::ExecutionSession::getActive();
	if (auto status = testSubmitAndWait(context); not status.isOk()) throw std::runtime_error("BenchMamba3: input initialization failed");

	oa::Vec<oa::Matrix> keepAlive;
	auto pre = measure(inEngine, "mamba_preprocess", inWarmup, inIterations, [&] {
		auto out = oa::FnMatrix::mamba3Preprocess(inputs.projected, inputs.dtBias, inputs.preprocessConfig);
		keepAlive = {out.x, out.z, out.bh, out.ch, out.dt, out.adt, out.trap, out.angle};
	});
	printSample(inShape, "preprocess", pre);

	auto preBwd = measure(inEngine, "mamba_preprocess_bwd", inWarmup, inIterations, [&] {
		auto out = oa::FnMatrix::mamba3PreprocessBwd(
			inputs.projected, inputs.dtBias, inputs.dZ, inputs.dX,
			inputs.dBh, inputs.dCh, inputs.dDT, inputs.dADT,
			inputs.dTrap, inputs.dAngle, inputs.preprocessConfig);
		keepAlive = {out.dProjected, out.dDtBias};
	});
	printSample(inShape, "preprocess_bwd", preBwd);

	auto fwd = measure(inEngine, "mamba_siso_fwd", inWarmup, inIterations, [&] {
		keepAlive = {oa::FnMatrix::mamba3Siso(
			inputs.c, inputs.b, inputs.x, inputs.z, inputs.adt, inputs.dt, inputs.trap,
			inputs.angle, inputs.cBias, inputs.bBias, inputs.d, inputs.scanConfig)};
	});
	printSample(inShape, "siso_fwd", fwd);

	auto bwd = measure(inEngine, "mamba_siso_bwd", inWarmup, inIterations, [&] {
		auto out = oa::FnMatrix::mamba3SisoBwd(
			inputs.dOut, inputs.c, inputs.b, inputs.x, inputs.z, inputs.adt, inputs.dt,
			inputs.trap, inputs.angle, inputs.cBias, inputs.bBias, inputs.d, inputs.scanConfig);
		keepAlive = {out.dC, out.dB, out.dX, out.dZ, out.dAdt, out.dDt, out.dTrap,
			out.dAngle, out.dCBias, out.dBBias, out.dD};
	});
	printSample(inShape, "siso_bwd", bwd);

	// The public backward result applies one sigmoid derivative and four bias/
	// head reductions after the core shader. time the same postlude separately
	// so the monolithic recurrence is not blamed for unrelated reduction cost.
	auto bwdPost = measure(inEngine, "mamba_bwd_post", inWarmup, inIterations, [&] {
		auto trapSigmoid = oa::FnMatrix::sigmoid(inputs.trap);
		auto dTrap = oa::FnMatrix::sigmoidBwd(trapSigmoid, inputs.dt);
		auto dAngle = oa::FnMatrix::sum(inputs.dAngleHead, 1)
			.reshape(oa::MatrixShape{inShape.batch, inShape.seq, kRopeAngles});
		auto dCBias = oa::FnMatrix::sum(
			inputs.c.reshape(oa::MatrixShape{inShape.batch * inShape.seq,
				inputs.scanConfig.nHeads * kDState}), 0);
		auto dBBias = oa::FnMatrix::sum(
			inputs.b.reshape(oa::MatrixShape{inShape.batch * inShape.seq,
				inputs.scanConfig.nHeads * kDState}), 0);
		auto dD = oa::FnMatrix::sum(
			inputs.dt.reshape(oa::MatrixShape{inShape.batch * inShape.seq,
				inputs.scanConfig.nHeads}), 0);
		keepAlive = {dTrap, dAngle, dCBias, dBBias, dD};
	});
	printSample(inShape, "bwd_post", bwdPost);

	oa::Mamba3Module module(inShape.dModel, kDState, kExpand, inShape.headDim, 1, 0.5F, false, 4,
		0.001F, 0.1F, 1.0e-4F, 1.0e-4F, true);
	auto blockInput = oa::FnMatrix::ones(
		oa::MatrixShape{inShape.batch, inShape.seq, inShape.dModel}, oa::ScalarType::Float32);
	if (auto status = testSubmitAndWait(context); not status.isOk()) throw std::runtime_error("BenchMamba3: module initialization failed");
	auto prefill = measure(inEngine, "mamba_prefill", inWarmup, inIterations, [&] {
		keepAlive = {module.forward(blockInput)};
	});
	printSample(inShape, "prefill", prefill);

	const oa::I32 normRows = inShape.batch * inShape.seq * nHeads;
	auto normX = oa::FnMatrix::ones(
		oa::MatrixShape{normRows, inShape.headDim}, oa::ScalarType::Float32);
	auto normZ = oa::FnMatrix::ones(normX.getShape(), oa::ScalarType::Float32);
	auto normGrad = oa::FnMatrix::ones(normX.getShape(), oa::ScalarType::Float32);
	auto normWeight = oa::FnMatrix::ones(
		oa::MatrixShape{inShape.headDim}, oa::ScalarType::Float32);
	auto normBias = oa::FnMatrix::zeros(
		oa::MatrixShape{inShape.headDim}, oa::ScalarType::Float32);
	if (auto status = testSubmitAndWait(context); not status.isOk()) {
		throw std::runtime_error("BenchMamba3: norm input initialization failed");
	}
	auto normBwd = measure(inEngine, "mamba_norm_bwd", inWarmup, inIterations, [&] {
		auto out = oa::FnMatrix::rmsNormGatedBwd(
			normX, normWeight, normBias, normZ, normGrad, 1.0e-5F);
		keepAlive = {out.dX, out.dWeight, out.dBias, out.dZ};
	});
	printSample(inShape, "norm_bwd", normBwd);
	auto enqueueNormBwd = [&] {
		auto out = oa::FnMatrix::rmsNormGatedBwd(
			normX, normWeight, normBias, normZ, normGrad, 1.0e-5F);
		keepAlive = {out.dX, out.dWeight, out.dBias, out.dZ};
	};
	auto normPair = measureFeaturePair(inEngine, "mamba_norm_bwd_pair",
		"OA_DISABLE_NARROW_ROW_KERNELS", inWarmup, inIterations, enqueueNormBwd);
	printSample(inShape, "norm_bwd_base", normPair.baseline);
	printSample(inShape, "norm_bwd_n32", normPair.optimized);
	std::printf("  %-14s mean=%8.3f %%   p50=%8.3f  p95=%8.3f\n",
		"norm_bwd_gain", normPair.gainPercent.mean,
		normPair.gainPercent.p50, normPair.gainPercent.p95);

	blockInput.setRequiresGrad(true);
	for (auto* parameter : module.allParameterPtrs()) {
		parameter->data.setRequiresGrad(true);
	}
	auto enqueueBlockTrain = [&] {
		oa::GradientTape tape;
		auto output = module.forward(blockInput);
		auto loss = oa::FnMatrix::sum(output);
		tape.backward(loss);
		keepAlive = {output, loss, blockInput.gradMatrix()};
	};
	enqueueBlockTrain();
	std::printf("  %-14s %u physical dispatches\n", "block_graph",
		context.nodeCount());
	if (envFlag("OA_BENCH_MAMBA_GRAPH")) {
		const auto* graph = context.graph();
		if (graph != nullptr) {
			for (oa::U32 index = 0; index < graph->nodeCount(); ++index) {
				const auto& node = graph->nodes()[index];
				std::printf("OAMAMBAGRAPH\t%u\t%s\t%s\t%u\t%u\t%u\n",
					index, node.operation.cStr(), node.shader.cStr(),
					node.groupsX, node.groupsY, node.groupsZ);
			}
		}
	}
	if (auto status = testSubmitAndWait(context); not status.isOk()) {
		throw std::runtime_error("BenchMamba3: block graph probe failed");
	}
	auto blockTrain = measure(inEngine, "mamba_block_train",
		inWarmup, inIterations, enqueueBlockTrain);
	printSample(inShape, "block_train", blockTrain);
	auto blockPair = measureFeaturePair(inEngine, "mamba_block_train_pair",
		"OA_DISABLE_NARROW_ROW_KERNELS", inWarmup, inIterations, enqueueBlockTrain);
	printSample(inShape, "block_base", blockPair.baseline);
	printSample(inShape, "block_n32", blockPair.optimized);
	std::printf("  %-14s mean=%8.3f %%   p50=%8.3f  p95=%8.3f\n",
		"block_gain", blockPair.gainPercent.mean,
		blockPair.gainPercent.p50, blockPair.gainPercent.p95);

	if (inShape.seq == 16) {
		Shape stepShape{inShape.tag, inShape.batch, 1, inShape.dModel, inShape.headDim};
		auto stepInputs = makeInputs(stepShape);
		if (auto status = testSubmitAndWait(context); not status.isOk()) throw std::runtime_error("BenchMamba3: step input initialization failed");
		auto step = measure(inEngine, "mamba_siso_step", inWarmup, inIterations, [&] {
			keepAlive = {oa::FnMatrix::mamba3SisoStep(
				stepInputs.c, stepInputs.b, stepInputs.x, stepInputs.z, stepInputs.adt,
				stepInputs.dt, stepInputs.trap, stepInputs.angle, stepInputs.cBias,
				stepInputs.bBias, stepInputs.d, stepInputs.ssmState, stepInputs.angleState,
				stepInputs.kState, stepInputs.vState, stepInputs.scanConfig)};
		});
		printSample(stepShape, "siso_step", step);
	}
}

} // namespace

TEST(BenchMamba3, KernelBreakdown) {
	ASSERT_TRUE(vkTestEngineOk()) << "VkTestEnvironment did not create oa::Engine";
	auto& engine = testEngine();
	const int warmup = envInt("OA_BENCH_MAMBA_WARMUP", 3);
	const int iterations = envInt("OA_BENCH_MAMBA_ITERS", 10);
	const int preheat = envInt("OA_BENCH_MAMBA_PREHEAT", 3);
	const char* shapeFilter = envText("OA_BENCH_MAMBA_SHAPE");
	const bool full = envFlag("OA_BENCH_MAMBA_FULL");
	ASSERT_FALSE(full && shapeFilter == nullptr)
		<< "full sweep must run one OA_BENCH_MAMBA_SHAPE per fresh process";

	std::printf("\nMamba-3 GPU kernel breakdown (FP32)\n");
	std::printf("  preheat=%d suite-backward iterations; warmup=%d iterations=%d\n", preheat, warmup, iterations);
	std::printf("  timestamps bracket queued GPU work only\n");
	std::printf("  default is the NLP-suite corner; OA_BENCH_MAMBA_FULL=1 enables B=1 S x D x P sweep\n\n");
	ASSERT_NO_THROW(preheatGpu(engine, preheat));

	if (shapeFilter == nullptr || std::strcmp(shapeFilter, "suite") == 0) {
		ASSERT_NO_THROW(benchmarkShape(engine, {"suite", 64, 16, 32, 16}, warmup, iterations));
	}
	bool matched = shapeFilter == nullptr || std::strcmp(shapeFilter, "suite") == 0;
	if (full) {
		static constexpr oa::I32 kSequences[] = {16, 64, 256, 1024};
		static constexpr oa::I32 kWidths[] = {32, 128, 512};
		static constexpr oa::I32 kHeadDims[] = {16, 32, 64};
		for (const oa::I32 headDim : kHeadDims) {
			for (const oa::I32 width : kWidths) {
				for (const oa::I32 sequence : kSequences) {
					char tag[40];
					std::snprintf(tag, sizeof(tag), "p%d_d%d_s%d", headDim, width, sequence);
					if (shapeFilter != nullptr && std::strcmp(shapeFilter, tag) != 0) continue;
					matched = true;
					ASSERT_NO_THROW(benchmarkShape(engine, {tag, 1, sequence, width, headDim}, warmup, iterations));
				}
			}
		}
	}
	ASSERT_TRUE(matched) << "OA_BENCH_MAMBA_SHAPE did not match suite or a full-sweep shape";
}
