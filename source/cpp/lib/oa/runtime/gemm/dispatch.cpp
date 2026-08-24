#include <oa/runtime/gemm/dispatch.h>
#include <oa/runtime/gemm/router.h>
#include <oa/core/validation.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/device.h>
#include <oa/runtime/dispatch.h>
#include <oa/runtime/spirv.h>
#include <oa/runtime/pipeline.h>
#include <oa/core/log.h>

#include <cstring>

namespace {

[[nodiscard]] oa::ScalarType storageDtypeForProblem(
	const oa::MatmulProblem& inProblem)
{
	return inProblem.aMaster == oa::StoragePrecision::Bf16
		or inProblem.bMaster == oa::StoragePrecision::Bf16
		or inProblem.requestedOutput == oa::StoragePrecision::Bf16
		? oa::ScalarType::BFloat16
		: oa::ScalarType::Float32;
}

} // namespace

oa::Status oa::GemmDispatch::init(oa::Engine& inRt)
{
	(void)inRt;
	return oa::Status::ok();
}

oa::Result<oa::MatmulKernelLaunch> oa::GemmDispatch::describeValidatedPlan(
	const oa::MatmulPlan& inPlan,
	const oa::MatmulProblem& inProblem)
{
	if (not inPlan) {
		return oa::Status::error("oa::GemmDispatch: cannot describe an empty matmul plan");
	}

	oa::MatmulKernelLaunch launch;
	launch.kernelName = inPlan.kernelName;
	launch.grid = inPlan.grid;
	launch.bufferCount = inProblem.epilogue == oa::GemmEpilogue::None ? 3U : 4U;

	auto copyPush = [&](const auto& inPush) {
		static_assert(sizeof(inPush) <= oa::MatmulKernelLaunch::MaxPushBytes);
		std::memcpy(launch.pushData, &inPush, sizeof(inPush));
		launch.pushSize = static_cast<oa::U32>(sizeof(inPush));
	};

	if (inPlan.path == oa::GemmPath::CoopVec) {
		if (launch.bufferCount != 3U) {
			return oa::Status::error(
				"oa::GemmDispatch: CoopVec does not implement fused epilogues");
		}
		// GemmCoopVec expects (matrix[N,K], vector[K], output[N]). OA's
		// MatMulNt problem order is (A[M,K], B[N,K], C[M,N]); for M=1 the
		// first two mathematical buffers must therefore be swapped.
		launch.bufferOrder[0] = 1U;
		launch.bufferOrder[1] = 0U;
		struct Push { oa::U32 n; oa::U32 k; } push{inProblem.n, inProblem.k};
		copyPush(push);
		return launch;
	}

	if (inPlan.kernel == oa::GemmKernel::StridedFp32
		or inPlan.kernel == oa::GemmKernel::StridedTiledFp32) {
		struct Push {
			oa::U32 M, N, K;
			oa::U32 aOffset, aRowStride, aColStride, aBatchStride;
			oa::U32 bOffset, bRowStride, bColStride, bBatchStride;
			oa::U32 cOffset, cRowStride, cColStride, cBatchStride;
		} push{
			inProblem.m, inProblem.n, inProblem.k,
			inProblem.a.offset, inProblem.a.rowStride, inProblem.a.colStride,
			inProblem.a.batchStride,
			inProblem.b.offset, inProblem.b.rowStride, inProblem.b.colStride,
			inProblem.b.batchStride,
			inProblem.c.offset, inProblem.c.rowStride, inProblem.c.colStride,
			inProblem.c.batchStride,
		};
		copyPush(push);
		return launch;
	}

	struct Push { oa::U32 M; oa::U32 N; oa::U32 K; } push{
		inProblem.m, inProblem.n, inProblem.k};
	copyPush(push);
	return launch;
}

oa::Status oa::GemmDispatch::executePlan(
	oa::Engine& inRt,
	const oa::MatmulPlan& inPlan,
	const oa::MatmulProblem& inProblem,
	oa::Span<oavk::Buffer> inBuffers)
{
	if (not oa::GemmRouter::validatePlan(inRt, inPlan, inProblem)) {
		return oa::Status::error("oa::GemmDispatch: stale or incompatible matmul plan");
	}
	auto described = describeValidatedPlan(inPlan, inProblem);
	if (not described.isOk()) return described.getStatus();
	const auto& launch = described.getValue();
	if (inBuffers.size() != launch.bufferCount) {
		return oa::Status::error(
			"oa::GemmDispatch: buffer count does not match matmul epilogue");
	}
	oavk::Buffer ordered[oa::MatmulKernelLaunch::MaxBuffers];
	for (oa::U32 index = 0; index < launch.bufferCount; ++index) {
		ordered[index] = inBuffers[launch.bufferOrder[index]];
	}
	return oavk::Dispatch::run(inRt, launch.kernelName,
		oa::Span<oavk::Buffer>{ordered, launch.bufferCount},
		launch.pushData, launch.pushSize,
		storageDtypeForProblem(inProblem),
		launch.grid.x, launch.grid.y, launch.grid.z);
}

oa::Status oa::GemmDispatch::recordPlan(
	oavk::Batch& inBatch,
	oa::Engine& inRt,
	const oa::MatmulPlan& inPlan,
	const oa::MatmulProblem& inProblem,
	oa::Span<oavk::Buffer> inBuffers)
{
	if (not oa::GemmRouter::validatePlan(inRt, inPlan, inProblem)) {
		return oa::Status::error("oa::GemmDispatch: stale or incompatible matmul plan");
	}
	auto described = describeValidatedPlan(inPlan, inProblem);
	if (not described.isOk()) return described.getStatus();
	const auto& launch = described.getValue();
	if (inBuffers.size() != launch.bufferCount) {
		return oa::Status::error(
			"oa::GemmDispatch: buffer count does not match matmul epilogue");
	}
	oavk::Buffer ordered[oa::MatmulKernelLaunch::MaxBuffers];
	for (oa::U32 index = 0; index < launch.bufferCount; ++index) {
		ordered[index] = inBuffers[launch.bufferOrder[index]];
	}
	return oavk::Dispatch::record(inBatch, inRt, launch.kernelName,
		oa::Span<oavk::Buffer>{ordered, launch.bufferCount},
		launch.pushData, launch.pushSize,
		storageDtypeForProblem(inProblem),
		launch.grid.x, launch.grid.y, launch.grid.z);
}

oa::Status oa::GemmDispatch::gemm(
	oa::Engine& inRt,
	oavk::Buffer inA,
	oavk::Buffer inB,
	oavk::Buffer outC,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK)
{
	// NOTE: Validation removed from hot path - was causing 2.6x slowdown in Debug builds
	// due to atomic loads in oa::Validation::isEnabled() on every call.
	// Callers are expected to validate dimensions before calling Gemm.
	
	auto problem = oa::GemmRouter::problemForRaw(
		inM, inN, inK, oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.training = false;
	const auto plan = oa::GemmRouter::plan(inRt, problem);
	oavk::Buffer bufs[] = { inA, inB, outC };
	return executePlan(inRt, plan, problem, bufs);
}

oa::Status oa::GemmDispatch::gemmRecord(
	oavk::Batch& inBatch,
	oa::Engine& inRt,
	oavk::Buffer inA,
	oavk::Buffer inB,
	oavk::Buffer outC,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK)
{
	OA_VALIDATE(inM > 0U, oa::ValidationSeverity::Error, oa::LogComponent::Compute, "GemmRecord: M must be > 0, got %u", inM);
	OA_VALIDATE(inN > 0U, oa::ValidationSeverity::Error, oa::LogComponent::Compute, "GemmRecord: N must be > 0, got %u", inN);
	OA_VALIDATE(inK > 0U, oa::ValidationSeverity::Error, oa::LogComponent::Compute, "GemmRecord: K must be > 0, got %u", inK);

	auto problem = oa::GemmRouter::problemForRaw(
		inM, inN, inK, oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.training = false;
	const auto plan = oa::GemmRouter::plan(inRt, problem);
	oavk::Buffer bufs[] = { inA, inB, outC };
	return recordPlan(inBatch, inRt, plan, problem, bufs);
}

oa::Status oa::GemmDispatch::gemmCmSgBf16Out(
	oa::Engine& inRt,
	oavk::Buffer inA,
	oavk::Buffer inB,
	oavk::Buffer outC,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK)
{
	return gemm(inRt, inA, inB, outC, inM, inN, inK);
}

oa::Status oa::GemmDispatch::transpose(
	oa::Engine& inRt,
	oavk::Buffer inX,
	oavk::Buffer outY,
	oa::U32 inRows,
	oa::U32 inCols)
{
	// stream.cpp prepends buffer indices automatically - only pass shader params
	struct Push { oa::U32 rows; oa::U32 cols; };
	static_assert(sizeof(Push) == 8, "Push size mismatch");
	Push push{ inRows, inCols };
	oavk::Buffer bufs[] = { inX, outY };

	static const oa::U32 TILE = 32;
	oa::U32 gx = (inCols + TILE - 1) / TILE;
	oa::U32 gy = (inRows + TILE - 1) / TILE;

	return oavk::Dispatch::run(
		inRt, "TransposeTiled", bufs, &push, sizeof(push),
		oa::ScalarType::Float32, gx, gy);
}

oa::Status oa::GemmDispatch::gemmSiluCoopMatBf16(
	oa::Engine& inRt,
	oavk::Buffer inA,
	oavk::Buffer inB,
	oavk::Buffer outPre,
	oavk::Buffer outAct,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK)
{
	oavk::Buffer bufs[] = { inA, inB, outPre, outAct };

	auto problem = oa::GemmRouter::problemForRaw(
		inM, inN, inK, oa::StoragePrecision::Bf16, oa::StoragePrecision::Bf16, true);
	problem.epilogue = oa::GemmEpilogue::SiluDual;
	problem.requiresPreActivation = true;
	problem.training = true;
	problem.precisionHint = oa::GemmPrecision::Bf16;
	const auto plan = oa::GemmRouter::plan(inRt, problem);
	if (not plan) {
		return oa::Status::error("GemmSiluCoopMatBf16: no legal dual-output SiLU variant");
	}
	return executePlan(inRt, plan, problem, bufs);
}

oa::Status oa::GemmDispatch::siluMul(
	oa::Engine& inRt,
	oavk::Buffer inFused,
	oavk::Buffer outY,
	oa::U32 inBatchSize,
	oa::U32 inIntermediateSize)
{
	struct Push { oa::U32 batch_size; oa::U32 intermediate_size; };
	static_assert(sizeof(Push) == 8, "Push size mismatch");
	Push push{ inBatchSize, inIntermediateSize };
	oavk::Buffer bufs[] = { inFused, outY };

	oa::U32 total_elements = inBatchSize * inIntermediateSize;
	oa::U32 gx = (total_elements + 255) / 256;
	oa::U32 gy = 1;

	return oavk::Dispatch::run(
		inRt, "SiluMul", bufs, &push, sizeof(push),
		oa::ScalarType::Float32, gx, gy);
}

oa::Status oa::GemmDispatch::geglu(
	oa::Engine& inRt,
	oavk::Buffer inFused,
	oavk::Buffer outY,
	oa::U32 inBatchSize,
	oa::U32 inIntermediateSize)
{
	struct Push { oa::U32 batch_size; oa::U32 intermediate_size; };
	static_assert(sizeof(Push) == 8, "Push size mismatch");
	Push push{ inBatchSize, inIntermediateSize };
	oavk::Buffer bufs[] = { inFused, outY };

	oa::U32 total_elements = inBatchSize * inIntermediateSize;
	oa::U32 gx = (total_elements + 255) / 256;
	oa::U32 gy = 1;

	return oavk::Dispatch::run(
		inRt, "Geglu", bufs, &push, sizeof(push),
		oa::ScalarType::Float32, gx, gy);
}

oa::Status oa::GemmDispatch::gemmBias(
	oa::Engine& inRt,
	oavk::Buffer inA,
	oavk::Buffer inB,
	oavk::Buffer inBias,
	oavk::Buffer outC,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK)
{
	oavk::Buffer bufs[] = { inA, inB, inBias, outC };

	auto problem = oa::GemmRouter::problemForRaw(
		inM, inN, inK, oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.epilogue = oa::GemmEpilogue::Bias;
	problem.training = false;
	const auto plan = oa::GemmRouter::plan(inRt, problem);
	if (not plan) {
		return oa::Status::error("GemmBias: no legal matmul variant");
	}
	return executePlan(inRt, plan, problem, bufs);
}

oa::Status oa::GemmDispatch::gemmBiasRelu(
	oa::Engine& inRt,
	oavk::Buffer inA,
	oavk::Buffer inB,
	oavk::Buffer inBias,
	oavk::Buffer outC,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK)
{
	oavk::Buffer bufs[] = { inA, inB, inBias, outC };
	auto problem = oa::GemmRouter::problemForRaw(
		inM, inN, inK, oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.epilogue = oa::GemmEpilogue::BiasRelu;
	problem.training = false;
	const auto plan = oa::GemmRouter::plan(inRt, problem);
	if (not plan) {
		return oa::Status::error("GemmBiasRelu: no legal matmul variant");
	}
	return executePlan(inRt, plan, problem, bufs);
}

oa::Status oa::GemmDispatch::gemmBiasGelu(
	oa::Engine& inRt,
	oavk::Buffer inA,
	oavk::Buffer inB,
	oavk::Buffer inBias,
	oavk::Buffer outC,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK)
{
	oavk::Buffer bufs[] = { inA, inB, inBias, outC };
	auto problem = oa::GemmRouter::problemForRaw(
		inM, inN, inK, oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.epilogue = oa::GemmEpilogue::BiasGelu;
	problem.training = false;
	const auto plan = oa::GemmRouter::plan(inRt, problem);
	if (not plan) {
		return oa::Status::error("GemmBiasGelu: no legal matmul variant");
	}
	return executePlan(inRt, plan, problem, bufs);
}
