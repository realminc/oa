#include <Oa/Runtime/Gemm/Dispatch.h>
#include <Oa/Runtime/Gemm/Router.h>
#include <Oa/Core/Validation.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Spirv.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Core/Log.h>

OaStatus OaGemmDispatch::Init(OaComputeEngine& InRt)
{
	(void)InRt;
	return OaStatus::Ok();
}

namespace {

template <typename TDispatch>
OaStatus LowerMatmulPlan(
	OaComputeEngine& InRt,
	const OaMatmulPlan& InPlan,
	const OaMatmulProblem& InProblem,
	OaSpan<OaVkBuffer> InBuffers,
	TDispatch&& InDispatch)
{
	if (not OaGemmRouter::ValidatePlan(InRt, InPlan, InProblem)) {
		return OaStatus::Error("OaGemmDispatch: stale or incompatible matmul plan");
	}
	const OaUsize expectedBuffers = InProblem.Epilogue == OaGemmEpilogue::None
		? 3U : 4U;
	if (InBuffers.size() != expectedBuffers) {
		return OaStatus::Error("OaGemmDispatch: buffer count does not match matmul epilogue");
	}
	if (InPlan.Path == OaGemmPath::CoopVec) {
		struct Push { OaU32 N; OaU32 K; } push{InProblem.N, InProblem.K};
		return InDispatch(InPlan.KernelName, InBuffers, &push, sizeof(push));
	}
	if (InPlan.Kernel == OaGemmKernel::StridedFp32) {
		struct Push {
			OaU32 M, N, K;
			OaU32 AOffset, ARowStride, AColStride, ABatchStride;
			OaU32 BOffset, BRowStride, BColStride, BBatchStride;
			OaU32 COffset, CRowStride, CColStride, CBatchStride;
		} push{
			InProblem.M, InProblem.N, InProblem.K,
			InProblem.A.Offset, InProblem.A.RowStride, InProblem.A.ColStride, InProblem.A.BatchStride,
			InProblem.B.Offset, InProblem.B.RowStride, InProblem.B.ColStride, InProblem.B.BatchStride,
			InProblem.C.Offset, InProblem.C.RowStride, InProblem.C.ColStride, InProblem.C.BatchStride,
		};
		return InDispatch(InPlan.KernelName, InBuffers, &push, sizeof(push));
	}
	struct Push { OaU32 M; OaU32 N; OaU32 K; } push{
		InProblem.M, InProblem.N, InProblem.K};
	return InDispatch(InPlan.KernelName, InBuffers, &push, sizeof(push));
}

} // namespace

OaStatus OaGemmDispatch::ExecutePlan(
	OaComputeEngine& InRt,
	const OaMatmulPlan& InPlan,
	const OaMatmulProblem& InProblem,
	OaSpan<OaVkBuffer> InBuffers)
{
	return LowerMatmulPlan(InRt, InPlan, InProblem, InBuffers,
		[&](const char* kernel, OaSpan<OaVkBuffer> buffers,
			const void* push, OaU32 pushSize) {
			return OaVkDispatch::Run(InRt, kernel, buffers, push, pushSize,
				InPlan.Grid.X, InPlan.Grid.Y, InPlan.Grid.Z);
		});
}

OaStatus OaGemmDispatch::RecordPlan(
	OaVkBatch& InBatch,
	OaComputeEngine& InRt,
	const OaMatmulPlan& InPlan,
	const OaMatmulProblem& InProblem,
	OaSpan<OaVkBuffer> InBuffers)
{
	return LowerMatmulPlan(InRt, InPlan, InProblem, InBuffers,
		[&](const char* kernel, OaSpan<OaVkBuffer> buffers,
			const void* push, OaU32 pushSize) {
			return OaVkDispatch::Record(InBatch, InRt, kernel, buffers, push, pushSize,
				InPlan.Grid.X, InPlan.Grid.Y, InPlan.Grid.Z);
		});
}

OaStatus OaGemmDispatch::Gemm(
	OaComputeEngine& InRt,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK)
{
	// NOTE: Validation removed from hot path - was causing 2.6x slowdown in Debug builds
	// due to atomic loads in OaValidation::IsEnabled() on every call.
	// Callers are expected to validate dimensions before calling Gemm.
	
	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.Training = false;
	const auto plan = OaGemmRouter::Plan(InRt, problem);
	OaVkBuffer bufs[] = { InA, InB, OutC };
	return ExecutePlan(InRt, plan, problem, bufs);
}

OaStatus OaGemmDispatch::GemmRecord(
	OaVkBatch& InBatch,
	OaComputeEngine& InRt,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK)
{
	OA_VALIDATE(InM > 0U, OaValidationSeverity::Error, OaLogComponent::Core, "GemmRecord: M must be > 0, got %u", InM);
	OA_VALIDATE(InN > 0U, OaValidationSeverity::Error, OaLogComponent::Core, "GemmRecord: N must be > 0, got %u", InN);
	OA_VALIDATE(InK > 0U, OaValidationSeverity::Error, OaLogComponent::Core, "GemmRecord: K must be > 0, got %u", InK);

	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.Training = false;
	const auto plan = OaGemmRouter::Plan(InRt, problem);
	OaVkBuffer bufs[] = { InA, InB, OutC };
	return RecordPlan(InBatch, InRt, plan, problem, bufs);
}

OaStatus OaGemmDispatch::GemmCmSgBf16Out(
	OaComputeEngine& InRt,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK)
{
	return Gemm(InRt, InA, InB, OutC, InM, InN, InK);
}

OaStatus OaGemmDispatch::Transpose(
	OaComputeEngine& InRt,
	OaVkBuffer InX,
	OaVkBuffer OutY,
	OaU32 InRows,
	OaU32 InCols)
{
	// Stream.cpp prepends buffer indices automatically - only pass shader params
	struct Push { OaU32 rows; OaU32 cols; };
	static_assert(sizeof(Push) == 8, "Push size mismatch");
	Push push{ InRows, InCols };
	OaVkBuffer bufs[] = { InX, OutY };

	static const OaU32 TILE = 32;
	OaU32 gx = (InCols + TILE - 1) / TILE;
	OaU32 gy = (InRows + TILE - 1) / TILE;

	return OaVkDispatch::Run(InRt, "TransposeTiled", bufs, &push, sizeof(push), gx, gy);
}

OaStatus OaGemmDispatch::GemmSiluCoopMatBf16(
	OaComputeEngine& InRt,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutPre,
	OaVkBuffer OutAct,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK)
{
	OaVkBuffer bufs[] = { InA, InB, OutPre, OutAct };

	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, OaStoragePrecision::Bf16, OaStoragePrecision::Bf16, true);
	problem.Epilogue = OaGemmEpilogue::SiluDual;
	problem.RequiresPreActivation = true;
	problem.Training = true;
	problem.PrecisionHint = OaGemmPrecision::Bf16;
	const auto plan = OaGemmRouter::Plan(InRt, problem);
	if (not plan) {
		return OaStatus::Error("GemmSiluCoopMatBf16: no legal dual-output SiLU variant");
	}
	return ExecutePlan(InRt, plan, problem, bufs);
}

OaStatus OaGemmDispatch::SiluMul(
	OaComputeEngine& InRt,
	OaVkBuffer InFused,
	OaVkBuffer OutY,
	OaU32 InBatchSize,
	OaU32 InIntermediateSize)
{
	struct Push { OaU32 batch_size; OaU32 intermediate_size; };
	static_assert(sizeof(Push) == 8, "Push size mismatch");
	Push push{ InBatchSize, InIntermediateSize };
	OaVkBuffer bufs[] = { InFused, OutY };

	OaU32 total_elements = InBatchSize * InIntermediateSize;
	OaU32 gx = (total_elements + 255) / 256;
	OaU32 gy = 1;

	return OaVkDispatch::Run(InRt, "SiluMul", bufs, &push, sizeof(push), gx, gy);
}

OaStatus OaGemmDispatch::Geglu(
	OaComputeEngine& InRt,
	OaVkBuffer InFused,
	OaVkBuffer OutY,
	OaU32 InBatchSize,
	OaU32 InIntermediateSize)
{
	struct Push { OaU32 batch_size; OaU32 intermediate_size; };
	static_assert(sizeof(Push) == 8, "Push size mismatch");
	Push push{ InBatchSize, InIntermediateSize };
	OaVkBuffer bufs[] = { InFused, OutY };

	OaU32 total_elements = InBatchSize * InIntermediateSize;
	OaU32 gx = (total_elements + 255) / 256;
	OaU32 gy = 1;

	return OaVkDispatch::Run(InRt, "Geglu", bufs, &push, sizeof(push), gx, gy);
}

OaStatus OaGemmDispatch::GemmBias(
	OaComputeEngine& InRt,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer InBias,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK)
{
	OaVkBuffer bufs[] = { InA, InB, InBias, OutC };

	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.Epilogue = OaGemmEpilogue::Bias;
	problem.Training = false;
	const auto plan = OaGemmRouter::Plan(InRt, problem);
	if (not plan) {
		return OaStatus::Error("GemmBias: no legal matmul variant");
	}
	return ExecutePlan(InRt, plan, problem, bufs);
}

OaStatus OaGemmDispatch::GemmBiasRelu(
	OaComputeEngine& InRt,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer InBias,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK)
{
	OaVkBuffer bufs[] = { InA, InB, InBias, OutC };
	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.Epilogue = OaGemmEpilogue::BiasRelu;
	problem.Training = false;
	const auto plan = OaGemmRouter::Plan(InRt, problem);
	if (not plan) {
		return OaStatus::Error("GemmBiasRelu: no legal matmul variant");
	}
	return ExecutePlan(InRt, plan, problem, bufs);
}

OaStatus OaGemmDispatch::GemmBiasGelu(
	OaComputeEngine& InRt,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer InBias,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK)
{
	OaVkBuffer bufs[] = { InA, InB, InBias, OutC };
	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.Epilogue = OaGemmEpilogue::BiasGelu;
	problem.Training = false;
	const auto plan = OaGemmRouter::Plan(InRt, problem);
	if (not plan) {
		return OaStatus::Error("GemmBiasGelu: no legal matmul variant");
	}
	return ExecutePlan(InRt, plan, problem, bufs);
}
