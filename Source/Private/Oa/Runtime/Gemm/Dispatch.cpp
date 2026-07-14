#include <Oa/Runtime/Gemm/Dispatch.h>
#include <Oa/Runtime/Gemm/Cache.h>
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
	
	const auto route = OaGemmRouter::Select(InRt, InM, InN, InK);

	if (route.Path == OaGemmPath::CoopVec) {
		// CoopVec GEMV: out[i] = sum_k A[i,k] * x[k]
		// Stream.cpp prepends buffer indices automatically - only pass shader params
		struct PushCoopVec { OaU32 N; OaU32 K; };
		static_assert(sizeof(PushCoopVec) == 8, "PushCoopVec size mismatch");
		PushCoopVec push{ InN, InK };
		OaVkBuffer bufs[] = { InA, InB, OutC };
		return OaVkDispatch::Run(InRt, route.KernelName, bufs, &push, sizeof(push), route.Gx, route.Gy);
	}
	
	// Standard path
	// Stream.cpp prepends buffer indices automatically - only pass shader params
	struct Push { OaU32 M; OaU32 N; OaU32 K; };
	static_assert(sizeof(Push) == 12, "Push size mismatch");
	Push push{ InM, InN, InK };
	OaVkBuffer bufs[] = { InA, InB, OutC };

	// Cache entries are populated only by OaGemmTuner using GPU timestamp
	// queries. Recording/submission duration is CPU overhead, not kernel time.
	return OaVkDispatch::Run(InRt, route.KernelName, bufs, &push, sizeof(push), route.Gx, route.Gy);
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

	const auto route = OaGemmRouter::Select(InRt, InM, InN, InK);

	if (route.Path == OaGemmPath::CoopVec) {
		// Stream.cpp prepends buffer indices automatically - only pass shader params
		struct PushCoopVec { OaU32 N; OaU32 K; };
		static_assert(sizeof(PushCoopVec) == 8, "PushCoopVec size mismatch");
		PushCoopVec push{ InN, InK };
		OaVkBuffer bufs[] = { InA, InB, OutC };
		return OaVkDispatch::Record(InBatch, InRt, route.KernelName, bufs, &push, sizeof(push), route.Gx, route.Gy);
	}
	
	// Standard path
	// Stream.cpp prepends buffer indices automatically - only pass shader params
	struct Push { OaU32 M; OaU32 N; OaU32 K; };
	static_assert(sizeof(Push) == 12, "Push size mismatch");
	Push push{ InM, InN, InK };
	OaVkBuffer bufs[] = { InA, InB, OutC };

	return OaVkDispatch::Record(InBatch, InRt, route.KernelName, bufs, &push, sizeof(push), route.Gx, route.Gy);
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
	// Stream.cpp prepends buffer indices automatically - only pass shader params
	struct Push { OaU32 M; OaU32 N; OaU32 K; };
	static_assert(sizeof(Push) == 12, "Push size mismatch");
	Push push{ InM, InN, InK };
	OaVkBuffer bufs[] = { InA, InB, OutPre, OutAct };

	const bool useWg = OaGemmRouter::IsGemmCmWgBf16Suitable(InRt, InM, InN, InK);
	const OaU32 BM = useWg ? 64U : 128U;
	const OaU32 BN = useWg ? 64U : 128U;
	OaU32 gx = (InM + BM - 1) / BM;
	OaU32 gy = (InN + BN - 1) / BN;

	const char* kernel = useWg ? "GemmSiluCmWgBf16" : "GemmSiluCmSgBf16";
	return OaVkDispatch::Run(InRt, kernel, bufs, &push, sizeof(push), gx, gy);
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
	const bool hasBf16CoopMat = InRt.Device.Info.Software.ShaderBfloat16CooperativeMatrixEnabled
		and InRt.Device.Info.Software.ShaderBfloat16TypeEnabled;

	struct Push { OaU32 M; OaU32 N; OaU32 K; };
	static_assert(sizeof(Push) == 12, "Push size mismatch");
	Push push{ InM, InN, InK };
	OaVkBuffer bufs[] = { InA, InB, InBias, OutC };

	const bool useWg = OaGemmRouter::IsGemmCmWgBf16Suitable(InRt, InM, InN, InK);
	const OaU32 BM = useWg ? 64U : 128U;
	const OaU32 BN = useWg ? 64U : 128U;
	OaU32 gx = (InM + BM - 1) / BM;
	OaU32 gy = (InN + BN - 1) / BN;

	const char* kernel = useWg ? "GemmBiasCmWgBf16"
	                  : (hasBf16CoopMat ? "GemmBiasCmSgBf16" : "GemmBiasTiled");
	return OaVkDispatch::Run(InRt, kernel, bufs, &push, sizeof(push), gx, gy);
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
	// Use BF16 CoopMat kernel if available, otherwise fall back to FP32 tiled
	const bool hasBf16CoopMat = InRt.Device.Info.Software.ShaderBfloat16CooperativeMatrixEnabled;
	
	if (hasBf16CoopMat) {
		// BF16 CooperativeMatrix path (tensor cores)
		struct Push { OaU32 M; OaU32 N; OaU32 K; };
		static_assert(sizeof(Push) == 12, "Push size mismatch");
		Push push{ InM, InN, InK };
		OaVkBuffer bufs[] = { InA, InB, InBias, OutC };

		const bool useWg = OaGemmRouter::IsGemmCmWgBf16Suitable(InRt, InM, InN, InK);
		const OaU32 BM = useWg ? 64U : 128U;
		const OaU32 BN = useWg ? 64U : 128U;
		OaU32 gx = (InM + BM - 1) / BM;
		OaU32 gy = (InN + BN - 1) / BN;

		const char* kernel = useWg ? "GemmBiasReluCmWgBf16" : "GemmBiasReluCmSgBf16";
		return OaVkDispatch::Run(InRt, kernel, bufs, &push, sizeof(push), gx, gy);
	} else {
		// FP32 tiled fallback
		struct Push { OaU32 M; OaU32 N; OaU32 K; };
		static_assert(sizeof(Push) == 12, "Push size mismatch");
		Push push{ InM, InN, InK };
		OaVkBuffer bufs[] = { InA, InB, InBias, OutC };

		static const OaU32 BM = 64;
		static const OaU32 BN = 64;
		OaU32 gx = (InM + BM - 1) / BM;
		OaU32 gy = (InN + BN - 1) / BN;

		return OaVkDispatch::Run(InRt, "GemmBiasReluTiled", bufs, &push, sizeof(push), gx, gy);
	}
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
	// Use BF16 CoopMat kernel if available, otherwise fall back to FP32 tiled
	const bool hasBf16CoopMat = InRt.Device.Info.Software.ShaderBfloat16CooperativeMatrixEnabled;
	
	if (hasBf16CoopMat) {
		// BF16 CooperativeMatrix path (tensor cores)
		struct Push { OaU32 M; OaU32 N; OaU32 K; };
		static_assert(sizeof(Push) == 12, "Push size mismatch");
		Push push{ InM, InN, InK };
		OaVkBuffer bufs[] = { InA, InB, InBias, OutC };

		const bool useWg = OaGemmRouter::IsGemmCmWgBf16Suitable(InRt, InM, InN, InK);
		const OaU32 BM = useWg ? 64U : 128U;
		const OaU32 BN = useWg ? 64U : 128U;
		OaU32 gx = (InM + BM - 1) / BM;
		OaU32 gy = (InN + BN - 1) / BN;

		const char* kernel = useWg ? "GemmBiasGeluCmWgBf16" : "GemmBiasGeluCmSgBf16";
		return OaVkDispatch::Run(InRt, kernel, bufs, &push, sizeof(push), gx, gy);
	} else {
		// FP32 tiled fallback
		struct Push { OaU32 M; OaU32 N; OaU32 K; };
		static_assert(sizeof(Push) == 12, "Push size mismatch");
		Push push{ InM, InN, InK };
		OaVkBuffer bufs[] = { InA, InB, InBias, OutC };

		static const OaU32 BM = 64;
		static const OaU32 BN = 64;
		OaU32 gx = (InM + BM - 1) / BM;
		OaU32 gy = (InN + BN - 1) / BN;

		return OaVkDispatch::Run(InRt, "GemmBiasGeluTiled", bufs, &push, sizeof(push), gx, gy);
	}
}
