// GEMM/linear lowering helpers used by OaContext's semantic convenience API.
// Generic dispatch ownership, dtype derivation, validation, and graph mutation
// stay in OaContext::Add/Record; this file only selects the concrete kernel.

#include "DefaultContext.h"
#include "ContextCore.h"

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Gemm/Router.h>
#include <Oa/Runtime/MatmulTypes.h>
#include <cassert>

namespace {

OaGemmPrecision ToGemmPrecision(OaContextMatMulPrecision InPrecision) {
	switch (InPrecision) {
		case OaContextMatMulPrecision::Auto:
			return OaGemmPrecision::Auto;
		case OaContextMatMulPrecision::Fp32:
			return OaGemmPrecision::Fp32;
		case OaContextMatMulPrecision::Bf16:
			return OaGemmPrecision::Bf16;
	}
	return OaGemmPrecision::Auto;
}

// Record-time DTYPE derivation — the single place a dispatch's storage dtype is decided.
// Returns 1 (BF16/FP16, 2-byte OaLoad/OaStore) if any operand tensor is a 16-bit float,
// else 0 (FP32, 4-byte). Buffers that are dtype-invariant (masks, loss accumulators) are
// read/written via the explicit OaLoadF32/OaStoreF32 helpers in-shader, so they are
// unaffected by this and correctly ignored here. This mirrors bindless buffer-index
// prepending: derived from the real tensors bound to the dispatch, never a global mode.
OaU32 DeriveNodeDtype(std::initializer_list<const OaMatrix*> InMatrices) {
	for (const OaMatrix* mat : InMatrices) {
		if (mat == nullptr) { continue; }
		const OaScalarType dt = mat->GetDtype();
		if (dt == OaScalarType::BFloat16 or dt == OaScalarType::Float16) { return 1u; }
	}
	return 0u;
}

void AddLinearActivation(
	OaContext& InCtx,
	OaGemmEpilogue InEpilogue,
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
);

} // namespace

void OaContext::AddMatMul(
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	AddMatMul(InA, InB, OutC, InM, InN, InK, OaContextMatMulPrecision::Auto);
}

void OaContext::AddMatMul(
	const OaMatrix& InA,
	const OaMatrix& InB,
	OaMatrix& OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	AddMatMul(InA, InB, OutC, InM, InN, InK, OaContextMatMulPrecision::Auto);
}

void OaContext::AddMatMul(
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaContextMatMulPrecision InPrecision
) {
	assert(GetEngine() and "Engine is null");

	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	// Preserve the raw-buffer route contract used by Select(M,N,K): these
	// dispatches are graph primitives, while training legality belongs to the
	// higher-level Linear/MatMul descriptors that carry matrix metadata.
	problem.Training = false;
	problem.PrecisionHint = ToGemmPrecision(InPrecision);
	const auto plan = OaGemmRouter::Plan(*GetEngine(), problem);
	if (not plan) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"AddMatMul: no legal matmul variant for raw buffers");
		return;
	}

	if (plan.Path == OaGemmPath::CoopVec) {
		// GemmCoopVec computes out[n] = sum_k matrix[n,k] * vector[k] and
		// expects (a_idx=matrix, x_idx=vector, out_idx=out) in that binding
		// order. For MatMul(A,B) with M=1, A=[1,K] is the vector and
		// B=[N,K] is the matrix, so we have to swap them at dispatch.
		struct PushCoopVec { OaU32 N; OaU32 K; } push{InN, InK};
		OaVkBuffer bufs[] = { InB, InA, OutC };
		OaBufferAccess access[] = {
			OaBufferAccess::Read,
			OaBufferAccess::Read,
			OaBufferAccess::Write
		};
		Add(plan.KernelName, bufs, access, &push, sizeof(push),
			plan.Grid.X, plan.Grid.Y, plan.Grid.Z, "MatMulNt", plan.Variant,
			plan.ProblemContractHash, plan.ShaderContentHash);
		return;
	}

	// The OaVkBuffer overload has no OaMatrix metadata; the router now returns
	// Standard-path Fp32AsBf16 kernels directly, so no pack/mirror fallback is
	// needed.
	struct Push { OaU32 M; OaU32 N; OaU32 K; } push{InM, InN, InK};
	OaVkBuffer bufs[] = { InA, InB, OutC };
	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write
	};
	Add(plan.KernelName, bufs, access, &push, sizeof(push),
		plan.Grid.X, plan.Grid.Y, plan.Grid.Z, "MatMulNt", plan.Variant,
		plan.ProblemContractHash, plan.ShaderContentHash);
}

void OaContext::AddMatMul(
	const OaMatrix& InA,
	const OaMatrix& InB,
	OaMatrix& OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaContextMatMulPrecision InPrecision
) {
	assert(GetEngine() and "Engine is null");

	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write
	};

	// BF16-stored inputs route through the GEMM router with Bf16 precision so
	// the router picks CmSg/CmWg (which support DTYPE-branched native bf16).
	// OA_GEMM_FORCE_FP32 keeps the exact tiled path for gradient checks.
	const bool isBf16Input = DeriveNodeDtype({&InA, &InB, &OutC}) != 0u;

	const OaStoragePrecision storage = isBf16Input
		? OaStoragePrecision::Bf16 : OaStoragePrecision::Fp32;
	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, storage, storage, true);
	problem.PrecisionHint = (isBf16Input and not OaEnvFlag::IsSet("OA_GEMM_FORCE_FP32"))
		? OaGemmPrecision::Bf16 : ToGemmPrecision(InPrecision);
	const OaU32 scalarBytes = static_cast<OaU32>(OaScalarSize(InA.GetDtype()));
	if (InA.Rank() == 2 and InB.Rank() == 2 and scalarBytes != 0U) {
		problem.A = {
			.Offset = static_cast<OaU32>(InA.ByteOffset() / scalarBytes),
			.RowStride = static_cast<OaU32>(InA.GetStride().StepElements(0)),
			.ColStride = static_cast<OaU32>(InA.GetStride().StepElements(1)),
			.BatchStride = InM * InK,
		};
		problem.B = {
			.Offset = static_cast<OaU32>(InB.ByteOffset() / scalarBytes),
			.RowStride = static_cast<OaU32>(InB.GetStride().StepElements(0)),
			.ColStride = static_cast<OaU32>(InB.GetStride().StepElements(1)),
			.BatchStride = InN * InK,
		};
		problem.C = {
			.Offset = static_cast<OaU32>(OutC.ByteOffset() / scalarBytes),
			.RowStride = static_cast<OaU32>(OutC.GetStride().StepElements(0)),
			.ColStride = static_cast<OaU32>(OutC.GetStride().StepElements(1)),
			.BatchStride = InM * InN,
		};
		problem.AContiguous = InA.GetStride().MatchesRowMajor(InA.GetShape());
		problem.BContiguous = InB.GetStride().MatchesRowMajor(InB.GetShape());
	}
	const auto plan = OaGemmRouter::Plan(*GetEngine(), problem);
	if (not plan) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"AddMatMul: no legal matmul variant for matrix layout");
		return;
	}

	if (plan.Path == OaGemmPath::CoopVec) {
		// See same-named branch in the OaVkBuffer overload above for the
		// rationale — GemmCoopVec wants (matrix, vector, out); MatMul(A,B)
		// for M=1 gives us (vector, matrix), so we swap.
		struct PushCoopVec { OaU32 N; OaU32 K; } push{InN, InK};
		Add(plan.KernelName, {&InB, &InA, &OutC}, access, &push, sizeof(push),
			plan.Grid.X, plan.Grid.Y, plan.Grid.Z, "MatMulNt", plan.Variant,
			plan.ProblemContractHash, plan.ShaderContentHash);
		return;
	}

	if (plan.Kernel == OaGemmKernel::StridedFp32) {
		struct Push {
			OaU32 M, N, K;
			OaU32 AOffset, ARowStride, AColStride, ABatchStride;
			OaU32 BOffset, BRowStride, BColStride, BBatchStride;
			OaU32 COffset, CRowStride, CColStride, CBatchStride;
		} push{
			InM, InN, InK,
			problem.A.Offset, problem.A.RowStride, problem.A.ColStride, problem.A.BatchStride,
			problem.B.Offset, problem.B.RowStride, problem.B.ColStride, problem.B.BatchStride,
			problem.C.Offset, problem.C.RowStride, problem.C.ColStride, problem.C.BatchStride,
		};
		Add(plan.KernelName, {&InA, &InB, &OutC}, access, &push, sizeof(push),
			plan.Grid.X, plan.Grid.Y, plan.Grid.Z, "MatMulNt", plan.Variant,
			plan.ProblemContractHash, plan.ShaderContentHash);
	} else {
		struct Push { OaU32 M; OaU32 N; OaU32 K; } push{InM, InN, InK};
		Add(plan.KernelName, {&InA, &InB, &OutC}, access, &push, sizeof(push),
			plan.Grid.X, plan.Grid.Y, plan.Grid.Z, "MatMulNt", plan.Variant,
			plan.ProblemContractHash, plan.ShaderContentHash);
	}
}

void OaContext::AddLinear(
	OaVkBuffer InX,
	OaVkBuffer InWeight,
	OaVkBuffer InBias,
	OaVkBuffer OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaBool InHasBias
) {
	AddMatMul(InX, InWeight, OutY, InM, InN, InK);

	if (not InHasBias) {
		return;
	}

	struct Push { OaU32 Rows; OaU32 Cols; } push{InM, InN};
	OaVkBuffer bufs[] = { OutY, InBias };
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Read
	};
	Add("BiasAdd", bufs, access, &push, sizeof(push), OaDivCeil(InM * InN, 256));
}

void OaContext::AddLinear(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix* InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	if (not InBias or InBias->IsEmpty()) {
		AddMatMul(InX, InWeight, OutY, InM, InN, InK);
		return;
	}

	assert(GetEngine() and "Engine is null");
	const bool isBf16Input = DeriveNodeDtype({&InX, &InWeight, InBias, &OutY}) != 0u;
	const OaStoragePrecision storage = isBf16Input
		? OaStoragePrecision::Bf16
		: OaStoragePrecision::Fp32;
	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, storage, storage, true);
	problem.Epilogue = OaGemmEpilogue::Bias;
	problem.Training = true;
	const auto plan = OaGemmRouter::Plan(*GetEngine(), problem);
	if (not plan) {
		OA_LOG_ERROR(OaLogComponent::Core, "AddLinear: no legal Bias matmul variant");
		return;
	}

	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write
	};
	struct Push { OaU32 M; OaU32 N; OaU32 K; } push{InM, InN, InK};
	Add(plan.KernelName, {&InX, &InWeight, InBias, &OutY}, access,
		&push, sizeof(push), plan.Grid.X, plan.Grid.Y, plan.Grid.Z,
		"Linear", plan.Variant, plan.ProblemContractHash, plan.ShaderContentHash);
}

void OaContext::AddLinearRelu(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	// out = act(A @ W^T + bias). Relu and Gelu share the identical BF16 /
	// Tiled dispatch skeleton — only the fused-GEMM kernel family differs.
	// Route both through one helper so the bf16-mirror cache + scratch-alloc
	// path is maintained in one place. OaContext::Record validates the transient
	// descriptor and OaComputeGraph::Add immediately copies its kernel name,
	// buffers, access list and push payload into graph-owned storage.
	AddLinearActivation(*this, OaGemmEpilogue::BiasRelu,
		InX, InWeight, InBias, OutY, InM, InN, InK);
}

void OaContext::AddLinearGelu(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	AddLinearActivation(*this, OaGemmEpilogue::BiasGelu,
		InX, InWeight, InBias, OutY, InM, InN, InK);
}

namespace {

void AddLinearActivation(
	OaContext& InCtx,
	OaGemmEpilogue InEpilogue,
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	assert(InCtx.GetEngine() and "Engine is null");

	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write
	};

	const bool isBf16Input = DeriveNodeDtype({&InX, &InWeight, &InBias, &OutY}) != 0u;
	const OaStoragePrecision storage = isBf16Input
		? OaStoragePrecision::Bf16
		: OaStoragePrecision::Fp32;
	auto problem = OaGemmRouter::ProblemForRaw(
		InM, InN, InK, storage, storage, true);
	problem.Epilogue = InEpilogue;
	problem.Training = true;
	const auto plan = OaGemmRouter::Plan(*InCtx.GetEngine(), problem);
	if (not plan) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"AddLinearActivation: no legal matmul variant for epilogue=%u",
			static_cast<OaU32>(InEpilogue));
		return;
	}

	struct Push { OaU32 M; OaU32 N; OaU32 K; } push{InM, InN, InK};
	const OaStringView operation = InEpilogue == OaGemmEpilogue::BiasRelu
		? OaStringView{"LinearRelu"} : OaStringView{"LinearGelu"};
	InCtx.Add(plan.KernelName, {&InX, &InWeight, &InBias, &OutY}, access,
		&push, sizeof(push), plan.Grid.X, plan.Grid.Y, plan.Grid.Z,
		operation, plan.Variant, plan.ProblemContractHash, plan.ShaderContentHash);
}

} // namespace

void OaContext::AddLinearBwdWeightBias(
	const OaMatrix& InInput,
	const OaMatrix& InGradOutput,
	OaMatrix& OutGradWeight,
	OaMatrix& OutGradBias,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	assert(GetEngine() and "Engine is null");

	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
		OaBufferAccess::Write,
	};
	const OaU32 tileN = OaDivCeil(InN, 32);
	const OaU32 tileK = OaDivCeil(InK, 32);
	// The cooperative reduction needs enough independent output tiles to fill
	// the device. Below that point, the scalar row-owned kernel is faster and
	// avoids workgroup barriers. This is an internal route; the semantic API and
	// output layout stay identical.
	const bool useTiled = InM >= 64 and tileN * tileK >= 9;
	if (useTiled) {
		struct TiledPush { OaU32 M, N, K; } push{InM, InN, InK};
		Add("LinearWeightBiasBwdTiled",
			{&InGradOutput, &InInput, &OutGradWeight, &OutGradBias},
			access, &push, sizeof(push), tileN, tileK, 1);
		return;
	}

	struct ScalarPush { OaU32 M, N, K, Total; }
		push{InM, InN, InK, InN * InK + InN};
	Add("LinearWeightBiasBwd",
		{&InGradOutput, &InInput, &OutGradWeight, &OutGradBias},
		access, &push, sizeof(push), OaDivCeil(push.Total, 256));
}

// ═════════════════════════════════════════════════════════════════════════════
// Per-domain default-context accessors
// ═════════════════════════════════════════════════════════════════════════════

namespace OaFnMatrix {
	OaContext& GetContext() {
		return OaContext::GetDefault();
	}
}

namespace OaFnLoss {
	OaContext& GetContext() {
		return OaContext::GetDefault();
	}
}

namespace OaFnAudio {
	OaContext& GetContext() {
		return OaContext::GetDefault();
	}
}

namespace OaFnUi {
	OaContext& GetContext() {
		return OaContext::GetDefault();
	}
}

namespace OaFnCrypto {
	OaContext& GetContext() {
		return OaContext::GetDefault();
	}
}
