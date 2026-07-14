// OaFnContext — context recording helpers for compute dispatches.
// These are the free-function implementations behind OaContext::Add* methods.

#include "FnContext.h"
#include "ContextCore.h"

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Gemm/Router.h>
#include <Oa/Runtime/MatmulTypes.h>
#include <Oa/Runtime/Spirv.h>
#include <Oa/Runtime/ComputeKernel.h>

#include <cassert>

static OaGemmPrecision ToGemmPrecision(OaContextMatMulPrecision InPrecision) {
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

namespace OaFnContext {

void Add(
	OaContext& InCtx,
	OaStringView InKernelName,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	// Always-compiled guard (OA_ASSERT/OA_VALIDATE are debug-only per OaValidation.md
	// §4.1). A null runtime/graph would segfault the shipped Release binary — log and
	// skip the dispatch instead. Debug still aborts at the source via assert.
	if (not InCtx.Runtime_ or not InCtx.Graph_) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnContext::Add '%.*s': null runtime/graph — dispatch skipped",
			static_cast<int>(InKernelName.Size()), InKernelName.Data());
		assert(false && "OaFnContext::Add: null runtime/graph");
		return;
	}

	AddOwned(InCtx,
		InKernelName,
		InBuffers,
		{},
		InAccess,
		InPush,
		InPushSize,
		InGroupsX,
		InGroupsY,
		InGroupsZ
	);
}

// Record-time DTYPE derivation — the single place a dispatch's storage dtype is decided.
// Returns 1 (BF16/FP16, 2-byte OaLoad/OaStore) if any operand tensor is a 16-bit float,
// else 0 (FP32, 4-byte). Buffers that are dtype-invariant (masks, loss accumulators) are
// read/written via the explicit OaLoadF32/OaStoreF32 helpers in-shader, so they are
// unaffected by this and correctly ignored here. This mirrors bindless buffer-index
// prepending: derived from the real tensors bound to the dispatch, never a global mode.
static OaU32 DeriveNodeDtype(std::initializer_list<const OaMatrix*> InMatrices) {
	for (const OaMatrix* mat : InMatrices) {
		if (mat == nullptr) { continue; }
		const OaScalarType dt = mat->GetDtype();
		if (dt == OaScalarType::BFloat16 or dt == OaScalarType::Float16) { return 1u; }
	}
	return 0u;
}

void Add(
	OaContext& InCtx,
	OaStringView InKernelName,
	std::initializer_list<const OaMatrix*> InMatrices,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	// Always-compiled guard: this overload dereferences InCtx.Graph_ below
	// (SetLastNodeDtype), so a null graph would segfault the Release binary. Log + skip.
	if (not InCtx.Runtime_ or not InCtx.Graph_) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnContext::Add '%.*s': null runtime/graph — dispatch skipped",
			static_cast<int>(InKernelName.Size()), InKernelName.Data());
		assert(false && "OaFnContext::Add: null runtime/graph");
		return;
	}

	const OaU32 nodeDtype = DeriveNodeDtype(InMatrices);

	OaVec<OaVkBuffer> buffers;
	OaVec<OaSharedPtr<OaVkBuffer>> owners;
	buffers.Reserve(InMatrices.size());
	owners.Reserve(InMatrices.size());

	for (const OaMatrix* mat : InMatrices) {
		if (mat and mat->VkBuf_) {
			buffers.PushBack(*mat->VkBuf_);
			owners.PushBack(mat->VkBuf_);
		} else {
			buffers.PushBack(OaVkBuffer{});
			owners.PushBack({});
		}
	}

	AddOwned(InCtx,
		InKernelName,
		buffers.Span(),
		owners.Span(),
		InAccess,
		InPush,
		InPushSize,
		InGroupsX,
		InGroupsY,
		InGroupsZ
	);

	// Stamp the storage dtype derived from the operand tensors onto the node AddOwned just
	// recorded, so the dispatch selects the matching pipeline variant (see OaComputeNode::Dtype).
	InCtx.Graph_->SetLastNodeDtype(nodeDtype);
}

void AddOwned(
	OaContext& InCtx,
	OaStringView InKernelName,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	// Always-compiled guard (OaValidation.md §4.1: OA_ASSERT/OA_VALIDATE are debug-only).
	// Null runtime/graph would segfault the Release binary — log + skip the dispatch.
	if (not InCtx.Runtime_ or not InCtx.Graph_) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnContext::AddOwned '%.*s': null runtime/graph — dispatch skipped",
			static_cast<int>(InKernelName.Size()), InKernelName.Data());
		assert(false && "OaFnContext::AddOwned: null runtime/graph");
		return;
	}

	// Buffer-binding validation (Session 3 fix): The number of bound buffers must match
	// the shader's declared buffer index count. Indices are auto-prepended in order, so
	// any extra buffer silently shifts every binding → wrong/absent gradients.
	// This assert catches the systemic bug pattern that affected SwigluBwd (6→5),
	// MaxPool2dBwd (5→3), RmsNormBwd (7→5), etc.
	//
	// Note: We can't validate the exact count without shader reflection, but we can
	// catch common mistakes: if InAccess.Size() != InBuffers.Size(), the caller likely
	// passed the wrong number of buffers or forgot to update the access array.
	// Always-compiled guard: proceeding with a buffer-count mismatch silently shifts
	// every auto-prepended bindless index -> wrong/absent gradients (the SwigluBwd 6->5
	// class). In Release that silent corruption is worse than a crash, so log + skip the
	// dispatch; debug still aborts at the source.
	if (InAccess.Size() != InBuffers.Size()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnContext::AddOwned '%.*s': buffer-count mismatch (access=%zu buffers=%zu) — "
			"dispatch skipped; see OaComputeGraph.md dispatch contract",
			static_cast<int>(InKernelName.Size()), InKernelName.Data(),
			InAccess.Size(), InBuffers.Size());
		assert(false && "buffer count mismatch: InAccess.Size() must equal InBuffers.Size()");
		return;
	}

#ifndef NDEBUG
	// Exact bindless-contract check (debug only). The GPU push the runtime
	// assembles is [numBuffers*4 bytes of auto-prepended buffer indices] ++ the
	// host push tail, and that must exactly fill the shader's declared
	// PushConstants block. Reflecting the block size from SPIR-V catches a wrong
	// buffer count OR a mismatched host push struct generically — including
	// hand-written kernels — at record time instead of as silent wrong results.
	// Only applies to kernels on the default bindless pipeline (image/present
	// kernels use a different, non-prepended push convention). Reflection returns
	// 0 when it can't size the block exactly, in which case we skip the check.
	{
		const OaString kernelName(InKernelName);
		if (OaComputeKernelUsesDefaultBindlessPipeline(kernelName.c_str())) {
			const OaU32 declared = OaSpvPushConstantBlockSizeByName(kernelName.c_str());
			if (declared != 0) {
				const OaU32 assembled =
					static_cast<OaU32>(InBuffers.Size()) * sizeof(OaU32) + InPushSize;
				if (assembled != declared) {
					// A mismatch means the shader reads buffer indices / push fields
					// from memory the host never wrote → silent wrong results. This is
					// fatal by default (debug builds): the bindless contract is exact.
					// Escape hatch for bring-up of a deliberately-partial kernel:
					// OA_DISABLE_PUSH_CHECK=1 downgrades to log-only.
					OA_LOG_ERROR(OaLogComponent::Core,
						"Bindless push mismatch for '%s': %u buffers * 4 + %u push tail = %u, "
						"but shader declares a %u-byte PushConstants block. Wrong buffer count "
						"or mismatched host push struct.",
						kernelName.c_str(), static_cast<OaU32>(InBuffers.Size()),
						InPushSize, assembled, declared);
					static const bool allowMismatch = OaEnvFlag::IsSet("OA_DISABLE_PUSH_CHECK");
					if (not allowMismatch) {
						assert(false && "Bindless push mismatch — 4*numBuffers + sizeof(hostPush) "
							"must equal the shader's declared PushConstants block size. See the "
							"OA_LOG_ERROR above; set OA_DISABLE_PUSH_CHECK=1 to bypass during bring-up.");
					}
				}
			}
		}
	}
#endif

	InCtx.Graph_->Add(
		InKernelName,
		InBuffers,
		InBufferOwners,
		InAccess,
		InPush,
		InPushSize,
		InGroupsX,
		InGroupsY,
		InGroupsZ
	);

	InCtx.Executed_ = false;
}

void AddMatMul(
	OaContext& InCtx,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	AddMatMul(InCtx, InA, InB, OutC, InM, InN, InK, OaContextMatMulPrecision::Auto);
}

void AddMatMul(
	OaContext& InCtx,
	const OaMatrix& InA,
	const OaMatrix& InB,
	OaMatrix& OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	AddMatMul(InCtx, InA, InB, OutC, InM, InN, InK, OaContextMatMulPrecision::Auto);
}

void AddMatMul(
	OaContext& InCtx,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaContextMatMulPrecision InPrecision
) {
	assert(InCtx.Runtime_ and "Runtime is null");

	const auto route = OaGemmRouter::Select(*InCtx.Runtime_, InM, InN, InK, ToGemmPrecision(InPrecision));

	if (route.Path == OaGemmPath::CoopVec) {
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
		Add(InCtx, route.KernelName, bufs, access, &push, sizeof(push),
			route.Gx, route.Gy, 1);
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
	Add(InCtx, route.KernelName, bufs, access, &push, sizeof(push), route.Gx, route.Gy, 1);
}

void AddMatMul(
	OaContext& InCtx,
	const OaMatrix& InA,
	const OaMatrix& InB,
	OaMatrix& OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaContextMatMulPrecision InPrecision
) {
	assert(InCtx.Runtime_ and "Runtime is null");

	OaVkBuffer bufs[] = {
		InA.VkBuf_ ? *InA.VkBuf_ : OaVkBuffer{},
		InB.VkBuf_ ? *InB.VkBuf_ : OaVkBuffer{},
		OutC.VkBuf_ ? *OutC.VkBuf_ : OaVkBuffer{}
	};
	OaSharedPtr<OaVkBuffer> owners[] = {InA.VkBuf_, InB.VkBuf_, OutC.VkBuf_};
	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write
	};

	// BF16-stored inputs route through the GEMM router with Bf16 precision so
	// the router picks CmSg/CmWg (which support DTYPE-branched native bf16).
	// OA_GEMM_FORCE_FP32 keeps the exact tiled path for gradient checks.
	const bool isBf16Input = DeriveNodeDtype({&InA, &InB, &OutC}) != 0u;

	const auto route = OaGemmRouter::Select(*InCtx.Runtime_, InM, InN, InK,
		(isBf16Input and not OaEnvFlag::IsSet("OA_GEMM_FORCE_FP32"))
			? OaGemmPrecision::Bf16
			: ToGemmPrecision(InPrecision));

	if (route.Path == OaGemmPath::CoopVec) {
		// See same-named branch in the OaVkBuffer overload above for the
		// rationale — GemmCoopVec wants (matrix, vector, out); MatMul(A,B)
		// for M=1 gives us (vector, matrix), so we swap.
		struct PushCoopVec { OaU32 N; OaU32 K; } push{InN, InK};
		OaVkBuffer cvBufs[] = { bufs[1], bufs[0], bufs[2] };
		OaSharedPtr<OaVkBuffer> cvOwners[] = { owners[1], owners[0], owners[2] };
		AddOwned(InCtx, route.KernelName, cvBufs, cvOwners, access, &push, sizeof(push),
			route.Gx, route.Gy, 1);
		return;
	}

	struct Push { OaU32 M; OaU32 N; OaU32 K; } push{InM, InN, InK};
	AddOwned(InCtx, route.KernelName, bufs, owners, access, &push, sizeof(push),
		route.Gx, route.Gy, 1);
	// Stamp the operand-derived storage dtype onto the node AddOwned just recorded.
	// GemmTiled/GemmNaive read A/B via the DTYPE-branched Storage helpers, so a
	// BF16-stored MatMul (routed here by the isBf16Input Fp32 force above) MUST run
	// with DTYPE=1 or it reads 2-byte bf16 as 4-byte fp32 → garbage (nan/inf/1e23).
	// The raw OaVkBuffer AddOwned can't derive this itself, unlike the OaMatrix Add.
	InCtx.Graph_->SetLastNodeDtype(isBf16Input ? 1U : 0U);
}

void AddLinear(
	OaContext& InCtx,
	OaVkBuffer InX,
	OaVkBuffer InWeight,
	OaVkBuffer InBias,
	OaVkBuffer OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaBool InHasBias
) {
	AddMatMul(InCtx, InX, InWeight, OutY, InM, InN, InK);

	if (not InHasBias) {
		return;
	}

	struct Push { OaU32 Rows; OaU32 Cols; } push{InM, InN};
	OaVkBuffer bufs[] = { OutY, InBias };
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Read
	};
	Add(InCtx, "BiasAdd", bufs, access, &push, sizeof(push), OaDivCeil(InM * InN, 256));
}

void AddLinear(
	OaContext& InCtx,
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix* InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	if (not InBias or InBias->IsEmpty()) {
		AddMatMul(InCtx, InX, InWeight, OutY, InM, InN, InK);
		return;
	}

	assert(InCtx.Runtime_ and "Runtime is null");
	const auto& dev = InCtx.Runtime_->Device.Info.Software;
	// OA_GEMM_FORCE_FP32 must reach the fused-bias Linear paths too. The
	// bf16 CoopMat route packs inputs to BF16 and never consults
	// OaGemmRouter (which is what honors the flag for plain MatMul), so without
	// this guard a biased Linear silently runs in bf16 even under the override —
	// its ~3-digit mantissa adds ~4e-3 forward error, corrupting finite-difference
	// gradient checks and bit-stable debugging. When forced, fall through to the
	// scalar AddMatMul (→ Router → exact fp32) + BiasAdd path below.
	const bool forceFp32 = OaEnvFlag::IsSet("OA_GEMM_FORCE_FP32");
	// BF16-stored inputs fall through to AddMatMul + BiasAdd (DTYPE-aware).
	const bool isBf16Input = DeriveNodeDtype({&InX, &InWeight, InBias, &OutY}) != 0u;

	// GemmBiasCmSgBf16: tuned KHR CoopMat path. Reads FP32 masters (or native bf16
	// when DTYPE=1) and converts/computes in bf16. 128×128 tile requires
	// M%16==0 and N%16==0; unaligned shapes fall through to AddMatMul + BiasAdd.
	const bool bf16Aligned = (InM % 16U) == 0U and (InN % 16U) == 0U;
	const bool useBf16Bias =
		not forceFp32
		and bf16Aligned
		and dev.ShaderBfloat16CooperativeMatrixEnabled
		and dev.ShaderBfloat16TypeEnabled;
	if (useBf16Bias) {
		OaVkBuffer bufs[] = {
			InX.VkBuf_ ? *InX.VkBuf_ : OaVkBuffer{},
			InWeight.VkBuf_ ? *InWeight.VkBuf_ : OaVkBuffer{},
			InBias->VkBuf_ ? *InBias->VkBuf_ : OaVkBuffer{},
			OutY.VkBuf_ ? *OutY.VkBuf_ : OaVkBuffer{}
		};
		OaSharedPtr<OaVkBuffer> owners[] = {InX.VkBuf_, InWeight.VkBuf_, InBias->VkBuf_, OutY.VkBuf_};
		OaBufferAccess access[] = {
			OaBufferAccess::Read, OaBufferAccess::Read,
			OaBufferAccess::Read, OaBufferAccess::Write
		};
		struct Push { OaU32 M; OaU32 N; OaU32 K; } push{InM, InN, InK};
		AddOwned(InCtx, "GemmBiasCmSgBf16", bufs, owners, access, &push, sizeof(push),
			OaDivCeil(InM, 128U), OaDivCeil(InN, 128U), 1);
		InCtx.Graph_->SetLastNodeDtype(isBf16Input ? 1U : 0U);
		return;
	}

	AddMatMul(InCtx, InX, InWeight, OutY, InM, InN, InK);

	struct Push { OaU32 Rows; OaU32 Cols; } push{InM, InN};
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Read
	};
	Add(InCtx, "BiasAdd", {&OutY, InBias}, access, &push, sizeof(push), OaDivCeil(InM * InN, 256));
}

void AddLinearRelu(
	OaContext& InCtx,
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
	// path is maintained in one place. Kernel names are passed as string
	// literals (static storage): AddOwned takes a non-owning OaStringView
	// resolved at Execute time, so a stack-built name would dangle under
	// deferred execution.
	AddLinearActivation(InCtx, "GemmBiasReluCmSgBf16", "GemmBiasReluCmWgBf16", "GemmBiasReluTiled", InX, InWeight, InBias, OutY, InM, InN, InK);
}

void AddLinearGelu(
	OaContext& InCtx,
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	AddLinearActivation(InCtx, "GemmBiasGeluCmSgBf16", "GemmBiasGeluCmWgBf16", "GemmBiasGeluTiled", InX, InWeight, InBias, OutY, InM, InN, InK);
}

void AddLinearActivation(
	OaContext& InCtx,
	const char* InBf16SgName,
	const char* InBf16WgName,
	const char* InTiledName,
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	assert(InCtx.Runtime_ and "Runtime is null");

	OaVkBuffer bufs[] = {
		InX.VkBuf_ ? *InX.VkBuf_ : OaVkBuffer{},
		InWeight.VkBuf_ ? *InWeight.VkBuf_ : OaVkBuffer{},
		InBias.VkBuf_ ? *InBias.VkBuf_ : OaVkBuffer{},
		OutY.VkBuf_ ? *OutY.VkBuf_ : OaVkBuffer{}
	};
	OaSharedPtr<OaVkBuffer> owners[] = {InX.VkBuf_, InWeight.VkBuf_, InBias.VkBuf_, OutY.VkBuf_};
	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write
	};

	const auto& dev = InCtx.Runtime_->Device.Info.Software;
	// OA_GEMM_FORCE_FP32 must reach the fused paths too (they bypass the Router,
	// which is what honors the flag for plain MatMul). When forced, drop to the
	// exact GemmBiasReluTiled fp32 kernel below — bf16's ~3-digit mantissa
	// otherwise corrupts finite-difference gradient checks. See AddLinear.
	const bool forceFp32 = OaEnvFlag::IsSet("OA_GEMM_FORCE_FP32");
	// BF16-stored inputs use the CoopMat1 BF16 kernels (InBf16SgName /
	// InBf16WgName) which read via the DTYPE-branched Storage module — with
	// DTYPE=1 they load native 2-byte bf16 directly, no pack — so bf16 input
	// uses tensor cores instead of the scalar tiled fallback.
	const bool isBf16Input = DeriveNodeDtype({&InX, &InWeight, &InBias, &OutY}) != 0u;
	const bool bf16Aligned = (InM % 16U) == 0U and (InN % 16U) == 0U;
	const bool hasBf16 =
		not forceFp32
		and bf16Aligned
		and dev.ShaderBfloat16CooperativeMatrixEnabled
		and dev.ShaderBfloat16TypeEnabled;
	// CmWg (workgroup-scope, 64×64 tile) wins on small/medium 32-aligned
	// shapes where CmSg's 128×128 tile under-occupies. Use CmWg when the
	// device reports the workgroup-scope BF16 shape and dims are 32-aligned
	// but NOT 128-aligned (128-aligned shapes use CmSg for better large-tile
	// throughput). Falls through to CmSg for 128-aligned or when CmWg is
	// unavailable.
	const OaU64 caps = InCtx.Runtime_->GemmCapsMask();
	const bool hasWg = (caps & kCapCoopMat1WorkgroupBf16) != 0;
	const bool wgAligned = (InM % 32U) == 0U and (InN % 32U) == 0U;
	const bool sgAligned = (InM % 128U) == 0U and (InN % 128U) == 0U;
	const bool useWg = hasBf16 and hasWg and wgAligned and not sgAligned;

	const char* kernel = hasBf16 ? (useWg ? InBf16WgName : InBf16SgName) : InTiledName;
	const OaU32 tile = hasBf16 ? (useWg ? 64U : 128U) : 64U;

	struct Push { OaU32 M; OaU32 N; OaU32 K; } push{InM, InN, InK};
	AddOwned(InCtx, kernel, bufs, owners, access, &push, sizeof(push),
		OaDivCeil(InM, tile), OaDivCeil(InN, tile), 1);
	// Stamp the operand-derived storage dtype onto the fused dispatch. The
	// Gemm*Bf16 kernels read through DTYPE-branched Storage, so they
	// carry the operand dtype. Only the tiled fallback needs the explicit
	// stamp to avoid reading 2-byte BF16 as 4-byte FP32.
	InCtx.Graph_->SetLastNodeDtype(isBf16Input and not hasBf16 ? 1U : 0U);
}

void AddLinearBwdWeightBias(
	OaContext& InCtx,
	const OaMatrix& InInput,
	const OaMatrix& InGradOutput,
	OaMatrix& OutGradWeight,
	OaMatrix& OutGradBias,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	assert(InCtx.Runtime_ and "Runtime is null");

	// Scalar fused backward: gradWeight[n, k] = sum_m gradOut[m, n] * input[m, k]
	//                        gradBias[n]      = sum_m gradOut[m, n]
	struct FallbackPush {
		OaU32 M;
		OaU32 N;
		OaU32 K;
		OaU32 Total;
	} push{InM, InN, InK, InN * InK + InN};
	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
		OaBufferAccess::Write,
	};
	Add(InCtx, "LinearWeightBiasBwd",
		{&InGradOutput, &InInput, &OutGradWeight, &OutGradBias},
		access, &push, sizeof(push), OaDivCeil(push.Total, 256));
}

} // namespace OaFnContext

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
