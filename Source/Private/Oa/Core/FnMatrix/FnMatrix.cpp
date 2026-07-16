// DeviceMatrixFn.cpp — Shared utilities and runtime management for OaMatrix operations
//
// This file contains:
// - Runtime global management (SetRuntime, GetRuntime, SetWeightDtype, GetWeightDtype)
// - Shared helper functions (DivCeil, OaMatrixFlatToElementOffset, etc.)
// - OaMatrix member functions (To, Clone, etc.)
//
// Actual matrix operations are modularized into category-specific files:
// - DeviceMatrixFnAlloc.cpp      — Empty, Zeros, Ones, Full, Rand, RandN, FromBytes, CausalMask, CopyToHost
// - DeviceMatrixFnElemwise.cpp   — Add, Sub, Mul, Div, Scale, Neg, Log, Sqrt, Pow
// - FnMatrixBlas.cpp             - MatMul and Linear semantic context ops
// - DeviceMatrixFnActivation.cpp — Gelu, Silu, Relu, Tanh, Sigmoid, LeakyRelu, Elu, Mish
// - DeviceMatrixFnReduce.cpp     — Sum, Mean, Max, Softmax, LogSoftmax
// - DeviceMatrixFnNorm.cpp       — LayerNorm, RmsNorm
// - DeviceMatrixFnPool.cpp       — AvgPool2d, MaxPool2d
// - FnMatrixIndex.cpp — Gather, Slice, Concat, Transpose, Reshape, RepeatInterleave,
//   CausalMask, Equal, CompactRows and ScatterRows
// - DeviceMatrixFnView.cpp       — View operations and shape manipulations
// - DeviceMatrixFnModules.cpp    — BiasAdd, Conv1d, Conv2d (NN layer operations)
// - DeviceMatrixFnOptim.cpp      — Optimizer operations (AdamW, etc.)
// - FnMatrixRng.cpp              — Legacy RNG functions removed (use Rand/RandN/etc. instead)

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Bindless.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/RuntimeGlobal.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/Topology.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <random>

// Shared Helper Functions
static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

static OaI64 OaMatrixFlatToElementOffset(const OaMatrixShape& InShape, const OaStride& InStride, OaI64 InFlatIdx) {
	OaI64 remainder = InFlatIdx;
	OaI64 elemOff = 0;
	for (OaI32 dimIdx = 0; dimIdx < InShape.Rank; ++dimIdx) {
		OaI64 later = 1;
		for (OaI32 kk = dimIdx + 1; kk < InShape.Rank; ++kk) {
			later *= InShape.Dims[kk];
		}
		const OaI64 coord = later > 0 ? remainder / later : 0;
		if (later > 0) {
			remainder %= later;
		}
		elemOff += coord * InStride.StepElements(dimIdx);
	}
	return elemOff;
}
static bool OaCoopMatEnabledViaEnv() {
	const char *val = std::getenv("OA_ENABLE_COOPMAT");
	return val and val[0] == '1';
}
static bool OaScalarIs16BitFloat(OaScalarType InDtype) {
	return InDtype == OaScalarType::BFloat16 or InDtype == OaScalarType::Float16;
}

// Runtime Global Management — symbol lives in OaRuntimeGlobal (see
// Source/Public/Oa/Runtime/RuntimeGlobal.h). Engine init wires itself in.
// OaFn bodies must NOT call these; use OaContext::GetDefault() instead.
static OaComputeEngine* GRuntime = nullptr;
static OaScalarType GWeightDtype = OaScalarType::Float32;
// Process-wide default context, owned here and bound to the active runtime.
// OaFn bodies record into OaContext::GetDefault(); without this the first
// recorded op dereferences a null context. Lifecycle is tied to SetRuntime so
// both engine init (SetAsGlobal) and manual OaRuntimeGlobal::SetRuntime callers
// get a valid default, and ClearGlobal()'s SetRuntime(nullptr) tears it down
// before device teardown.
static OaContext* GDefaultContext = nullptr;

void OaRuntimeGlobal::SetRuntime(OaComputeEngine* InRuntime) {
	GRuntime = InRuntime;
	if (InRuntime) {
		GWeightDtype = OaPrecisionDtype(InRuntime->GetPrecision());
		// Rebind if the runtime pointer changed (e.g. engine moved).
		// Destroying the context here causes it to be created/destroyed
		// ~8 times during init because the engine is moved from the
		// stack into the result, into the app member, and into derived
		// graphics engines. Just rebind the internal pointer instead.
		// Also clear any stale graph state from a previous test/app.
		if (GDefaultContext and GDefaultContext->GetRuntime() != InRuntime) {
			GDefaultContext->Clear();
			GDefaultContext->Runtime_ = InRuntime;
		}
		if (not GDefaultContext) {
			GDefaultContext = OaContext::Create(InRuntime);
		}
		OaContext::SetDefault(GDefaultContext);
	} else {
		GWeightDtype = OaScalarType::Float32;
		OaContext::SetDefault(nullptr);
		// Do NOT destroy GDefaultContext — it's a stable singleton that survives
		// engine teardown. Just clear the runtime pointer so any late accesses
		// fail safely. The next SetRuntime(this) will rebind it.
		if (GDefaultContext) {
			GDefaultContext->Runtime_ = nullptr;
		}
	}
}

OaComputeEngine* OaRuntimeGlobal::GetRuntime() { return GRuntime; }

void OaFnMatrix::SetWeightDtype(OaScalarType InDtype) { GWeightDtype = InDtype; }

OaScalarType OaFnMatrix::GetWeightDtype() { return GWeightDtype; }

// OaMatrix Member Functions
OaF32 OaFnMatrix::Scalar(const OaMatrix& InSrc) {
	assert(InSrc.NumElements() == 1 && "Scalar() requires single-element tensor");
	auto& ctx = OaContext::GetDefault();
	auto executeStatus = ctx.Execute();
	assert(executeStatus.IsOk() && "OaContext::Execute failed before Scalar readback");
	auto syncStatus = ctx.Sync();
	assert(syncStatus.IsOk() && "OaContext::Sync failed before Scalar readback");
	return InSrc.At(0);
}

OaMatrix OaMatrix::Clone() const {
	if (not VkBuf_) return OaMatrix();
	
	OaMatrix t;
	t.Shape_ = Shape_;
	t.Stride_ = Stride_;
	t.ByteOffset_ = 0;
	t.Dtype_ = Dtype_;
	t.Device_ = Device_;

	OaI64 bytes = Shape_.NumElements() * static_cast<OaI64>(OaScalarSize(Dtype_));
	if (bytes <= 0) {
		t.SyncMatrixDescriptor();
		return t;
	}

	auto res = GRuntime->Allocator.AllocHostVisible(static_cast<OaU64>(bytes));
	if (not res) {
		t.SyncMatrixDescriptor();
		return t;
	}

	OaSharedPtr<OaVkBuffer> buf(
		new OaVkBuffer(std::move(*res)),
		[](OaVkBuffer* InPtr) {
			if (not InPtr) return;
			if (GRuntime) {
				GRuntime->FreeBuffer(*InPtr);
			}
			delete InPtr;
		});
	buf->NodeIndex = static_cast<OaU32>(Device_.Index);
	if (GRuntime->IsMultiDevice()) {
		GRuntime->RegisterBufferForOwnedNode(*buf);
	}
	t.VkBuf_ = buf;
	t.Data_ = OaSharedPtr<void>(buf->MappedPtr, [](void*) {});

	// Copy data
	if (VkBuf_ and buf) {
		std::memcpy(buf->MappedPtr, static_cast<const OaU8*>(VkBuf_->MappedPtr) + ByteOffset_,
			static_cast<OaU64>(bytes));
	}

	t.SyncMatrixDescriptor();
	return t;
}

OaMatrix OaMatrix::To(OaU32 InNodeIndex) const {
	if (static_cast<OaU32>(Device_.Index) == InNodeIndex) return *this;

	if (not GRuntime or not GRuntime->IsMultiDevice()) {
		OaMatrix t = Clone();
		t.Device_.Index = static_cast<OaI32>(InNodeIndex);
		return t;
	}

	OaMatrix dst = OaFnMatrix::EmptyOn(Shape_, Dtype_, InNodeIndex);
	if (not dst.VkBuf_ or not VkBuf_) return dst;

	auto* mesh = GRuntime->GetMesh();
	if (mesh) {
		if (auto s = mesh->CopyBuffer(static_cast<OaU32>(Device_.Index), *VkBuf_, InNodeIndex, *dst.VkBuf_,
			static_cast<OaU64>(Shape_.NumElements() * OaScalarSize(Dtype_))); !s.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core, "ToDevice: CopyBuffer failed: %s", s.GetMessage().c_str());
		}
	}
	return dst;
}

// In-Place Operations
void OaFnMatrix::AddInPlace(OaMatrix& InSelf, const OaMatrix& InOther) {
	auto& ctx = OaContext::GetDefault();
	if (InSelf.GetShape() == InOther.GetShape()) {
		OaU32 n = static_cast<OaU32>(InSelf.NumElements());
		struct { OaU32 Count; } push{n};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("Add", {&InSelf, &InOther, &InSelf}, access, &push, sizeof(push), DivCeil(n, 256));
		return;
	}
	auto bcastResult = InSelf.Shape_.Broadcast(InOther.Shape_);
	assert(bcastResult.IsOk() && "AddInPlace: shapes are not broadcast-compatible");
	OaMatrixShape outShape = bcastResult.GetValue();
	assert(outShape == InSelf.Shape_ && "AddInPlace: InSelf must already be broadcast shape");

	auto aStrides = InSelf.Shape_.BroadcastStrides(outShape);
	auto bStrides = InOther.Shape_.BroadcastStrides(outShape);
	struct PushBcast {
		OaU32 Total;
		OaU32 Rank;
		OaU32 OutDims[OA_MAX_TENSOR_DIMS];
		OaU32 AStrides[OA_MAX_TENSOR_DIMS];
		OaU32 BStrides[OA_MAX_TENSOR_DIMS];
	};
	PushBcast push{};
	push.Total = static_cast<OaU32>(outShape.NumElements());
	push.Rank = static_cast<OaU32>(outShape.Rank);
	for (OaI32 d = 0; d < outShape.Rank; ++d) {
		push.OutDims[d] = static_cast<OaU32>(outShape.Dims[d]);
		push.AStrides[d] = static_cast<OaU32>(aStrides[d]);
		push.BStrides[d] = static_cast<OaU32>(bStrides[d]);
	}
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("AddBcast", {&InSelf, &InOther, &InSelf}, access, &push, sizeof(push), DivCeil(push.Total, 256));
}

void OaFnMatrix::ScaleInPlace(OaMatrix& InSelf, OaF32 InScalar) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InSelf.NumElements());
	struct { OaU32 Count; OaF32 Alpha; } push{n, InScalar};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Scale", {&InSelf, &InSelf}, access, &push, sizeof(push), DivCeil(n, 256));
}

void OaFnMatrix::AddScalarInPlace(OaMatrix& InSelf, OaF32 InScalar) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InSelf.NumElements());
	struct { OaU32 Count; OaF32 Scalar; } push{n, InScalar};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("AddScalar", {&InSelf, &InSelf}, access, &push, sizeof(push), DivCeil(n, 256));
}

void OaFnMatrix::SubScalarInPlace(OaMatrix& InSelf, OaF32 InScalar) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InSelf.NumElements());
	struct { OaU32 Count; OaF32 Scalar; } push{n, InScalar};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("SubScalar", {&InSelf, &InSelf}, access, &push, sizeof(push), DivCeil(n, 256));
}

void OaFnMatrix::DivScalarInPlace(OaMatrix& InSelf, OaF32 InScalar) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InSelf.NumElements());
	struct { OaU32 Count; OaF32 Scalar; } push{n, InScalar};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("DivScalar", {&InSelf, &InSelf}, access, &push, sizeof(push), DivCeil(n, 256));
}

void OaFnMatrix::Fill(OaMatrix& InSelf, OaF32 InValue) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InSelf.NumElements());
	struct { OaU32 Count; OaF32 Value; } push{n, InValue};
	// The "Fill" shader is nullary_scalar: only needs output buffer
	OaBufferAccess access[] = {OaBufferAccess::Write};
	ctx.Add("Fill", {&InSelf}, access, &push, sizeof(push), DivCeil(n, 256));
}

// Multi-tensor batch operations — fuse N dispatches into one dispatch.
// Pattern-matched to AdamwMany4: up to 4 tensors, dispatch size = max count.

void OaFnMatrix::MultiFill(OaSpan<OaMatrix> InTensors, OaF32 InValue) {
	auto& ctx = OaContext::GetDefault();
	const OaU32 total = static_cast<OaU32>(InTensors.Size());
	if (total == 0) return;

	// The MultiMatrixFill shader hardcodes 4 bound buffers. The runtime prepends
	// one bindless index per *bound* buffer, so the push layout only matches when
	// exactly 4 buffers are bound. Use the fused dispatch for full groups of 4
	// and fall back to individual Fill for the 1–3 remainder — otherwise the
	// indices/counts/Value misalign and the fill silently writes nothing (this
	// was the AdamW.ZeroGrad() no-op bug for models with <4 parameters).
	OaU32 i = 0;
	for (; i + 4 <= total; i += 4) {
		struct Push {
			OaU32 count0;
			OaU32 count1;
			OaU32 count2;
			OaU32 count3;
			OaF32 Value;
		} push{};
		OaU32 maxCount = 0;
		OaVkBuffer bufs[4];
		for (OaU32 j = 0; j < 4; ++j) {
			bufs[j] = InTensors[i + j].GetVkBuffer();
			OaU32 c = static_cast<OaU32>(InTensors[i + j].NumElements());
			if (c > maxCount) maxCount = c;
			if (j == 0) push.count0 = c;
			if (j == 1) push.count1 = c;
			if (j == 2) push.count2 = c;
			if (j == 3) push.count3 = c;
		}
		push.Value = InValue;
		OaBufferAccess access[4] = {
			OaBufferAccess::Write, OaBufferAccess::Write,
			OaBufferAccess::Write, OaBufferAccess::Write};
		ctx.Add("MultiMatrixFill", OaSpan<OaVkBuffer>(bufs, 4), access, &push, sizeof(push), DivCeil(maxCount, 256));
	}
	for (; i < total; ++i) Fill(InTensors[i], InValue);
}

void OaFnMatrix::MultiAdd(OaSpan<OaMatrix> InDst, OaSpan<const OaMatrix> InSrc) {
	auto& ctx = OaContext::GetDefault();
	OaU32 N = static_cast<OaU32>(InDst.Size());
	if (N == 0 || N != static_cast<OaU32>(InSrc.Size())) return;
	if (N == 1) { AddInPlace(InDst[0], InSrc[0]); return; }
	if (N > 4) N = 4;
	// User push — buffer indices are auto-prepended by the runtime.
	struct Push {
		OaU32 count0;
		OaU32 count1;
		OaU32 count2;
		OaU32 count3;
	} push{};
	OaU32 maxCount = 0;
	OaVkBuffer bufs[8];
	for (OaU32 i = 0; i < N; ++i) {
		bufs[i * 2 + 0] = InDst[i].GetVkBuffer();
		bufs[i * 2 + 1] = InSrc[i].GetVkBuffer();
		OaU32 c = static_cast<OaU32>(InDst[i].NumElements());
		if (c > maxCount) maxCount = c;
		if (i == 0) push.count0 = c;
		if (i == 1) push.count1 = c;
		if (i == 2) push.count2 = c;
		if (i == 3) push.count3 = c;
	}
	if (N <= 1) push.count1 = 0;
	if (N <= 2) push.count2 = 0;
	if (N <= 3) push.count3 = 0;
	OaBufferAccess access[8] = {};
	for (OaU32 i = 0; i < N; ++i) {
		access[i * 2 + 0] = OaBufferAccess::ReadWrite;
		access[i * 2 + 1] = OaBufferAccess::Read;
	}
	ctx.Add("MultiMatrixAdd", OaSpan<OaVkBuffer>(bufs, N * 2), access, &push, sizeof(push), DivCeil(maxCount, 256));
}

// SSM — Selective State-Space Scan

// Note: The actual SSM implementation is in Source/Private/Oa/Ml/FnMatrix/Ssm/FnMatrixSsm.cpp
// TopK — top-k values and indices along the last dimension. Supports 1D
// ([E] → [K]) and 2D ([T,E] → [T,K]) input, matching torch.topk's rank
// behavior. GPU kernel (one workgroup per row); k beyond the groupshared
// "taken" bound is rejected rather than silently falling back to the CPU.
OaTopKResult OaFnMatrix::TopK(const OaMatrix& InA, OaI32 InK, OaI32 InDim) {
	// VQ seeding needs the highest-norm NumCodes rows (currently up to 512).
	// The cooperative kernel is intentionally correctness-first for large K;
	// routing uses tiny K and remains on its fast path through the same primitive.
	constexpr OaI32 kTopKMaxGpu = 1024;

	const OaI32 rank = InA.Rank();
	const OaI32 dim = (InDim < 0) ? (rank + InDim) : InDim;

	// Validate rank/dim/k explicitly: asserts compile out in Release, and the
	// old assert-only path read Size(1) on a rank-1 tensor → OOB → SIGSEGV.
	if ((rank != 1 and rank != 2) or dim != rank - 1 or InK < 0) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"TopK: unsupported args (rank=%d, dim=%d, k=%d); expects 1D/2D input, "
			"last-dim only, k>=0. Returning empty result.", rank, dim, InK);
		return {OaMatrix{}, OaMatrix{}};
	}

	const OaI64 T = (rank == 2) ? InA.Size(0) : 1;
	const OaI64 E = (rank == 2) ? InA.Size(1) : InA.Size(0);
	const OaI32 k = (static_cast<OaI64>(InK) <= E) ? InK : static_cast<OaI32>(E);

	const OaMatrixShape outShape = (rank == 2) ? OaMatrixShape{T, k} : OaMatrixShape{k};
	OaMatrix values  = OaFnMatrix::Empty(outShape, InA.GetDtype());
	OaMatrix indices = OaFnMatrix::Empty(outShape, OaScalarType::Int32);

	if (k == 0) return {values, indices};

	// GPU path: one workgroup per row, k cooperative-argmax passes per row.
	if (k <= kTopKMaxGpu) {
		auto& ctx = OaContext::GetDefault();
		struct { OaU32 T; OaU32 E; OaU32 K; } push{
			static_cast<OaU32>(T), static_cast<OaU32>(E), static_cast<OaU32>(k)};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write, OaBufferAccess::Write};
		ctx.Add("TopK", {&InA, &values, &indices}, access, &push, sizeof(push),
			static_cast<OaU32>(T));
		// TopK is a normal deferred GPU operation. Its consumers establish the
		// required graph dependency; only an explicit host readback may execute.
		return {values, indices};
	}

	OA_LOG_ERROR(OaLogComponent::ML,
		"TopK: k=%d exceeds the GPU limit %d; refusing a hidden CPU fallback",
		k, kTopKMaxGpu);
	return {OaMatrix{}, OaMatrix{}};
}

OaMatrix OaFnMatrix::TopKMask(const OaMatrix& InIndices, OaI32 InNumExperts) {
	if (InIndices.Rank() != 2 or InIndices.GetDtype() != OaScalarType::Int32 or
		InIndices.Size(1) <= 0 or InNumExperts <= 0) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"TopKMask: expects Int32 [T,K] indices, K>0, E>0");
		return {};
	}
	const OaU32 T = static_cast<OaU32>(InIndices.Size(0));
	const OaU32 K = static_cast<OaU32>(InIndices.Size(1));
	const OaU32 E = static_cast<OaU32>(InNumExperts);
	OaMatrix out = OaFnMatrix::Empty(
		OaMatrixShape{static_cast<OaI64>(T), static_cast<OaI64>(E)}, GetWeightDtype());
	struct { OaU32 T, E, K; } push{T, E, K};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("TopKMask", {&InIndices, &out}, access, &push, sizeof(push), DivCeil(T * E, 256));
	return out;
}
