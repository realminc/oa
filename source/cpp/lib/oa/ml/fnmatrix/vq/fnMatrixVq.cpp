// oa::FnMatrix — vector-quantization ops (VQ-VAE) + the detach stop-gradient primitive.
//
//   detach    — metadata-only stop-gradient (the STE primitive).
//   VqAssign  — on-GPU nearest-code argmin + gather (replaces a host loop).
//
// Both keep the VQ training step fully GPU-resident: encode → VqAssign → STE
// (detach) → decode → loss → backward records into one context graph, one submit.

#include <oa/ml/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/op.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/executionSession.h>

#include <assert.h>

static oa::U32 vqDivCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1u) / inB; }

namespace oa {

namespace FnMatrix {

oa::Matrix detach(const oa::Matrix& inSelf) {
	// Copy the descriptor (shares data_/vkBuf_/heapSlot_/shape_ by value/shared_ptr)
	// and drop the autograd tape entry. The result aliases the same device buffer
	// but is a leaf with no gradFn → backward stops here. forward value is identical.
	oa::Matrix out = inSelf;
	oa::MatrixAccess::autograd(out) = nullptr;
	return out;
}

oa::VqAssignResult vqAssign(const oa::Matrix& inZe, const oa::Matrix& inCodebook) {
	assert(inZe.rank() == 2 && "VqAssign: inZe must be 2D [N, D]");
	assert(inCodebook.rank() == 2 && "VqAssign: inCodebook must be 2D [K, D]");
	assert(inZe.size(1) == inCodebook.size(1) && "VqAssign: latent dim mismatch (D)");

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	const oa::I64 N = inZe.size(0);
	const oa::I64 D = inZe.size(1);
	const oa::I64 K = inCodebook.size(0);

	oa::VqAssignResult result;
	result.idx = oa::FnMatrix::empty(oa::MatrixShape{N}, oa::ScalarType::Int32);
	result.zq  = oa::FnMatrix::empty(oa::MatrixShape{N, D}, inZe.getDtype());

	struct { oa::U32 N, D, K; } push{
		static_cast<oa::U32>(N), static_cast<oa::U32>(D), static_cast<oa::U32>(K)
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // ze
		oa::BufferAccess::Read,   // codebook
		oa::BufferAccess::Write,  // idx
		oa::BufferAccess::Write,  // zq
	};

	// One thread per row; thread n scans all K codes for its argmin.
	ctx.add( "VqAssign",
		{&inZe, &inCodebook, &result.idx, &result.zq},
		access, &push, sizeof(push),
		vqDivCeil(static_cast<oa::U32>(N), 256u));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::vqAssign,
		{&inZe, &inCodebook},
		{&result.idx, &result.zq}).isOk())
	{
		return {};
	}
	return result;
}

void vqEmaUpdate(const oa::Matrix& inZe, const oa::Matrix& inIdx,
                 oa::Matrix& inOutEmbedSum, oa::Matrix& inOutClusterSize, oa::Matrix& outCodebook,
                 oa::F32 inDecay, oa::F32 inEps, oa::F32 inDeadThreshold, oa::U32 inSeed,
                 bool inNormalize) {
	assert(inZe.rank() == 2 && "VqEmaUpdate: inZe must be 2D [N, D]");
	assert(outCodebook.rank() == 2 && "VqEmaUpdate: outCodebook must be 2D [K, D]");
	assert(inZe.size(1) == outCodebook.size(1) && "VqEmaUpdate: latent dim mismatch (D)");
	assert(inOutClusterSize.numElements() == outCodebook.size(0) && "VqEmaUpdate: cluster_size must be [K]");

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	const oa::I64 N = inZe.size(0);
	const oa::I64 D = inZe.size(1);
	const oa::I64 K = outCodebook.size(0);

	struct { oa::U32 N, D, K; oa::F32 decay, eps, deadThresh; oa::U32 seed, normalize; } push{
		static_cast<oa::U32>(N), static_cast<oa::U32>(D), static_cast<oa::U32>(K),
		inDecay, inEps, inDeadThreshold, inSeed, inNormalize ? 1u : 0u
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,       // ze
		oa::BufferAccess::Read,       // idx
		oa::BufferAccess::ReadWrite,  // embed_sum
		oa::BufferAccess::ReadWrite,  // cluster_size
		oa::BufferAccess::Write,      // codebook
	};

	// One thread per code.
	ctx.add( "VqEmaUpdate",
		{&inZe, &inIdx, &inOutEmbedSum, &inOutClusterSize, &outCodebook},
		access, &push, sizeof(push),
		vqDivCeil(static_cast<oa::U32>(K), 256u));
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnMatrix::vqEmaUpdate,
		{&inZe, &inIdx, &inOutEmbedSum, &inOutClusterSize, &outCodebook},
		{},
		{oa::OpAttribute::fromFloat("decay", inDecay),
			oa::OpAttribute::fromFloat("eps", inEps),
			oa::OpAttribute::fromFloat("deadThreshold", inDeadThreshold),
			oa::OpAttribute::fromUnsignedInteger("seed", inSeed),
			oa::OpAttribute::fromBoolean("normalize", inNormalize)});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"VqEmaUpdate semantic lowering failed: %s",
			status.getMessage().cStr());
	}
}

} // namespace FnMatrix

} // namespace oa
