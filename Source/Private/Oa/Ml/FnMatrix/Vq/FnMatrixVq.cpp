// OaFnMatrix — vector-quantization ops (VQ-VAE) + the Detach stop-gradient primitive.
//
//   Detach    — metadata-only stop-gradient (the STE primitive).
//   VqAssign  — on-GPU nearest-code argmin + gather (replaces a host loop).
//
// Both keep the VQ training step fully GPU-resident: encode → VqAssign → STE
// (Detach) → decode → loss → backward records into one context graph, one submit.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>

#include <cassert>

static OaU32 OaVqDivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1u) / InB; }

namespace OaFnMatrix {

OaMatrix Detach(const OaMatrix& InSelf) {
	// Copy the descriptor (shares Data_/VkBuf_/HeapSlot_/Shape_ by value/shared_ptr)
	// and drop the autograd tape entry. The result aliases the same device buffer
	// but is a leaf with no GradFn → Backward stops here. Forward value is identical.
	OaMatrix out = InSelf;
	out.Autograd_ = nullptr;
	return out;
}

OaVqAssignResult VqAssign(const OaMatrix& InZe, const OaMatrix& InCodebook) {
	assert(InZe.Rank() == 2 && "VqAssign: InZe must be 2D [N, D]");
	assert(InCodebook.Rank() == 2 && "VqAssign: InCodebook must be 2D [K, D]");
	assert(InZe.Size(1) == InCodebook.Size(1) && "VqAssign: latent dim mismatch (D)");

	auto& ctx = OaContext::GetDefault();

	const OaI64 N = InZe.Size(0);
	const OaI64 D = InZe.Size(1);
	const OaI64 K = InCodebook.Size(0);

	OaVqAssignResult result;
	result.Idx = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Int32);
	result.Zq  = OaFnMatrix::Empty(OaMatrixShape{N, D}, InZe.Dtype_);

	struct { OaU32 N, D, K; } push{
		static_cast<OaU32>(N), static_cast<OaU32>(D), static_cast<OaU32>(K)
	};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // ze
		OaBufferAccess::Read,   // codebook
		OaBufferAccess::Write,  // idx
		OaBufferAccess::Write,  // zq
	};

	// One thread per row; thread n scans all K codes for its argmin.
	ctx.Add("VqAssign",
		{&InZe, &InCodebook, &result.Idx, &result.Zq},
		access, &push, sizeof(push),
		OaVqDivCeil(static_cast<OaU32>(N), 256u));

	return result;
}

void VqEmaUpdate(const OaMatrix& InZe, const OaMatrix& InIdx,
                 OaMatrix& IoEmbedSum, OaMatrix& IoClusterSize, OaMatrix& OutCodebook,
                 OaF32 InDecay, OaF32 InEps, OaF32 InDeadThreshold, OaU32 InSeed,
                 bool InNormalize) {
	assert(InZe.Rank() == 2 && "VqEmaUpdate: InZe must be 2D [N, D]");
	assert(OutCodebook.Rank() == 2 && "VqEmaUpdate: OutCodebook must be 2D [K, D]");
	assert(InZe.Size(1) == OutCodebook.Size(1) && "VqEmaUpdate: latent dim mismatch (D)");
	assert(IoClusterSize.NumElements() == OutCodebook.Size(0) && "VqEmaUpdate: cluster_size must be [K]");

	auto& ctx = OaContext::GetDefault();

	const OaI64 N = InZe.Size(0);
	const OaI64 D = InZe.Size(1);
	const OaI64 K = OutCodebook.Size(0);

	struct { OaU32 N, D, K; OaF32 Decay, Eps, DeadThresh; OaU32 Seed, Normalize; } push{
		static_cast<OaU32>(N), static_cast<OaU32>(D), static_cast<OaU32>(K),
		InDecay, InEps, InDeadThreshold, InSeed, InNormalize ? 1u : 0u
	};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,       // ze
		OaBufferAccess::Read,       // idx
		OaBufferAccess::ReadWrite,  // embed_sum
		OaBufferAccess::ReadWrite,  // cluster_size
		OaBufferAccess::Write,      // codebook
	};

	// One thread per code.
	ctx.Add("VqEmaUpdate",
		{&InZe, &InIdx, &IoEmbedSum, &IoClusterSize, &OutCodebook},
		access, &push, sizeof(push),
		OaVqDivCeil(static_cast<OaU32>(K), 256u));
}

} // namespace OaFnMatrix
