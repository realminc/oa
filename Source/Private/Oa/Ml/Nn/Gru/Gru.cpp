#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>

#include <cassert>
#include <cstdio>

// OaFnMatrix::Linear records forward kernels only (pure dispatch); the matching
// OaGradLinear must be attached at the call site, exactly like OaLinear::Forward.
// Skipping this makes a projection a graph leaf, so gradients stop there and the
// recurrent weights never learn — the failure mode behind the GRU's ln(vocab) loss.
static void AttachLinearGrad(OaMatrix& InOut,
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix* InBias) {
	if (not OaFnAutograd::IsEnabled()) return;
	if (not (InX.RequiresGrad() or InWeight.RequiresGrad() or (InBias and InBias->RequiresGrad()))) return;
	auto gradFn = OaMakeSharedPtr<OaGradLinear>();
	gradFn->Saved_ = OaVec<OaMatrix>{InX, InWeight};
	gradFn->SetGraphInputs(InBias ? OaVec<OaMatrix>{InX, InWeight, *InBias}
	                              : OaVec<OaMatrix>{InX, InWeight});
	gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
	gradFn->OutputShape_ = InOut.GetShape();
	InOut.MutAutograd().GradFn = gradFn;
}

OaGruCell::OaGruCell(OaI32 InInputSize, OaI32 InHiddenSize, bool InBias)
	: InputSize_(InInputSize)
	, HiddenSize_(InHiddenSize)
	, HasBias_(InBias) {
	auto wd = OaFnMatrix::GetWeightDtype();
	RegisterParameter("weight_ih", OaFnMatrix::RandXavier(OaMatrixShape{3 * HiddenSize_, InputSize_}, wd));
	RegisterParameter("weight_hh", OaFnMatrix::RandXavier(OaMatrixShape{3 * HiddenSize_, HiddenSize_}, wd));
	if (HasBias_) {
		RegisterParameter("bias_ih", OaFnMatrix::Zeros(OaMatrixShape{3 * HiddenSize_}, wd));
		RegisterParameter("bias_hh", OaFnMatrix::Zeros(OaMatrixShape{3 * HiddenSize_}, wd));
	}
}

OaMatrix OaGruCell::ZeroState(OaI32 InBatch) const {
	return OaFnMatrix::Zeros(OaMatrixShape{InBatch, HiddenSize_}, Params_[0].Data.GetDtype());
}

// Input projection only: gatesI = Linear(x, W_ih, b_ih). Has NO recurrent
// dependency, so OaGru hoists this over the whole sequence as one batched GEMM
// ([B*T, in] @ W_ih → [B*T, 3H]) instead of T tiny per-timestep GEMMs. Linear is
// pure dispatch, so attach the OaGradLinear node here (one node for the batched
// projection; the per-step Slice nodes scatter its grad back).
OaMatrix OaGruCell::InputProjection(const OaMatrix& InInput) {
	const OaMatrix& weightIh = Params_[0].Data;
	const OaMatrix* biasIh = HasBias_ ? &Params_[2].Data : nullptr;
	OaMatrix gatesI = biasIh ? OaFnMatrix::Linear(InInput, weightIh, *biasIh) : OaFnMatrix::Linear(InInput, weightIh);
	AttachLinearGrad(gatesI, InInput, weightIh, biasIh);
	return gatesI;
}

// Recurrent half of the cell: consumes a precomputed gatesI [B*T, 3H] at row
// offset timeOffset, computes the hidden projection gatesH = Linear(h, W_hh),
// and applies the fused pointwise update in a single dispatch. timeOffset
// indexes into the full [B*T, 3H] gatesI buffer without a Slice dispatch.
OaMatrix OaGruCell::StepWithGatesI(const OaMatrix& InGatesI, const OaMatrix& InHidden, OaU32 InTimeOffset, OaU32 InBatchStride) {
	const OaMatrix& weightHh = Params_[1].Data;
	const OaMatrix* biasHh = HasBias_ ? &Params_[3].Data : nullptr;

	OaMatrix gatesH;
	auto newHidden = OaFnMatrix::GruCellLinear(InGatesI, InHidden, weightHh, biasHh, HiddenSize_, InTimeOffset, InBatchStride, &gatesH);

	// Attach the same backward nodes as before: Linear on gatesH, pointwise on
	// newHidden. The fused forward produced the same gatesH values, so the
	// unchanged backward math is still correct.
	AttachLinearGrad(gatesH, InHidden, weightHh, biasHh);

	if (OaFnAutograd::IsEnabled() and (InGatesI.RequiresGrad() or gatesH.RequiresGrad() or InHidden.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradGruCellPointwise>();
		gradFn->Saved_ = OaVec<OaMatrix>{InGatesI, gatesH, InHidden};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InGatesI, gatesH, InHidden});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = newHidden.GetShape();
		gradFn->HiddenSize_ = HiddenSize_;
		gradFn->TimeOffset_ = InTimeOffset;
		gradFn->BatchStride_ = InBatchStride;
		newHidden.MutAutograd().GradFn = gradFn;
	}
	return newHidden;
}

OaMatrix OaGruCell::Step(const OaMatrix& InInput, const OaMatrix& InHidden) {
	// Single-token convenience: project this one token then run the recurrent half.
	return StepWithGatesI(InputProjection(InInput), InHidden);
}

OaMatrix OaGruCell::Forward(const OaMatrix& InInput) {
	return Step(InInput, ZeroState(static_cast<OaI32>(InInput.Size(0))));
}

OaGru::OaGru(OaI32 InInputSize, OaI32 InHiddenSize, OaI32 InNumLayers, bool InBias)
	: InputSize_(InInputSize)
	, HiddenSize_(InHiddenSize)
	, NumLayers_(InNumLayers)
	, HasBias_(InBias)
{
	for (OaI32 i = 0; i < NumLayers_; ++i) {
		const OaI32 layerInputSize = (i == 0) ? InputSize_ : HiddenSize_;
		auto layer = OaMakeSharedPtr<OaGruCell>(layerInputSize, HiddenSize_, HasBias_);
		Layers_.PushBack(layer);
		char layerName[32];
		std::snprintf(layerName, sizeof(layerName), "layer%d", i);
		RegisterModule(layerName, layer);
	}
}

OaMatrix OaGru::ZeroState(OaI32 InBatch, OaI32 InLayer) const {
	return Layers_[InLayer]->ZeroState(InBatch);
}

OaMatrix OaGru::Step(const OaMatrix& InInput, OaMatrix& InOutHidden, OaI32 InLayer) {
	InOutHidden = Layers_[InLayer]->Step(InInput, InOutHidden);
	return InOutHidden;
}

OaMatrix OaGru::Forward(const OaMatrix& InInput) {
	// Input shape: [batch, seq_len, input_size]. This contract is load-bearing: the
	// timestep loop slices dim 1, so a flattened rank-2 [batch*seq, input] (what
	// OaEmbedding/Gather returns) would be read as batch=batch*seq, seq=input_size —
	// slicing the wrong axis into O(seq^2) buffers with garbage gradients. NDEBUG
	// strips asserts, so guard loudly at runtime regardless of build type.
	if (InInput.Rank() != 3) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaGru::Forward expects rank-3 [batch, seq, input], got rank=%d — "
			"reshape the embedding output (Gather flattens to [batch*seq, embed])",
			InInput.Rank());
		assert(false && "OaGru::Forward requires rank-3 [batch, seq, input]");
		// Release strips the assert — do NOT fall through to the timestep loop, which
		// would slice the wrong axis into O(seq^2) garbage with bad gradients. Return
		// empty: a loud, non-corrupting failure that errors cleanly downstream.
		return {};
	}
	const OaI32 batch = static_cast<OaI32>(InInput.Size(0));
	const OaI32 seqLen = static_cast<OaI32>(InInput.Size(1));
	OaMatrix layerInput = InInput;
	const bool useScan = not OaEnvFlag::IsSet("OA_DISABLE_GRU_SCAN");

	// Process each layer
	for (OaI32 layerIndex = 0; layerIndex < NumLayers_; ++layerIndex) {
		const OaI32 inputDim = (layerIndex == 0) ? InputSize_ : HiddenSize_;

		// Precompute the input projection for the WHOLE sequence in one batched GEMM:
		// [batch*seqLen, inputDim] @ W_ih → [batch*seqLen, 3H]. The input projection
		// has no recurrent dependency, so this replaces seqLen tiny per-timestep GEMMs
		// (the dispatch-starvation that pinned recurrent CPU overhead near the kernel
		// launch ceiling) with a single large one. Only W_hh @ h_t must stay recurrent.
		// Mathematically identical — slicing rows of the batched result equals the
		// per-row GEMM — so numerical gradients are unchanged.
		auto flatInput = layerInput.Reshape(OaMatrixShape{static_cast<OaI64>(batch) * seqLen, inputDim});
		auto gatesIAll = Layers_[layerIndex]->InputProjection(flatInput);   // [B*S, 3H]

		if (useScan) {
			// Whole-sequence recurrent scan: one dispatch runs every timestep
			// (h lives in groupshared across t). The backward is likewise one
			// BPTT dispatch + one LinearWeightBiasBwd. Collapses the S-dispatch
			// recurrent loop that dominated RNN/GRU CPU overhead.
			auto& cellParams = Layers_[layerIndex]->Parameters();
			const OaMatrix& wHh = cellParams[1].Data;           // weight_hh [3H, H]
			const OaMatrix* bHh = HasBias_ ? &cellParams[3].Data : nullptr;  // bias_hh [3H]
			auto scan = OaFnMatrix::GruScan(gatesIAll, wHh, bHh, HiddenSize_, seqLen, batch);

			if (OaFnAutograd::IsEnabled() and
				(gatesIAll.RequiresGrad() or wHh.RequiresGrad() or (bHh and bHh->RequiresGrad()))) {
				auto gradFn = OaMakeSharedPtr<OaGradGruScan>();
				gradFn->Saved_ = OaVec<OaMatrix>{gatesIAll, wHh, bHh ? *bHh : gatesIAll, scan.Hprev};
				gradFn->SetGraphInputs(bHh ? OaVec<OaMatrix>{gatesIAll, wHh, *bHh}
				                           : OaVec<OaMatrix>{gatesIAll, wHh});
				gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
				gradFn->OutputShape_ = scan.Out.GetShape();
				gradFn->HiddenSize_  = HiddenSize_;
				gradFn->SeqLen_      = seqLen;
				gradFn->Batch_       = batch;
				gradFn->HasBias_     = (bHh != nullptr);
				scan.Out.MutAutograd().GradFn = gradFn;
			}
			layerInput = scan.Out;   // [B, S, H]
		} else {
			// Compatibility route for drivers that cannot compile the large
			// groupshared GruScanBwd kernel. It retains the hoisted input GEMM,
			// but records the recurrent cell and its smaller backward kernels per
			// timestep. The environment switch is capability-oriented so mobile,
			// desktop, and future backends share the same implementation.
			auto hidden = Layers_[layerIndex]->ZeroState(batch);
			OaVec<OaMatrix> outputs;
			outputs.Reserve(seqLen);
			for (OaI32 time = 0; time < seqLen; ++time) {
				hidden = Layers_[layerIndex]->StepWithGatesI(
					gatesIAll, hidden, static_cast<OaU32>(time), static_cast<OaU32>(seqLen));
				outputs.PushBack(OaFnMatrix::Reshape(hidden, OaMatrixShape{batch, 1, HiddenSize_}));
			}
			layerInput = OaFnMatrix::Concat(OaSpan<OaMatrix>(outputs), 1);
		}
	}

	return layerInput;
}
