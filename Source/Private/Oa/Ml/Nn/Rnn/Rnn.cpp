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
// recurrent weights never learn. (Same helper / rule as OaGruCell::Step.)
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

OaRnnCell::OaRnnCell(OaI32 InInputSize, OaI32 InHiddenSize, bool InBias)
	: InputSize_(InInputSize)
	, HiddenSize_(InHiddenSize)
	, HasBias_(InBias)
{
	auto wd = OaFnMatrix::GetWeightDtype();
	RegisterParameter("weight_ih", OaFnMatrix::RandXavier(OaMatrixShape{HiddenSize_, InputSize_}, wd));
	RegisterParameter("weight_hh", OaFnMatrix::RandXavier(OaMatrixShape{HiddenSize_, HiddenSize_}, wd));
	if (HasBias_) {
		RegisterParameter("bias_ih", OaFnMatrix::Zeros(OaMatrixShape{HiddenSize_}, wd));
		RegisterParameter("bias_hh", OaFnMatrix::Zeros(OaMatrixShape{HiddenSize_}, wd));
	}
}

OaMatrix OaRnnCell::ZeroState(OaI32 InBatch) const {
	return OaFnMatrix::Zeros(OaMatrixShape{InBatch, HiddenSize_}, Params_[0].Data.GetDtype());
}

// Input projection only: gi = Linear(x, W_ih, b_ih). No recurrent dependency, so
// OaRnn hoists this over the whole sequence as one batched GEMM ([B*T, in] @ W_ih
// → [B*T, H]) instead of T tiny per-timestep GEMMs. Linear is pure dispatch, so
// attach the OaGradLinear node here.
OaMatrix OaRnnCell::InputProjection(const OaMatrix& InInput) {
	const OaMatrix& weightIh = Params_[0].Data;
	const OaMatrix* biasIh = HasBias_ ? &Params_[2].Data : nullptr;
	OaMatrix gi = biasIh ? OaFnMatrix::Linear(InInput, weightIh, *biasIh)
	                     : OaFnMatrix::Linear(InInput, weightIh);
	AttachLinearGrad(gi, InInput, weightIh, biasIh);
	return gi;
}

// Recurrent half: consumes a precomputed gi [B*T, H] at row offset timeOffset,
// computes the hidden projection gh = Linear(h, W_hh), and applies the fused tanh
// pointwise: h_new = tanh(gi + gh) in a single dispatch. timeOffset indexes into the
// full [B*T, H] gi buffer without a per-step Slice dispatch.
OaMatrix OaRnnCell::StepWithGi(const OaMatrix& InGi, const OaMatrix& InHidden, OaU32 InTimeOffset, OaU32 InBatchStride) {
	const OaMatrix& weightHh = Params_[1].Data;
	const OaMatrix* biasHh = HasBias_ ? &Params_[3].Data : nullptr;

	OaMatrix gh;
	auto newHidden = OaFnMatrix::RnnCellLinear(InGi, InHidden, weightHh, biasHh, InTimeOffset, InBatchStride, &gh);

	// Attach the same backward nodes as before: Linear on gh, pointwise on
	// newHidden. The fused forward produced the same gh values, so the unchanged
	// backward math is still correct.
	AttachLinearGrad(gh, InHidden, weightHh, biasHh);

	if (OaFnAutograd::IsEnabled() and (InGi.RequiresGrad() or gh.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradRnnCellPointwise>();
		gradFn->Saved_ = OaVec<OaMatrix>{InGi, gh};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InGi, gh});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = newHidden.GetShape();
		gradFn->HiddenSize_  = HiddenSize_;
		gradFn->TimeOffset_  = InTimeOffset;
		gradFn->BatchStride_ = InBatchStride;
		newHidden.MutAutograd().GradFn = gradFn;
	}
	return newHidden;
}

OaMatrix OaRnnCell::Step(const OaMatrix& InInput, const OaMatrix& InHidden) {
	// Single-token convenience: project this one token then run the recurrent half.
	return StepWithGi(InputProjection(InInput), InHidden);
}

OaMatrix OaRnnCell::Forward(const OaMatrix& InInput) {
	return Step(InInput, ZeroState(static_cast<OaI32>(InInput.Size(0))));
}

OaRnn::OaRnn(OaI32 InInputSize, OaI32 InHiddenSize, OaI32 InNumLayers, bool InBias)
	: InputSize_(InInputSize)
	, HiddenSize_(InHiddenSize)
	, NumLayers_(InNumLayers)
	, HasBias_(InBias)
{
	for (OaI32 i = 0; i < NumLayers_; ++i) {
		const OaI32 layerInputSize = (i == 0) ? InputSize_ : HiddenSize_;
		auto layer = OaMakeSharedPtr<OaRnnCell>(layerInputSize, HiddenSize_, HasBias_);
		Layers_.PushBack(layer);
		char layerName[32];
		std::snprintf(layerName, sizeof(layerName), "layer%d", i);
		RegisterModule(layerName, layer);
	}
}

OaMatrix OaRnn::ZeroState(OaI32 InBatch, OaI32 InLayer) const {
	return Layers_[InLayer]->ZeroState(InBatch);
}

OaMatrix OaRnn::Step(const OaMatrix& InInput, OaMatrix& InOutHidden, OaI32 InLayer) {
	InOutHidden = Layers_[InLayer]->Step(InInput, InOutHidden);
	return InOutHidden;
}

OaMatrix OaRnn::Forward(const OaMatrix& InInput) {
	// Input shape: [batch, seq_len, input_size]. Load-bearing contract: the timestep
	// loop slices dim 1, so a flattened rank-2 [batch*seq, input] (what OaEmbedding/
	// Gather returns) would be read as batch=batch*seq, seq=input_size — wrong axis,
	// O(seq^2) buffers, garbage grads. NDEBUG strips asserts, so guard at runtime.
	if (InInput.Rank() != 3) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaRnn::Forward expects rank-3 [batch, seq, input], got rank=%d — "
			"reshape the embedding output (Gather flattens to [batch*seq, embed])",
			InInput.Rank());
		assert(false && "OaRnn::Forward requires rank-3 [batch, seq, input]");
		// Release strips the assert — do NOT fall through to the timestep loop, which
		// would slice the wrong axis into O(seq^2) garbage with bad gradients. Return
		// empty: a loud, non-corrupting failure that errors cleanly downstream.
		return {};
	}
	const OaI32 batch = static_cast<OaI32>(InInput.Size(0));
	const OaI32 seqLen = static_cast<OaI32>(InInput.Size(1));
	OaMatrix layerInput = InInput;
	const bool useScan = not OaEnvFlag::IsSet("OA_DISABLE_RNN_SCAN");

	for (OaI32 layerIndex = 0; layerIndex < NumLayers_; ++layerIndex) {
		const OaI32 inputDim = (layerIndex == 0) ? InputSize_ : HiddenSize_;

		// Precompute the input projection for the WHOLE sequence in one batched GEMM:
		// [batch*seqLen, inputDim] @ W_ih → [batch*seqLen, H]. No recurrent dependency,
		// so this replaces seqLen tiny per-timestep GEMMs (the dispatch starvation that
		// pinned recurrent CPU overhead near the launch ceiling) with one large GEMM.
		// Only W_hh @ h_t stays in the loop. Slicing rows of the batched result equals
		// the per-row GEMM — numerics and gradients are unchanged.
		auto flatInput = layerInput.Reshape(OaMatrixShape{static_cast<OaI64>(batch) * seqLen, inputDim});
		auto giAll     = Layers_[layerIndex]->InputProjection(flatInput);      // [B*T, H]
		const OaMatrix& wHh = Layers_[layerIndex]->Parameters()[1].Data;
		const OaMatrix* bHh = HasBias_ ? &Layers_[layerIndex]->Parameters()[3].Data : nullptr;

		if (useScan) {
			auto scan = OaFnMatrix::RnnScan(giAll, wHh, bHh, HiddenSize_, seqLen, batch);
			if (OaFnAutograd::IsEnabled() and (giAll.RequiresGrad() or wHh.RequiresGrad() or (bHh and bHh->RequiresGrad()))) {
				auto gradFn = OaMakeSharedPtr<OaGradRnnScan>();
				gradFn->Saved_ = OaVec<OaMatrix>{giAll, wHh, bHh ? *bHh : giAll, scan.Hprev};
				gradFn->SetGraphInputs(bHh ? OaVec<OaMatrix>{giAll, wHh, *bHh} : OaVec<OaMatrix>{giAll, wHh});
				gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
				gradFn->OutputShape_ = scan.Out.GetShape();
				gradFn->HiddenSize_  = HiddenSize_;
				gradFn->SeqLen_      = seqLen;
				gradFn->Batch_       = batch;
				gradFn->HasBias_     = (bHh != nullptr);
				scan.Out.MutAutograd().GradFn = gradFn;
			}
			layerInput = scan.Out;
		} else {
			auto hidden = Layers_[layerIndex]->ZeroState(batch);
			OaVec<OaMatrix> outputs;
			outputs.Reserve(seqLen);
			for (OaI32 time = 0; time < seqLen; ++time) {
				hidden = Layers_[layerIndex]->StepWithGi(
					giAll, hidden, static_cast<OaU32>(time), static_cast<OaU32>(seqLen));
				outputs.PushBack(OaFnMatrix::Reshape(
					hidden, OaMatrixShape{batch, 1, HiddenSize_}));
			}
			layerInput = OaFnMatrix::Concat(OaSpan<OaMatrix>(outputs), 1);
		}
	}

	return layerInput;
}
