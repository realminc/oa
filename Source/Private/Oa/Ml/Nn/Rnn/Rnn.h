#pragma once

// OaRnnCell / OaRnn — vanilla Elman recurrent module.
//
//   h_new = tanh( W_ih x + b_ih + W_hh h + b_hh )
//
// Mirrors the OaGru / OaGruCell API 1:1 (ZeroState / Step / Forward, stacked
// layers) so the two recurrent modules are interchangeable in a model. Like the
// GRU it fuses its pointwise tail (Add + Tanh) into one RnnCellPointwise kernel.
// The two Linear projections stay separate dispatches; since OaFnMatrix::Linear
// is pure dispatch, each needs a manual grad-node attach — see AttachLinearGrad
// in Rnn.cpp.

class OaRnnCell : public OaModule {
public:
	OaRnnCell(OaI32 InInputSize, OaI32 InHiddenSize, bool InBias = true);

	[[nodiscard]] OaI32 InputSize() const { return InputSize_; }
	[[nodiscard]] OaI32 HiddenSize() const { return HiddenSize_; }
	[[nodiscard]] OaMatrix ZeroState(OaI32 InBatch) const;
	[[nodiscard]] OaMatrix Step(const OaMatrix& InInput, const OaMatrix& InHidden);

	// Split projections so OaRnn can hoist the (recurrence-free) input projection
	// out of the timestep loop into one batched GEMM.
	//   InputProjection: gi = Linear(x, W_ih, b_ih) for any row count → [*, H].
	//   StepWithGi:      consumes a precomputed gi [B*T, H] at row offset timeOffset
	//                    and runs only the recurrent gh = Linear(h, W_hh) + fused
	//                    tanh pointwise (no per-step Slice).
	[[nodiscard]] OaMatrix InputProjection(const OaMatrix& InInput);
	[[nodiscard]] OaMatrix StepWithGi(const OaMatrix& InGi, const OaMatrix& InHidden, OaU32 InTimeOffset = 0, OaU32 InBatchStride = 1);
	OaMatrix Forward(const OaMatrix& InInput) override;

private:
	OaI32 InputSize_;
	OaI32 HiddenSize_;
	bool HasBias_;
};

class OaRnn : public OaModule {
public:
	OaRnn(OaI32 InInputSize, OaI32 InHiddenSize, OaI32 InNumLayers = 1, bool InBias = true);

	[[nodiscard]] OaI32 InputSize() const { return InputSize_; }
	[[nodiscard]] OaI32 HiddenSize() const { return HiddenSize_; }
	[[nodiscard]] OaI32 NumLayers() const { return NumLayers_; }
	[[nodiscard]] OaMatrix ZeroState(OaI32 InBatch, OaI32 InLayer = 0) const;
	[[nodiscard]] OaMatrix Step(const OaMatrix& InInput, OaMatrix& InOutHidden, OaI32 InLayer = 0);
	OaMatrix Forward(const OaMatrix& InInput) override;

private:
	OaI32 InputSize_;
	OaI32 HiddenSize_;
	OaI32 NumLayers_;
	bool HasBias_;
	OaVec<OaSharedPtr<OaRnnCell>> Layers_;
};
