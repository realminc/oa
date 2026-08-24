#pragma once

namespace oa {

// RnnCell / Rnn — vanilla Elman recurrent module.
//
//   h_new = tanh( W_ih x + b_ih + W_hh h + b_hh )
//
// Mirrors the Gru / GruCell API 1:1 (zeroState / step / forward, stacked
// layers) so the two recurrent modules are interchangeable in a model. Like the
// GRU it fuses its pointwise tail (Add + Tanh) into one RnnCellPointwise kernel.
// The two Linear projections stay separate dispatches; since oa::FnMatrix::Linear
// is pure dispatch, each needs a manual grad-node attach — see AttachLinearGrad
// in Rnn.cpp.

class RnnCell : public oa::Module {
public:
	RnnCell(oa::I32 inInputSize, oa::I32 inHiddenSize, bool inBias = true);

	[[nodiscard]] oa::I32 inputSize() const { return inputSize_; }
	[[nodiscard]] oa::I32 hiddenSize() const { return hiddenSize_; }
	[[nodiscard]] oa::Matrix zeroState(oa::I32 inBatch) const;
	[[nodiscard]] oa::Matrix step(const oa::Matrix& inInput, const oa::Matrix& inHidden);

	// split projections so Rnn can hoist the (recurrence-free) input projection
	// out of the timestep loop into one batched GEMM.
	//   inputProjection: gi = Linear(x, W_ih, b_ih) for any row count → [*, H].
	//   stepWithGi:      consumes a precomputed gi [B*T, H] at row offset timeOffset
	//                    and runs only the recurrent gh = Linear(h, W_hh) + fused
	//                    tanh pointwise (no per-step Slice).
	[[nodiscard]] oa::Matrix inputProjection(const oa::Matrix& inInput);
	[[nodiscard]] oa::Matrix stepWithGi(const oa::Matrix& inGi, const oa::Matrix& inHidden, oa::U32 inTimeOffset = 0, oa::U32 inBatchStride = 1);
	oa::Matrix forward(const oa::Matrix& inInput) override;

private:
	oa::I32 inputSize_;
	oa::I32 hiddenSize_;
	bool hasBias_;
};

class Rnn : public oa::Module {
public:
	Rnn(oa::I32 inInputSize, oa::I32 inHiddenSize, oa::I32 inNumLayers = 1, bool inBias = true);

	[[nodiscard]] oa::I32 inputSize() const { return inputSize_; }
	[[nodiscard]] oa::I32 hiddenSize() const { return hiddenSize_; }
	[[nodiscard]] oa::I32 numLayers() const { return numLayers_; }
	[[nodiscard]] oa::Matrix zeroState(oa::I32 inBatch, oa::I32 inLayer = 0) const;
	[[nodiscard]] oa::Matrix step(const oa::Matrix& inInput, oa::Matrix& inOutHidden, oa::I32 inLayer = 0);
	oa::Matrix forward(const oa::Matrix& inInput) override;

private:
	oa::I32 inputSize_;
	oa::I32 hiddenSize_;
	oa::I32 numLayers_;
	bool hasBias_;
	oa::Vec<oa::SharedPtr<RnnCell>> layers_;
};

} // namespace oa
