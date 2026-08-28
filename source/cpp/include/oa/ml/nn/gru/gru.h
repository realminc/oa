#pragma once

namespace oa {

class GruCell : public oa::Module {
public:
	GruCell(oa::I32 inInputSize, oa::I32 inHiddenSize, bool inBias = true);

	[[nodiscard]] oa::I32 inputSize() const { return inputSize_; }
	[[nodiscard]] oa::I32 hiddenSize() const { return hiddenSize_; }
	[[nodiscard]] oa::Matrix zeroState(oa::I32 inBatch) const;
	[[nodiscard]] oa::Matrix step(const oa::Matrix& inInput, const oa::Matrix& inHidden);

	// split projections so Gru can hoist the (recurrence-free) input projection
	// out of the timestep loop into one batched GEMM.
	//   inputProjection: gatesI = Linear(x, W_ih, b_ih) for any row count → [*, 3H].
	//   stepWithGatesI:  consumes a precomputed gatesI [B*T, 3H] at row offset
	//                    timeOffset and runs only the recurrent gatesH + pointwise.
	[[nodiscard]] oa::Matrix inputProjection(const oa::Matrix& inInput);
	[[nodiscard]] oa::Matrix stepWithGatesI(const oa::Matrix& inGatesI, const oa::Matrix& inHidden, oa::U32 inTimeOffset = 0, oa::U32 inBatchStride = 1);
	oa::Matrix forward(const oa::Matrix& inInput) override;

private:
	oa::I32 inputSize_;
	oa::I32 hiddenSize_;
	bool hasBias_;
};

class Gru : public oa::Module {
public:
	Gru(oa::I32 inInputSize, oa::I32 inHiddenSize, oa::I32 inNumLayers = 1, bool inBias = true);

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
	oa::Vector<oa::SharedPtr<GruCell>> layers_;
};

} // namespace oa
