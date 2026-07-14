#pragma once

class OaGruCell : public OaModule {
public:
	OaGruCell(OaI32 InInputSize, OaI32 InHiddenSize, bool InBias = true);

	[[nodiscard]] OaI32 InputSize() const { return InputSize_; }
	[[nodiscard]] OaI32 HiddenSize() const { return HiddenSize_; }
	[[nodiscard]] OaMatrix ZeroState(OaI32 InBatch) const;
	[[nodiscard]] OaMatrix Step(const OaMatrix& InInput, const OaMatrix& InHidden);

	// Split projections so OaGru can hoist the (recurrence-free) input projection
	// out of the timestep loop into one batched GEMM.
	//   InputProjection: gatesI = Linear(x, W_ih, b_ih) for any row count → [*, 3H].
	//   StepWithGatesI:  consumes a precomputed gatesI [B*T, 3H] at row offset
	//                    timeOffset and runs only the recurrent gatesH + pointwise.
	[[nodiscard]] OaMatrix InputProjection(const OaMatrix& InInput);
	[[nodiscard]] OaMatrix StepWithGatesI(const OaMatrix& InGatesI, const OaMatrix& InHidden, OaU32 InTimeOffset = 0, OaU32 InBatchStride = 1);
	OaMatrix Forward(const OaMatrix& InInput) override;

private:
	OaI32 InputSize_;
	OaI32 HiddenSize_;
	bool HasBias_;
};

class OaGru : public OaModule {
public:
	OaGru(OaI32 InInputSize, OaI32 InHiddenSize, OaI32 InNumLayers = 1, bool InBias = true);

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
	OaVec<OaSharedPtr<OaGruCell>> Layers_;
};
