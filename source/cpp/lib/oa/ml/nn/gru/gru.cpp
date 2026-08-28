#include <oa/ml/nn.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/envFlag.h>
#include <oa/core/log.h>
#include <oa/core/std/format.h>

#include <assert.h>

oa::GruCell::GruCell(oa::I32 inInputSize, oa::I32 inHiddenSize, bool inBias)
	: inputSize_(inInputSize)
	, hiddenSize_(inHiddenSize)
	, hasBias_(inBias) {
	auto wd = oa::FnMatrix::weightDtype();
	registerParameter("weight_ih", oa::FnMatrix::randXavier(oa::MatrixShape{3 * hiddenSize_, inputSize_}, wd));
	registerParameter("weight_hh", oa::FnMatrix::randXavier(oa::MatrixShape{3 * hiddenSize_, hiddenSize_}, wd));
	if (hasBias_) {
		registerParameter("bias_ih", oa::FnMatrix::zeros(oa::MatrixShape{3 * hiddenSize_}, wd));
		registerParameter("bias_hh", oa::FnMatrix::zeros(oa::MatrixShape{3 * hiddenSize_}, wd));
	}
}

oa::Matrix oa::GruCell::zeroState(oa::I32 inBatch) const {
	return oa::FnMatrix::zeros(oa::MatrixShape{inBatch, hiddenSize_}, params_[0].data.getDtype());
}

// input projection only: gatesI = Linear(x, W_ih, b_ih). Has NO recurrent
// dependency, so oa::Gru hoists this over the whole sequence as one batched GEMM
// ([B*T, in] @ W_ih → [B*T, 3H]) instead of T tiny per-timestep GEMMs. Linear is
// the projection; the per-step recurrent adjoint scatters its grad back.
oa::Matrix oa::GruCell::inputProjection(const oa::Matrix& inInput) {
	const oa::Matrix& weightIh = params_[0].data;
	const oa::Matrix* biasIh = hasBias_ ? &params_[2].data : nullptr;
	return biasIh
		? oa::FnMatrix::linear(inInput, weightIh, *biasIh)
		: oa::FnMatrix::linear(inInput, weightIh);
}

// Recurrent half of the cell: consumes a precomputed gatesI [B*T, 3H] at row
// offset timeOffset, computes the hidden projection gatesH = Linear(h, W_hh),
// and applies the fused pointwise update in a single dispatch. timeOffset
// indexes into the full [B*T, 3H] gatesI buffer without a Slice dispatch.
oa::Matrix oa::GruCell::stepWithGatesI(const oa::Matrix& inGatesI, const oa::Matrix& inHidden, oa::U32 inTimeOffset, oa::U32 inBatchStride) {
	const oa::Matrix& weightHh = params_[1].data;
	const oa::Matrix* biasHh = hasBias_ ? &params_[3].data : nullptr;

	return oa::FnMatrix::gruCellLinear(
		inGatesI, inHidden, weightHh, hiddenSize_, inTimeOffset, inBatchStride,
		biasHh ? *biasHh : oa::Matrix{});
}

oa::Matrix oa::GruCell::step(const oa::Matrix& inInput, const oa::Matrix& inHidden) {
	// Single-token convenience: project this one token then run the recurrent half.
	return stepWithGatesI(inputProjection(inInput), inHidden);
}

oa::Matrix oa::GruCell::forward(const oa::Matrix& inInput) {
	return step(inInput, zeroState(static_cast<oa::I32>(inInput.size(0))));
}

oa::Gru::Gru(oa::I32 inInputSize, oa::I32 inHiddenSize, oa::I32 inNumLayers, bool inBias)
	: inputSize_(inInputSize)
	, hiddenSize_(inHiddenSize)
	, numLayers_(inNumLayers)
	, hasBias_(inBias)
{
	for (oa::I32 i = 0; i < numLayers_; ++i) {
		const oa::I32 layerInputSize = (i == 0) ? inputSize_ : hiddenSize_;
		auto layer = oa::makeShared<oa::GruCell>(layerInputSize, hiddenSize_, hasBias_);
		layers_.pushBack(layer);
		const oa::String layerName = oa::format("layer%d", i);
		registerModule(layerName.cStr(), layer);
	}
}

oa::Matrix oa::Gru::zeroState(oa::I32 inBatch, oa::I32 inLayer) const {
	return layers_[inLayer]->zeroState(inBatch);
}

oa::Matrix oa::Gru::step(const oa::Matrix& inInput, oa::Matrix& inOutHidden, oa::I32 inLayer) {
	inOutHidden = layers_[inLayer]->step(inInput, inOutHidden);
	return inOutHidden;
}

oa::Matrix oa::Gru::forward(const oa::Matrix& inInput) {
	// input shape: [batch, seq_len, input_size]. This contract is load-bearing: the
	// timestep loop slices dim 1, so a flattened rank-2 [batch*seq, input] (what
	// oa::Embedding/Gather returns) would be read as batch=batch*seq, seq=input_size —
	// slicing the wrong axis into O(seq^2) buffers with garbage gradients. NDEBUG
	// strips asserts, so guard loudly at runtime regardless of build type.
	if (inInput.rank() != 3) {
		OaLogError(oa::LogComponent::Ml,
			"oa::Gru::forward expects rank-3 [batch, seq, input], got rank=%d — "
			"reshape the embedding output (Gather flattens to [batch*seq, embed])",
			inInput.rank());
		assert(false && "oa::Gru::forward requires rank-3 [batch, seq, input]");
		// release strips the assert — do NOT fall through to the timestep loop, which
		// would slice the wrong axis into O(seq^2) garbage with bad gradients. Return
		// empty: a loud, non-corrupting failure that errors cleanly downstream.
		return {};
	}
	const oa::I32 batch = static_cast<oa::I32>(inInput.size(0));
	const oa::I32 seqLen = static_cast<oa::I32>(inInput.size(1));
	oa::Matrix layerInput = inInput;
	const bool useScan = not oa::EnvFlag::isSet("OA_DISABLE_GRU_SCAN");

	// process each layer
	for (oa::I32 layerIndex = 0; layerIndex < numLayers_; ++layerIndex) {
		const oa::I32 inputDim = (layerIndex == 0) ? inputSize_ : hiddenSize_;

		// Precompute the input projection for the WHOLE sequence in one batched GEMM:
		// [batch*seqLen, inputDim] @ W_ih → [batch*seqLen, 3H]. The input projection
		// has no recurrent dependency, so this replaces seqLen tiny per-timestep GEMMs
		// (the dispatch-starvation that pinned recurrent CPU overhead near the kernel
		// launch ceiling) with a single large one. Only W_hh @ h_t must stay recurrent.
		// Mathematically identical — slicing rows of the batched result equals the
		// per-row GEMM — so numerical gradients are unchanged.
		auto flatInput = layerInput.reshape(oa::MatrixShape{static_cast<oa::I64>(batch) * seqLen, inputDim});
		auto gatesIAll = layers_[layerIndex]->inputProjection(flatInput);   // [B*S, 3H]

		if (useScan) {
			// Whole-sequence recurrent scan: one dispatch runs every timestep
			// (h lives in groupshared across t). The backward is likewise one
			// BPTT dispatch + one LinearWeightBiasBwd. Collapses the S-dispatch
			// recurrent loop that dominated RNN/GRU CPU overhead.
			auto& cellParams = layers_[layerIndex]->parameters();
			const oa::Matrix& wHh = cellParams[1].data;           // weight_hh [3H, H]
			const oa::Matrix* bHh = hasBias_ ? &cellParams[3].data : nullptr;  // bias_hh [3H]
			auto scan = oa::FnMatrix::gruScan(
			 	gatesIAll, wHh, hiddenSize_, seqLen, batch,
			 	bHh ? *bHh : oa::Matrix{});
			layerInput = scan.out;   // [B, S, H]
		} else {
			// Compatibility route for drivers that cannot compile the large
			// groupshared GruScanBwd kernel. It retains the hoisted input GEMM,
			// but records the recurrent cell and its smaller backward kernels per
			// timestep. The environment switch is capability-oriented so mobile,
			// desktop, and future backends share the same implementation.
			auto hidden = layers_[layerIndex]->zeroState(batch);
			oa::Vector<oa::Matrix> outputs;
			outputs.reserve(seqLen);
			for (oa::I32 time = 0; time < seqLen; ++time) {
				hidden = layers_[layerIndex]->stepWithGatesI(
				 	gatesIAll, hidden, static_cast<oa::U32>(time), static_cast<oa::U32>(seqLen));
				outputs.pushBack(oa::FnMatrix::reshape(hidden, oa::MatrixShape{batch, 1, hiddenSize_}));
			}
			layerInput = oa::FnMatrix::concat(outputs.span(), 1);
		}
	}

	return layerInput;
}
