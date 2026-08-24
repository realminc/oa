#include <oa/ml/nn.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/envFlag.h>
#include <oa/core/log.h>

#include <cassert>
#include <cstdio>

oa::RnnCell::RnnCell(oa::I32 inInputSize, oa::I32 inHiddenSize, bool inBias)
	: inputSize_(inInputSize)
	, hiddenSize_(inHiddenSize)
	, hasBias_(inBias)
{
	auto wd = oa::FnMatrix::weightDtype();
	registerParameter("weight_ih", oa::FnMatrix::randXavier(oa::MatrixShape{hiddenSize_, inputSize_}, wd));
	registerParameter("weight_hh", oa::FnMatrix::randXavier(oa::MatrixShape{hiddenSize_, hiddenSize_}, wd));
	if (hasBias_) {
		registerParameter("bias_ih", oa::FnMatrix::zeros(oa::MatrixShape{hiddenSize_}, wd));
		registerParameter("bias_hh", oa::FnMatrix::zeros(oa::MatrixShape{hiddenSize_}, wd));
	}
}

oa::Matrix oa::RnnCell::zeroState(oa::I32 inBatch) const {
	return oa::FnMatrix::zeros(oa::MatrixShape{inBatch, hiddenSize_}, params_[0].data.getDtype());
}

// input projection only: gi = Linear(x, W_ih, b_ih). No recurrent dependency, so
// oa::Rnn hoists this over the whole sequence as one batched GEMM ([B*T, in] @ W_ih
// → [B*T, H]) instead of T tiny per-timestep GEMMs.
oa::Matrix oa::RnnCell::inputProjection(const oa::Matrix& inInput) {
	const oa::Matrix& weightIh = params_[0].data;
	const oa::Matrix* biasIh = hasBias_ ? &params_[2].data : nullptr;
	return biasIh
		? oa::FnMatrix::linear(inInput, weightIh, *biasIh)
		: oa::FnMatrix::linear(inInput, weightIh);
}

// Recurrent half: consumes a precomputed gi [B*T, H] at row offset timeOffset,
// computes the hidden projection gh = Linear(h, W_hh), and applies the fused tanh
// pointwise: h_new = tanh(gi + gh) in a single dispatch. timeOffset indexes into the
// full [B*T, H] gi buffer without a per-step Slice dispatch.
oa::Matrix oa::RnnCell::stepWithGi(const oa::Matrix& inGi, const oa::Matrix& inHidden, oa::U32 inTimeOffset, oa::U32 inBatchStride) {
	const oa::Matrix& weightHh = params_[1].data;
	const oa::Matrix* biasHh = hasBias_ ? &params_[3].data : nullptr;

	return oa::FnMatrix::rnnCellLinear(
		inGi, inHidden, weightHh, inTimeOffset, inBatchStride,
		biasHh ? *biasHh : oa::Matrix{});
}

oa::Matrix oa::RnnCell::step(const oa::Matrix& inInput, const oa::Matrix& inHidden) {
	// Single-token convenience: project this one token then run the recurrent half.
	return stepWithGi(inputProjection(inInput), inHidden);
}

oa::Matrix oa::RnnCell::forward(const oa::Matrix& inInput) {
	return step(inInput, zeroState(static_cast<oa::I32>(inInput.size(0))));
}

oa::Rnn::Rnn(oa::I32 inInputSize, oa::I32 inHiddenSize, oa::I32 inNumLayers, bool inBias)
	: inputSize_(inInputSize)
	, hiddenSize_(inHiddenSize)
	, numLayers_(inNumLayers)
	, hasBias_(inBias)
{
	for (oa::I32 i = 0; i < numLayers_; ++i) {
		const oa::I32 layerInputSize = (i == 0) ? inputSize_ : hiddenSize_;
		auto layer = oa::makeShared<oa::RnnCell>(layerInputSize, hiddenSize_, hasBias_);
		layers_.pushBack(layer);
		char layerName[32];
		std::snprintf(layerName, sizeof(layerName), "layer%d", i);
		registerModule(layerName, layer);
	}
}

oa::Matrix oa::Rnn::zeroState(oa::I32 inBatch, oa::I32 inLayer) const {
	return layers_[inLayer]->zeroState(inBatch);
}

oa::Matrix oa::Rnn::step(const oa::Matrix& inInput, oa::Matrix& inOutHidden, oa::I32 inLayer) {
	inOutHidden = layers_[inLayer]->step(inInput, inOutHidden);
	return inOutHidden;
}

oa::Matrix oa::Rnn::forward(const oa::Matrix& inInput) {
	// input shape: [batch, seq_len, input_size]. load-bearing contract: the timestep
	// loop slices dim 1, so a flattened rank-2 [batch*seq, input] (what oa::Embedding/
	// Gather returns) would be read as batch=batch*seq, seq=input_size — wrong axis,
	// O(seq^2) buffers, garbage grads. NDEBUG strips asserts, so guard at runtime.
	if (inInput.rank() != 3) {
		OaLogError(oa::LogComponent::Ml,
			"oa::Rnn::forward expects rank-3 [batch, seq, input], got rank=%d — "
			"reshape the embedding output (Gather flattens to [batch*seq, embed])",
			inInput.rank());
		assert(false && "oa::Rnn::forward requires rank-3 [batch, seq, input]");
		// release strips the assert — do NOT fall through to the timestep loop, which
		// would slice the wrong axis into O(seq^2) garbage with bad gradients. Return
		// empty: a loud, non-corrupting failure that errors cleanly downstream.
		return {};
	}
	const oa::I32 batch = static_cast<oa::I32>(inInput.size(0));
	const oa::I32 seqLen = static_cast<oa::I32>(inInput.size(1));
	oa::Matrix layerInput = inInput;
	const bool useScan = not oa::EnvFlag::isSet("OA_DISABLE_RNN_SCAN");

	for (oa::I32 layerIndex = 0; layerIndex < numLayers_; ++layerIndex) {
		const oa::I32 inputDim = (layerIndex == 0) ? inputSize_ : hiddenSize_;

		// Precompute the input projection for the WHOLE sequence in one batched GEMM:
		// [batch*seqLen, inputDim] @ W_ih → [batch*seqLen, H]. No recurrent dependency,
		// so this replaces seqLen tiny per-timestep gEMMs (the dispatch starvation that
		// pinned recurrent CPU overhead near the launch ceiling) with one large GEMM.
		// Only W_hh @ h_t stays in the loop. Slicing rows of the batched result equals
		// the per-row GEMM — numerics and gradients are unchanged.
		auto flatInput = layerInput.reshape(oa::MatrixShape{static_cast<oa::I64>(batch) * seqLen, inputDim});
		auto giAll     = layers_[layerIndex]->inputProjection(flatInput);      // [B*T, H]
		const oa::Matrix& wHh = layers_[layerIndex]->parameters()[1].data;
		const oa::Matrix* bHh = hasBias_ ? &layers_[layerIndex]->parameters()[3].data : nullptr;

		if (useScan) {
			auto scan = oa::FnMatrix::rnnScan(
				giAll, wHh, hiddenSize_, seqLen, batch,
				bHh ? *bHh : oa::Matrix{});
			layerInput = scan.out;
		} else {
			auto hidden = layers_[layerIndex]->zeroState(batch);
			oa::Vec<oa::Matrix> outputs;
			outputs.reserve(seqLen);
			for (oa::I32 time = 0; time < seqLen; ++time) {
				hidden = layers_[layerIndex]->stepWithGi(
					giAll, hidden, static_cast<oa::U32>(time), static_cast<oa::U32>(seqLen));
				outputs.pushBack(oa::FnMatrix::reshape(
					hidden, oa::MatrixShape{batch, 1, hiddenSize_}));
			}
			layerInput = oa::FnMatrix::concat(oa::Span<oa::Matrix>(outputs), 1);
		}
	}

	return layerInput;
}
