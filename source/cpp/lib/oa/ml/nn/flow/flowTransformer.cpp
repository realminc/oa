#include <oa/ml/nn/flow/flowTransformer.h>
#include <oa/core/assert.h>
#include <oa/core/std/format.h>

namespace oa {

namespace {

void validate(const FlowTransformerConfig& inConfig) {
	OA_REQUIRE_MSG(inConfig.dModel > 0 && inConfig.hiddenDim > 0
		&& inConfig.sequenceLength > 0 && inConfig.numLayers > 0
		&& inConfig.numHeads > 0
		&& (inConfig.dModel % inConfig.numHeads) == 0
		&& inConfig.epsilon > 0.0F,
		"FlowTransformer requires positive dimensions, layers and epsilon, with dModel divisible by numHeads");
	OA_REQUIRE_MSG(inConfig.numExperts >= 0 && inConfig.expertsPerToken >= 0
		&& ((inConfig.numExperts == 0 && inConfig.expertsPerToken == 0)
			|| (inConfig.numExperts > 0 && inConfig.expertsPerToken > 0
				&& inConfig.expertsPerToken <= inConfig.numExperts)),
		"FlowTransformer MoE requires 0/0 experts for dense or 0 < expertsPerToken <= numExperts");
}

} // namespace

FlowTransformer::FlowTransformer(const FlowTransformerConfig& inConfig)	: config_(inConfig) {
	validate(config_);
	blocks_.reserve(config_.numLayers);
	for (oa::I32 index = 0; index < config_.numLayers; ++index) {
		oa::SharedPtr<oa::TransformerBlock> block;
		if (isMoe()) {
			block = oa::makeShared<oa::TransformerBlock>(
				config_.dModel, config_.hiddenDim, config_.sequenceLength,
				config_.numHeads, config_.numExperts, config_.expertsPerToken,
				config_.epsilon
			);
		} else {
			block = oa::makeShared<oa::TransformerBlock>(
				config_.dModel, config_.hiddenDim, config_.sequenceLength,
				config_.numHeads, config_.epsilon
			);
		}
		block->setAttentionMode(oa::AttentionMode::Bidirectional);
		if (config_.adaptiveConditioning) {
			block->enableAdaptiveConditioning(config_.dModel);
		}
		const oa::String name = oa::format("block%d", index);
		registerModule(name.cStr(), block);
		blocks_.pushBack(oa::move(block));
	}
	outputNorm_ = oa::makeShared<oa::LayerNorm>(config_.dModel, config_.epsilon);
	registerModule("output_norm", outputNorm_);
}

oa::Matrix FlowTransformer::forward(const oa::Matrix& inTokens) {
	return forwardImpl(inTokens, nullptr, nullptr);
}

oa::Matrix FlowTransformer::forwardMasked(const oa::Matrix& inTokens,	const oa::Matrix& inTokenMask) {
	return forwardImpl(inTokens, &inTokenMask, nullptr);
}

oa::Matrix FlowTransformer::forwardConditioned(
	const oa::Matrix& inTokens,
	const oa::Matrix& inCondition,
	const oa::Matrix& inTokenMask) {
	return forwardImpl(inTokens,inTokenMask.isEmpty() ? nullptr : &inTokenMask, &inCondition);
}

oa::Matrix FlowTransformer::forwardImpl(const oa::Matrix& inTokens,	const oa::Matrix* inTokenMask, const oa::Matrix* inCondition) {
	OA_REQUIRE_MSG(inTokens.rank() == 2 || inTokens.rank() == 3,
		"FlowTransformer expects [B*S,D] or [B,S,D] tokens");
	const bool batched = inTokens.rank() == 3;
	const oa::I64 rows = batched ? inTokens.size(0) * inTokens.size(1) : inTokens.size(0);
	const oa::I64 sequence = batched ? inTokens.size(1) : config_.sequenceLength;
	const oa::I64 features = inTokens.size(inTokens.rank() - 1);
	OA_REQUIRE_MSG(sequence == config_.sequenceLength
		&& features == config_.dModel
		&& rows % config_.sequenceLength == 0,
		"FlowTransformer token shape does not match configured sequence length and model dimension");

	const oa::I64 batch = rows / config_.sequenceLength;
	if (inCondition) {
		OA_REQUIRE_MSG(config_.adaptiveConditioning && inCondition->rank() == 2
			&& inCondition->size(0) == batch
			&& inCondition->size(1) == config_.dModel
			&& inCondition->getDtype() == inTokens.getDtype(),
			"FlowTransformer condition must match enabled [B,dModel] adaptive conditioning");
	}
	oa::Matrix additiveMask;
	if (inTokenMask) {
		const bool mask2 = inTokenMask->rank() == 2
			&& inTokenMask->size(0) == batch
			&& inTokenMask->size(1) == config_.sequenceLength;
		const bool mask3 = inTokenMask->rank() == 3
			&& inTokenMask->size(0) == batch
			&& inTokenMask->size(1) == config_.sequenceLength
			&& inTokenMask->size(2) == 1;
		OA_REQUIRE_MSG((mask2 || mask3)
			&& inTokenMask->getDtype() == inTokens.getDtype(),
			"FlowTransformer token mask must match [B,S] or [B,S,1] and token dtype");
		auto keyMask = inTokenMask->reshape(oa::MatrixShape{
			batch, 1, config_.sequenceLength});
		keyMask = (keyMask - 1.0F) * 1.0e4F;
		additiveMask = oa::FnMatrix::repeatInterleave(
			keyMask, config_.numHeads * config_.sequenceLength, 1)
			.reshape(oa::MatrixShape{
				batch * config_.numHeads * config_.sequenceLength,
				config_.sequenceLength}
			);
	}

	auto output = batched
		? inTokens.reshape(oa::MatrixShape{rows, config_.dModel}) : inTokens;
	for (auto& block : blocks_) {
		if (inCondition) {
			output = block->forwardConditioned(output, *inCondition, additiveMask);
		} else {
			output = inTokenMask
				? block->forwardMasked(output, additiveMask)
				: block->forward(output);
		}
	}
	output = outputNorm_->forward(output);
	return batched ? output.reshape(inTokens.getShape()) : output;
}

void FlowTransformer::setSequenceLength(oa::I32 inSequenceLength) {
	OA_REQUIRE_MSG(inSequenceLength > 0,
		"FlowTransformer sequence length must be positive");
	if (config_.sequenceLength == inSequenceLength) return;
	config_.sequenceLength = inSequenceLength;
	for (auto& block : blocks_) block->setSeqLen(inSequenceLength);
}

oa::TransformerBlock& FlowTransformer::block(oa::I32 inIndex) {
	OA_REQUIRE_MSG(inIndex >= 0 && inIndex < config_.numLayers,
		"FlowTransformer block index is out of range");
	return *blocks_[static_cast<oa::Usize>(inIndex)];
}

const oa::TransformerBlock& FlowTransformer::block(oa::I32 inIndex) const {
	OA_REQUIRE_MSG(inIndex >= 0 && inIndex < config_.numLayers,
		"FlowTransformer block index is out of range");
	return *blocks_[static_cast<oa::Usize>(inIndex)];
}

} // namespace oa
