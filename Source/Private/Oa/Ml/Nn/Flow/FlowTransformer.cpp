#include <Oa/Ml/Nn/Flow/FlowTransformer.h>

#include <stdexcept>

namespace {

void Validate(const OaFlowTransformerConfig& InConfig) {
	if (InConfig.DModel <= 0 || InConfig.HiddenDim <= 0
		|| InConfig.SequenceLength <= 0 || InConfig.NumLayers <= 0
		|| InConfig.NumHeads <= 0
		|| (InConfig.DModel % InConfig.NumHeads) != 0
		|| InConfig.Epsilon <= 0.0F) {
		throw std::invalid_argument("FlowTransformer requires positive dimensions, layers and epsilon, with DModel divisible by NumHeads");
	}
	if (InConfig.NumExperts < 0 || InConfig.ExpertsPerToken < 0
		|| (InConfig.NumExperts == 0 && InConfig.ExpertsPerToken != 0)
		|| (InConfig.NumExperts > 0	&& (InConfig.ExpertsPerToken == 0	|| InConfig.ExpertsPerToken > InConfig.NumExperts))) {
		throw std::invalid_argument("FlowTransformer MoE requires 0/0 experts for dense or 0 < ExpertsPerToken <= NumExperts");
	}
}

} // namespace

OaFlowTransformer::OaFlowTransformer(const OaFlowTransformerConfig& InConfig)	: Config_(InConfig) {
	Validate(Config_);
	Blocks_.Reserve(Config_.NumLayers);
	for (OaI32 index = 0; index < Config_.NumLayers; ++index) {
		OaSharedPtr<OaTransformerBlock> block;
		if (IsMoe()) {
			block = OaMakeSharedPtr<OaTransformerBlock>(
				Config_.DModel, Config_.HiddenDim, Config_.SequenceLength,
				Config_.NumHeads, Config_.NumExperts, Config_.ExpertsPerToken,
				Config_.Epsilon
			);
		} else {
			block = OaMakeSharedPtr<OaTransformerBlock>(
				Config_.DModel, Config_.HiddenDim, Config_.SequenceLength,
				Config_.NumHeads, Config_.Epsilon
			);
		}
		block->SetAttentionMode(OaAttentionMode::Bidirectional);
		if (Config_.AdaptiveConditioning) {
			block->EnableAdaptiveConditioning(Config_.DModel);
		}
		RegisterModule((OaString("block") + std::to_string(index)).c_str(), block);
		Blocks_.PushBack(OaStdMove(block));
	}
	OutputNorm_ = OaMakeSharedPtr<OaLayerNorm>(Config_.DModel, Config_.Epsilon);
	RegisterModule("output_norm", OutputNorm_);
}

OaMatrix OaFlowTransformer::Forward(const OaMatrix& InTokens) {
	return ForwardImpl(InTokens, nullptr, nullptr);
}

OaMatrix OaFlowTransformer::ForwardMasked(const OaMatrix& InTokens,	const OaMatrix& InTokenMask) {
	return ForwardImpl(InTokens, &InTokenMask, nullptr);
}

OaMatrix OaFlowTransformer::ForwardConditioned(
	const OaMatrix& InTokens,
	const OaMatrix& InCondition,
	const OaMatrix& InTokenMask) {
	return ForwardImpl(InTokens,InTokenMask.IsEmpty() ? nullptr : &InTokenMask, &InCondition);
}

OaMatrix OaFlowTransformer::ForwardImpl(const OaMatrix& InTokens,	const OaMatrix* InTokenMask, const OaMatrix* InCondition) {
	if (InTokens.Rank() != 2 && InTokens.Rank() != 3) {
		throw std::invalid_argument("FlowTransformer expects [B*S,D] or [B,S,D] tokens");
	}
	const bool batched = InTokens.Rank() == 3;
	const OaI64 rows = batched ? InTokens.Size(0) * InTokens.Size(1) : InTokens.Size(0);
	const OaI64 sequence = batched ? InTokens.Size(1) : Config_.SequenceLength;
	const OaI64 features = InTokens.Size(InTokens.Rank() - 1);
	if (sequence != Config_.SequenceLength || features != Config_.DModel
		|| rows % Config_.SequenceLength != 0) {
		throw std::invalid_argument(
			"FlowTransformer token shape does not match configured sequence length and model dimension");
	}

	const OaI64 batch = rows / Config_.SequenceLength;
	if (InCondition) {
		if (!Config_.AdaptiveConditioning || InCondition->Rank() != 2
			|| InCondition->Size(0) != batch
			|| InCondition->Size(1) != Config_.DModel
			|| InCondition->GetDtype() != InTokens.GetDtype()
		) {
			throw std::invalid_argument("FlowTransformer condition must match enabled [B,DModel] adaptive conditioning");
		}
	}
	OaMatrix additiveMask;
	if (InTokenMask) {
		const bool mask2 = InTokenMask->Rank() == 2
			&& InTokenMask->Size(0) == batch
			&& InTokenMask->Size(1) == Config_.SequenceLength;
		const bool mask3 = InTokenMask->Rank() == 3
			&& InTokenMask->Size(0) == batch
			&& InTokenMask->Size(1) == Config_.SequenceLength
			&& InTokenMask->Size(2) == 1;
		if ((!mask2 && !mask3)
			|| InTokenMask->GetDtype() != InTokens.GetDtype()) {
			throw std::invalid_argument(
				"FlowTransformer token mask must match [B,S] or [B,S,1] and token dtype");
		}
		auto keyMask = InTokenMask->Reshape(OaMatrixShape{
			batch, 1, Config_.SequenceLength});
		keyMask = (keyMask - 1.0F) * 1.0e4F;
		additiveMask = OaFnMatrix::RepeatInterleave(
			keyMask, Config_.NumHeads * Config_.SequenceLength, 1)
			.Reshape(OaMatrixShape{
				batch * Config_.NumHeads * Config_.SequenceLength,
				Config_.SequenceLength}
			);
	}

	auto output = batched
		? InTokens.Reshape(OaMatrixShape{rows, Config_.DModel}) : InTokens;
	for (auto& block : Blocks_) {
		if (InCondition) {
			output = block->ForwardConditioned(output, *InCondition, additiveMask);
		} else {
			output = InTokenMask
				? block->ForwardMasked(output, additiveMask)
				: block->Forward(output);
		}
	}
	output = OutputNorm_->Forward(output);
	return batched ? output.Reshape(InTokens.GetShape()) : output;
}

void OaFlowTransformer::SetSequenceLength(OaI32 InSequenceLength) {
	if (InSequenceLength <= 0) {
		throw std::invalid_argument("FlowTransformer sequence length must be positive");
	}
	if (Config_.SequenceLength == InSequenceLength) return;
	Config_.SequenceLength = InSequenceLength;
	for (auto& block : Blocks_) block->SetSeqLen(InSequenceLength);
}

OaTransformerBlock& OaFlowTransformer::Block(OaI32 InIndex) {
	if (InIndex < 0 || InIndex >= Config_.NumLayers) {
		throw std::out_of_range("FlowTransformer block index is out of range");
	}
	return *Blocks_[static_cast<size_t>(InIndex)];
}

const OaTransformerBlock& OaFlowTransformer::Block(OaI32 InIndex) const {
	if (InIndex < 0 || InIndex >= Config_.NumLayers) {
		throw std::out_of_range("FlowTransformer block index is out of range");
	}
	return *Blocks_[static_cast<size_t>(InIndex)];
}
