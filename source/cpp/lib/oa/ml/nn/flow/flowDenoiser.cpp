#include <oa/ml/nn/flow/flowDenoiser.h>

#include <cmath>
#include <stdexcept>

namespace oa {

namespace {

void validate(const FlowDenoiserConfig& inConfig) {
	if (inConfig.inputDim <= 0 || inConfig.conditionDim < 0
		|| inConfig.backbone.dModel <= 0
		|| inConfig.backbone.sequenceLength <= 0
		|| inConfig.timeMaxPeriod <= 1.0F || inConfig.timeScale <= 0.0F
		|| inConfig.conditionDropoutP < 0.0F
		|| inConfig.conditionDropoutP >= 1.0F) {
		throw std::invalid_argument(
			"FlowDenoiser requires positive input/model/sequence/time dimensions and non-negative condition dimension");
	}
}

} // namespace

FlowDenoiser::FlowDenoiser(const FlowDenoiserConfig& inConfig)
	: config_(inConfig) {
	validate(config_);
	inputProjection_ = oa::makeShared<oa::Linear>(
		config_.inputDim, config_.backbone.dModel);
	timeEmbedding_ = oa::makeShared<FlowTimeEmbedding>(
		config_.backbone.dModel, config_.timeMaxPeriod, config_.timeScale);
	if (config_.conditionDim > 0) {
		conditionProjection_ = oa::makeShared<oa::Linear>(
			config_.conditionDim, config_.backbone.dModel);
	}
	backbone_ = oa::makeShared<FlowTransformer>(config_.backbone);
	outputProjection_ = oa::makeShared<oa::Linear>(
		config_.backbone.dModel, config_.inputDim);

	registerModule("input_projection", inputProjection_);
	registerModule("time_embedding", timeEmbedding_);
	if (conditionProjection_) {
		registerModule("condition_projection", conditionProjection_);
	}
	registerModule("backbone", backbone_);
	registerModule("output_projection", outputProjection_);
	position_ = oa::FnMatrix::randN(oa::MatrixShape{
		config_.backbone.sequenceLength, config_.backbone.dModel},
		oa::FnMatrix::weightDtype()) * 0.02F;
	registerParameter("position", position_);
}

oa::Matrix FlowDenoiser::forward(const oa::Matrix& inSample) {
	if (inSample.rank() != 3) {
		throw std::invalid_argument("FlowDenoiser expects [B,S,inputDim]");
	}
	auto time = oa::FnMatrix::zeros(
		oa::MatrixShape{inSample.size(0), 1}, inSample.getDtype());
	oa::Matrix condition;
	if (config_.conditionDim > 0) {
		condition = oa::FnMatrix::zeros(
			oa::MatrixShape{inSample.size(0), config_.conditionDim},
			inSample.getDtype());
	}
	return forwardConditioned(inSample, time, condition);
}

oa::Matrix FlowDenoiser::forwardConditioned(
	const oa::Matrix& inSample,
	const oa::Matrix& inTime,
	const oa::Matrix& inCondition,
	const oa::Matrix& inTokenMask) {
	if (inSample.rank() != 3
		|| inSample.size(1) != config_.backbone.sequenceLength
		|| inSample.size(2) != config_.inputDim) {
		throw std::invalid_argument(
			"FlowDenoiser sample must match configured [B,S,inputDim]");
	}
	const oa::I64 batch = inSample.size(0);
	if (inTime.rank() < 1 || inTime.rank() > 2
		|| inTime.size(0) != batch
		|| (inTime.rank() == 2 && inTime.size(1) != 1)) {
		throw std::invalid_argument("FlowDenoiser time must be [B] or [B,1]");
	}
	if (config_.conditionDim == 0) {
		if (!inCondition.isEmpty()) {
			throw std::invalid_argument(
				"FlowDenoiser was configured without condition features");
		}
	} else if (inCondition.rank() != 2 || inCondition.size(0) != batch
		|| inCondition.size(1) != config_.conditionDim) {
		throw std::invalid_argument(
			"FlowDenoiser condition must match configured [B,conditionDim]");
	}

	auto rows = inSample.reshape(oa::MatrixShape{
		batch * config_.backbone.sequenceLength, config_.inputDim});
	auto tokens = inputProjection_->forward(rows).reshape(oa::MatrixShape{
		batch, config_.backbone.sequenceLength, config_.backbone.dModel});
	tokens = tokens + position_.reshape(oa::MatrixShape{
		1, config_.backbone.sequenceLength, config_.backbone.dModel});
	auto time = timeEmbedding_->forward(inTime).reshape(oa::MatrixShape{
		batch, 1, config_.backbone.dModel});
	auto context = time;
	if (conditionProjection_) {
		auto conditionInput = inCondition;
		if (isTraining() && config_.conditionDropoutP > 0.0F) {
			auto keep = oa::FnMatrix::greaterEqual(
				oa::FnMatrix::philoxUniform(oa::FnMatrix::empty(
					oa::MatrixShape{batch, 1}, inCondition.getDtype()),
					0.0F, 1.0F, 0),
				config_.conditionDropoutP);
			conditionInput = conditionInput * keep;
		}
		auto condition = conditionProjection_->forward(conditionInput).reshape(
			oa::MatrixShape{batch, 1, config_.backbone.dModel});
		context = context + condition;
	}
	auto hidden = backbone_->forwardConditioned(
		tokens, context.reshape(oa::MatrixShape{batch, config_.backbone.dModel}),
		inTokenMask).reshape(oa::MatrixShape{
		batch * config_.backbone.sequenceLength, config_.backbone.dModel});
	return outputProjection_->forward(hidden).reshape(inSample.getShape());
}

oa::Matrix FlowDenoiser::forwardGuided(
	const oa::Matrix& inSample,
	const oa::Matrix& inTime,
	const oa::Matrix& inCondition,
	oa::F32 inGuidanceScale,
	const oa::Matrix& inTokenMask) {
	if (!conditionProjection_ || !std::isfinite(inGuidanceScale)
		|| inGuidanceScale < 0.0F) {
		throw std::invalid_argument(
			"FlowDenoiser guidance requires configured conditions and a finite non-negative scale");
	}
	oa::Module::ScopedEval eval(*this);
	auto unconditional = forwardConditioned(
		inSample, inTime,
		oa::FnMatrix::zeros(inCondition.getShape(), inCondition.getDtype()),
		inTokenMask);
	auto conditional = forwardConditioned(
		inSample, inTime, inCondition, inTokenMask);
	return unconditional
		+ (conditional - unconditional) * inGuidanceScale;
}

} // namespace oa
