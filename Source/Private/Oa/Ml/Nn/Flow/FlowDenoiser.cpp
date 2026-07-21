#include "FlowDenoiser.h"

#include <cmath>
#include <stdexcept>

namespace {

void Validate(const OaFlowDenoiserConfig& InConfig) {
	if (InConfig.InputDim <= 0 || InConfig.ConditionDim < 0
		|| InConfig.Backbone.DModel <= 0
		|| InConfig.Backbone.SequenceLength <= 0
		|| InConfig.TimeMaxPeriod <= 1.0F || InConfig.TimeScale <= 0.0F
		|| InConfig.ConditionDropoutP < 0.0F
		|| InConfig.ConditionDropoutP >= 1.0F) {
		throw std::invalid_argument(
			"FlowDenoiser requires positive input/model/sequence/time dimensions and non-negative condition dimension");
	}
}

} // namespace

OaFlowDenoiser::OaFlowDenoiser(const OaFlowDenoiserConfig& InConfig)
	: Config_(InConfig) {
	Validate(Config_);
	InputProjection_ = OaMakeSharedPtr<OaLinear>(
		Config_.InputDim, Config_.Backbone.DModel);
	TimeEmbedding_ = OaMakeSharedPtr<OaFlowTimeEmbedding>(
		Config_.Backbone.DModel, Config_.TimeMaxPeriod, Config_.TimeScale);
	if (Config_.ConditionDim > 0) {
		ConditionProjection_ = OaMakeSharedPtr<OaLinear>(
			Config_.ConditionDim, Config_.Backbone.DModel);
	}
	Backbone_ = OaMakeSharedPtr<OaFlowTransformer>(Config_.Backbone);
	OutputProjection_ = OaMakeSharedPtr<OaLinear>(
		Config_.Backbone.DModel, Config_.InputDim);

	RegisterModule("input_projection", InputProjection_);
	RegisterModule("time_embedding", TimeEmbedding_);
	if (ConditionProjection_) {
		RegisterModule("condition_projection", ConditionProjection_);
	}
	RegisterModule("backbone", Backbone_);
	RegisterModule("output_projection", OutputProjection_);
	Position_ = OaFnMatrix::RandN(OaMatrixShape{
		Config_.Backbone.SequenceLength, Config_.Backbone.DModel},
		OaFnMatrix::GetWeightDtype()) * 0.02F;
	RegisterParameter("position", Position_);
}

OaMatrix OaFlowDenoiser::Forward(const OaMatrix& InSample) {
	if (InSample.Rank() != 3) {
		throw std::invalid_argument("FlowDenoiser expects [B,S,InputDim]");
	}
	auto time = OaFnMatrix::Zeros(
		OaMatrixShape{InSample.Size(0), 1}, InSample.GetDtype());
	OaMatrix condition;
	if (Config_.ConditionDim > 0) {
		condition = OaFnMatrix::Zeros(
			OaMatrixShape{InSample.Size(0), Config_.ConditionDim},
			InSample.GetDtype());
	}
	return ForwardConditioned(InSample, time, condition);
}

OaMatrix OaFlowDenoiser::ForwardConditioned(
	const OaMatrix& InSample,
	const OaMatrix& InTime,
	const OaMatrix& InCondition,
	const OaMatrix& InTokenMask) {
	if (InSample.Rank() != 3
		|| InSample.Size(1) != Config_.Backbone.SequenceLength
		|| InSample.Size(2) != Config_.InputDim) {
		throw std::invalid_argument(
			"FlowDenoiser sample must match configured [B,S,InputDim]");
	}
	const OaI64 batch = InSample.Size(0);
	if (InTime.Rank() < 1 || InTime.Rank() > 2
		|| InTime.Size(0) != batch
		|| (InTime.Rank() == 2 && InTime.Size(1) != 1)) {
		throw std::invalid_argument("FlowDenoiser time must be [B] or [B,1]");
	}
	if (Config_.ConditionDim == 0) {
		if (!InCondition.IsEmpty()) {
			throw std::invalid_argument(
				"FlowDenoiser was configured without condition features");
		}
	} else if (InCondition.Rank() != 2 || InCondition.Size(0) != batch
		|| InCondition.Size(1) != Config_.ConditionDim) {
		throw std::invalid_argument(
			"FlowDenoiser condition must match configured [B,ConditionDim]");
	}

	auto rows = InSample.Reshape(OaMatrixShape{
		batch * Config_.Backbone.SequenceLength, Config_.InputDim});
	auto tokens = InputProjection_->Forward(rows).Reshape(OaMatrixShape{
		batch, Config_.Backbone.SequenceLength, Config_.Backbone.DModel});
	tokens = tokens + Position_.Reshape(OaMatrixShape{
		1, Config_.Backbone.SequenceLength, Config_.Backbone.DModel});
	auto time = TimeEmbedding_->Forward(InTime).Reshape(OaMatrixShape{
		batch, 1, Config_.Backbone.DModel});
	auto context = time;
	if (ConditionProjection_) {
		auto conditionInput = InCondition;
		if (IsTraining() && Config_.ConditionDropoutP > 0.0F) {
			auto keep = OaFnMatrix::GreaterEqual(
				OaFnMatrix::PhiloxUniform(OaFnMatrix::Empty(
					OaMatrixShape{batch, 1}, InCondition.GetDtype()),
					0.0F, 1.0F, 0),
				Config_.ConditionDropoutP);
			conditionInput = conditionInput * keep;
		}
		auto condition = ConditionProjection_->Forward(conditionInput).Reshape(
			OaMatrixShape{batch, 1, Config_.Backbone.DModel});
		context = context + condition;
	}
	auto hidden = Backbone_->ForwardConditioned(
		tokens, context.Reshape(OaMatrixShape{batch, Config_.Backbone.DModel}),
		InTokenMask).Reshape(OaMatrixShape{
		batch * Config_.Backbone.SequenceLength, Config_.Backbone.DModel});
	return OutputProjection_->Forward(hidden).Reshape(InSample.GetShape());
}

OaMatrix OaFlowDenoiser::ForwardGuided(
	const OaMatrix& InSample,
	const OaMatrix& InTime,
	const OaMatrix& InCondition,
	OaF32 InGuidanceScale,
	const OaMatrix& InTokenMask) {
	if (!ConditionProjection_ || !std::isfinite(InGuidanceScale)
		|| InGuidanceScale < 0.0F) {
		throw std::invalid_argument(
			"FlowDenoiser guidance requires configured conditions and a finite non-negative scale");
	}
	OaModule::ScopedEval eval(*this);
	auto unconditional = ForwardConditioned(
		InSample, InTime,
		OaFnMatrix::Zeros(InCondition.GetShape(), InCondition.GetDtype()),
		InTokenMask);
	auto conditional = ForwardConditioned(
		InSample, InTime, InCondition, InTokenMask);
	return unconditional
		+ (conditional - unconditional) * InGuidanceScale;
}
