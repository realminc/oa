#include <Ml/Nn/Alm/ClipTextAg.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Runtime/Context.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

class OaClipResidualBlock final : public OaModule {
public:
	OaClipResidualBlock(const OaClipTextConfig& InConfig) : Config_(InConfig) {
		LayerNorm1_ = OaMakeSharedPtr<OaLayerNorm>(Config_.HiddenSize, Config_.LayerNormEps);
		SelfAttention_ = OaMakeSharedPtr<OaMultiHeadAttention>(
			Config_.HiddenSize, Config_.NumHeads, 0.0F, true);
		SelfAttention_->SetSeqLen(Config_.ContextLength);
		LayerNorm2_ = OaMakeSharedPtr<OaLayerNorm>(Config_.HiddenSize, Config_.LayerNormEps);
		Fc1_ = OaMakeSharedPtr<OaLinear>(Config_.HiddenSize, Config_.IntermediateSize, true);
		Fc2_ = OaMakeSharedPtr<OaLinear>(Config_.IntermediateSize, Config_.HiddenSize, true);

		RegisterModule("layer_norm1", LayerNorm1_);
		RegisterModule("self_attn", SelfAttention_);
		RegisterModule("layer_norm2", LayerNorm2_);
		RegisterModule("mlp.fc1", Fc1_);
		RegisterModule("mlp.fc2", Fc2_);
	}

	OaMatrix Forward(const OaMatrix& InInput) override {
		auto x = OaFnMatrix::Add(InInput, SelfAttention_->Forward(LayerNorm1_->Forward(InInput)));
		auto h = Fc1_->Forward(LayerNorm2_->Forward(x));
		// OpenAI CLIP QuickGELU: x * sigmoid(1.702 * x). This composition is
		// entirely GPU-native and remains differentiable for parity gradchecks.
		h = OaFnMatrix::Mul(h, OaFnMatrix::Sigmoid(
			OaFnMatrix::Scale(h, Config_.QuickGeluAlpha)));
		return OaFnMatrix::Add(x, Fc2_->Forward(h));
	}

private:
	OaClipTextConfig Config_;
	OaSharedPtr<OaLayerNorm> LayerNorm1_;
	OaSharedPtr<OaMultiHeadAttention> SelfAttention_;
	OaSharedPtr<OaLayerNorm> LayerNorm2_;
	OaSharedPtr<OaLinear> Fc1_;
	OaSharedPtr<OaLinear> Fc2_;
};

} // namespace

OaStatus OaClipTextConfig::Validate() const {
	if (VocabSize <= 0 or ContextLength <= 0 or HiddenSize <= 0 or
		IntermediateSize <= 0 or NumHeads <= 0 or NumLayers <= 0 or
		ProjectionDim <= 0 or HiddenSize % NumHeads != 0 or LayerNormEps <= 0.0F or
		QuickGeluAlpha <= 0.0F or BosToken < 0 or EosToken < 0 or PadToken < 0 or
		BosToken >= VocabSize or EosToken >= VocabSize or PadToken >= VocabSize) {
		return OaStatus::InvalidArgument("invalid CLIP text configuration");
	}
	return OaStatus::Ok();
}

OaClipTextConfig OaClipTextConfig::ViTL14() { return {}; }

OaClipTextAg::OaClipTextAg(const OaClipTextConfig& InConfig) : Config_(InConfig) {
	const auto valid = Config_.Validate();
	if (not valid.IsOk()) throw std::invalid_argument(valid.GetMessage().CStr());

	TokenEmbedding_ = OaMakeSharedPtr<OaEmbedding>(Config_.VocabSize, Config_.HiddenSize);
	PositionEmbedding_ = OaMakeSharedPtr<OaEmbedding>(Config_.ContextLength, Config_.HiddenSize);
	RegisterModule("text_model.embeddings.token_embedding", TokenEmbedding_);
	RegisterModule("text_model.embeddings.position_embedding", PositionEmbedding_);

	for (OaI32 i = 0; i < Config_.NumLayers; ++i) {
		auto layer = OaMakeSharedPtr<OaClipResidualBlock>(Config_);
		Layers_.PushBack(layer);
		char name[64];
		std::snprintf(name, sizeof(name), "text_model.encoder.layers.%d", i);
		RegisterModule(name, layer);
	}

	FinalLayerNorm_ = OaMakeSharedPtr<OaLayerNorm>(Config_.HiddenSize, Config_.LayerNormEps);
	TextProjection_ = OaMakeSharedPtr<OaLinear>(Config_.HiddenSize, Config_.ProjectionDim, false);
	RegisterModule("text_model.final_layer_norm", FinalLayerNorm_);
	RegisterModule("text_projection", TextProjection_);

	std::vector<OaI32> positions(static_cast<size_t>(Config_.ContextLength));
	for (OaI32 i = 0; i < Config_.ContextLength; ++i) positions[static_cast<size_t>(i)] = i;
	PositionIds_ = OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(positions.data(), positions.size()),
		OaMatrixShape{Config_.ContextLength}, OaScalarType::Int32);
	Freeze();
}

void OaClipTextAg::Freeze() {
	for (auto& named : AllNamedParameterPtrs()) {
		named.Param->RequiresGrad = false;
		named.Param->Data.SetRequiresGrad(false);
	}
}

OaMatrix OaClipTextAg::ForwardTokens(
	const OaMatrix& InTokenIds, const OaMatrix& InFlatEosRows) {
	if ((InTokenIds.GetDtype() != OaScalarType::Int32 and
		 InTokenIds.GetDtype() != OaScalarType::UInt32) or
		InTokenIds.Rank() != 2 or InTokenIds.Size(1) != Config_.ContextLength) {
		throw std::invalid_argument("OaClipTextAg token IDs must be [B,ContextLength]");
	}
	const OaI32 batch = static_cast<OaI32>(InTokenIds.Size(0));
	if ((InFlatEosRows.GetDtype() != OaScalarType::Int32 and
		 InFlatEosRows.GetDtype() != OaScalarType::UInt32) or
		batch <= 0 or InFlatEosRows.Rank() != 1 or InFlatEosRows.Size(0) != batch) {
		throw std::invalid_argument("OaClipTextAg EOS rows must be [B]");
	}

	auto token = TokenEmbedding_->Forward(InTokenIds).Reshape(
		OaMatrixShape{batch, Config_.ContextLength, Config_.HiddenSize});
	auto position = PositionEmbedding_->Forward(PositionIds_).Reshape(
		OaMatrixShape{1, Config_.ContextLength, Config_.HiddenSize});
	auto x = OaFnMatrix::Add(token, position).Reshape(
		OaMatrixShape{static_cast<OaI64>(batch) * Config_.ContextLength, Config_.HiddenSize});
	for (auto& layer : Layers_) x = layer->Forward(x);
	x = FinalLayerNorm_->Forward(x);
	auto pooled = OaFnMatrix::Gather(x, InFlatEosRows);
	return TextProjection_->Forward(pooled);
}

OaMatrix OaClipTextAg::Forward(const OaMatrix& InTokenIds) {
	if (InTokenIds.Rank() != 2 or InTokenIds.Size(1) != Config_.ContextLength) {
		throw std::invalid_argument("OaClipTextAg token IDs must be [B,ContextLength]");
	}
	// OpenAI CLIP's EOS token is the largest vocabulary ID, so row-wise argmax
	// identifies its position. This fallback is intentionally synchronized; the
	// production tokenizer provides EOS rows directly to ForwardTokens.
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
	const OaI32 batch = static_cast<OaI32>(InTokenIds.Size(0));
	std::vector<OaI32> rows(static_cast<size_t>(batch));
	if (InTokenIds.GetDtype() == OaScalarType::UInt32) {
		const auto* ids = InTokenIds.DataAs<const OaU32>();
		for (OaI32 b = 0; b < batch; ++b) {
			OaI32 best = 0;
			for (OaI32 t = 1; t < Config_.ContextLength; ++t)
				if (ids[static_cast<OaI64>(b) * Config_.ContextLength + t] >
					ids[static_cast<OaI64>(b) * Config_.ContextLength + best]) best = t;
			rows[static_cast<size_t>(b)] = b * Config_.ContextLength + best;
		}
	} else {
		const auto* ids = InTokenIds.DataAs<const OaI32>();
		for (OaI32 b = 0; b < batch; ++b) {
			OaI32 best = 0;
			for (OaI32 t = 1; t < Config_.ContextLength; ++t)
				if (ids[static_cast<OaI64>(b) * Config_.ContextLength + t] >
					ids[static_cast<OaI64>(b) * Config_.ContextLength + best]) best = t;
			rows[static_cast<size_t>(b)] = b * Config_.ContextLength + best;
		}
	}
	auto eosRows = OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(rows.data(), rows.size()), OaMatrixShape{batch}, OaScalarType::Int32);
	return ForwardTokens(InTokenIds, eosRows);
}

OaResult<OaSharedPtr<OaClipTextAg>> OaClipTextAg::LoadOam(const OaString& InPath) {
	auto loaded = OamModel::Load(InPath);
	if (loaded.IsError()) return loaded.GetStatus();
	auto oam = OaStdMove(loaded.GetValue());
	if (std::strncmp(oam.Config.Architecture, "OaClipTextAg", sizeof(oam.Config.Architecture)) != 0 or
		oam.Config.ConfigVersion != 1 or oam.ArchConfig.Size() != sizeof(OaClipTextConfig)) {
		return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
			"checkpoint is not an OaClipTextAg v1 model");
	}
	OaClipTextConfig config;
	std::memcpy(&config, oam.ArchConfig.Data(), sizeof(config));
	if (auto valid = config.Validate(); not valid.IsOk()) return valid;
	auto model = OaMakeSharedPtr<OaClipTextAg>(config);
	const auto expected = model->AllNamedParameterPtrs();
	if (expected.Size() != oam.WeightIndex.Size()) {
		return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
			"OaClipTextAg checkpoint tensor count mismatch");
	}
	for (const auto& named : expected) {
		const auto* entry = oam.FindWeight(named.Path.CStr());
		if (entry == nullptr or entry->Dtype != named.Param->Data.GetDtype() or
			entry->Rank != named.Param->Data.Rank() or
			entry->NumBytes != static_cast<OaU64>(named.Param->Data.ByteSize())) {
			return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
				OaString("OaClipTextAg tensor mismatch: ") + named.Path);
		}
		for (OaI32 d = 0; d < named.Param->Data.Rank(); ++d)
			if (entry->Shape[d] != static_cast<OaU64>(named.Param->Data.Size(d)))
				return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
					OaString("OaClipTextAg shape mismatch: ") + named.Path);
	}
	auto& ctx = OaContext::GetDefault();
	OA_RETURN_IF_ERROR(ctx.Execute());
	OA_RETURN_IF_ERROR(ctx.Sync());
	ctx.Clear();
	model->LoadFrom(oam);
	model->Freeze();
	return model;
}
