#include <ml/nn/alm/clipTextAg.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml/modelFile.h>
#include <oa/runtime/executionSession.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

class ClipResidualBlock final : public oa::Module {
public:
	ClipResidualBlock(const oa::ClipTextConfig& inConfig) : config_(inConfig) {
		layerNorm1_ = oa::makeShared<oa::LayerNorm>(config_.hiddenSize, config_.layerNormEps);
		selfAttention_ = oa::makeShared<oa::MultiHeadAttention>(
			config_.hiddenSize, config_.numHeads, 0.0F, true);
		selfAttention_->setSeqLen(config_.contextLength);
		layerNorm2_ = oa::makeShared<oa::LayerNorm>(config_.hiddenSize, config_.layerNormEps);
		fc1_ = oa::makeShared<oa::Linear>(config_.hiddenSize, config_.intermediateSize, true);
		fc2_ = oa::makeShared<oa::Linear>(config_.intermediateSize, config_.hiddenSize, true);

		registerModule("layer_norm1", layerNorm1_);
		registerModule("self_attn", selfAttention_);
		registerModule("layer_norm2", layerNorm2_);
		registerModule("mlp.fc1", fc1_);
		registerModule("mlp.fc2", fc2_);
	}

	oa::Matrix forward(const oa::Matrix& inInput) override {
		auto x = oa::FnMatrix::add(inInput, selfAttention_->forward(layerNorm1_->forward(inInput)));
		auto h = fc1_->forward(layerNorm2_->forward(x));
		// OpenAI CLIP QuickGELU: x * sigmoid(1.702 * x). This composition is
		// entirely GPU-native and remains differentiable for parity gradchecks.
		h = oa::FnMatrix::mul(h, oa::FnMatrix::sigmoid(
			oa::FnMatrix::scale(h, config_.quickGeluAlpha)));
		return oa::FnMatrix::add(x, fc2_->forward(h));
	}

private:
	oa::ClipTextConfig config_;
	oa::SharedPtr<oa::LayerNorm> layerNorm1_;
	oa::SharedPtr<oa::MultiHeadAttention> selfAttention_;
	oa::SharedPtr<oa::LayerNorm> layerNorm2_;
	oa::SharedPtr<oa::Linear> fc1_;
	oa::SharedPtr<oa::Linear> fc2_;
};

} // namespace

oa::Status oa::ClipTextConfig::validate() const {
	if (vocabSize <= 0 or contextLength <= 0 or hiddenSize <= 0 or
		intermediateSize <= 0 or numHeads <= 0 or numLayers <= 0 or
		projectionDim <= 0 or hiddenSize % numHeads != 0 or layerNormEps <= 0.0F or
		quickGeluAlpha <= 0.0F or bosToken < 0 or eosToken < 0 or padToken < 0 or
		bosToken >= vocabSize or eosToken >= vocabSize or padToken >= vocabSize) {
		return oa::Status::invalidArgument("invalid CLIP text configuration");
	}
	return oa::Status::ok();
}

oa::ClipTextConfig oa::ClipTextConfig::viTL14() { return {}; }

oa::ClipTextAg::ClipTextAg(const oa::ClipTextConfig& inConfig) : config_(inConfig) {
	const auto valid = config_.validate();
	if (not valid.isOk()) throw std::invalid_argument(valid.getMessage().cStr());

	tokenEmbedding_ = oa::makeShared<oa::Embedding>(config_.vocabSize, config_.hiddenSize);
	positionEmbedding_ = oa::makeShared<oa::Embedding>(config_.contextLength, config_.hiddenSize);
	registerModule("text_model.embeddings.token_embedding", tokenEmbedding_);
	registerModule("text_model.embeddings.position_embedding", positionEmbedding_);

	for (oa::I32 i = 0; i < config_.numLayers; ++i) {
		auto layer = oa::makeShared<ClipResidualBlock>(config_);
		layers_.pushBack(layer);
		char name[64];
		std::snprintf(name, sizeof(name), "text_model.encoder.layers.%d", i);
		registerModule(name, layer);
	}

	finalLayerNorm_ = oa::makeShared<oa::LayerNorm>(config_.hiddenSize, config_.layerNormEps);
	textProjection_ = oa::makeShared<oa::Linear>(config_.hiddenSize, config_.projectionDim, false);
	registerModule("text_model.final_layer_norm", finalLayerNorm_);
	registerModule("text_projection", textProjection_);

	std::vector<oa::I32> positions(static_cast<size_t>(config_.contextLength));
	for (oa::I32 i = 0; i < config_.contextLength; ++i) positions[static_cast<size_t>(i)] = i;
	positionIds_ = oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(positions.data(), positions.size()),
		oa::MatrixShape{config_.contextLength}, oa::ScalarType::Int32);
	freeze();
}

void oa::ClipTextAg::freeze() {
	for (auto& named : allNamedParameterPtrs()) {
		named.param->requiresGrad = false;
		named.param->data.setRequiresGrad(false);
	}
}

oa::Matrix oa::ClipTextAg::forwardTokens(
	const oa::Matrix& inTokenIds, const oa::Matrix& inFlatEosRows) {
	if ((inTokenIds.getDtype() != oa::ScalarType::Int32 and
		 inTokenIds.getDtype() != oa::ScalarType::UInt32) or
		inTokenIds.rank() != 2 or inTokenIds.size(1) != config_.contextLength) {
		throw std::invalid_argument("oa::ClipTextAg token IDs must be [B,contextLength]");
	}
	const oa::I32 batch = static_cast<oa::I32>(inTokenIds.size(0));
	if ((inFlatEosRows.getDtype() != oa::ScalarType::Int32 and
		 inFlatEosRows.getDtype() != oa::ScalarType::UInt32) or
		batch <= 0 or inFlatEosRows.rank() != 1 or inFlatEosRows.size(0) != batch) {
		throw std::invalid_argument("oa::ClipTextAg EOS rows must be [B]");
	}

	auto token = tokenEmbedding_->forward(inTokenIds).reshape(
		oa::MatrixShape{batch, config_.contextLength, config_.hiddenSize});
	auto position = positionEmbedding_->forward(positionIds_).reshape(
		oa::MatrixShape{1, config_.contextLength, config_.hiddenSize});
	auto x = oa::FnMatrix::add(token, position).reshape(
		oa::MatrixShape{static_cast<oa::I64>(batch) * config_.contextLength, config_.hiddenSize});
	for (auto& layer : layers_) x = layer->forward(x);
	x = finalLayerNorm_->forward(x);
	auto pooled = oa::FnMatrix::gather(x, inFlatEosRows);
	return textProjection_->forward(pooled);
}

oa::Matrix oa::ClipTextAg::forward(const oa::Matrix& inTokenIds) {
	if (inTokenIds.rank() != 2 or inTokenIds.size(1) != config_.contextLength) {
		throw std::invalid_argument("oa::ClipTextAg token IDs must be [B,contextLength]");
	}
	// OpenAI CLIP's EOS token is the largest vocabulary ID, so row-wise argmax
	// identifies its position. This fallback is intentionally synchronized; the
	// production tokenizer provides EOS rows directly to forwardTokens.
	auto& ctx = oa::ExecutionSession::getActive();
	(void)ctx.submitAndWait();
	const oa::I32 batch = static_cast<oa::I32>(inTokenIds.size(0));
	std::vector<oa::I32> rows(static_cast<size_t>(batch));
	if (inTokenIds.getDtype() == oa::ScalarType::UInt32) {
		const auto* ids = inTokenIds.dataAs<const oa::U32>();
		for (oa::I32 b = 0; b < batch; ++b) {
			oa::I32 best = 0;
			for (oa::I32 t = 1; t < config_.contextLength; ++t)
				if (ids[static_cast<oa::I64>(b) * config_.contextLength + t] >
					ids[static_cast<oa::I64>(b) * config_.contextLength + best]) best = t;
			rows[static_cast<size_t>(b)] = b * config_.contextLength + best;
		}
	} else {
		const auto* ids = inTokenIds.dataAs<const oa::I32>();
		for (oa::I32 b = 0; b < batch; ++b) {
			oa::I32 best = 0;
			for (oa::I32 t = 1; t < config_.contextLength; ++t)
				if (ids[static_cast<oa::I64>(b) * config_.contextLength + t] >
					ids[static_cast<oa::I64>(b) * config_.contextLength + best]) best = t;
			rows[static_cast<size_t>(b)] = b * config_.contextLength + best;
		}
	}
	auto eosRows = oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(rows.data(), rows.size()), oa::MatrixShape{batch}, oa::ScalarType::Int32);
	return forwardTokens(inTokenIds, eosRows);
}

oa::Result<oa::SharedPtr<oa::ClipTextAg>> oa::ClipTextAg::loadArchive(
	oa::Engine& inEngine, const oa::String& inPath)
{
	auto& ctx = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(ctx);
	auto loaded = oa::ModelFile::load(inPath);
	if (loaded.isError()) return loaded.getStatus();
	auto oam = oa::move(loaded.getValue());
	if (std::strncmp(oam.config.architecture, "OaClipTextAg", sizeof(oam.config.architecture)) != 0 or
		oam.config.configVersion != 1 or oam.archConfig.size() != sizeof(oa::ClipTextConfig)) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
			"checkpoint is not an oa::ClipTextAg v1 model");
	}
	oa::ClipTextConfig config;
	std::memcpy(&config, oam.archConfig.data(), sizeof(config));
	if (auto valid = config.validate(); not valid.isOk()) return valid;
	auto model = oa::makeShared<oa::ClipTextAg>(config);
	const auto expected = model->allNamedParameterPtrs();
	if (expected.size() != oam.weightIndex.size()) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
			"oa::ClipTextAg checkpoint tensor count mismatch");
	}
	for (const auto& named : expected) {
		const auto* entry = oam.findWeight(named.path.cStr());
		if (entry == nullptr or entry->dtype != named.param->data.getDtype() or
			entry->rank != named.param->data.rank() or
			entry->numBytes != static_cast<oa::U64>(named.param->data.byteSize())) {
			return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
				oa::String("oa::ClipTextAg tensor mismatch: ") + named.path);
		}
		for (oa::I32 d = 0; d < named.param->data.rank(); ++d)
			if (entry->shape[d] != static_cast<oa::U64>(named.param->data.size(d)))
				return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
					oa::String("oa::ClipTextAg shape mismatch: ") + named.path);
	}
	OA_RETURN_IF_ERROR(ctx.submitAndWait());
	ctx.clear();
	OA_RETURN_IF_ERROR(model->loadFrom(inEngine, oam));
	model->freeze();
	return model;
}
