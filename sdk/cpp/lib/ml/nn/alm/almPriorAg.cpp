// oa::AlmPriorAg — stage 2 autoregressive language model implementation.

#include <ml/nn/alm/almPriorAg.h>
#include <ml/nn/alm/almTokenizerAg.h>
#include <oa/ml/nn.h>
#include <oa/core/log.h>
#include <oa/core/fnMatrix.h>
#include <oa/ml/fnMatrix.h>
#include <oa/runtime/executionSession.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// oa::AlmPriorAg

oa::AlmPriorAg::AlmPriorAg(const oa::AlmPriorConfig& inConfig) : config_(inConfig) {
	if (config_.dModel <= 0 or config_.numHeads <= 0 or config_.dModel % config_.numHeads != 0 or
		config_.dFfn <= 0 or config_.numLayers <= 0 or
		config_.seqLen <= 0 or config_.maxSeqLen <= 0 or config_.seqLen > config_.maxSeqLen) {
		throw std::invalid_argument("oa::AlmPriorAg dimensions must be positive, dModel divisible by numHeads, and sequence lengths internally consistent");
	}
	if (config_.ffnType != oa::AlmFfnType::Dense and
		(config_.moeNumExperts <= 0 or config_.moeExpertsPerToken <= 0)) {
		throw std::invalid_argument("oa::AlmPriorAg MoE expert counts must be positive");
	}
	if (config_.ffnType == oa::AlmFfnType::Hybrid and config_.moeEvery <= 0) {
		throw std::invalid_argument("oa::AlmPriorAg hybrid MoE cadence must be positive");
	}
	maxSeqLen_ = config_.maxSeqLen;

	// Token embedding
	tokenEmbed_ = oa::makeShared<oa::Embedding>(config_.vocabSize, config_.dModel);
	if (config_.textFeatureDim < 0) {
		throw std::invalid_argument("oa::AlmPriorAg text feature dimension cannot be negative");
	}
	if (config_.textFeatureDim > 0) {
		textProjection_ = oa::makeShared<oa::Linear>(config_.textFeatureDim, config_.dModel);
		registerModule("text_projection", textProjection_);
	}

	// Learned positional embedding [maxSeqLen, dModel]
	posEmbed_ = oa::makeShared<oa::Embedding>(maxSeqLen_, config_.dModel);

	// Decoder layers
	for (oa::I32 i = 0; i < config_.numLayers; ++i) {
		oa::SharedPtr<oa::TransformerBlock> layer;
		if (config_.usesMoe(i)) {
			layer = oa::makeShared<oa::TransformerBlock>(config_.dModel, config_.dFfn, config_.seqLen,
				config_.numHeads, config_.moeNumExperts, config_.moeExpertsPerToken, 1e-5F);
			layer->moe()->setBalanceRate(config_.moeBalanceRate);
			layer->moe()->setAuxLossAlpha(config_.moeAuxLossAlpha);
			layer->moe()->setRouterZLossBeta(config_.moeRouterZLossBeta);
		} else {
			layer = oa::makeShared<oa::TransformerBlock>(
				config_.dModel, config_.dFfn, config_.seqLen, config_.numHeads, 1e-5F);
		}
		layers_.pushBack(layer);
		char buf[32];
		std::snprintf(buf, sizeof(buf), "layer%d", i);
		registerModule(buf, layer);
	}

	// Final norm
	finalNorm_ = oa::makeShared<oa::RmsNorm>(config_.dModel);

	// output head (no bias)
	outputHead_ = oa::makeShared<oa::Linear>(config_.dModel, config_.vocabSize, false);

	registerModule("token_embed", tokenEmbed_);
	registerModule("pos_embed", posEmbed_);
	registerModule("final_norm", finalNorm_);
	registerModule("output_head", outputHead_);

	const char* ffn = config_.ffnType == oa::AlmFfnType::Dense ? "dense"
		: (config_.ffnType == oa::AlmFfnType::Moe ? "moe" : "hybrid");
	OaLogInfo(oa::LogComponent::Ml,
		"oa::AlmPriorAg initialized: transformer, %s FFN, %d layers, %d heads, %d dim, %d vocab, maxseq=%d",
		ffn, config_.numLayers, config_.numHeads, config_.dModel, config_.vocabSize, maxSeqLen_);
}

oa::Matrix oa::AlmPriorAg::moeAuxLoss() const {
	oa::Matrix total;
	for (const auto& layer : layers_) {
		const oa::Moe* moe = layer->moe();
		if (moe == nullptr) continue;
		total = total.isEmpty() ? moe->auxLoss() : oa::FnMatrix::add(total, moe->auxLoss());
	}
	return total;
}

void oa::AlmPriorAg::updateMoeRoutingBias() {
	for (auto& layer : layers_) if (layer->moe() != nullptr) layer->moe()->updateRoutingBias();
}

oa::Vector<oa::MoeRouteStats> oa::AlmPriorAg::moeRouteStats() const {
	oa::Vector<oa::MoeRouteStats> stats;
	for (const auto& layer : layers_) if (layer->moe() != nullptr) stats.pushBack(layer->moe()->routeStats());
	return stats;
}

oa::Matrix oa::AlmPriorAg::forward(const oa::Matrix& inTokenIds) {
	if (config_.textFeatureDim > 0) {
		throw std::invalid_argument("oa::AlmPriorAg conditioned model requires forwardConditioned");
	}
	return forwardImpl(inTokenIds, nullptr);
}

oa::Matrix oa::AlmPriorAg::forwardConditioned(
	const oa::Matrix& inTokenIds, const oa::Matrix& inTextFeatures) {
	if (config_.textFeatureDim <= 0) {
		throw std::invalid_argument("oa::AlmPriorAg was not configured for text features");
	}
	return forwardImpl(inTokenIds, &inTextFeatures);
}

oa::Matrix oa::AlmPriorAg::forwardImpl(
	const oa::Matrix& inTokenIds, const oa::Matrix* inTextFeatures) {
	const oa::I32 B = static_cast<oa::I32>(inTokenIds.size(0));
	const oa::I32 T = static_cast<oa::I32>(inTokenIds.size(1));
	const oa::I32 prefix = inTextFeatures != nullptr ? 1 : 0;
	const oa::I32 totalT = T + prefix;
	if (B <= 0 or T <= 0 or totalT > maxSeqLen_) {
		throw std::invalid_argument("oa::AlmPriorAg input must be non-empty and fit the configured maximum sequence length");
	}

	// Token embedding: [B, T, dModel]
	auto emb = tokenEmbed_->forward(inTokenIds);
	if (inTextFeatures != nullptr) {
		if (inTextFeatures->rank() != 2 or inTextFeatures->size(0) != B
			or inTextFeatures->size(1) != config_.textFeatureDim) {
			throw std::invalid_argument("oa::AlmPriorAg text features must be [B, textFeatureDim]");
		}
		auto text = textProjection_->forward(*inTextFeatures).reshape(
			oa::MatrixShape{B, 1, config_.dModel});
		emb = emb.reshape(oa::MatrixShape{B, T, config_.dModel});
		oa::Matrix parts[] = {text, emb};
		emb = oa::FnMatrix::concat(oa::Span<oa::Matrix>(parts), 1).reshape(
			oa::MatrixShape{static_cast<oa::I64>(B) * totalT, config_.dModel});
	}

	// Positional embedding: position indices [B, T] are [0..t-1] per row (generation
	// feeds the whole growing prefix, so positions are always absolute-from-zero).
	// Rebuild + upload only when (B, T) changes — a one-time cost in training (§10.7 ③).
	if (B != cachedPosB_ || totalT != cachedPosT_) {
		std::vector<oa::U32> posIds(static_cast<size_t>(B) * static_cast<size_t>(totalT));
		for (oa::I32 b = 0; b < B; ++b) {
			for (oa::I32 t = 0; t < totalT; ++t) {
				posIds[static_cast<size_t>(b) * static_cast<size_t>(totalT) + static_cast<size_t>(t)] =
					static_cast<oa::U32>(t);
			}
		}
		posIdxCache_ = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(posIds.data()), posIds.size() * sizeof(oa::U32)),
			oa::MatrixShape{B, totalT}, oa::ScalarType::UInt32);
		cachedPosB_ = B;
		cachedPosT_ = totalT;
	}
	auto posEmb = posEmbed_->forward(posIdxCache_);  // [B, T, dModel]

	// Add token + position embeddings, then flatten to [B*T, dModel]
	auto x = oa::FnMatrix::add(emb, posEmb);
	x = x.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * totalT, config_.dModel});

	for (oa::I32 i = 0; i < config_.numLayers; ++i) {
		layers_[static_cast<oa::Usize>(i)]->setSeqLen(totalT);
		x = layers_[static_cast<oa::Usize>(i)]->forward(x);
	}

	x = finalNorm_->forward(x);
	if (prefix != 0) {
		x = x.reshape(oa::MatrixShape{B, totalT, config_.dModel});
		x = oa::FnMatrix::slice(x, 1, 1, totalT).reshape(
			oa::MatrixShape{static_cast<oa::I64>(B) * T, config_.dModel});
	}
	auto logits2d = outputHead_->forward(x);
	auto logits = logits2d.reshape(oa::MatrixShape{B, T * config_.vocabSize});
	return logits.reshape(oa::MatrixShape{B, T, config_.vocabSize});
}

oa::Matrix oa::AlmPriorAg::generate(
	oa::I32 inBatchSize,
	oa::F32 inTemperature,
	oa::I32 inTopK,
	oa::F32 inTopP,
	oa::I32 inMaxLen,
	bool inUseCache
) {
	return generateImpl(nullptr, inBatchSize, inTemperature, inTopK, inTopP,
		inMaxLen, inUseCache);
}

oa::Matrix oa::AlmPriorAg::generateConditioned(
	const oa::Matrix& inTextFeatures,
	oa::F32 inTemperature,
	oa::I32 inTopK,
	oa::F32 inTopP,
	oa::I32 inMaxLen,
	bool inUseCache
) {
	if (config_.textFeatureDim <= 0 or inTextFeatures.rank() != 2) {
		throw std::invalid_argument("oa::AlmPriorAg conditioned generation requires [B,textFeatureDim]");
	}
	return generateImpl(&inTextFeatures, static_cast<oa::I32>(inTextFeatures.size(0)),
		inTemperature, inTopK, inTopP, inMaxLen, inUseCache);
}

oa::Matrix oa::AlmPriorAg::generateImpl(
	const oa::Matrix* inTextFeatures,
	oa::I32 inBatchSize,
	oa::F32 inTemperature,
	oa::I32 inTopK,
	oa::F32 inTopP,
	oa::I32 inMaxLen,
	bool inUseCache
) {
	const oa::I32 prefix = inTextFeatures != nullptr ? 1 : 0;
	if (inBatchSize <= 0 or inMaxLen <= 0 or inMaxLen + prefix > maxSeqLen_) {
		throw std::invalid_argument("oa::AlmPriorAg generation batch/length must be positive and fit maxSeqLen");
	}
	if (inTextFeatures != nullptr and
		(inTextFeatures->size(0) != inBatchSize or
		 inTextFeatures->size(1) != config_.textFeatureDim)) {
		throw std::invalid_argument("oa::AlmPriorAg generation text features must be [B, textFeatureDim]");
	}
	OaLogInfo(oa::LogComponent::Ml, "oa::AlmPriorAg::generate — batch=%d, maxlen=%d, cache=%d",
		inBatchSize, inMaxLen, inUseCache ? 1 : 0);
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I64 vocabSize = config_.vocabSize;

	// Per-row token sequences (growing). Done rows are padded with [PAD] to keep the
	// returned tensor rectangular; their logits are never sampled.
	std::vector<std::vector<oa::I32>> rows(static_cast<size_t>(inBatchSize),
		std::vector<oa::I32>{config_.somToken});
	std::vector<bool> done(static_cast<size_t>(inBatchSize), false);

	// KV-cache is not implemented yet. Feed the growing prefix and sample the last
	// position each step; the Transformer blocks reuse their weights and rebuild only
	// the sequence-length-dependent causal mask.
	(void)inUseCache;
	for (oa::I32 step = 0; step < inMaxLen; ++step) {
		const oa::I32 curLen = step + 1;   // uniform dense length this step
		std::vector<oa::I32> dense(static_cast<size_t>(inBatchSize) * static_cast<size_t>(curLen), config_.padToken);
		for (oa::I32 b = 0; b < inBatchSize; ++b) {
			const auto& r = rows[static_cast<size_t>(b)];
			const oa::I32 n = static_cast<oa::I32>(r.size());   // active row: n==curLen; done row: n<curLen (rest stays [PAD])
			for (oa::I32 t = 0; t < n; ++t) {
				dense[static_cast<size_t>(b) * curLen + t] = r[static_cast<size_t>(t)];
			}
		}
		oa::Matrix ids = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(dense.data()), dense.size() * sizeof(oa::I32)),
			oa::MatrixShape{inBatchSize, curLen}, oa::ScalarType::Int32);

		auto logits = inTextFeatures != nullptr
			? forwardConditioned(ids, *inTextFeatures)
			: forward(ids);   // [B, curLen, vocabSize]
		auto lastLogits = oa::FnMatrix::reshape(
			oa::FnMatrix::slice(logits, 1, curLen - 1, curLen),
			oa::MatrixShape{inBatchSize, vocabSize});
		auto sampled = oa::FnMatrix::sampleLogits(
			lastLogits, inTemperature, inTopK, inTopP);
		(void)ctx.submitAndWait();
		const oa::I32* sampledHost = sampled.dataAs<const oa::I32>();
		bool allDone = true;
		for (oa::I32 b = 0; b < inBatchSize; ++b) {
			if (done[static_cast<size_t>(b)]) { continue; }
			const oa::I32 next = sampledHost[b];
			rows[static_cast<size_t>(b)].push_back(next);
			if (next == config_.eomToken) { done[static_cast<size_t>(b)] = true; }
			else { allDone = false; }
		}
		if (allDone) { break; }
	}

	// Flatten to [B, maxLen] (rows padded with [PAD] to the longest row for a dense tensor).
	size_t maxRow = 0;
	for (const auto& r : rows) maxRow = std::max(maxRow, r.size());
	const oa::I32 outLen = static_cast<oa::I32>(maxRow);
	std::vector<oa::I32> flat(static_cast<size_t>(inBatchSize) * maxRow, config_.padToken);
	for (oa::I32 b = 0; b < inBatchSize; ++b) {
		const auto& r = rows[static_cast<size_t>(b)];
		for (size_t t = 0; t < r.size(); ++t) flat[static_cast<size_t>(b) * maxRow + t] = r[t];
	}
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(flat.data()), flat.size() * sizeof(oa::I32)),
		oa::MatrixShape{inBatchSize, outLen}, oa::ScalarType::Int32);
}

oa::Matrix oa::AlmPriorAg::decodeToMotion(
	const oa::Matrix& inTokenIds,
	oa::AlmTokenizerAg& inTokenizer
) {
	auto& ctx = oa::ExecutionSession::getActive();
	(void)ctx.submitAndWait();
	const oa::I32 batch = static_cast<oa::I32>(inTokenIds.size(0));
	const oa::I32 len   = static_cast<oa::I32>(inTokenIds.size(1));
	const oa::I32* ids  = inTokenIds.dataAs<const oa::I32>();
	if (batch == 0 or len == 0) return {};

	oa::Vector<oa::Matrix> levelIds;
	oa::Vector<oa::I32>    seqLens;
	seqLens.reserve(batch);

	for (oa::I32 b = 0; b < batch; ++b) {
		oa::I32 seqLen = len;
		for (oa::I32 t = 0; t < len; ++t) {
			if (ids[b * len + t] == config_.eomToken) { seqLen = t; break; }
		}
		seqLens.pushBack(seqLen);
	}

	const oa::I32 minSeqLen = *std::min_element(seqLens.begin(), seqLens.end());
	const oa::I32 tokLen = std::max<oa::I32>(0, minSeqLen - 1);  // exclude [SOM]
	if (tokLen <= 0) return {};

	std::vector<oa::I32> flat;
	flat.reserve(static_cast<size_t>(batch) * tokLen);
	for (oa::I32 b = 0; b < batch; ++b) {
		for (oa::I32 t = 0; t < tokLen; ++t) {
			flat.push_back(ids[b * len + t + 1]);  // skip [SOM]
		}
	}
	oa::Matrix idx = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(flat.data()), flat.size() * sizeof(oa::I32)),
		oa::MatrixShape{static_cast<oa::I64>(batch) * tokLen, 1}, oa::ScalarType::Int32);

	levelIds.pushBack(idx);
	return inTokenizer.detokenize(levelIds, batch, tokLen);
}
