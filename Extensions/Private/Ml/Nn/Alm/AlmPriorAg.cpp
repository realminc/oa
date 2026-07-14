// OaAlmPriorAg — Stage 2 autoregressive language model implementation.

#include <Ml/Nn/Alm/AlmPriorAg.h>
#include <Ml/Nn/Alm/AlmTokenizerAg.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Runtime/Context.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// OaAlmPriorAg

OaAlmPriorAg::OaAlmPriorAg(const OaAlmPriorConfig& InConfig) : Config_(InConfig) {
	if (Config_.DModel <= 0 or Config_.NumHeads <= 0 or Config_.DModel % Config_.NumHeads != 0 or
		Config_.DFfn <= 0 or Config_.NumLayers <= 0 or
		Config_.SeqLen <= 0 or Config_.MaxSeqLen <= 0 or Config_.SeqLen > Config_.MaxSeqLen) {
		throw std::invalid_argument("OaAlmPriorAg dimensions must be positive, DModel divisible by NumHeads, and sequence lengths internally consistent");
	}
	if (Config_.FfnType != OaAlmFfnType::Dense and
		(Config_.MoeNumExperts <= 0 or Config_.MoeExpertsPerToken <= 0)) {
		throw std::invalid_argument("OaAlmPriorAg MoE expert counts must be positive");
	}
	if (Config_.FfnType == OaAlmFfnType::Hybrid and Config_.MoeEvery <= 0) {
		throw std::invalid_argument("OaAlmPriorAg hybrid MoE cadence must be positive");
	}
	MaxSeqLen_ = Config_.MaxSeqLen;

	// Token embedding
	TokenEmbed_ = OaMakeSharedPtr<OaEmbedding>(Config_.VocabSize, Config_.DModel);
	if (Config_.TextFeatureDim < 0) {
		throw std::invalid_argument("OaAlmPriorAg text feature dimension cannot be negative");
	}
	if (Config_.TextFeatureDim > 0) {
		TextProjection_ = OaMakeSharedPtr<OaLinear>(Config_.TextFeatureDim, Config_.DModel);
		RegisterModule("text_projection", TextProjection_);
	}

	// Learned positional embedding [MaxSeqLen, DModel]
	PosEmbed_ = OaMakeSharedPtr<OaEmbedding>(MaxSeqLen_, Config_.DModel);

	// Decoder layers
	for (OaI32 i = 0; i < Config_.NumLayers; ++i) {
		OaSharedPtr<OaTransformerBlock> layer;
		if (Config_.UsesMoe(i)) {
			layer = OaMakeSharedPtr<OaTransformerBlock>(Config_.DModel, Config_.DFfn, Config_.SeqLen,
				Config_.NumHeads, Config_.MoeNumExperts, Config_.MoeExpertsPerToken, 1e-5F);
			layer->Moe()->SetBalanceRate(Config_.MoeBalanceRate);
			layer->Moe()->SetAuxLossAlpha(Config_.MoeAuxLossAlpha);
			layer->Moe()->SetRouterZLossBeta(Config_.MoeRouterZLossBeta);
		} else {
			layer = OaMakeSharedPtr<OaTransformerBlock>(
				Config_.DModel, Config_.DFfn, Config_.SeqLen, Config_.NumHeads, 1e-5F);
		}
		Layers_.PushBack(layer);
		char buf[32];
		std::snprintf(buf, sizeof(buf), "layer%d", i);
		RegisterModule(buf, layer);
	}

	// Final norm
	FinalNorm_ = OaMakeSharedPtr<OaRmsNorm>(Config_.DModel);

	// Output head (no bias)
	OutputHead_ = OaMakeSharedPtr<OaLinear>(Config_.DModel, Config_.VocabSize, false);

	RegisterModule("token_embed", TokenEmbed_);
	RegisterModule("pos_embed", PosEmbed_);
	RegisterModule("final_norm", FinalNorm_);
	RegisterModule("output_head", OutputHead_);

	const char* ffn = Config_.FfnType == OaAlmFfnType::Dense ? "dense"
		: (Config_.FfnType == OaAlmFfnType::Moe ? "moe" : "hybrid");
	OA_LOG_INFO(OaLogComponent::ML,
		"OaAlmPriorAg initialized: transformer, %s FFN, %d layers, %d heads, %d dim, %d vocab, maxseq=%d",
		ffn, Config_.NumLayers, Config_.NumHeads, Config_.DModel, Config_.VocabSize, MaxSeqLen_);
}

OaMatrix OaAlmPriorAg::MoeAuxLoss() const {
	OaMatrix total;
	for (const auto& layer : Layers_) {
		const OaMoE* moe = layer->Moe();
		if (moe == nullptr) continue;
		total = total.IsEmpty() ? moe->AuxLoss() : OaFnMatrix::Add(total, moe->AuxLoss());
	}
	return total;
}

void OaAlmPriorAg::UpdateMoeRoutingBias() {
	for (auto& layer : Layers_) if (layer->Moe() != nullptr) layer->Moe()->UpdateRoutingBias();
}

OaVec<OaMoeRouteStats> OaAlmPriorAg::MoeRouteStats() const {
	OaVec<OaMoeRouteStats> stats;
	for (const auto& layer : Layers_) if (layer->Moe() != nullptr) stats.PushBack(layer->Moe()->RouteStats());
	return stats;
}

OaMatrix OaAlmPriorAg::Forward(const OaMatrix& InTokenIds) {
	if (Config_.TextFeatureDim > 0) {
		throw std::invalid_argument("OaAlmPriorAg conditioned model requires ForwardConditioned");
	}
	return ForwardImpl(InTokenIds, nullptr);
}

OaMatrix OaAlmPriorAg::ForwardConditioned(
	const OaMatrix& InTokenIds, const OaMatrix& InTextFeatures) {
	if (Config_.TextFeatureDim <= 0) {
		throw std::invalid_argument("OaAlmPriorAg was not configured for text features");
	}
	return ForwardImpl(InTokenIds, &InTextFeatures);
}

OaMatrix OaAlmPriorAg::ForwardImpl(
	const OaMatrix& InTokenIds, const OaMatrix* InTextFeatures) {
	const OaI32 B = static_cast<OaI32>(InTokenIds.Size(0));
	const OaI32 T = static_cast<OaI32>(InTokenIds.Size(1));
	const OaI32 prefix = InTextFeatures != nullptr ? 1 : 0;
	const OaI32 totalT = T + prefix;
	if (B <= 0 or T <= 0 or totalT > MaxSeqLen_) {
		throw std::invalid_argument("OaAlmPriorAg input must be non-empty and fit the configured maximum sequence length");
	}

	// Token embedding: [B, T, DModel]
	auto emb = TokenEmbed_->Forward(InTokenIds);
	if (InTextFeatures != nullptr) {
		if (InTextFeatures->Rank() != 2 or InTextFeatures->Size(0) != B
			or InTextFeatures->Size(1) != Config_.TextFeatureDim) {
			throw std::invalid_argument("OaAlmPriorAg text features must be [B, TextFeatureDim]");
		}
		auto text = TextProjection_->Forward(*InTextFeatures).Reshape(
			OaMatrixShape{B, 1, Config_.DModel});
		emb = emb.Reshape(OaMatrixShape{B, T, Config_.DModel});
		OaMatrix parts[] = {text, emb};
		emb = OaFnMatrix::Concat(OaSpan<OaMatrix>(parts), 1).Reshape(
			OaMatrixShape{static_cast<OaI64>(B) * totalT, Config_.DModel});
	}

	// Positional embedding: position indices [B, T] are [0..T-1] per row (generation
	// feeds the whole growing prefix, so positions are always absolute-from-zero).
	// Rebuild + upload only when (B, T) changes — a one-time cost in training (§10.7 ③).
	if (B != CachedPosB_ || totalT != CachedPosT_) {
		std::vector<OaU32> posIds(static_cast<size_t>(B) * static_cast<size_t>(totalT));
		for (OaI32 b = 0; b < B; ++b) {
			for (OaI32 t = 0; t < totalT; ++t) {
				posIds[static_cast<size_t>(b) * static_cast<size_t>(totalT) + static_cast<size_t>(t)] =
					static_cast<OaU32>(t);
			}
		}
		PosIdxCache_ = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(posIds.data()), posIds.size() * sizeof(OaU32)),
			OaMatrixShape{B, totalT}, OaScalarType::UInt32);
		CachedPosB_ = B;
		CachedPosT_ = totalT;
	}
	auto posEmb = PosEmbed_->Forward(PosIdxCache_);  // [B, T, DModel]

	// Add token + position embeddings, then flatten to [B*T, DModel]
	auto x = OaFnMatrix::Add(emb, posEmb);
	x = x.Reshape(OaMatrixShape{static_cast<OaI64>(B) * totalT, Config_.DModel});

	for (OaI32 i = 0; i < Config_.NumLayers; ++i) {
		Layers_[static_cast<OaUsize>(i)]->SetSeqLen(totalT);
		x = Layers_[static_cast<OaUsize>(i)]->Forward(x);
	}

	x = FinalNorm_->Forward(x);
	if (prefix != 0) {
		x = x.Reshape(OaMatrixShape{B, totalT, Config_.DModel});
		x = OaFnMatrix::Slice(x, 1, 1, totalT).Reshape(
			OaMatrixShape{static_cast<OaI64>(B) * T, Config_.DModel});
	}
	auto logits2d = OutputHead_->Forward(x);
	auto logits = logits2d.Reshape(OaMatrixShape{B, T * Config_.VocabSize});
	return logits.Reshape(OaMatrixShape{B, T, Config_.VocabSize});
}

OaMatrix OaAlmPriorAg::Generate(
	OaI32 InBatchSize,
	OaF32 InTemperature,
	OaI32 InTopK,
	OaF32 InTopP,
	OaI32 InMaxLen,
	bool InUseCache
) {
	return GenerateImpl(nullptr, InBatchSize, InTemperature, InTopK, InTopP,
		InMaxLen, InUseCache);
}

OaMatrix OaAlmPriorAg::GenerateConditioned(
	const OaMatrix& InTextFeatures,
	OaF32 InTemperature,
	OaI32 InTopK,
	OaF32 InTopP,
	OaI32 InMaxLen,
	bool InUseCache
) {
	if (Config_.TextFeatureDim <= 0 or InTextFeatures.Rank() != 2) {
		throw std::invalid_argument("OaAlmPriorAg conditioned generation requires [B,TextFeatureDim]");
	}
	return GenerateImpl(&InTextFeatures, static_cast<OaI32>(InTextFeatures.Size(0)),
		InTemperature, InTopK, InTopP, InMaxLen, InUseCache);
}

OaMatrix OaAlmPriorAg::GenerateImpl(
	const OaMatrix* InTextFeatures,
	OaI32 InBatchSize,
	OaF32 InTemperature,
	OaI32 InTopK,
	OaF32 InTopP,
	OaI32 InMaxLen,
	bool InUseCache
) {
	const OaI32 prefix = InTextFeatures != nullptr ? 1 : 0;
	if (InBatchSize <= 0 or InMaxLen <= 0 or InMaxLen + prefix > MaxSeqLen_) {
		throw std::invalid_argument("OaAlmPriorAg generation batch/length must be positive and fit MaxSeqLen");
	}
	if (InTextFeatures != nullptr and
		(InTextFeatures->Size(0) != InBatchSize or
		 InTextFeatures->Size(1) != Config_.TextFeatureDim)) {
		throw std::invalid_argument("OaAlmPriorAg generation text features must be [B, TextFeatureDim]");
	}
	OA_LOG_INFO(OaLogComponent::ML, "OaAlmPriorAg::Generate — batch=%d, maxlen=%d, cache=%d",
		InBatchSize, InMaxLen, InUseCache ? 1 : 0);
	auto& ctx = OaContext::GetDefault();
	const OaI64 vocabSize = Config_.VocabSize;

	// Per-row token sequences (growing). Done rows are padded with [PAD] to keep the
	// returned tensor rectangular; their logits are never sampled.
	std::vector<std::vector<OaI32>> rows(static_cast<size_t>(InBatchSize),
		std::vector<OaI32>{Config_.SomToken});
	std::vector<bool> done(static_cast<size_t>(InBatchSize), false);

	// KV-cache is not implemented yet. Feed the growing prefix and sample the last
	// position each step; the Transformer blocks reuse their weights and rebuild only
	// the sequence-length-dependent causal mask.
	(void)InUseCache;
	for (OaI32 step = 0; step < InMaxLen; ++step) {
		const OaI32 curLen = step + 1;   // uniform dense length this step
		std::vector<OaI32> dense(static_cast<size_t>(InBatchSize) * static_cast<size_t>(curLen), Config_.PadToken);
		for (OaI32 b = 0; b < InBatchSize; ++b) {
			const auto& r = rows[static_cast<size_t>(b)];
			const OaI32 n = static_cast<OaI32>(r.size());   // active row: n==curLen; done row: n<curLen (rest stays [PAD])
			for (OaI32 t = 0; t < n; ++t) {
				dense[static_cast<size_t>(b) * curLen + t] = r[static_cast<size_t>(t)];
			}
		}
		OaMatrix ids = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(dense.data()), dense.size() * sizeof(OaI32)),
			OaMatrixShape{InBatchSize, curLen}, OaScalarType::Int32);

		auto logits = InTextFeatures != nullptr
			? ForwardConditioned(ids, *InTextFeatures)
			: Forward(ids);   // [B, curLen, VocabSize]
		auto lastLogits = OaFnMatrix::Reshape(
			OaFnMatrix::Slice(logits, 1, curLen - 1, curLen),
			OaMatrixShape{InBatchSize, vocabSize});
		auto sampled = OaFnMatrix::SampleLogits(
			lastLogits, InTemperature, InTopK, InTopP);
		(void)ctx.Execute(); (void)ctx.Sync();
		const OaI32* sampledHost = sampled.DataAs<const OaI32>();
		bool allDone = true;
		for (OaI32 b = 0; b < InBatchSize; ++b) {
			if (done[static_cast<size_t>(b)]) { continue; }
			const OaI32 next = sampledHost[b];
			rows[static_cast<size_t>(b)].push_back(next);
			if (next == Config_.EomToken) { done[static_cast<size_t>(b)] = true; }
			else { allDone = false; }
		}
		if (allDone) { break; }
	}

	// Flatten to [B, maxLen] (rows padded with [PAD] to the longest row for a dense tensor).
	size_t maxRow = 0;
	for (const auto& r : rows) maxRow = std::max(maxRow, r.size());
	const OaI32 outLen = static_cast<OaI32>(maxRow);
	std::vector<OaI32> flat(static_cast<size_t>(InBatchSize) * maxRow, Config_.PadToken);
	for (OaI32 b = 0; b < InBatchSize; ++b) {
		const auto& r = rows[static_cast<size_t>(b)];
		for (size_t t = 0; t < r.size(); ++t) flat[static_cast<size_t>(b) * maxRow + t] = r[t];
	}
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(flat.data()), flat.size() * sizeof(OaI32)),
		OaMatrixShape{InBatchSize, outLen}, OaScalarType::Int32);
}

OaMatrix OaAlmPriorAg::DecodeToMotion(
	const OaMatrix& InTokenIds,
	OaAlmTokenizerAg& InTokenizer
) {
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute(); (void)ctx.Sync();
	const OaI32 batch = static_cast<OaI32>(InTokenIds.Size(0));
	const OaI32 len   = static_cast<OaI32>(InTokenIds.Size(1));
	const OaI32* ids  = InTokenIds.DataAs<const OaI32>();
	if (batch == 0 or len == 0) return {};

	OaVec<OaMatrix> levelIds;
	OaVec<OaI32>    seqLens;
	seqLens.Reserve(batch);

	for (OaI32 b = 0; b < batch; ++b) {
		OaI32 seqLen = len;
		for (OaI32 t = 0; t < len; ++t) {
			if (ids[b * len + t] == Config_.EomToken) { seqLen = t; break; }
		}
		seqLens.PushBack(seqLen);
	}

	const OaI32 minSeqLen = *std::min_element(seqLens.begin(), seqLens.end());
	const OaI32 tokLen = std::max<OaI32>(0, minSeqLen - 1);  // exclude [SOM]
	if (tokLen <= 0) return {};

	std::vector<OaI32> flat;
	flat.reserve(static_cast<size_t>(batch) * tokLen);
	for (OaI32 b = 0; b < batch; ++b) {
		for (OaI32 t = 0; t < tokLen; ++t) {
			flat.push_back(ids[b * len + t + 1]);  // skip [SOM]
		}
	}
	OaMatrix idx = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(flat.data()), flat.size() * sizeof(OaI32)),
		OaMatrixShape{static_cast<OaI64>(batch) * tokLen, 1}, OaScalarType::Int32);

	levelIds.PushBack(idx);
	return InTokenizer.Detokenize(levelIds, batch, tokLen);
}
