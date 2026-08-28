#include <ml/nn/alm/almAg.h>

#include <oa/ml/modelFile.h>
#include <oa/runtime/executionSession.h>

#include <cstring>
#include <stdexcept>

namespace {

constexpr oa::U32 kAlmAgBundleMagic = 0x47414D41U; // "AMAG"
constexpr oa::U32 kAlmAgBundleVersion = 3;

#pragma pack(push, 1)
struct AlmAgBundleConfigV3 {
	oa::U32 magic = kAlmAgBundleMagic;
	oa::U32 version = kAlmAgBundleVersion;
	oa::I32 inputDim = 0;
	oa::I32 width = 0;
	oa::I32 codeDim = 0;
	oa::I32 numCodes = 0;
	oa::I32 downT = 0;
	oa::I32 depth = 0;
	oa::F32 commitBeta = 0.0F;
	oa::F32 emaDecay = 0.0F;
	oa::F32 emaEps = 0.0F;
	oa::F32 deadThresh = 0.0F;
	oa::I32 dModel = 0;
	oa::I32 numHeads = 0;
	oa::I32 numLayers = 0;
	oa::I32 dFfn = 0;
	oa::I32 textFeatureDim = 0;
	oa::U8 ffnType = 0;
	oa::I32 moeNumExperts = 0;
	oa::I32 moeExpertsPerToken = 0;
	oa::I32 moeEvery = 0;
	oa::F32 moeBalanceRate = 0.0F;
	oa::F32 moeAuxLossAlpha = 0.0F;
	oa::F32 moeRouterZLossBeta = 0.0F;
	oa::I32 seqLen = 0;
	oa::I32 maxSeqLen = 0;
	oa::I32 maxGenLen = 0;
	oa::U32 clipMergesBytes = 0;
	char textEncoder[96] = {};
};
#pragma pack(pop)

AlmAgBundleConfigV3 encodeConfig(const oa::AlmAgConfig& inConfig) {
	AlmAgBundleConfigV3 out;
	const auto& t = inConfig.tokenizer;
	const auto& p = inConfig.prior;
	out.inputDim = t.inputDim;
	out.width = t.width;
	out.codeDim = t.codeDim;
	out.numCodes = t.numCodes;
	out.downT = t.downT;
	out.depth = t.depth;
	out.commitBeta = t.commitBeta;
	out.emaDecay = t.emaDecay;
	out.emaEps = t.emaEps;
	out.deadThresh = t.deadThresh;
	out.dModel = p.dModel;
	out.numHeads = p.numHeads;
	out.numLayers = p.numLayers;
	out.dFfn = p.dFfn;
	out.textFeatureDim = p.textFeatureDim;
	out.ffnType = static_cast<oa::U8>(p.ffnType);
	out.moeNumExperts = p.moeNumExperts;
	out.moeExpertsPerToken = p.moeExpertsPerToken;
	out.moeEvery = p.moeEvery;
	out.moeBalanceRate = p.moeBalanceRate;
	out.moeAuxLossAlpha = p.moeAuxLossAlpha;
	out.moeRouterZLossBeta = p.moeRouterZLossBeta;
	out.seqLen = p.seqLen;
	out.maxSeqLen = p.maxSeqLen;
	out.maxGenLen = p.maxGenLen;
	out.clipMergesBytes = inConfig.clipMergesBytes;
	std::strncpy(out.textEncoder, inConfig.textEncoder.cStr(), sizeof(out.textEncoder) - 1);
	return out;
}

oa::AlmAgConfig decodeConfig(const AlmAgBundleConfigV3& inConfig) {
	oa::AlmAgConfig out;
	auto& t = out.tokenizer;
	auto& p = out.prior;
	t.inputDim = inConfig.inputDim;
	t.width = inConfig.width;
	t.codeDim = inConfig.codeDim;
	t.numCodes = inConfig.numCodes;
	t.downT = inConfig.downT;
	t.depth = inConfig.depth;
	t.commitBeta = inConfig.commitBeta;
	t.emaDecay = inConfig.emaDecay;
	t.emaEps = inConfig.emaEps;
	t.deadThresh = inConfig.deadThresh;
	p.syncVocab(t.numCodes);
	p.dModel = inConfig.dModel;
	p.numHeads = inConfig.numHeads;
	p.numLayers = inConfig.numLayers;
	p.dFfn = inConfig.dFfn;
	p.textFeatureDim = inConfig.textFeatureDim;
	p.ffnType = static_cast<oa::AlmFfnType>(inConfig.ffnType);
	p.moeNumExperts = inConfig.moeNumExperts;
	p.moeExpertsPerToken = inConfig.moeExpertsPerToken;
	p.moeEvery = inConfig.moeEvery;
	p.moeBalanceRate = inConfig.moeBalanceRate;
	p.moeAuxLossAlpha = inConfig.moeAuxLossAlpha;
	p.moeRouterZLossBeta = inConfig.moeRouterZLossBeta;
	p.seqLen = inConfig.seqLen;
	p.maxSeqLen = inConfig.maxSeqLen;
	p.maxGenLen = inConfig.maxGenLen;
	out.clipMergesBytes = inConfig.clipMergesBytes;
	out.textEncoder = inConfig.textEncoder;
	return out;
}

oa::Status validateConfig(const oa::AlmAgConfig& inConfig) {
	if (inConfig.tokenizer.numCodes <= 0 or
		inConfig.prior.numCodes != inConfig.tokenizer.numCodes or
		inConfig.prior.vocabSize != inConfig.tokenizer.numCodes + 3) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::AlmAg tokenizer/prior vocabulary contract does not match");
	}
	if (inConfig.prior.textFeatureDim > 0 and inConfig.textEncoder.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::AlmAg conditioned prior requires an exact text encoder identity");
	}
	if (inConfig.prior.numHeads <= 0 or
		inConfig.prior.dModel % inConfig.prior.numHeads != 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::AlmAg prior dModel must be divisible by a positive attention-head count");
	}
	if (inConfig.clipMergesBytes > 0 and
		(inConfig.prior.textFeatureDim != oa::ClipTextConfig::viTL14().projectionDim or
		 inConfig.textEncoder != "openai/clip-vit-large-patch14")) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"native oa::AlmAg CLIP bundle requires the pinned ViT-L/14 identity and feature dimension");
	}
	return oa::Status::ok();
}

oa::Status validateWeights(oa::Module& inModule, const oa::ModelFile& inFile) {
	const auto expected = inModule.allNamedParameterPtrs();
	if (inFile.weightIndex.size() != expected.size()) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
			"oa::AlmAg bundle parameter count does not match its architecture");
	}
	for (const auto& named : expected) {
		const oa::ModelTensorEntry* entry = inFile.findWeight(named.path.cStr());
		if (entry == nullptr or entry->dtype != named.param->data.getDtype() or
			entry->rank != named.param->data.rank() or
			entry->numBytes != static_cast<oa::U64>(named.param->data.byteSize())) {
			return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
				oa::String("oa::AlmAg bundle tensor mismatch: ") + named.path);
		}
		for (oa::I32 d = 0; d < named.param->data.rank(); ++d) {
			if (entry->shape[d] != static_cast<oa::U64>(named.param->data.size(d))) {
				return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
					oa::String("oa::AlmAg bundle shape mismatch: ") + named.path);
			}
		}
	}
	struct NamedBuffer { oa::String path; const oa::ModuleBuffer* buffer = nullptr; };
	oa::Vector<NamedBuffer> buffers;
	auto collect = [&](auto&& self, const oa::Module& module, const oa::String& prefix) -> void {
		for (const auto& buffer : module.buffers()) {
			if (not buffer.persistent or buffer.data.isEmpty()) continue;
			buffers.pushBack({prefix.empty() ? buffer.name : prefix + "." + buffer.name, &buffer});
		}
		for (const auto& child : module.children()) {
			const oa::String path = prefix.empty() ? child.name : prefix + "." + child.name;
			self(self, *child.module, path);
		}
	};
	collect(collect, inModule, oa::String());
	if (inFile.stateIndex.size() != buffers.size()) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
			"oa::AlmAg bundle persistent-state count does not match its architecture");
	}
	for (const auto& named : buffers) {
		const oa::ModelTensorEntry* entry = inFile.findState(named.path.cStr());
		const oa::Matrix& data = named.buffer->data;
		if (entry == nullptr or entry->dtype != data.getDtype() or
			entry->rank != data.rank() or entry->numBytes != static_cast<oa::U64>(data.byteSize())) {
			return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
				oa::String("oa::AlmAg bundle state mismatch: ") + named.path);
		}
		for (oa::I32 d = 0; d < data.rank(); ++d) {
			if (entry->shape[d] != static_cast<oa::U64>(data.size(d))) {
				return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
					oa::String("oa::AlmAg bundle state shape mismatch: ") + named.path);
			}
		}
	}
	return oa::Status::ok();
}

} // namespace

oa::AlmAg::AlmAg(const oa::AlmAgConfig& inConfig) : config_(inConfig) {
	const oa::Status valid = validateConfig(config_);
	if (not valid.isOk()) throw std::invalid_argument(valid.getMessage().cStr());
	tokenizer_ = oa::makeShared<oa::AlmTokenizerAg>(config_.tokenizer);
	prior_ = oa::makeShared<oa::AlmPriorAg>(config_.prior);
	if (config_.clipMergesBytes > 0) {
		textEncoder_ = oa::makeShared<oa::ClipTextAg>(oa::ClipTextConfig::viTL14());
		registerBuffer("text_tokenizer_merges", oa::FnMatrix::empty(
			oa::MatrixShape{config_.clipMergesBytes}, oa::ScalarType::UInt8), true);
	}
	registerChildren();
}

oa::AlmAg::AlmAg(oa::SharedPtr<oa::AlmTokenizerAg> inTokenizer,
	oa::SharedPtr<oa::AlmPriorAg> inPrior, oa::StringView inTextEncoder)
	: tokenizer_(std::move(inTokenizer)), prior_(std::move(inPrior)) {
	if (not tokenizer_ or not prior_) throw std::invalid_argument("oa::AlmAg children cannot be null");
	config_.tokenizer = tokenizer_->config();
	config_.prior = prior_->config();
	config_.textEncoder = oa::String(inTextEncoder);
	const oa::Status valid = validateConfig(config_);
	if (not valid.isOk()) throw std::invalid_argument(valid.getMessage().cStr());
	registerChildren();
}

oa::AlmAg::AlmAg(oa::SharedPtr<oa::AlmTokenizerAg> inTokenizer,
	oa::SharedPtr<oa::AlmPriorAg> inPrior, oa::SharedPtr<oa::ClipTextAg> inTextEncoder,
	oa::Span<const oa::U8> inClipMerges, oa::StringView inTextEncoderIdentity)
	: tokenizer_(std::move(inTokenizer)), prior_(std::move(inPrior)),
	  textEncoder_(std::move(inTextEncoder)) {
	if (not tokenizer_ or not prior_ or not textEncoder_ or inClipMerges.empty())
		throw std::invalid_argument("native oa::AlmAg children and CLIP merges cannot be empty");
	config_.tokenizer = tokenizer_->config();
	config_.prior = prior_->config();
	config_.textEncoder = oa::String(inTextEncoderIdentity);
	config_.clipMergesBytes = static_cast<oa::U32>(inClipMerges.size());
	const oa::Status valid = validateConfig(config_);
	if (not valid.isOk()) throw std::invalid_argument(valid.getMessage().cStr());
	registerBuffer("text_tokenizer_merges", oa::FnMatrix::fromBytes(
		inClipMerges, oa::MatrixShape{config_.clipMergesBytes}, oa::ScalarType::UInt8), true);
	registerChildren();
}

void oa::AlmAg::registerChildren() {
	registerModule("tokenizer", tokenizer_);
	registerModule("prior", prior_);
	if (textEncoder_) registerModule("text_encoder", textEncoder_);
}

oa::Matrix oa::AlmAg::forward(const oa::Matrix& inTokenIds) {
	return prior_->forward(inTokenIds);
}

oa::Matrix oa::AlmAg::forwardConditioned(
	const oa::Matrix& inTokenIds, const oa::Matrix& inTextFeatures) {
	return prior_->forwardConditioned(inTokenIds, inTextFeatures);
}

oa::Vector<oa::Matrix> oa::AlmAg::tokenize(
	const oa::Matrix& inMotion, oa::I32 inBatch, oa::I32 inFrames) {
	return tokenizer_->tokenize(inMotion, inBatch, inFrames);
}

oa::Matrix oa::AlmAg::detokenize(
	const oa::Vector<oa::Matrix>& inTokenIds, oa::I32 inBatch, oa::I32 inTokenLength) {
	return tokenizer_->detokenize(inTokenIds, inBatch, inTokenLength);
}

oa::Matrix oa::AlmAg::generateMotion(oa::I32 inBatchSize, oa::F32 inTemperature,
	oa::I32 inTopK, oa::F32 inTopP, oa::I32 inMaxTokens) {
	auto tokens = prior_->generate(
		inBatchSize, inTemperature, inTopK, inTopP, inMaxTokens);
	return prior_->decodeToMotion(tokens, *tokenizer_);
}

oa::Matrix oa::AlmAg::generateMotionConditioned(const oa::Matrix& inTextFeatures,
	oa::F32 inTemperature, oa::I32 inTopK, oa::F32 inTopP, oa::I32 inMaxTokens) {
	auto tokens = prior_->generateConditioned(
		inTextFeatures, inTemperature, inTopK, inTopP, inMaxTokens);
	return prior_->decodeToMotion(tokens, *tokenizer_);
}

oa::Result<oa::Matrix> oa::AlmAg::encodePrompt(oa::StringView inPrompt) {
	const oa::ModuleBuffer* mergesBuffer = nullptr;
	for (const auto& buffer : buffers_) {
		if (buffer.name == "text_tokenizer_merges") {
			mergesBuffer = &buffer;
			break;
		}
	}
	if (not textEncoder_ or mergesBuffer == nullptr)
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::AlmAg bundle has no native CLIP text encoder/tokenizer");
	if (not clipTokenizer_) {
		clipTokenizer_ = oa::makeUnique<oa::ClipTokenizer>();
		const auto& merges = mergesBuffer->data;
		auto& ctx = oa::ExecutionSession::getActive();
		OA_RETURN_IF_ERROR(ctx.submitAndWait());
		OA_RETURN_IF_ERROR(clipTokenizer_->loadMerges(oa::Span<const oa::U8>(
			merges.dataAs<const oa::U8>(), static_cast<oa::Usize>(merges.numElements()))));
	}
	const oa::String prompt(inPrompt);
	auto encoded = clipTokenizer_->encode(oa::Span<const oa::String>(&prompt, 1),
		textEncoder_->config().contextLength, true);
	if (encoded.isError()) return encoded.getStatus();
	const auto& batch = encoded.getValue();
	auto ids = oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(batch.tokenIds.data(), batch.tokenIds.size()),
		oa::MatrixShape{1, batch.contextLength}, oa::ScalarType::Int32);
	auto eos = oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(batch.flatEosRows.data(), batch.flatEosRows.size()),
		oa::MatrixShape{1}, oa::ScalarType::Int32);
	return textEncoder_->forwardTokens(ids, eos);
}

oa::Result<oa::Matrix> oa::AlmAg::generateMotionPrompt(oa::StringView inPrompt,
	oa::F32 inTemperature, oa::I32 inTopK, oa::F32 inTopP, oa::I32 inMaxTokens) {
	auto feature = encodePrompt(inPrompt);
	if (feature.isError()) return feature.getStatus();
	return generateMotionConditioned(feature.getValue(), inTemperature, inTopK, inTopP, inMaxTokens);
}

oa::Status oa::AlmAg::saveBundle(
	oa::Engine& inEngine, const oa::String& inPath) const
{
	oa::ModelFile oam;
	std::strncpy(oam.config.architecture, "OaAlmAg", sizeof(oam.config.architecture) - 1);
	oam.config.configVersion = kAlmAgBundleVersion;
	oam.config.dModel = static_cast<oa::U32>(config_.prior.dModel);
	oam.config.nLayers = static_cast<oa::U32>(config_.prior.numLayers);
	oam.config.dVocab = static_cast<oa::U32>(config_.prior.vocabSize);
	oam.config.weightDtype = static_cast<oa::U8>(oa::FnMatrix::weightDtype());
	const auto arch = encodeConfig(config_);
	oam.archConfig.resize(sizeof(arch));
	std::memcpy(oam.archConfig.data(), &arch, sizeof(arch));
	oam.config.archConfigSize = static_cast<oa::U32>(oam.archConfig.size());
	OA_RETURN_IF_ERROR(saveTo(inEngine, oam));
	return oam.save(inPath);
}

oa::Result<oa::SharedPtr<oa::AlmAg>> oa::AlmAg::loadBundle(
	oa::Engine& inEngine, const oa::String& inPath)
{
	auto& ctx = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(ctx);
	auto loaded = oa::ModelFile::load(inPath);
	if (not loaded.isOk()) return loaded.getStatus();
	auto oam = std::move(loaded).getValue();
	if (std::strncmp(oam.config.architecture, "OaAlmAg", sizeof(oam.config.architecture)) != 0 or
		oam.config.configVersion != kAlmAgBundleVersion or
		oam.archConfig.size() != sizeof(AlmAgBundleConfigV3)) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
			"checkpoint is not a supported oa::AlmAg bundle");
	}
	AlmAgBundleConfigV3 encoded;
	std::memcpy(&encoded, oam.archConfig.data(), sizeof(encoded));
	if (encoded.magic != kAlmAgBundleMagic or encoded.version != kAlmAgBundleVersion) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
			"oa::AlmAg architecture payload has an invalid version");
	}
	auto config = decodeConfig(encoded);
	const oa::Status valid = validateConfig(config);
	if (not valid.isOk()) return valid;
	auto model = oa::makeShared<oa::AlmAg>(config);
	const oa::Status weights = validateWeights(*model, oam);
	if (not weights.isOk()) return weights;
	// Constructors enqueue deferred parameter initialization. Complete it before
	// the host checkpoint copy so a later submission cannot overwrite restored bytes.
	OA_RETURN_IF_ERROR(ctx.submitAndWait());
	ctx.clear();
	OA_RETURN_IF_ERROR(model->loadFrom(inEngine, oam));
	return model;
}
