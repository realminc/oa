#include <ml/nn/gptOss/gptOss.h>

#include <oa/core/yaml.h>

#include <cmath>

namespace {

bool near(oa::F32 inA, oa::F32 inB) {
	return std::abs(inA - inB) <= 1e-6F * std::max<oa::F32>(1.0F, std::abs(inB));
}

oa::Status validatePublishedJson(const oa::Yaml::Node& inRoot, const oa::GptOssConfig& inConfig) {
#ifdef OA_HAS_YAML_CPP
	if (oa::Yaml::get<oa::String>(inRoot, "model_type", "") != "gpt_oss")
		return oa::Status::error(oa::StatusCode::InvalidArgument, "gpt-oss config: model_type must be gpt_oss");
	if (oa::Yaml::get<bool>(inRoot, "attention_bias", false) != inConfig.attentionBias)
		return oa::Status::error(oa::StatusCode::InvalidArgument, "gpt-oss config: attention_bias mismatch");
	if (oa::Yaml::get<bool>(inRoot, "tie_word_embeddings", true) != inConfig.tieWordEmbeddings)
		return oa::Status::error(oa::StatusCode::InvalidArgument, "gpt-oss config: tied embeddings are unsupported");

	const auto layers = inRoot["layer_types"];
	if (not layers or not layers.IsSequence()
		or layers.size() != static_cast<size_t>(inConfig.numLayers)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "gpt-oss config: invalid layer_types count");
	}
	for (oa::I32 i = 0; i < inConfig.numLayers; ++i) {
		const oa::String expected = inConfig.layerUsesSlidingAttention(i)
			? "sliding_attention" : "full_attention";
		if (layers[static_cast<size_t>(i)].as<oa::String>() != expected) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"gpt-oss config: layer_types must alternate sliding/full from layer zero");
		}
	}
	const auto quant = inRoot["quantization_config"];
	if (not quant or oa::Yaml::get<oa::String>(quant, "quant_method", "") != "mxfp4")
		return oa::Status::error(oa::StatusCode::InvalidArgument, "gpt-oss config: expected MXFP4 experts");
	return oa::Status::ok();
#else
	(void)inRoot; (void)inConfig;
	return oa::Status::error(oa::StatusCode::Unavailable, "gpt-oss config parsing requires yaml-cpp");
#endif
}

} // namespace

oa::GptOssConfig oa::GptOssConfig::preset20B() { return {}; }

oa::GptOssConfig oa::GptOssConfig::preset120B() {
	auto out = preset20B();
	out.numLayers = 36;
	out.numExperts = 128;
	return out;
}

oa::Status oa::GptOssConfig::validate() const {
	if (vocabSize <= 0 or numLayers <= 0 or hiddenSize <= 0 or intermediateSize <= 0
		or numAttentionHeads <= 0 or numKvHeads <= 0 or headDim <= 0
		or numExperts <= 0 or expertsPerToken <= 0 or expertsPerToken > numExperts
		or slidingWindow <= 0 or originalContextLength <= 0
		or maxPositionEmbeddings < originalContextLength) {
		return oa::Status::invalidArgument("gpt-oss config: dimensions/counts are inconsistent");
	}
	if (numAttentionHeads % numKvHeads != 0 or queryWidth() <= 0 or kvWidth() <= 0)
		return oa::Status::invalidArgument("gpt-oss config: attention heads must form integer GQA groups");
	if (headDim % 2 != 0 or ropeTheta <= 0.0F or ropeScalingFactor < 1.0F
		or ropeNtkAlpha <= 0.0F or ropeNtkBeta <= ropeNtkAlpha)
		return oa::Status::invalidArgument("gpt-oss config: invalid YaRN/RoPE parameters");
	if (rmsNormEps <= 0.0F or swiGluAlpha <= 0.0F or swiGluLimit <= 0.0F)
		return oa::Status::invalidArgument("gpt-oss config: invalid normalization/activation parameters");
	if (padToken < 0 or padToken >= vocabSize or eosToken < 0 or eosToken >= vocabSize)
		return oa::Status::invalidArgument("gpt-oss config: special token is outside vocabulary");
	if (not attentionBias or tieWordEmbeddings or not mxFp4Experts)
		return oa::Status::invalidArgument("gpt-oss config: published bias/embedding/MXFP4 contract changed");
	return oa::Status::ok();
}

oa::Result<oa::GptOssConfig> oa::GptOssConfig::fromJson(const oa::String& inPath) {
#ifdef OA_HAS_YAML_CPP
	try {
		const auto root = oa::Yaml::loadFile(inPath);
		oa::GptOssConfig out;
		out.vocabSize = oa::Yaml::get<oa::I32>(root, "vocab_size", 0);
		out.numLayers = oa::Yaml::get<oa::I32>(root, "num_hidden_layers", 0);
		out.hiddenSize = oa::Yaml::get<oa::I32>(root, "hidden_size", 0);
		out.intermediateSize = oa::Yaml::get<oa::I32>(root, "intermediate_size", 0);
		out.numAttentionHeads = oa::Yaml::get<oa::I32>(root, "num_attention_heads", 0);
		out.numKvHeads = oa::Yaml::get<oa::I32>(root, "num_key_value_heads", 0);
		out.headDim = oa::Yaml::get<oa::I32>(root, "head_dim", 0);
		out.numExperts = oa::Yaml::get<oa::I32>(root, "num_local_experts", 0);
		out.expertsPerToken = oa::Yaml::get<oa::I32>(root, "num_experts_per_tok",
			oa::Yaml::get<oa::I32>(root, "experts_per_token", 0));
		out.slidingWindow = oa::Yaml::get<oa::I32>(root, "sliding_window", 0);
		out.originalContextLength = oa::Yaml::get<oa::I32>(root, "initial_context_length", 0);
		out.maxPositionEmbeddings = oa::Yaml::get<oa::I32>(root, "max_position_embeddings", 0);
		out.ropeTheta = oa::Yaml::get<oa::F32>(root, "rope_theta", 0.0F);
		out.rmsNormEps = oa::Yaml::get<oa::F32>(root, "rms_norm_eps", 0.0F);
		out.swiGluLimit = oa::Yaml::get<oa::F32>(root, "swiglu_limit", 0.0F);
		out.padToken = oa::Yaml::get<oa::I32>(root, "pad_token_id", -1);
		out.eosToken = oa::Yaml::get<oa::I32>(root, "eos_token_id", -1);
		out.attentionBias = oa::Yaml::get<bool>(root, "attention_bias", false);
		out.tieWordEmbeddings = oa::Yaml::get<bool>(root, "tie_word_embeddings", true);
		const auto rope = root["rope_scaling"];
		out.ropeScalingFactor = oa::Yaml::get<oa::F32>(rope, "factor", 0.0F);
		out.ropeNtkAlpha = oa::Yaml::get<oa::F32>(rope, "beta_slow", 0.0F);
		out.ropeNtkBeta = oa::Yaml::get<oa::F32>(rope, "beta_fast", 0.0F);

		if (auto status = out.validate(); not status.isOk()) return status;
		if (auto status = validatePublishedJson(root, out); not status.isOk()) return status;
		return out;
	} catch (const oa::Yaml::Exception& error) {
		return oa::Status::error(oa::StatusCode::FileCorrupt,
			oa::String("gpt-oss config parse failed: ") + error.what());
	}
#else
	(void)inPath;
	return oa::Status::error(oa::StatusCode::Unavailable, "gpt-oss config parsing requires yaml-cpp");
#endif
}

bool oa::GptOssConfig::isPublished20B() const {
	const auto p = preset20B();
	return vocabSize == p.vocabSize and numLayers == p.numLayers
		and hiddenSize == p.hiddenSize and intermediateSize == p.intermediateSize
		and numAttentionHeads == p.numAttentionHeads and numKvHeads == p.numKvHeads
		and headDim == p.headDim and numExperts == p.numExperts
		and expertsPerToken == p.expertsPerToken and slidingWindow == p.slidingWindow
		and originalContextLength == p.originalContextLength
		and maxPositionEmbeddings == p.maxPositionEmbeddings
		and near(ropeTheta, p.ropeTheta) and near(ropeScalingFactor, p.ropeScalingFactor)
		and near(ropeNtkAlpha, p.ropeNtkAlpha) and near(ropeNtkBeta, p.ropeNtkBeta)
		and near(rmsNormEps, p.rmsNormEps) and near(swiGluLimit, p.swiGluLimit)
		and padToken == p.padToken and eosToken == p.eosToken;
}

bool oa::GptOssConfig::isPublished120B() const {
	auto copy = *this;
	copy.numLayers = 24;
	copy.numExperts = 32;
	return numLayers == 36 and numExperts == 128 and copy.isPublished20B();
}
