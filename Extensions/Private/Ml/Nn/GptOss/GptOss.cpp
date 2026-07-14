#include <Ml/Nn/GptOss/GptOss.h>

#include <Oa/Core/Yaml.h>

#include <cmath>

namespace {

bool Near(OaF32 InA, OaF32 InB) {
	return std::abs(InA - InB) <= 1e-6F * std::max<OaF32>(1.0F, std::abs(InB));
}

OaStatus ValidatePublishedJson(const OaYaml::Node& InRoot, const OaGptOssConfig& InConfig) {
#ifdef OA_HAS_YAML_CPP
	if (OaYaml::Get<OaString>(InRoot, "model_type", "") != "gpt_oss")
		return OaStatus::Error(OaStatusCode::InvalidArgument, "gpt-oss config: model_type must be gpt_oss");
	if (OaYaml::Get<bool>(InRoot, "attention_bias", false) != InConfig.AttentionBias)
		return OaStatus::Error(OaStatusCode::InvalidArgument, "gpt-oss config: attention_bias mismatch");
	if (OaYaml::Get<bool>(InRoot, "tie_word_embeddings", true) != InConfig.TieWordEmbeddings)
		return OaStatus::Error(OaStatusCode::InvalidArgument, "gpt-oss config: tied embeddings are unsupported");

	const auto layers = InRoot["layer_types"];
	if (not layers or not layers.IsSequence()
		or layers.size() != static_cast<size_t>(InConfig.NumLayers)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "gpt-oss config: invalid layer_types count");
	}
	for (OaI32 i = 0; i < InConfig.NumLayers; ++i) {
		const OaString expected = InConfig.LayerUsesSlidingAttention(i)
			? "sliding_attention" : "full_attention";
		if (layers[static_cast<size_t>(i)].as<OaString>() != expected) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"gpt-oss config: layer_types must alternate sliding/full from layer zero");
		}
	}
	const auto quant = InRoot["quantization_config"];
	if (not quant or OaYaml::Get<OaString>(quant, "quant_method", "") != "mxfp4")
		return OaStatus::Error(OaStatusCode::InvalidArgument, "gpt-oss config: expected MXFP4 experts");
	return OaStatus::Ok();
#else
	(void)InRoot; (void)InConfig;
	return OaStatus::Error(OaStatusCode::Unavailable, "gpt-oss config parsing requires yaml-cpp");
#endif
}

} // namespace

OaGptOssConfig OaGptOssConfig::Preset20B() { return {}; }

OaGptOssConfig OaGptOssConfig::Preset120B() {
	auto out = Preset20B();
	out.NumLayers = 36;
	out.NumExperts = 128;
	return out;
}

OaStatus OaGptOssConfig::Validate() const {
	if (VocabSize <= 0 or NumLayers <= 0 or HiddenSize <= 0 or IntermediateSize <= 0
		or NumAttentionHeads <= 0 or NumKvHeads <= 0 or HeadDim <= 0
		or NumExperts <= 0 or ExpertsPerToken <= 0 or ExpertsPerToken > NumExperts
		or SlidingWindow <= 0 or OriginalContextLength <= 0
		or MaxPositionEmbeddings < OriginalContextLength) {
		return OaStatus::InvalidArgument("gpt-oss config: dimensions/counts are inconsistent");
	}
	if (NumAttentionHeads % NumKvHeads != 0 or QueryWidth() <= 0 or KvWidth() <= 0)
		return OaStatus::InvalidArgument("gpt-oss config: attention heads must form integer GQA groups");
	if (HeadDim % 2 != 0 or RopeTheta <= 0.0F or RopeScalingFactor < 1.0F
		or RopeNtkAlpha <= 0.0F or RopeNtkBeta <= RopeNtkAlpha)
		return OaStatus::InvalidArgument("gpt-oss config: invalid YaRN/RoPE parameters");
	if (RmsNormEps <= 0.0F or SwiGluAlpha <= 0.0F or SwiGluLimit <= 0.0F)
		return OaStatus::InvalidArgument("gpt-oss config: invalid normalization/activation parameters");
	if (PadToken < 0 or PadToken >= VocabSize or EosToken < 0 or EosToken >= VocabSize)
		return OaStatus::InvalidArgument("gpt-oss config: special token is outside vocabulary");
	if (not AttentionBias or TieWordEmbeddings or not MxFp4Experts)
		return OaStatus::InvalidArgument("gpt-oss config: published bias/embedding/MXFP4 contract changed");
	return OaStatus::Ok();
}

OaResult<OaGptOssConfig> OaGptOssConfig::FromJson(const OaString& InPath) {
#ifdef OA_HAS_YAML_CPP
	try {
		const auto root = OaYaml::LoadFile(InPath);
		OaGptOssConfig out;
		out.VocabSize = OaYaml::Get<OaI32>(root, "vocab_size", 0);
		out.NumLayers = OaYaml::Get<OaI32>(root, "num_hidden_layers", 0);
		out.HiddenSize = OaYaml::Get<OaI32>(root, "hidden_size", 0);
		out.IntermediateSize = OaYaml::Get<OaI32>(root, "intermediate_size", 0);
		out.NumAttentionHeads = OaYaml::Get<OaI32>(root, "num_attention_heads", 0);
		out.NumKvHeads = OaYaml::Get<OaI32>(root, "num_key_value_heads", 0);
		out.HeadDim = OaYaml::Get<OaI32>(root, "head_dim", 0);
		out.NumExperts = OaYaml::Get<OaI32>(root, "num_local_experts", 0);
		out.ExpertsPerToken = OaYaml::Get<OaI32>(root, "num_experts_per_tok",
			OaYaml::Get<OaI32>(root, "experts_per_token", 0));
		out.SlidingWindow = OaYaml::Get<OaI32>(root, "sliding_window", 0);
		out.OriginalContextLength = OaYaml::Get<OaI32>(root, "initial_context_length", 0);
		out.MaxPositionEmbeddings = OaYaml::Get<OaI32>(root, "max_position_embeddings", 0);
		out.RopeTheta = OaYaml::Get<OaF32>(root, "rope_theta", 0.0F);
		out.RmsNormEps = OaYaml::Get<OaF32>(root, "rms_norm_eps", 0.0F);
		out.SwiGluLimit = OaYaml::Get<OaF32>(root, "swiglu_limit", 0.0F);
		out.PadToken = OaYaml::Get<OaI32>(root, "pad_token_id", -1);
		out.EosToken = OaYaml::Get<OaI32>(root, "eos_token_id", -1);
		out.AttentionBias = OaYaml::Get<bool>(root, "attention_bias", false);
		out.TieWordEmbeddings = OaYaml::Get<bool>(root, "tie_word_embeddings", true);
		const auto rope = root["rope_scaling"];
		out.RopeScalingFactor = OaYaml::Get<OaF32>(rope, "factor", 0.0F);
		out.RopeNtkAlpha = OaYaml::Get<OaF32>(rope, "beta_slow", 0.0F);
		out.RopeNtkBeta = OaYaml::Get<OaF32>(rope, "beta_fast", 0.0F);

		if (auto status = out.Validate(); not status.IsOk()) return status;
		if (auto status = ValidatePublishedJson(root, out); not status.IsOk()) return status;
		return out;
	} catch (const OaYaml::Exception& error) {
		return OaStatus::Error(OaStatusCode::FileCorrupt,
			OaString("gpt-oss config parse failed: ") + error.what());
	}
#else
	(void)InPath;
	return OaStatus::Error(OaStatusCode::Unavailable, "gpt-oss config parsing requires yaml-cpp");
#endif
}

bool OaGptOssConfig::IsPublished20B() const {
	const auto p = Preset20B();
	return VocabSize == p.VocabSize and NumLayers == p.NumLayers
		and HiddenSize == p.HiddenSize and IntermediateSize == p.IntermediateSize
		and NumAttentionHeads == p.NumAttentionHeads and NumKvHeads == p.NumKvHeads
		and HeadDim == p.HeadDim and NumExperts == p.NumExperts
		and ExpertsPerToken == p.ExpertsPerToken and SlidingWindow == p.SlidingWindow
		and OriginalContextLength == p.OriginalContextLength
		and MaxPositionEmbeddings == p.MaxPositionEmbeddings
		and Near(RopeTheta, p.RopeTheta) and Near(RopeScalingFactor, p.RopeScalingFactor)
		and Near(RopeNtkAlpha, p.RopeNtkAlpha) and Near(RopeNtkBeta, p.RopeNtkBeta)
		and Near(RmsNormEps, p.RmsNormEps) and Near(SwiGluLimit, p.SwiGluLimit)
		and PadToken == p.PadToken and EosToken == p.EosToken;
}

bool OaGptOssConfig::IsPublished120B() const {
	auto copy = *this;
	copy.NumLayers = 24;
	copy.NumExperts = 32;
	return NumLayers == 36 and NumExperts == 128 and copy.IsPublished20B();
}
