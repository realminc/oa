#include <ml/nn/alm/clipTextModelTranslator.h>

#include <ml/nn/alm/clipTextAg.h>

namespace {

class ClipTextModelTranslator final : public oa::ModelTranslator {
public:
	[[nodiscard]] oa::StringView name() const noexcept override { return "clip-text"; }

	[[nodiscard]] oa::Result<oa::WeightMap>
	buildMap(const oa::WeightSource& inSource) const override {
		const auto cfg = oa::ClipTextConfig::viTL14();
		oa::WeightMap map;
		map.architecture = "OaClipTextAg";
		map.configVersion = 1;
		map.config.dModel = static_cast<oa::U32>(cfg.hiddenSize);
		map.config.nLayers = static_cast<oa::U32>(cfg.numLayers);
		map.config.dVocab = static_cast<oa::U32>(cfg.vocabSize);
		map.config.weightDtype = static_cast<oa::U8>(oa::ScalarType::Float32);
		map.archConfig.resize(sizeof(cfg));
		oa::memcpy(map.archConfig.data(), &cfg, sizeof(cfg));
		// The published file is a full image+text CLIP checkpoint. Vision and
		// logit_scale weights are deliberately outside this translator, while every
		// text-tower tensor is checked below.
		map.requireAllSourceWeights = false;

		auto add = [&](oa::StringView inName, oa::Vector<oa::I64> shape) -> oa::Status {
			const auto* info = inSource.find(inName);
			if (info == nullptr)
				return oa::Status::notFound(oa::String("CLIP tensor missing: ") + inName);
			if (info->shape.size() != shape.size())
				return oa::Status::error(oa::StatusCode::ShapeMismatch,
										 oa::String("CLIP rank mismatch: ") + inName);
			for (oa::Usize i = 0; i < shape.size(); ++i)
				if (info->shape[i] != shape[i])
					return oa::Status::error(oa::StatusCode::ShapeMismatch,
											 oa::String("CLIP shape mismatch: ") + inName);
			if (info->dtype != oa::ScalarType::Float32)
				return oa::Status::error(oa::StatusCode::DtypeMismatch,
										 oa::String("CLIP FP32 tensor required: ") + inName);
			oa::WeightMapping mapping;
			mapping.sources.pushBack(oa::String(inName));
			mapping.target = oa::String(inName);
			mapping.targetShape = oa::move(shape);
			mapping.targetDtype = oa::ScalarType::Float32;
			map.mappings.pushBack(oa::move(mapping));
			return oa::Status::ok();
		};

		OA_RETURN_IF_ERROR(
			add("text_model.embeddings.token_embedding.weight", {cfg.vocabSize, cfg.hiddenSize}));
		OA_RETURN_IF_ERROR(add("text_model.embeddings.position_embedding.weight",
							   {cfg.contextLength, cfg.hiddenSize}));
		for (oa::I32 layer = 0; layer < cfg.numLayers; ++layer) {
			const oa::String root = "text_model.encoder.layers."
				+ oa::toString(static_cast<oa::I64>(layer));
			for (const char* projection : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
				const oa::String base = root + ".self_attn." + projection;
				OA_RETURN_IF_ERROR(add(base + ".weight", {cfg.hiddenSize, cfg.hiddenSize}));
				OA_RETURN_IF_ERROR(add(base + ".bias", {cfg.hiddenSize}));
			}
			for (const char* norm : {"layer_norm1", "layer_norm2"}) {
				const oa::String base = root + "." + norm;
				OA_RETURN_IF_ERROR(add(base + ".weight", {cfg.hiddenSize}));
				OA_RETURN_IF_ERROR(add(base + ".bias", {cfg.hiddenSize}));
			}
			OA_RETURN_IF_ERROR(
				add(root + ".mlp.fc1.weight", {cfg.intermediateSize, cfg.hiddenSize}));
			OA_RETURN_IF_ERROR(add(root + ".mlp.fc1.bias", {cfg.intermediateSize}));
			OA_RETURN_IF_ERROR(
				add(root + ".mlp.fc2.weight", {cfg.hiddenSize, cfg.intermediateSize}));
			OA_RETURN_IF_ERROR(add(root + ".mlp.fc2.bias", {cfg.hiddenSize}));
		}
		OA_RETURN_IF_ERROR(add("text_model.final_layer_norm.weight", {cfg.hiddenSize}));
		OA_RETURN_IF_ERROR(add("text_model.final_layer_norm.bias", {cfg.hiddenSize}));
		OA_RETURN_IF_ERROR(add("text_projection.weight", {cfg.projectionDim, cfg.hiddenSize}));

		oa::HashSet<oa::String> expected;
		for (const auto& mapping : map.mappings)
			expected.insert(mapping.sources[0]);
		for (const auto& info : inSource.list()) {
			if (info.name == "text_model.embeddings.position_ids") {
				if (info.shape.size() != 2 or info.shape[0] != 1 or
					info.shape[1] != cfg.contextLength) {
					return oa::Status::error(oa::StatusCode::ShapeMismatch,
											 "CLIP position_ids buffer has the wrong shape");
				}
				continue;
			}
			const oa::StringView name = info.name;
			const bool textTensor = name.find("text_model.") == 0 or
				name.find("text_projection") == 0;
			if (textTensor and not expected.contains(info.name)) {
				return oa::Status::error(oa::StatusCode::FailedPrecondition,
										 oa::String("unexpected CLIP text tensor: ") + info.name);
			}
		}
		return map;
	}
};

} // namespace

oa::Status oa::registerClipTextModelTranslator() {
	if (oa::findModelTranslator("clip-text") != nullptr)
		return oa::Status::ok();
	return oa::registerModelTranslator(oa::makeUnique<ClipTextModelTranslator>());
}
