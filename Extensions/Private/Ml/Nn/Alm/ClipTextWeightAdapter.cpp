#include <Ml/Nn/Alm/ClipTextWeightAdapter.h>

#include <Ml/Nn/Alm/ClipTextAg.h>

#include <cstring>

namespace {

class OaClipTextWeightAdapter final : public OaModelWeightAdapter {
public:
	[[nodiscard]] OaStringView Name() const noexcept override { return "clip-text"; }

	[[nodiscard]] OaResult<OaWeightMap> BuildMap(
		const OaWeightSource& InSource) const override {
		const auto cfg = OaClipTextConfig::ViTL14();
		OaWeightMap map;
		map.Architecture = "OaClipTextAg";
		map.ConfigVersion = 1;
		map.Config.DModel = static_cast<OaU32>(cfg.HiddenSize);
		map.Config.NLayers = static_cast<OaU32>(cfg.NumLayers);
		map.Config.DVocab = static_cast<OaU32>(cfg.VocabSize);
		map.Config.WeightDtype = static_cast<OaU8>(OaScalarType::Float32);
		map.ArchConfig.Resize(sizeof(cfg));
		std::memcpy(map.ArchConfig.Data(), &cfg, sizeof(cfg));
		// The published file is a full image+text CLIP checkpoint. Vision and
		// logit_scale weights are deliberately outside this adapter, while every
		// text-tower tensor is checked below.
		map.RequireAllSourceWeights = false;

		auto add = [&](OaStringView InName, std::initializer_list<OaI64> InShape) -> OaStatus {
			const auto* info = InSource.Find(InName);
			if (info == nullptr) return OaStatus::NotFound(OaString("CLIP tensor missing: ") + InName);
			OaVec<OaI64> shape(InShape);
			if (info->Shape.Size() != shape.Size())
				return OaStatus::Error(OaStatusCode::ShapeMismatch, OaString("CLIP rank mismatch: ") + InName);
			for (OaUsize i = 0; i < shape.Size(); ++i)
				if (info->Shape[i] != shape[i])
					return OaStatus::Error(OaStatusCode::ShapeMismatch, OaString("CLIP shape mismatch: ") + InName);
			if (info->Dtype != OaScalarType::Float32)
				return OaStatus::Error(OaStatusCode::DtypeMismatch, OaString("CLIP FP32 tensor required: ") + InName);
			OaWeightMapping mapping;
			mapping.Sources.PushBack(OaString(InName));
			mapping.Target = OaString(InName);
			mapping.TargetShape = OaStdMove(shape);
			mapping.TargetDtype = OaScalarType::Float32;
			map.Mappings.PushBack(OaStdMove(mapping));
			return OaStatus::Ok();
		};

		OA_RETURN_IF_ERROR(add("text_model.embeddings.token_embedding.weight", {cfg.VocabSize, cfg.HiddenSize}));
		OA_RETURN_IF_ERROR(add("text_model.embeddings.position_embedding.weight", {cfg.ContextLength, cfg.HiddenSize}));
		for (OaI32 layer = 0; layer < cfg.NumLayers; ++layer) {
			const OaString root = "text_model.encoder.layers." + OaString(std::to_string(layer).c_str());
			for (const char* projection : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
				const OaString base = root + ".self_attn." + projection;
				OA_RETURN_IF_ERROR(add(base + ".weight", {cfg.HiddenSize, cfg.HiddenSize}));
				OA_RETURN_IF_ERROR(add(base + ".bias", {cfg.HiddenSize}));
			}
			for (const char* norm : {"layer_norm1", "layer_norm2"}) {
				const OaString base = root + "." + norm;
				OA_RETURN_IF_ERROR(add(base + ".weight", {cfg.HiddenSize}));
				OA_RETURN_IF_ERROR(add(base + ".bias", {cfg.HiddenSize}));
			}
			OA_RETURN_IF_ERROR(add(root + ".mlp.fc1.weight", {cfg.IntermediateSize, cfg.HiddenSize}));
			OA_RETURN_IF_ERROR(add(root + ".mlp.fc1.bias", {cfg.IntermediateSize}));
			OA_RETURN_IF_ERROR(add(root + ".mlp.fc2.weight", {cfg.HiddenSize, cfg.IntermediateSize}));
			OA_RETURN_IF_ERROR(add(root + ".mlp.fc2.bias", {cfg.HiddenSize}));
		}
		OA_RETURN_IF_ERROR(add("text_model.final_layer_norm.weight", {cfg.HiddenSize}));
		OA_RETURN_IF_ERROR(add("text_model.final_layer_norm.bias", {cfg.HiddenSize}));
		OA_RETURN_IF_ERROR(add("text_projection.weight", {cfg.ProjectionDim, cfg.HiddenSize}));

		OaHashSet<OaString> expected;
		for (const auto& mapping : map.Mappings) expected.Insert(mapping.Sources[0]);
		for (const auto& info : InSource.List()) {
			if (info.Name == "text_model.embeddings.position_ids") {
				if (info.Shape.Size() != 2 or info.Shape[0] != 1 or
					info.Shape[1] != cfg.ContextLength) {
					return OaStatus::Error(OaStatusCode::ShapeMismatch,
						"CLIP position_ids buffer has the wrong shape");
				}
				continue;
			}
			const bool textTensor = info.Name.StdStr().starts_with("text_model.") or
				info.Name.StdStr().starts_with("text_projection");
			if (textTensor and not expected.Contains(info.Name)) {
				return OaStatus::Error(OaStatusCode::FailedPrecondition,
					OaString("unexpected CLIP text tensor: ") + info.Name);
			}
		}
		return map;
	}
};

} // namespace

OaStatus OaRegisterClipTextWeightAdapter() {
	if (OaFindModelWeightAdapter("clip-text") != nullptr) return OaStatus::Ok();
	return OaRegisterModelWeightAdapter(OaMakeUniquePtr<OaClipTextWeightAdapter>());
}
