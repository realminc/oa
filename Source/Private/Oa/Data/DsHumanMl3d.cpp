// OaDsHumanMl3d / OaDsCombatMotionProcessed implementation.

#include <Oa/Data/DsHumanMl3d.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Vlm.h>

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

// Minimal NumPy .npy reader (v1.0/2.0, little-endian float32, C-order). The
// HumanML3D pipeline only ever writes that, so we assert it rather than handle
// the full format zoo.
bool NpyLoadF32(const std::string& InPath, std::vector<OaI64>& OutShape, std::vector<float>& OutData) {
	std::ifstream f(InPath, std::ios::binary);
	if (!f) return false;
	char magic[6];
	f.read(magic, 6);
	if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
	OaU8 major = 0, minor = 0;
	f.read(reinterpret_cast<char*>(&major), 1);
	f.read(reinterpret_cast<char*>(&minor), 1);
	OaU32 headerLen = 0;
	if (major == 1) { OaU16 h = 0; f.read(reinterpret_cast<char*>(&h), 2); headerLen = h; }
	else            { f.read(reinterpret_cast<char*>(&headerLen), 4); }
	std::string header(headerLen, '\0');
	f.read(header.data(), static_cast<std::streamsize>(headerLen));
	if (header.find("'<f4'") == std::string::npos) return false;     // little-endian float32
	if (header.find("'fortran_order': True") != std::string::npos) return false;

	const size_t sp = header.find("'shape':");
	const size_t lp = header.find('(', sp);
	const size_t rp = header.find(')', lp);
	if (sp == std::string::npos || lp == std::string::npos || rp == std::string::npos) return false;
	OutShape.clear();
	const std::string dims = header.substr(lp + 1, rp - lp - 1);
	for (size_t i = 0; i < dims.size();) {
		while (i < dims.size() && !std::isdigit(static_cast<unsigned char>(dims[i]))) ++i;
		if (i >= dims.size()) break;
		OaI64 v = 0;
		while (i < dims.size() && std::isdigit(static_cast<unsigned char>(dims[i]))) { v = v * 10 + (dims[i] - '0'); ++i; }
		OutShape.push_back(v);
	}
	OaI64 n = 1; for (OaI64 d : OutShape) n *= d;
	OutData.resize(static_cast<size_t>(n));
	f.read(reinterpret_cast<char*>(OutData.data()), n * static_cast<OaI64>(sizeof(float)));
	return static_cast<bool>(f);
}

std::vector<std::string> ReadLines(const std::string& InPath) {
	std::vector<std::string> out;
	std::ifstream f(InPath);
	std::string line;
	while (std::getline(f, line)) {
		while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) line.pop_back();
		if (!line.empty()) out.push_back(line);
	}
	return out;
}

std::string ReadText(const std::string& InPath) {
	std::ifstream f(InPath);
	if (!f) return {};
	return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Tiny strict reader for the bake_clip_text.py manifest we own. This avoids
// making the dataset loader depend on a general JSON package while still making
// encoder identity and dimensionality part of the data contract.
std::string JsonStringField(const std::string& InJson, const char* InKey) {
	const std::string key = std::string("\"") + InKey + "\"";
	const size_t kp = InJson.find(key);
	const size_t colon = kp == std::string::npos ? kp : InJson.find(':', kp + key.size());
	const size_t quote = colon == std::string::npos ? colon : InJson.find('"', colon + 1);
	const size_t end = quote == std::string::npos ? quote : InJson.find('"', quote + 1);
	return quote == std::string::npos or end == std::string::npos
		? std::string{} : InJson.substr(quote + 1, end - quote - 1);
}

OaI32 JsonIntField(const std::string& InJson, const char* InKey) {
	const std::string key = std::string("\"") + InKey + "\"";
	const size_t kp = InJson.find(key);
	const size_t colon = kp == std::string::npos ? kp : InJson.find(':', kp + key.size());
	if (colon == std::string::npos) return 0;
	char* end = nullptr;
	const long value = std::strtol(InJson.c_str() + colon + 1, &end, 10);
	return end == InJson.c_str() + colon + 1 or value <= 0
		? 0 : static_cast<OaI32>(value);
}

// HumanML3D stores one or more "text#pos-tags#start#end" records. A zero/zero
// range denotes the full clip; non-zero ranges denote captioned sub-clips.
OaVec<OaHumanMl3dCaption> ReadCaptions(const std::string& InPath) {
	OaVec<OaHumanMl3dCaption> out;
	std::ifstream f(InPath);
	std::string line;
	while (std::getline(f, line)) {
		if (line.empty()) continue;
		const size_t h0 = line.find('#');
		const size_t h1 = h0 == std::string::npos ? h0 : line.find('#', h0 + 1);
		const size_t h2 = h1 == std::string::npos ? h1 : line.find('#', h1 + 1);
		OaHumanMl3dCaption caption;
		caption.Text = OaString((h0 == std::string::npos ? line : line.substr(0, h0)).c_str());
		if (h1 != std::string::npos && h2 != std::string::npos) {
			const std::string start = line.substr(h1 + 1, h2 - h1 - 1);
			const std::string end = line.substr(h2 + 1);
			char* startEnd = nullptr;
			char* endEnd = nullptr;
			caption.StartSec = std::strtof(start.c_str(), &startEnd);
			caption.EndSec = std::strtof(end.c_str(), &endEnd);
			const bool parsed = startEnd != start.c_str() && endEnd != end.c_str()
				&& std::isfinite(caption.StartSec) && std::isfinite(caption.EndSec);
			if (!parsed) {
				caption.StartSec = 0.0F;
				caption.EndSec = 0.0F;
			}
			caption.HasRange = parsed
				&& (caption.StartSec != 0.0F || caption.EndSec != 0.0F);
		}
		out.PushBack(std::move(caption));
	}
	return out;
}

// HumanML3D feature dim → joint count. 263 ↔ SMPL-22 (HumanML3D / CMP),
// 251 ↔ 21 (KIT-ML). Unknown dims default to 0.
OaI32 JointsForFeatDimImpl(OaI32 InFeatDim) {
	switch (InFeatDim) {
		case 263: return 22;
		case 251: return 21;
		default:  return 0;
	}
}

} // namespace

OaVec<OaF32> OaHumanMl3dRecoverWorldJoints(
	OaSpan<const OaF32> InFeatures, OaI32 InFrames, OaI32 InFeatDim) {
	const OaI32 joints = JointsForFeatDimImpl(InFeatDim);
	if (InFrames <= 0 or joints <= 0 or
		InFeatures.Size() < static_cast<OaUsize>(InFrames) * InFeatDim) {
		return {};
	}

	const OaI32 ricCount = (joints - 1) * 3;
	OaVec<OaF32> positions;
	positions.Reserve(static_cast<OaUsize>(InFrames) * joints * 3);

	// HumanML3D stores half-angle yaw deltas. This mirrors the reference
	// recover_root_rot_pos/recover_from_ric quaternion convention.
	OaVec<OaF32> yaw(static_cast<OaUsize>(InFrames), 0.0F);
	for (OaI32 t = 1; t < InFrames; ++t) {
		yaw[static_cast<OaUsize>(t)] = yaw[static_cast<OaUsize>(t - 1)]
			+ InFeatures[static_cast<OaUsize>(t - 1) * InFeatDim];
	}

	OaVec<VlmVec3> root(static_cast<OaUsize>(InFrames));
	for (OaI32 t = 0; t < InFrames; ++t) {
		VlmVec3 delta = {};
		if (t > 0) {
			delta.X = InFeatures[static_cast<OaUsize>(t - 1) * InFeatDim + 1];
			delta.Z = InFeatures[static_cast<OaUsize>(t - 1) * InFeatDim + 2];
		}
		const OaF32 angle = yaw[static_cast<OaUsize>(t)];
		const VlmQuat inverseYaw = {0.0F, -std::sin(angle), 0.0F, std::cos(angle)};
		root[static_cast<OaUsize>(t)] = Vlm::RotateVector(inverseYaw, delta);
	}
	for (OaI32 t = 1; t < InFrames; ++t) {
		root[static_cast<OaUsize>(t)] = Vlm::Add(
			root[static_cast<OaUsize>(t)], root[static_cast<OaUsize>(t - 1)]);
	}
	for (OaI32 t = 0; t < InFrames; ++t) {
		root[static_cast<OaUsize>(t)].Y =
			InFeatures[static_cast<OaUsize>(t) * InFeatDim + 3];
	}

	for (OaI32 t = 0; t < InFrames; ++t) {
		const OaF32 angle = yaw[static_cast<OaUsize>(t)];
		const VlmQuat inverseYaw = {0.0F, -std::sin(angle), 0.0F, std::cos(angle)};
		const VlmVec3 rootPos = root[static_cast<OaUsize>(t)];
		positions.PushBack(rootPos.X);
		positions.PushBack(rootPos.Y);
		positions.PushBack(rootPos.Z);
		const OaF32* ric = InFeatures.Data()
			+ static_cast<OaUsize>(t) * InFeatDim + 4;
		for (OaI32 j = 0; j < ricCount; j += 3) {
			VlmVec3 local = {ric[j], ric[j + 1], ric[j + 2]};
			VlmVec3 world = Vlm::RotateVector(inverseYaw, local);
			world.X += rootPos.X;
			world.Z += rootPos.Z;
			positions.PushBack(world.X);
			positions.PushBack(world.Y);
			positions.PushBack(world.Z);
		}
	}
	return positions;
}

OaF64 OaHumanMl3dMpjpeCm(
	OaSpan<const OaF32> InPredWorld, OaSpan<const OaF32> InTargetWorld) {
	if (InPredWorld.Size() == 0 or InPredWorld.Size() != InTargetWorld.Size()
		or InPredWorld.Size() % 3 != 0) {
		return std::numeric_limits<OaF64>::quiet_NaN();
	}
	OaF64 sum = 0.0;
	for (OaUsize i = 0; i < InPredWorld.Size(); i += 3) {
		const OaF64 dx = static_cast<OaF64>(InPredWorld[i]) - InTargetWorld[i];
		const OaF64 dy = static_cast<OaF64>(InPredWorld[i + 1]) - InTargetWorld[i + 1];
		const OaF64 dz = static_cast<OaF64>(InPredWorld[i + 2]) - InTargetWorld[i + 2];
		sum += std::sqrt(dx * dx + dy * dy + dz * dz);
	}
	return 100.0 * sum / static_cast<OaF64>(InPredWorld.Size() / 3);
}

OaI32 OaDsHumanMl3d::JointsForFeatDim(OaI32 InFeatDim) {
	return JointsForFeatDimImpl(InFeatDim);
}

OaDsHumanMl3d::OaDsHumanMl3d(const OaString& InDataDir, const OaString& InSplit,
		OaI32 InMaxClips, OaI32 InFeatDim)
	: FeatDim_(InFeatDim)
	, NumJoints_(JointsForFeatDimImpl(InFeatDim)) {
	Ok_ = Load(InDataDir, InSplit, InMaxClips);
}

bool OaDsHumanMl3d::Load(const OaString& InDataDir, const OaString& InSplit, OaI32 InMaxClips) {
	const std::string dir = std::string(InDataDir.c_str());
	const std::string textManifest = ReadText(dir + "/text_feats/manifest.json");
	if (not textManifest.empty()) {
		TextFeatureFormat_ = OaString(JsonStringField(textManifest, "format").c_str());
		TextFeatureModel_ = OaString(JsonStringField(textManifest, "model").c_str());
		TextFeatureManifestDim_ = JsonIntField(textManifest, "dim");
		const std::string dtype = JsonStringField(textManifest, "dtype");
		const std::string feature = JsonStringField(textManifest, "feature");
		if (TextFeatureFormat_ != "oa_clip_text_v1" or TextFeatureModel_.Empty() or
			TextFeatureManifestDim_ <= 0 or dtype != "float32" or
			feature != "CLIPTextModelWithProjection.text_embeds") {
			OA_LOG_WARN(OaLogComponent::ML,
				"DsHumanMl3d: ignoring incompatible text_feats/manifest.json in %s", dir.c_str());
			TextFeatureFormat_ = {};
			TextFeatureModel_ = {};
			TextFeatureManifestDim_ = 0;
		}
	}

	std::vector<OaI64> ms, ss;
	std::vector<float> md, sd;
	if (!NpyLoadF32(dir + "/Mean.npy", ms, md) || !NpyLoadF32(dir + "/Std.npy", ss, sd)) {
		OA_LOG_ERROR(OaLogComponent::ML, "DsHumanMl3d: cannot read Mean/Std in %s", dir.c_str());
		return false;
	}
	if (static_cast<OaI32>(md.size()) != FeatDim_ || static_cast<OaI32>(sd.size()) != FeatDim_) {
		OA_LOG_ERROR(OaLogComponent::ML, "DsHumanMl3d: Mean/Std dim %zu != FeatDim %d", md.size(), FeatDim_);
		return false;
	}
	Mean_.Append(md.data(), md.size());
	Std_.Append(sd.data(), sd.size());

	const auto ids = ReadLines(dir + "/" + std::string(InSplit.c_str()) + ".txt");
	if (ids.empty()) {
		OA_LOG_ERROR(OaLogComponent::ML, "DsHumanMl3d: empty/missing split %s in %s", InSplit.c_str(), dir.c_str());
		return false;
	}

	Offsets_.PushBack(0);
	TextFeatureOffsets_.PushBack(0);
	OaI32 loaded = 0;
	for (const auto& id : ids) {
		if (InMaxClips > 0 && loaded >= InMaxClips) break;
		std::vector<OaI64> shape;
		std::vector<float> data;
		if (!NpyLoadF32(dir + "/new_joint_vecs/" + id + ".npy", shape, data)) continue;
		if (shape.size() != 2 || shape[1] != FeatDim_) continue;
		const OaI64 frames = shape[0];

		// Standardize in place: (x - Mean) / Std.
		for (OaI64 t = 0; t < frames; ++t)
			for (OaI32 d = 0; d < FeatDim_; ++d) {
				const size_t k = static_cast<size_t>(t) * FeatDim_ + d;
				data[k] = (data[k] - Mean_[d]) / Std_[d];
			}
		Feat_.Append(data.data(), data.size());
		TotalFrames_ += frames;
		Offsets_.PushBack(TotalFrames_);
		Ids_.PushBack(OaString(id.c_str()));
		auto captions = ReadCaptions(dir + "/texts/" + id + ".txt");
		Texts_.PushBack(captions.Empty() ? OaString{} : captions[0].Text);

		std::vector<OaI64> textShape;
		std::vector<float> textData;
		if (TextFeatureManifestDim_ > 0 and
			NpyLoadF32(dir + "/text_feats/" + id + ".npy", textShape, textData)) {
			const OaI64 rows = textShape.size() == 1 ? 1
				: (textShape.size() == 2 ? textShape[0] : 0);
			const OaI64 dim = textShape.size() == 1 ? textShape[0]
				: (textShape.size() == 2 ? textShape[1] : 0);
			if (rows <= 0 or dim <= 0 or rows != static_cast<OaI64>(captions.Size())) {
				OA_LOG_WARN(OaLogComponent::ML,
					"DsHumanMl3d: ignoring text feature rows for %s (shape/caption mismatch)", id.c_str());
			} else if (dim != TextFeatureManifestDim_ or
				(TextFeatureDim_ != 0 and dim != TextFeatureDim_)) {
				OA_LOG_WARN(OaLogComponent::ML,
					"DsHumanMl3d: ignoring text features for %s (dim %lld != %d)",
					id.c_str(), static_cast<long long>(dim), TextFeatureDim_);
			} else {
				TextFeatureDim_ = static_cast<OaI32>(dim);
				TextFeatures_.Append(textData.data(), textData.size());
			}
		}
		Captions_.PushBack(std::move(captions));
		TextFeatureOffsets_.PushBack(TextFeatureDim_ > 0
			? static_cast<OaI64>(TextFeatures_.Size() / static_cast<OaUsize>(TextFeatureDim_))
			: 0);
		++loaded;
	}
	NumClips_ = static_cast<OaI64>(Ids_.Size());
	if (NumClips_ == 0) {
		OA_LOG_ERROR(OaLogComponent::ML, "DsHumanMl3d: loaded 0 clips from %s", dir.c_str());
		return false;
	}
	return true;
}

OaI32 OaDsHumanMl3d::ClipFrames(OaI64 InIndex) const {
	return static_cast<OaI32>(Offsets_[InIndex + 1] - Offsets_[InIndex]);
}

const OaF32* OaDsHumanMl3d::ClipData(OaI64 InIndex) const {
	return Feat_.Data() + Offsets_[InIndex] * FeatDim_;
}

OaMatrix OaDsHumanMl3d::GetItem(OaI64 InIndex) const {
	const OaI32 frames = ClipFrames(InIndex);
	const OaF32* p = ClipData(InIndex);
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(p),
			static_cast<size_t>(frames) * FeatDim_ * sizeof(float)),
		OaMatrixShape{frames, FeatDim_}, OaScalarType::Float32);
}

void OaDsHumanMl3d::Denormalize(OaF32* InOutFeat, OaI64 InFrames) const {
	for (OaI64 t = 0; t < InFrames; ++t)
		for (OaI32 d = 0; d < FeatDim_; ++d) {
			const size_t k = static_cast<size_t>(t) * FeatDim_ + d;
			InOutFeat[k] = InOutFeat[k] * Std_[d] + Mean_[d];
		}
}
