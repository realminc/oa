// HumanML3D and CombatMotionProcessed dataset implementation.

#include <data/dsHumanMl3d.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/vlm.h>

#include <algorithm>
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

namespace oa {

namespace {

// Minimal NumPy .npy reader (v1.0/2.0, little-endian float32, C-order). The
// HumanML3D pipeline only ever writes that, so we assert it rather than handle
// the full format zoo.
bool npyLoadF32(const std::string& inPath, std::vector<oa::I64>& outShape, std::vector<float>& outData) {
	std::ifstream f(inPath, std::ios::binary);
	if (!f) return false;
	char magic[6];
	f.read(magic, 6);
	if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
	oa::U8 major = 0, minor = 0;
	f.read(reinterpret_cast<char*>(&major), 1);
	f.read(reinterpret_cast<char*>(&minor), 1);
	oa::U32 headerLen = 0;
	if (major == 1) { oa::U16 h = 0; f.read(reinterpret_cast<char*>(&h), 2); headerLen = h; }
	else            { f.read(reinterpret_cast<char*>(&headerLen), 4); }
	std::string header(headerLen, '\0');
	f.read(header.data(), static_cast<std::streamsize>(headerLen));
	if (header.find("'<f4'") == std::string::npos) return false;     // little-endian float32
	if (header.find("'fortran_order': True") != std::string::npos) return false;

	const size_t sp = header.find("'shape':");
	const size_t lp = header.find('(', sp);
	const size_t rp = header.find(')', lp);
	if (sp == std::string::npos || lp == std::string::npos || rp == std::string::npos) return false;
	outShape.clear();
	const std::string dims = header.substr(lp + 1, rp - lp - 1);
	for (size_t i = 0; i < dims.size();) {
		while (i < dims.size() && !std::isdigit(static_cast<unsigned char>(dims[i]))) ++i;
		if (i >= dims.size()) break;
		oa::I64 v = 0;
		while (i < dims.size() && std::isdigit(static_cast<unsigned char>(dims[i]))) { v = v * 10 + (dims[i] - '0'); ++i; }
		outShape.push_back(v);
	}
	oa::I64 n = 1; for (oa::I64 d : outShape) n *= d;
	outData.resize(static_cast<size_t>(n));
	f.read(reinterpret_cast<char*>(outData.data()), n * static_cast<oa::I64>(sizeof(float)));
	return static_cast<bool>(f);
}

std::vector<std::string> readLines(const std::string& inPath) {
	std::vector<std::string> out;
	std::ifstream f(inPath);
	std::string line;
	while (std::getline(f, line)) {
		while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) line.pop_back();
		if (!line.empty()) out.push_back(line);
	}
	return out;
}

std::string readText(const std::string& inPath) {
	std::ifstream f(inPath);
	if (!f) return {};
	return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Tiny strict reader for the bake_clip_text.py manifest we own. This avoids
// making the dataset loader depend on a general JSON package while still making
// encoder identity and dimensionality part of the data contract.
std::string jsonStringField(const std::string& inJson, const char* inKey) {
	const std::string key = std::string("\"") + inKey + "\"";
	const size_t kp = inJson.find(key);
	const size_t colon = kp == std::string::npos ? kp : inJson.find(':', kp + key.size());
	const size_t quote = colon == std::string::npos ? colon : inJson.find('"', colon + 1);
	const size_t end = quote == std::string::npos ? quote : inJson.find('"', quote + 1);
	return quote == std::string::npos or end == std::string::npos
		? std::string{} : inJson.substr(quote + 1, end - quote - 1);
}

oa::I32 jsonIntField(const std::string& inJson, const char* inKey) {
	const std::string key = std::string("\"") + inKey + "\"";
	const size_t kp = inJson.find(key);
	const size_t colon = kp == std::string::npos ? kp : inJson.find(':', kp + key.size());
	if (colon == std::string::npos) return 0;
	char* end = nullptr;
	const long value = std::strtol(inJson.c_str() + colon + 1, &end, 10);
	return end == inJson.c_str() + colon + 1 or value <= 0
		? 0 : static_cast<oa::I32>(value);
}

// HumanML3D stores one or more "text#pos-tags#start#end" records. A zero/zero
// range denotes the full clip; non-zero ranges denote captioned sub-clips.
oa::Vector<HumanMl3dCaption> readCaptions(const std::string& inPath) {
	oa::Vector<HumanMl3dCaption> out;
	std::ifstream f(inPath);
	std::string line;
	while (std::getline(f, line)) {
		if (line.empty()) continue;
		const size_t h0 = line.find('#');
		const size_t h1 = h0 == std::string::npos ? h0 : line.find('#', h0 + 1);
		const size_t h2 = h1 == std::string::npos ? h1 : line.find('#', h1 + 1);
		HumanMl3dCaption caption;
		caption.text = oa::String((h0 == std::string::npos ? line : line.substr(0, h0)).c_str());
		if (h1 != std::string::npos && h2 != std::string::npos) {
			const std::string start = line.substr(h1 + 1, h2 - h1 - 1);
			const std::string end = line.substr(h2 + 1);
			char* startEnd = nullptr;
			char* endEnd = nullptr;
			caption.startSec = std::strtof(start.c_str(), &startEnd);
			caption.endSec = std::strtof(end.c_str(), &endEnd);
			const bool parsed = startEnd != start.c_str() && endEnd != end.c_str()
				&& std::isfinite(caption.startSec) && std::isfinite(caption.endSec);
			if (!parsed) {
				caption.startSec = 0.0F;
				caption.endSec = 0.0F;
			}
			caption.hasRange = parsed
				&& (caption.startSec != 0.0F || caption.endSec != 0.0F);
		}
		out.pushBack(std::move(caption));
	}
	return out;
}

// HumanML3D feature dim → joint count. 263 ↔ SMPL-22 (HumanML3D / CMP),
// 251 ↔ 21 (KIT-ML). Unknown dims default to 0.
oa::I32 jointsForFeatDimImpl(oa::I32 inFeatDim) {
	switch (inFeatDim) {
		case 263: return 22;
		case 251: return 21;
		default:  return 0;
	}
}

} // namespace

oa::Vector<oa::F32> humanMl3dRecoverWorldJoints(
	oa::Span<const oa::F32> inFeatures, oa::I32 inFrames, oa::I32 inFeatDim) {
	const oa::I32 joints = jointsForFeatDimImpl(inFeatDim);
	if (inFrames <= 0 or joints <= 0 or
		inFeatures.size() < static_cast<oa::Usize>(inFrames) * inFeatDim) {
		return {};
	}

	const oa::I32 ricCount = (joints - 1) * 3;
	oa::Vector<oa::F32> positions;
	positions.reserve(static_cast<oa::Usize>(inFrames) * joints * 3);

	// HumanML3D stores half-angle yaw deltas. This mirrors the reference
	// recover_root_rot_pos/recover_from_ric quaternion convention.
	oa::Vector<oa::F32> yaw(static_cast<oa::Usize>(inFrames), 0.0F);
	for (oa::I32 t = 1; t < inFrames; ++t) {
		yaw[static_cast<oa::Usize>(t)] = yaw[static_cast<oa::Usize>(t - 1)]
			+ inFeatures[static_cast<oa::Usize>(t - 1) * inFeatDim];
	}

	oa::Vector<oa::vlm::Vec3> root(static_cast<oa::Usize>(inFrames));
	for (oa::I32 t = 0; t < inFrames; ++t) {
		oa::vlm::Vec3 delta = {};
		if (t > 0) {
			delta.x = inFeatures[static_cast<oa::Usize>(t - 1) * inFeatDim + 1];
			delta.z = inFeatures[static_cast<oa::Usize>(t - 1) * inFeatDim + 2];
		}
		const oa::F32 angle = yaw[static_cast<oa::Usize>(t)];
		const oa::vlm::Quat inverseYaw = {0.0F, -std::sin(angle), 0.0F, std::cos(angle)};
		root[static_cast<oa::Usize>(t)] = oa::vlm::rotateVector(inverseYaw, delta);
	}
	for (oa::I32 t = 1; t < inFrames; ++t) {
		root[static_cast<oa::Usize>(t)] = oa::vlm::add(
			root[static_cast<oa::Usize>(t)], root[static_cast<oa::Usize>(t - 1)]);
	}
	for (oa::I32 t = 0; t < inFrames; ++t) {
		root[static_cast<oa::Usize>(t)].y =
			inFeatures[static_cast<oa::Usize>(t) * inFeatDim + 3];
	}

	for (oa::I32 t = 0; t < inFrames; ++t) {
		const oa::F32 angle = yaw[static_cast<oa::Usize>(t)];
		const oa::vlm::Quat inverseYaw = {0.0F, -std::sin(angle), 0.0F, std::cos(angle)};
		const oa::vlm::Vec3 rootPos = root[static_cast<oa::Usize>(t)];
		positions.pushBack(rootPos.x);
		positions.pushBack(rootPos.y);
		positions.pushBack(rootPos.z);
		const oa::F32* ric = inFeatures.data()
			+ static_cast<oa::Usize>(t) * inFeatDim + 4;
		for (oa::I32 j = 0; j < ricCount; j += 3) {
			oa::vlm::Vec3 local = {ric[j], ric[j + 1], ric[j + 2]};
			oa::vlm::Vec3 world = oa::vlm::rotateVector(inverseYaw, local);
			world.x += rootPos.x;
			world.z += rootPos.z;
			positions.pushBack(world.x);
			positions.pushBack(world.y);
			positions.pushBack(world.z);
		}
	}
	return positions;
}

oa::F64 humanMl3dMpjpeCm(
	oa::Span<const oa::F32> inPredWorld, oa::Span<const oa::F32> inTargetWorld) {
	if (inPredWorld.size() == 0 or inPredWorld.size() != inTargetWorld.size()
		or inPredWorld.size() % 3 != 0) {
		return std::numeric_limits<oa::F64>::quiet_NaN();
	}
	oa::F64 sum = 0.0;
	for (oa::Usize i = 0; i < inPredWorld.size(); i += 3) {
		const oa::F64 dx = static_cast<oa::F64>(inPredWorld[i]) - inTargetWorld[i];
		const oa::F64 dy = static_cast<oa::F64>(inPredWorld[i + 1]) - inTargetWorld[i + 1];
		const oa::F64 dz = static_cast<oa::F64>(inPredWorld[i + 2]) - inTargetWorld[i + 2];
		sum += std::sqrt(dx * dx + dy * dy + dz * dz);
	}
	return 100.0 * sum / static_cast<oa::F64>(inPredWorld.size() / 3);
}

HumanMl3dMotionMetrics humanMl3dEvaluateMotion(
	oa::Span<const oa::F32> inPredFeatures,
	oa::Span<const oa::F32> inTargetFeatures,
	oa::I32 inFrames,
	oa::I32 inFeatDim,
	oa::F32 inContactThreshold) {
	HumanMl3dMotionMetrics out;
	const oa::I32 joints = jointsForFeatDimImpl(inFeatDim);
	const oa::Usize expected = static_cast<oa::Usize>(std::max(0, inFrames))
		* static_cast<oa::Usize>(std::max(0, inFeatDim));
	if (inFrames <= 0 || joints <= 0 || inPredFeatures.size() != expected
		|| inTargetFeatures.size() != expected
		|| !std::isfinite(inContactThreshold)) {
		return out;
	}
	auto predWorld = humanMl3dRecoverWorldJoints(
		inPredFeatures, inFrames, inFeatDim);
	auto targetWorld = humanMl3dRecoverWorldJoints(
		inTargetFeatures, inFrames, inFeatDim);
	if (predWorld.empty() || predWorld.size() != targetWorld.size()) return out;

	out.mpjpeCm = humanMl3dMpjpeCm(
		oa::Span<const oa::F32>(predWorld.data(), predWorld.size()),
		oa::Span<const oa::F32>(targetWorld.data(), targetWorld.size()));

	oa::F64 velocityError = 0.0;
	oa::I64 velocityCount = 0;
	for (oa::I32 frame = 1; frame < inFrames; ++frame) {
		for (oa::I32 joint = 0; joint < joints; ++joint) {
			const oa::Usize current = (static_cast<oa::Usize>(frame) * joints + joint) * 3;
			const oa::Usize previous = current - static_cast<oa::Usize>(joints) * 3;
			const oa::F64 dx = (predWorld[current] - predWorld[previous])
				- (targetWorld[current] - targetWorld[previous]);
			const oa::F64 dy = (predWorld[current + 1] - predWorld[previous + 1])
				- (targetWorld[current + 1] - targetWorld[previous + 1]);
			const oa::F64 dz = (predWorld[current + 2] - predWorld[previous + 2])
				- (targetWorld[current + 2] - targetWorld[previous + 2]);
			velocityError += std::sqrt(dx * dx + dy * dy + dz * dz);
			++velocityCount;
		}
	}
	out.velocityErrorCmPerFrame = velocityCount > 0
		? 100.0 * velocityError / static_cast<oa::F64>(velocityCount) : 0.0;

	constexpr oa::I32 contactDims = 4;
	const oa::I32 contactOffset = inFeatDim - contactDims;
	const oa::I32 footJoints[contactDims] = {7, 10, 8, 11};
	oa::I64 contactCorrect = 0;
	oa::I64 contactCount = 0;
	oa::F64 footSkate = 0.0;
	oa::I64 footSkateCount = 0;
	for (oa::I32 frame = 0; frame < inFrames; ++frame) {
		for (oa::I32 contact = 0; contact < contactDims; ++contact) {
			const oa::Usize feature = static_cast<oa::Usize>(frame) * inFeatDim
				+ contactOffset + contact;
			const bool predicted = inPredFeatures[feature] >= inContactThreshold;
			const bool target = inTargetFeatures[feature] >= inContactThreshold;
			contactCorrect += predicted == target ? 1 : 0;
			++contactCount;
			if (frame == 0 || !target || footJoints[contact] >= joints) continue;
			const oa::Usize current = (static_cast<oa::Usize>(frame) * joints
				+ footJoints[contact]) * 3;
			const oa::Usize previous = current - static_cast<oa::Usize>(joints) * 3;
			const oa::F64 dx = predWorld[current] - predWorld[previous];
			const oa::F64 dz = predWorld[current + 2] - predWorld[previous + 2];
			footSkate += std::sqrt(dx * dx + dz * dz);
			++footSkateCount;
		}
	}
	out.contactAccuracy = contactCount > 0
		? static_cast<oa::F64>(contactCorrect) / static_cast<oa::F64>(contactCount) : 0.0;
	out.footSkateCmPerFrame = footSkateCount > 0
		? 100.0 * footSkate / static_cast<oa::F64>(footSkateCount) : 0.0;
	out.ok = std::isfinite(out.mpjpeCm)
		&& std::isfinite(out.velocityErrorCmPerFrame)
		&& std::isfinite(out.contactAccuracy)
		&& std::isfinite(out.footSkateCmPerFrame);
	return out;
}

oa::I32 DsHumanMl3d::jointsForFeatDim(oa::I32 inFeatDim) {
	return jointsForFeatDimImpl(inFeatDim);
}

DsHumanMl3d::DsHumanMl3d(const oa::String& inDataDir, const oa::String& inSplit,
		oa::I32 inMaxClips, oa::I32 inFeatDim)
	: featDim_(inFeatDim)
	, numJoints_(jointsForFeatDimImpl(inFeatDim)) {
	ok_ = load(inDataDir, inSplit, inMaxClips);
}

bool DsHumanMl3d::load(const oa::String& inDataDir, const oa::String& inSplit, oa::I32 inMaxClips) {
	const std::string dir = std::string(inDataDir.cStr());
	const std::string textManifest = readText(dir + "/text_feats/manifest.json");
	if (not textManifest.empty()) {
		textFeatureFormat_ = oa::String(jsonStringField(textManifest, "format").c_str());
		textFeatureModel_ = oa::String(jsonStringField(textManifest, "model").c_str());
		textFeatureManifestDim_ = jsonIntField(textManifest, "dim");
		const std::string dtype = jsonStringField(textManifest, "dtype");
		const std::string feature = jsonStringField(textManifest, "feature");
		if (textFeatureFormat_ != "oa_clip_text_v1" or textFeatureModel_.empty() or
			textFeatureManifestDim_ <= 0 or dtype != "float32" or
			feature != "CLIPTextModelWithProjection.text_embeds") {
			OaLogWarn(oa::LogComponent::Data,
				"DsHumanMl3d: ignoring incompatible text_feats/manifest.json in %s", dir.c_str());
			textFeatureFormat_ = {};
			textFeatureModel_ = {};
			textFeatureManifestDim_ = 0;
		}
	}

	std::vector<oa::I64> ms, ss;
	std::vector<float> md, sd;
	if (!npyLoadF32(dir + "/Mean.npy", ms, md) || !npyLoadF32(dir + "/Std.npy", ss, sd)) {
		OaLogError(oa::LogComponent::Data, "DsHumanMl3d: cannot read Mean/Std in %s", dir.c_str());
		return false;
	}
	if (static_cast<oa::I32>(md.size()) != featDim_ || static_cast<oa::I32>(sd.size()) != featDim_) {
		OaLogError(oa::LogComponent::Data, "DsHumanMl3d: Mean/Std dim %zu != featDim %d", md.size(), featDim_);
		return false;
	}
	mean_.append(md.data(), md.size());
	std_.append(sd.data(), sd.size());

	const auto ids = readLines(dir + "/" + std::string(inSplit.cStr()) + ".txt");
	if (ids.empty()) {
		OaLogError(oa::LogComponent::Data, "DsHumanMl3d: empty/missing split %s in %s", inSplit.cStr(), dir.c_str());
		return false;
	}

	offsets_.pushBack(0);
	textFeatureOffsets_.pushBack(0);
	oa::I32 loaded = 0;
	for (const auto& id : ids) {
		if (inMaxClips > 0 && loaded >= inMaxClips) break;
		std::vector<oa::I64> shape;
		std::vector<float> data;
		if (!npyLoadF32(dir + "/new_joint_vecs/" + id + ".npy", shape, data)) continue;
		if (shape.size() != 2 || shape[1] != featDim_) continue;
		const oa::I64 frames = shape[0];

		// Standardize in place: (x - Mean) / std.
		for (oa::I64 t = 0; t < frames; ++t)
			for (oa::I32 d = 0; d < featDim_; ++d) {
				const size_t k = static_cast<size_t>(t) * featDim_ + d;
				data[k] = (data[k] - mean_[d]) / std_[d];
			}
		feat_.append(data.data(), data.size());
		totalFrames_ += frames;
		offsets_.pushBack(totalFrames_);
		ids_.pushBack(oa::String(id.c_str()));
		auto captions = readCaptions(dir + "/texts/" + id + ".txt");
		texts_.pushBack(captions.empty() ? oa::String{} : captions[0].text);

		std::vector<oa::I64> textShape;
		std::vector<float> textData;
		if (textFeatureManifestDim_ > 0 and
			npyLoadF32(dir + "/text_feats/" + id + ".npy", textShape, textData)) {
			const oa::I64 rows = textShape.size() == 1 ? 1
				: (textShape.size() == 2 ? textShape[0] : 0);
			const oa::I64 dim = textShape.size() == 1 ? textShape[0]
				: (textShape.size() == 2 ? textShape[1] : 0);
			if (rows <= 0 or dim <= 0 or rows != static_cast<oa::I64>(captions.size())) {
				OaLogWarn(oa::LogComponent::Data,
					"DsHumanMl3d: ignoring text feature rows for %s (shape/caption mismatch)", id.c_str());
			} else if (dim != textFeatureManifestDim_ or
				(textFeatureDim_ != 0 and dim != textFeatureDim_)) {
				OaLogWarn(oa::LogComponent::Data,
					"DsHumanMl3d: ignoring text features for %s (dim %lld != %d)",
					id.c_str(), static_cast<long long>(dim), textFeatureDim_);
			} else {
				textFeatureDim_ = static_cast<oa::I32>(dim);
				textFeatures_.append(textData.data(), textData.size());
			}
		}
		captions_.pushBack(std::move(captions));
		textFeatureOffsets_.pushBack(textFeatureDim_ > 0
			? static_cast<oa::I64>(textFeatures_.size() / static_cast<oa::Usize>(textFeatureDim_))
			: 0);
		++loaded;
	}
	numClips_ = static_cast<oa::I64>(ids_.size());
	if (numClips_ == 0) {
		OaLogError(oa::LogComponent::Data, "DsHumanMl3d: loaded 0 clips from %s", dir.c_str());
		return false;
	}
	return true;
}

oa::I32 DsHumanMl3d::clipFrames(oa::I64 inIndex) const {
	return static_cast<oa::I32>(offsets_[inIndex + 1] - offsets_[inIndex]);
}

const oa::F32* DsHumanMl3d::clipData(oa::I64 inIndex) const {
	return feat_.data() + offsets_[inIndex] * featDim_;
}

oa::Matrix DsHumanMl3d::getItem(oa::I64 inIndex) const {
	const oa::I32 frames = clipFrames(inIndex);
	const oa::F32* p = clipData(inIndex);
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(p),
			static_cast<size_t>(frames) * featDim_ * sizeof(float)),
		oa::MatrixShape{frames, featDim_}, oa::ScalarType::Float32);
}

void DsHumanMl3d::denormalize(oa::F32* inOutFeat, oa::I64 inFrames) const {
	for (oa::I64 t = 0; t < inFrames; ++t)
		for (oa::I32 d = 0; d < featDim_; ++d) {
			const size_t k = static_cast<size_t>(t) * featDim_ + d;
			inOutFeat[k] = inOutFeat[k] * std_[d] + mean_[d];
		}
}

} // namespace oa
