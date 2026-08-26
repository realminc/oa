#include <anim/poseClip.h>
#include <core/streamText.h>

#include <cstring>
#include <iomanip>
#include <sstream>

namespace {

constexpr oa::U8 kMagic[4] = { '3', 'D', 'A', 'N' };
constexpr oa::Usize kHeaderSize = 4 + sizeof(oa::U32) * 5 + sizeof(oa::F32);

template <typename T>
void appendPod(oa::Vec<oa::U8>& out, const T& inValue) {
	const auto* bytes = reinterpret_cast<const oa::U8*>(&inValue);
	for (oa::Usize i = 0; i < sizeof(T); ++i) {
		out.pushBack(bytes[i]);
	}
}

template <typename T>
bool readPod(const oa::Vec<oa::U8>& inData, oa::Usize& inOutOffset, T& outValue) {
	if (inOutOffset + sizeof(T) > inData.size()) {
		return false;
	}
	std::memcpy(&outValue, inData.data() + inOutOffset, sizeof(T));
	inOutOffset += sizeof(T);
	return true;
}

} // namespace

bool oa::PoseClip::isValid() const noexcept {
	return version == formatVersion
		&& frameCount > 0
		&& poseDim > 0
		&& fps > 0.0f
		&& samples.size() == valueCount();
}

oa::Result<oa::PoseClip> oa::PoseClip::create(
	oa::U32 inFrameCount,
	oa::U32 inPoseDim,
	oa::F32 inFps,
	oa::U32 inSkeletonId,
	oa::Span<const oa::F32> inSamples,
	oa::U32 inFlags)
{
	if (inFrameCount == 0 || inPoseDim == 0 || inFps <= 0.0f) {
		return oa::Status::invalidArgument("oa::PoseClip::Create: invalid metadata");
	}

	const oa::Usize expected =
		static_cast<oa::Usize>(inFrameCount) * static_cast<oa::Usize>(inPoseDim);
	if (inSamples.size() != expected) {
		return oa::Status::invalidArgument("oa::PoseClip::Create: sample count mismatch");
	}

	oa::PoseClip clip;
	clip.flags      = inFlags;
	clip.frameCount = inFrameCount;
	clip.poseDim    = inPoseDim;
	clip.fps        = inFps;
	clip.skeletonId = inSkeletonId;
	clip.samples.resize(expected);
	std::memcpy(clip.samples.data(), inSamples.data(), expected * sizeof(oa::F32));
	return clip;
}

oa::Status oa::PoseClip::write3dAnim(const oa::Path& inPath) const {
	if (!isValid()) {
		return oa::Status::invalidArgument("oa::PoseClip::write3dAnim: invalid clip");
	}

	oa::Vec<oa::U8> bytes;
	bytes.reserve(kHeaderSize + samples.size() * sizeof(oa::F32));
	for (oa::U8 b : kMagic) {
		bytes.pushBack(b);
	}
	appendPod(bytes, version);
	appendPod(bytes, flags);
	appendPod(bytes, frameCount);
	appendPod(bytes, poseDim);
	appendPod(bytes, fps);
	appendPod(bytes, skeletonId);

	const auto* payload = reinterpret_cast<const oa::U8*>(samples.data());
	const oa::Usize payloadBytes = samples.size() * sizeof(oa::F32);
	for (oa::Usize i = 0; i < payloadBytes; ++i) {
		bytes.pushBack(payload[i]);
	}

	return oa::Filesystem::writeBinary(inPath, oa::Span<const oa::U8>(bytes.data(), bytes.size()));
}

oa::Result<oa::PoseClip> oa::PoseClip::read3dAnim(const oa::Path& inPath) {
	auto bytesResult = oa::Filesystem::readBinary(inPath);
	if (!bytesResult.isOk()) {
		return bytesResult.getStatus();
	}
	const auto& bytes = *bytesResult;
	if (bytes.size() < kHeaderSize || std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
		return oa::Status::invalidArgument("oa::PoseClip::Read3dAnim: invalid magic/header");
	}

	oa::PoseClip clip;
	oa::Usize offset = sizeof(kMagic);
	if (!readPod(bytes, offset, clip.version)
		|| !readPod(bytes, offset, clip.flags)
		|| !readPod(bytes, offset, clip.frameCount)
		|| !readPod(bytes, offset, clip.poseDim)
		|| !readPod(bytes, offset, clip.fps)
		|| !readPod(bytes, offset, clip.skeletonId)) {
		return oa::Status::invalidArgument("oa::PoseClip::Read3dAnim: truncated header");
	}
	if (clip.version != formatVersion || clip.frameCount == 0 || clip.poseDim == 0 || clip.fps <= 0.0f) {
		return oa::Status::invalidArgument("oa::PoseClip::Read3dAnim: unsupported metadata");
	}

	const oa::Usize valueCount = clip.valueCount();
	const oa::Usize payloadBytes = valueCount * sizeof(oa::F32);
	if (bytes.size() - offset != payloadBytes) {
		return oa::Status::invalidArgument("oa::PoseClip::Read3dAnim: payload size mismatch");
	}

	clip.samples.resize(valueCount);
	std::memcpy(clip.samples.data(), bytes.data() + offset, payloadBytes);
	if (!clip.isValid()) {
		return oa::Status::invalidArgument("oa::PoseClip::Read3dAnim: invalid clip");
	}
	return clip;
}

oa::Status oa::PoseClip::writeTxt(const oa::Path& inPath) const {
	if (!isValid()) {
		return oa::Status::invalidArgument("oa::PoseClip::writeTxt: invalid clip");
	}

	std::ostringstream out;
	out << "# 3DAN version " << version
		<< " frames " << frameCount
		<< " d_pose " << poseDim
		<< " fps " << fps
		<< " skeleton " << skeletonId
		<< " flags " << flags << "\n";
	out << std::setprecision(9);
	for (oa::U32 f = 0; f < frameCount; ++f) {
		const oa::Usize base = static_cast<oa::Usize>(f) * poseDim;
		for (oa::U32 d = 0; d < poseDim; ++d) {
			if (d > 0) {
				out << ' ';
			}
			out << samples[base + d];
		}
		out << '\n';
	}

	return oa::Filesystem::writeText(inPath, oa::sdk::fromStdString(out.str()));
}
