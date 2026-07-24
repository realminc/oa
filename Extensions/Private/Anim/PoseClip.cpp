#include <Anim/PoseClip.h>

#include <cstring>
#include <iomanip>
#include <sstream>

namespace {

constexpr OaU8 kMagic[4] = { '3', 'D', 'A', 'N' };
constexpr OaUsize kHeaderSize = 4 + sizeof(OaU32) * 5 + sizeof(OaF32);

template <typename T>
void AppendPod(OaVec<OaU8>& Out, const T& InValue) {
	const auto* bytes = reinterpret_cast<const OaU8*>(&InValue);
	for (OaUsize i = 0; i < sizeof(T); ++i) {
		Out.PushBack(bytes[i]);
	}
}

template <typename T>
bool ReadPod(const OaVec<OaU8>& InData, OaUsize& InOutOffset, T& OutValue) {
	if (InOutOffset + sizeof(T) > InData.Size()) {
		return false;
	}
	std::memcpy(&OutValue, InData.Data() + InOutOffset, sizeof(T));
	InOutOffset += sizeof(T);
	return true;
}

} // namespace

bool OaPoseClip::IsValid() const noexcept {
	return Version == FormatVersion
		&& FrameCount > 0
		&& PoseDim > 0
		&& Fps > 0.0f
		&& Samples.Size() == ValueCount();
}

OaResult<OaPoseClip> OaPoseClip::Create(
	OaU32 InFrameCount,
	OaU32 InPoseDim,
	OaF32 InFps,
	OaU32 InSkeletonId,
	OaSpan<const OaF32> InSamples,
	OaU32 InFlags)
{
	if (InFrameCount == 0 || InPoseDim == 0 || InFps <= 0.0f) {
		return OaStatus::InvalidArgument("OaPoseClip::Create: invalid metadata");
	}

	const OaUsize expected =
		static_cast<OaUsize>(InFrameCount) * static_cast<OaUsize>(InPoseDim);
	if (InSamples.size() != expected) {
		return OaStatus::InvalidArgument("OaPoseClip::Create: sample count mismatch");
	}

	OaPoseClip clip;
	clip.Flags      = InFlags;
	clip.FrameCount = InFrameCount;
	clip.PoseDim    = InPoseDim;
	clip.Fps        = InFps;
	clip.SkeletonId = InSkeletonId;
	clip.Samples.Resize(expected);
	std::memcpy(clip.Samples.Data(), InSamples.data(), expected * sizeof(OaF32));
	return clip;
}

OaStatus OaPoseClip::Write3dAnim(const OaPath& InPath) const {
	if (!IsValid()) {
		return OaStatus::InvalidArgument("OaPoseClip::Write3dAnim: invalid clip");
	}

	OaVec<OaU8> bytes;
	bytes.Reserve(kHeaderSize + Samples.Size() * sizeof(OaF32));
	for (OaU8 b : kMagic) {
		bytes.PushBack(b);
	}
	AppendPod(bytes, Version);
	AppendPod(bytes, Flags);
	AppendPod(bytes, FrameCount);
	AppendPod(bytes, PoseDim);
	AppendPod(bytes, Fps);
	AppendPod(bytes, SkeletonId);

	const auto* payload = reinterpret_cast<const OaU8*>(Samples.Data());
	const OaUsize payloadBytes = Samples.Size() * sizeof(OaF32);
	for (OaUsize i = 0; i < payloadBytes; ++i) {
		bytes.PushBack(payload[i]);
	}

	return OaFilesystem::WriteBinary(InPath, OaSpan<const OaU8>(bytes.Data(), bytes.Size()));
}

OaResult<OaPoseClip> OaPoseClip::Read3dAnim(const OaPath& InPath) {
	auto bytesResult = OaFilesystem::ReadBinary(InPath);
	if (!bytesResult.IsOk()) {
		return bytesResult.GetStatus();
	}
	const auto& bytes = *bytesResult;
	if (bytes.Size() < kHeaderSize || std::memcmp(bytes.Data(), kMagic, sizeof(kMagic)) != 0) {
		return OaStatus::InvalidArgument("OaPoseClip::Read3dAnim: invalid magic/header");
	}

	OaPoseClip clip;
	OaUsize offset = sizeof(kMagic);
	if (!ReadPod(bytes, offset, clip.Version)
		|| !ReadPod(bytes, offset, clip.Flags)
		|| !ReadPod(bytes, offset, clip.FrameCount)
		|| !ReadPod(bytes, offset, clip.PoseDim)
		|| !ReadPod(bytes, offset, clip.Fps)
		|| !ReadPod(bytes, offset, clip.SkeletonId)) {
		return OaStatus::InvalidArgument("OaPoseClip::Read3dAnim: truncated header");
	}
	if (clip.Version != FormatVersion || clip.FrameCount == 0 || clip.PoseDim == 0 || clip.Fps <= 0.0f) {
		return OaStatus::InvalidArgument("OaPoseClip::Read3dAnim: unsupported metadata");
	}

	const OaUsize valueCount = clip.ValueCount();
	const OaUsize payloadBytes = valueCount * sizeof(OaF32);
	if (bytes.Size() - offset != payloadBytes) {
		return OaStatus::InvalidArgument("OaPoseClip::Read3dAnim: payload size mismatch");
	}

	clip.Samples.Resize(valueCount);
	std::memcpy(clip.Samples.Data(), bytes.Data() + offset, payloadBytes);
	if (!clip.IsValid()) {
		return OaStatus::InvalidArgument("OaPoseClip::Read3dAnim: invalid clip");
	}
	return clip;
}

OaStatus OaPoseClip::WriteTxt(const OaPath& InPath) const {
	if (!IsValid()) {
		return OaStatus::InvalidArgument("OaPoseClip::WriteTxt: invalid clip");
	}

	std::ostringstream out;
	out << "# 3DAN version " << Version
		<< " frames " << FrameCount
		<< " d_pose " << PoseDim
		<< " fps " << Fps
		<< " skeleton " << SkeletonId
		<< " flags " << Flags << "\n";
	out << std::setprecision(9);
	for (OaU32 f = 0; f < FrameCount; ++f) {
		const OaUsize base = static_cast<OaUsize>(f) * PoseDim;
		for (OaU32 d = 0; d < PoseDim; ++d) {
			if (d > 0) {
				out << ' ';
			}
			out << Samples[base + d];
		}
		out << '\n';
	}

	return OaFilesystem::WriteText(InPath, OaString(out.str()));
}
