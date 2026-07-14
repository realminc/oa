#pragma once

// PoseClip - canonical skeletal-motion clip container for Gen3dAnim.
//
// .3danim is the compact training/runtime format:
//   magic "3DAN", version, flags, n_frames, d_pose, fps, skeleton_id, payload.
// Payload is row-major f32[n_frames * d_pose] in canonical channel order.

#include <Oa/Core/FileIo.h>
#include <Oa/Core/Types.h>

struct OaPoseClip {
	static constexpr OaU32 FormatVersion = 1;

	OaU32 Version    = FormatVersion;
	OaU32 Flags      = 0;
	OaU32 FrameCount = 0;
	OaU32 PoseDim    = 0;
	OaF32 Fps        = 30.0f;
	OaU32 SkeletonId = 0;
	OaVec<OaF32> Samples;

	[[nodiscard]] bool IsValid() const noexcept;
	[[nodiscard]] OaUsize ValueCount() const noexcept {
		return static_cast<OaUsize>(FrameCount) * static_cast<OaUsize>(PoseDim);
	}

	[[nodiscard]] static OaResult<OaPoseClip> Create(
		OaU32 InFrameCount,
		OaU32 InPoseDim,
		OaF32 InFps,
		OaU32 InSkeletonId,
		OaSpan<const OaF32> InSamples,
		OaU32 InFlags = 0);

	[[nodiscard]] static OaResult<OaPoseClip> Read3dAnim(const OaPath& InPath);
	[[nodiscard]] OaStatus Write3dAnim(const OaPath& InPath) const;
	[[nodiscard]] OaStatus WriteTxt(const OaPath& InPath) const;
};
