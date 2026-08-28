#pragma once

// PoseClip - canonical skeletal-motion clip container for Gen3dAnim.
//
// .3danim is the compact training/runtime format:
//   magic "3DAN", version, flags, n_frames, d_pose, fps, skeleton_id, payload.
// Payload is row-major f32[n_frames * d_pose] in canonical channel order.

#include <oa/core/filesystem.h>
#include <oa/core/types.h>

namespace oa {

struct PoseClip {
	static constexpr oa::U32 formatVersion = 1;

	oa::U32 version    = formatVersion;
	oa::U32 flags      = 0;
	oa::U32 frameCount = 0;
	oa::U32 poseDim    = 0;
	oa::F32 fps        = 30.0f;
	oa::U32 skeletonId = 0;
	oa::Vector<oa::F32> samples;

	[[nodiscard]] bool isValid() const noexcept;
	[[nodiscard]] oa::Usize valueCount() const noexcept {
		return static_cast<oa::Usize>(frameCount) * static_cast<oa::Usize>(poseDim);
	}

	[[nodiscard]] static oa::Result<PoseClip> create(
		oa::U32 inFrameCount,
		oa::U32 inPoseDim,
		oa::F32 inFps,
		oa::U32 inSkeletonId,
		oa::Span<const oa::F32> inSamples,
		oa::U32 inFlags = 0);

	[[nodiscard]] static oa::Result<PoseClip> read3dAnim(const oa::Path& inPath);
	[[nodiscard]] oa::Status write3dAnim(const oa::Path& inPath) const;
	[[nodiscard]] oa::Status writeTxt(const oa::Path& inPath) const;
};

}  // namespace oa
