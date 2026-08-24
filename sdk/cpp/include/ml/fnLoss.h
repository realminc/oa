#pragma once

// oa::FnLoss - extension loss functions and metrics (Gen3dAnim, etc.).
// Extends the core oa::FnLoss namespace from source/cpp/include/oa/Ml/FnLoss.h

#include <oa/core/matrix.h>
#include <oa/core/types.h>
#include <rig/skeleton.h>

namespace oa {

// Gen3dAnimLossConfig - Configuration for the Gen3dAnim motion loss.
// Per-channel-type path (compact ChannelSpec layout, poseDim 272): set
// `channelWeights` to a length-poseDim vector (build it with oa::FnGen3dAnim::MakeChannelWeights).
// Leave channelWeights empty for the legacy uniform layout.
struct Gen3dAnimLossConfig {
	oa::I32 poseDim;                 // required
	oa::I32 seqLen;                  // required (context length T)
	oa::I32 contactDims  = 2;        // trailing channels treated as foot contacts
	oa::I32 rootDims     = 9;        // leading channels treated as root (trans3 + rot6)
	oa::F32 poseWeight   = 1.0f;
	oa::F32 velWeight    = 0.5f;
	oa::F32 contactWeight= 2.0f;
	oa::F32 rootWeight   = 0.0f;
	oa::Vec<oa::F32> channelWeights;   // optional; size==poseDim => per-channel-type path
};

// SkPoseFkLossConfig - differentiable skeletal FK world-position loss.
// inPredFlat: [B*T, poseDim] raw pose channels. inTargetWorld: [B*T, J, 3] world
// positions in cm, usually precomputed from the target raw clip with the same skeleton.
struct SkPoseFkLossConfig {
	oa::I32 poseDim;                 // required
	oa::I32 seqLen;                  // required for velocity term
	oa::F32 positionWeight = 1.0f;   // MPJPE-style world-position MSE
	oa::F32 velWeight      = 0.5f;   // world-position delta consistency along time
};

// Full-pose skeletal FK animation metrics. Inputs are raw oa::PoseClip channel
// frames, not standardized z-score features.
//
// Two families:
//   - rotation geodesic (deg): the honest per-joint measure. Every non-root joint
//     is rotation-only (6D / hinge → quaternion), so we compare the unpacked
//     per-joint LOCAL quaternions directly — no FK, no bone length, no position.
//     This is what actually drives the pose; MPJPE just propagates fixed rest
//     offsets and conflates a wrist twist with a shoulder twist.
//   - FK/world position (cm): kept for root tracking + foot-skate, where world
//     position genuinely matters (root + pelvis are the only translated joints).
struct SkPoseFkMetrics {
	// rotation geodesic error (degrees), over joints with a live rotation channel.
	oa::F64 meanJointRotDeg = 0.0;         // mean per-joint local-quaternion geodesic error
	oa::F64 maxJointRotDeg  = 0.0;         // worst joint
	oa::I32 rotJointsScored = 0;           // joints with rotDof != 0 that were compared

	// root + pelvis full transform — the only two joints with a live translate
	// channel, so the only joints where world position genuinely matters. Each
	// gets BOTH translation (FK world-position error, cm) and rotation (local
	// quaternion geodesic, deg).
	oa::F64 rootRotDeg    = 0.0;           // root orientation error (deg)
	oa::F64 rootMeanCm    = 0.0;           // mean root world-position error (cm)
	oa::F64 rootMaxCm     = 0.0;           // worst root world-position error (cm)
	oa::F64 pelvisRotDeg  = 0.0;           // pelvis orientation error (deg)
	oa::F64 pelvisMeanCm  = 0.0;           // mean pelvis world-position error (cm)
	oa::F64 pelvisMaxCm   = 0.0;           // worst pelvis world-position error (cm)

	// FK/world position (cm) — whole-body + foot contact.
	oa::F64 mpjpeCm = 0.0;                 // mean per-joint position error, cm
	oa::F64 footSkateCmPerFrame = 0.0;     // generated foot horizontal motion while target contact is planted
	oa::Bool ok = false;
};

} // namespace oa

namespace oa {

namespace FnLoss {

// SmoothL1Mean - smoothL1(A, B) reduced to a scalar mean in one dispatch.
// Replaces the SmoothL1 + Mean pair (2 dispatches) with a single fused kernel.
[[nodiscard]] oa::Matrix smoothL1Mean(
	const oa::Matrix& inA,
	const oa::Matrix& inB);

// VelSmoothL1 - velocity SmoothL1 loss reduced to a scalar mean in one dispatch.
// Computes finite-difference velocity along dim 1 of [B, T, D], applies SmoothL1,
// and reduces. Replaces Slice + Sub + Slice + Sub + SmoothL1 + mean (6 dispatches).
[[nodiscard]] oa::Matrix velSmoothL1(
	const oa::Matrix& inPred,
	const oa::Matrix& inTarget);

// backward kernels for the fused losses above. Called by the autograd nodes;
// not intended for direct use.
[[nodiscard]] oa::Matrix smoothL1MeanBwd(
	const oa::Matrix& inA,
	const oa::Matrix& inB,
	const oa::Matrix& inDOut);

[[nodiscard]] oa::Matrix velSmoothL1Bwd(
	const oa::Matrix& inPred,
	const oa::Matrix& inTarget,
	const oa::Matrix& inDOut);

// Gen3dAnim - motion loss for 3D animation (pose-MSE + velocity + contact + root).
// inPredFlat / inTargetFlat: [B*T, D_pose] (same flat layout the head emits).
// Records into the active oa::GradientTape; returns the scalar loss matrix.
[[nodiscard]] oa::Matrix gen3dAnim(
	const oa::Matrix& inPredFlat,
	const oa::Matrix& inTargetFlat,
	const Gen3dAnimLossConfig& inConfig);

// SkPoseFkLoss - differentiable FK world-position loss for tokenizer/reconstruction training.
// Differentiable through live translate channels, full 6D rotation joints, and
// 1D hinge joints (via oa::FnMatrix::sin/Cos). locked joints use rest orientation.
[[nodiscard]] oa::Matrix skPoseFkLoss(
	const oa::Matrix& inPredFlat,
	const oa::Matrix& inTargetWorld,
	const oa::Skeleton& inSkeleton,
	const SkPoseFkLossConfig& inConfig);

// Build the constant target tensor consumed by SkPoseFkLoss from raw oa::PoseClip
// channels. Shape: [frameCount, jointCount, 3], units cm.
[[nodiscard]] oa::Matrix skPoseFkTargetWorld(
	oa::Span<const oa::F32> inRaw,
	oa::I32 inFrameCount,
	oa::I32 inPoseDim,
	oa::F32 inFps,
	const oa::Skeleton& inSkeleton);

// SkPoseFk - skeletal-pose FK/world-space diagnostic metric for generated clips.
// inPredRaw / inTargetRaw: row-major raw oa::PoseClip samples [frameCount, poseDim].
// Unlike gen3dAnim(), this is not differentiable and does not record autograd;
// it is for validation, logging, and visual-quality gates.
// Sets oa::FnLoss::lastName() to "skposefk_loss" for the standard metric/callback
// naming path used by unnamed oa::MetricLoss instances.
[[nodiscard]] SkPoseFkMetrics skPoseFk(
	oa::Span<const oa::F32> inPredRaw,
	oa::Span<const oa::F32> inTargetRaw,
	oa::I32 inFrameCount,
	oa::I32 inPoseDim,
	oa::F32 inFps,
	const oa::Skeleton& inSkeleton,
	oa::F32 inContactThreshold = 0.5f);

} // namespace FnLoss

} // namespace oa
