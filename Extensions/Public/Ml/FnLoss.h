#pragma once

// OaFnLoss - Extension loss functions and metrics (Gen3dAnim, etc.).
// Extends the core OaFnLoss namespace from Source/Public/Oa/Ml/FnLoss.h

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Rig/Skeleton.h>

// OaGen3dAnimLossConfig - Configuration for the Gen3dAnim motion loss.
// Per-channel-type path (compact ChannelSpec layout, PoseDim 272): set
// `ChannelWeights` to a length-PoseDim vector (build it with OaFnGen3dAnim::MakeChannelWeights).
// Leave ChannelWeights empty for the legacy uniform layout.
struct OaGen3dAnimLossConfig {
	OaI32 PoseDim;                 // required
	OaI32 SeqLen;                  // required (context length T)
	OaI32 ContactDims  = 2;        // trailing channels treated as foot contacts
	OaI32 RootDims     = 9;        // leading channels treated as root (trans3 + rot6)
	OaF32 PoseWeight   = 1.0f;
	OaF32 VelWeight    = 0.5f;
	OaF32 ContactWeight= 2.0f;
	OaF32 RootWeight   = 0.0f;
	OaVec<OaF32> ChannelWeights;   // optional; size==PoseDim => per-channel-type path
};

// OaSkPoseFkLossConfig - differentiable skeletal FK world-position loss.
// InPredFlat: [B*T, PoseDim] raw pose channels. InTargetWorld: [B*T, J, 3] world
// positions in cm, usually precomputed from the target raw clip with the same skeleton.
struct OaSkPoseFkLossConfig {
	OaI32 PoseDim;                 // required
	OaI32 SeqLen;                  // required for velocity term
	OaF32 PositionWeight = 1.0f;   // MPJPE-style world-position MSE
	OaF32 VelWeight      = 0.5f;   // world-position delta consistency along time
};

// Full-pose skeletal FK animation metrics. Inputs are raw OaPoseClip channel
// frames, not standardized z-score features.
//
// Two families:
//   - Rotation geodesic (deg): the honest per-joint measure. Every non-root joint
//     is rotation-only (6D / hinge → quaternion), so we compare the unpacked
//     per-joint LOCAL quaternions directly — no FK, no bone length, no position.
//     This is what actually drives the pose; MPJPE just propagates fixed rest
//     offsets and conflates a wrist twist with a shoulder twist.
//   - FK/world position (cm): kept for root tracking + foot-skate, where world
//     position genuinely matters (root + pelvis are the only translated joints).
struct OaSkPoseFkMetrics {
	// Rotation geodesic error (degrees), over joints with a live rotation channel.
	OaF64 MeanJointRotDeg = 0.0;         // mean per-joint local-quaternion geodesic error
	OaF64 MaxJointRotDeg  = 0.0;         // worst joint
	OaI32 RotJointsScored = 0;           // joints with RotDof != 0 that were compared

	// Root + pelvis full transform — the only two joints with a live translate
	// channel, so the only joints where world position genuinely matters. Each
	// gets BOTH translation (FK world-position error, cm) and rotation (local
	// quaternion geodesic, deg).
	OaF64 RootRotDeg    = 0.0;           // root orientation error (deg)
	OaF64 RootMeanCm    = 0.0;           // mean root world-position error (cm)
	OaF64 RootMaxCm     = 0.0;           // worst root world-position error (cm)
	OaF64 PelvisRotDeg  = 0.0;           // pelvis orientation error (deg)
	OaF64 PelvisMeanCm  = 0.0;           // mean pelvis world-position error (cm)
	OaF64 PelvisMaxCm   = 0.0;           // worst pelvis world-position error (cm)

	// FK/world position (cm) — whole-body + foot contact.
	OaF64 MpjpeCm = 0.0;                 // mean per-joint position error, cm
	OaF64 FootSkateCmPerFrame = 0.0;     // generated foot horizontal motion while target contact is planted
	OaBool Ok = false;
};

using OaGen3dAnimPoseMetrics [[deprecated("Use OaSkPoseFkMetrics and OaFnLoss::SkPoseFk")]] = OaSkPoseFkMetrics;

namespace OaFnLoss {

// SmoothL1Mean - SmoothL1(A, B) reduced to a scalar mean in one dispatch.
// Replaces the SmoothL1 + Mean pair (2 dispatches) with a single fused kernel.
[[nodiscard]] OaMatrix SmoothL1Mean(
	const OaMatrix& InA,
	const OaMatrix& InB);

// VelSmoothL1 - velocity SmoothL1 loss reduced to a scalar mean in one dispatch.
// Computes finite-difference velocity along dim 1 of [B, T, D], applies SmoothL1,
// and reduces. Replaces Slice + Sub + Slice + Sub + SmoothL1 + Mean (6 dispatches).
[[nodiscard]] OaMatrix VelSmoothL1(
	const OaMatrix& InPred,
	const OaMatrix& InTarget);

// Backward kernels for the fused losses above. Called by the autograd nodes;
// not intended for direct use.
[[nodiscard]] OaMatrix SmoothL1MeanBwd(
	const OaMatrix& InA,
	const OaMatrix& InB,
	const OaMatrix& InDOut);

[[nodiscard]] OaMatrix VelSmoothL1Bwd(
	const OaMatrix& InPred,
	const OaMatrix& InTarget,
	const OaMatrix& InDOut);

// Gen3dAnim - Motion loss for 3D animation (pose-MSE + velocity + contact + root).
// InPredFlat / InTargetFlat: [B*T, D_pose] (same flat layout the head emits).
// Records into the active OaGradientTape; returns the scalar loss matrix.
[[nodiscard]] OaMatrix Gen3dAnim(
	const OaMatrix& InPredFlat,
	const OaMatrix& InTargetFlat,
	const OaGen3dAnimLossConfig& InConfig);

// SkPoseFkLoss - differentiable FK world-position loss for tokenizer/reconstruction training.
// Differentiable through live translate channels, full 6D rotation joints, and
// 1D hinge joints (via OaFnMatrix::Sin/Cos). Locked joints use rest orientation.
[[nodiscard]] OaMatrix SkPoseFkLoss(
	const OaMatrix& InPredFlat,
	const OaMatrix& InTargetWorld,
	const OaSkeleton& InSkeleton,
	const OaSkPoseFkLossConfig& InConfig);

// Build the constant target tensor consumed by SkPoseFkLoss from raw OaPoseClip
// channels. Shape: [FrameCount, JointCount, 3], units cm.
[[nodiscard]] OaMatrix SkPoseFkTargetWorld(
	OaSpan<const OaF32> InRaw,
	OaI32 InFrameCount,
	OaI32 InPoseDim,
	OaF32 InFps,
	const OaSkeleton& InSkeleton);

// SkPoseFk - skeletal-pose FK/world-space diagnostic metric for generated clips.
// InPredRaw / InTargetRaw: row-major raw OaPoseClip samples [FrameCount, PoseDim].
// Unlike Gen3dAnim(), this is not differentiable and does not record autograd;
// it is for validation, logging, and visual-quality gates.
// Sets OaFnLoss::LastName() to "skposefk_loss" for the standard metric/callback
// naming path used by unnamed OaMetricLoss instances.
[[nodiscard]] OaSkPoseFkMetrics SkPoseFk(
	OaSpan<const OaF32> InPredRaw,
	OaSpan<const OaF32> InTargetRaw,
	OaI32 InFrameCount,
	OaI32 InPoseDim,
	OaF32 InFps,
	const OaSkeleton& InSkeleton,
	OaF32 InContactThreshold = 0.5f);

[[deprecated("Use OaFnLoss::SkPoseFk")]]
[[nodiscard]] OaSkPoseFkMetrics Gen3dAnimPoseMetrics(
	OaSpan<const OaF32> InPredRaw,
	OaSpan<const OaF32> InTargetRaw,
	OaI32 InFrameCount,
	OaI32 InPoseDim,
	OaF32 InFps,
	const OaSkeleton& InSkeleton,
	OaF32 InContactThreshold = 0.5f);

} // namespace OaFnLoss
