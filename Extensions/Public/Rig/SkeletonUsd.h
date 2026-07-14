#pragma once

// OaSkeleton <-> USD bridge — skeleton-agnostic.
//
// Turns world joint positions into a valid UsdSkel clip using ANY OaSkeleton's
// joint names + hierarchy for the joint paths. This is what lets the USD output
// "still work" when the skeleton definition is swapped (OaSkMetaHuman,
// OaSkHumanMl3d, OaSkHumanIk, ...): the writer (OaUsd::WriteUsda) is already
// skeleton-free, and this bridge supplies the joints/paths from the chosen def.

#include <Rig/Skeleton.h>
#include <Anim/Usd.h>
#include <Oa/Core/Types.h>

// Build a positions-only UsdSkel clip from world joint positions.
//   InWorldXyz : flat [Frames * JointCount * 3], frame-major, joint order ==
//                skeleton joint order. (CMP/HumanML3D `new_joints` is exactly this.)
//   InScale    : unit conversion applied to every coordinate (HumanML3D metres ->
//                cm: 100.0).
//   InUpAxis   : 1 = Y-up (HumanML3D/SMPL), 2 = Z-up.
// Rotations are identity; each joint's per-frame local translation is
// world_j - world_parent; bind/rest are seeded from frame 0. Swap InSkel to
// retarget the same positions onto a different skeleton definition.
[[nodiscard]] OaUsdSkelClip OaUsdClipFromWorldJoints(const OaSkeleton& InSkel,
                                                     OaSpan<const OaF32> InWorldXyz,
                                                     OaI32 InFrames,
                                                     OaF32 InFps   = 20.0f,
                                                     OaI32 InUpAxis = 1,
                                                     OaF32 InScale  = 100.0f);
