#pragma once

// Skeleton <-> USD bridge — skeleton-agnostic.
//
// Turns world joint positions into a valid UsdSkel clip using ANY Skeleton's
// joint names + hierarchy for the joint paths. This is what lets the USD output
// "still work" when the skeleton definition is swapped (skMetaHuman,
// skHumanMl3d, skHumanIk, ...): the writer (Usd::WriteUsda) is already
// skeleton-free, and this bridge supplies the joints/paths from the chosen def.

#include <rig/skeleton.h>
#include <anim/usd.h>
#include <oa/core/types.h>

// Build a positions-only UsdSkel clip from world joint positions.
//   inWorldXyz : flat [frames * jointCount * 3], frame-major, joint order ==
//                skeleton joint order. (CMP/HumanML3D `new_joints` is exactly this.)
//   inScale    : unit conversion applied to every coordinate (HumanML3D metres ->
//                cm: 100.0).
//   inUpAxis   : 1 = Y-up (HumanML3D/SMPL), 2 = Z-up.
// rotations are identity; each joint's per-frame local translation is
// world_j - world_parent; bind/rest are seeded from frame 0. swap inSkel to
// retarget the same positions onto a different skeleton definition.
namespace oa {

[[nodiscard]] UsdSkelClip usdClipFromWorldJoints(const Skeleton& inSkel,
                                                     oa::Span<const oa::F32> inWorldXyz,
                                                     oa::I32 inFrames,
                                                     oa::F32 inFps   = 20.0f,
                                                     oa::I32 inUpAxis = 1,
                                                     oa::F32 inScale  = 100.0f);

}  // namespace oa
