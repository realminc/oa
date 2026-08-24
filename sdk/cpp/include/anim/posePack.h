#pragma once

// PosePack — the lossy bridge between USD SkelAnimation and the model's packed
// canonical pose channels used by the ALM motion representation.
//
// This is the only place quat↔6D conversion, channel ordering, and contact
// derivation live, so Usd stays a pure file parser and PoseClip stays a
// dumb tensor. joint identity comes from Skeleton; the USD clip's joints are
// matched to skeleton joints by leaf bone name.
//
// Canonical channel layout (matches oa::Gen3dAnimLossConfig: rootDims=9 leading,
// contactDims trailing):
//   [0:3)            root translation xyz (cm)
//   [3:9)            root orientation, 6D (first two columns of its rotation)
//   [9 + 6k : +6)    6D rotation of non-root skeleton joint k (k = 1..n-1)
//   [D-C : D)        C soft foot-contact channels (one per skeleton ContactJoint)

#include <anim/poseClip.h>
#include <rig/skeleton.h>
#include <anim/usd.h>

namespace oa {

namespace PosePack {

// USD SkelAnimation clip → packed PoseClip (quat→6D, channel-pack, derive
// contacts via forward kinematics on foot height + speed). inUsd's joints must
// cover every skeleton joint (matched by leaf name).
[[nodiscard]] oa::Result<PoseClip> pack(const UsdSkelClip& inUsd, const Skeleton& inSkel);

// Packed PoseClip → USD SkelAnimation clip (6D→quat; non-root translations
// taken from the skeleton rest pose; bind/rest filled from the skeleton;
// contacts dropped). The inverse of Pack up to the 6D↔quat round-trip and the
// discarded contact channels.
[[nodiscard]] oa::Result<UsdSkelClip> unpack(const PoseClip& inClip, const Skeleton& inSkel);

} // namespace PosePack

}  // namespace oa
