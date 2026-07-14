#pragma once

// OaPosePack — the lossy bridge between USD SkelAnimation and the model's packed
// canonical pose channels used by the ALM motion representation.
//
// This is the only place quat↔6D conversion, channel ordering, and contact
// derivation live, so OaUsd stays a pure file parser and OaPoseClip stays a
// dumb tensor. Joint identity comes from OaSkeleton; the USD clip's joints are
// matched to skeleton joints by leaf bone name.
//
// Canonical channel layout (matches OaGen3dAnimLossConfig: RootDims=9 leading,
// ContactDims trailing):
//   [0:3)            root translation xyz (cm)
//   [3:9)            root orientation, 6D (first two columns of its rotation)
//   [9 + 6k : +6)    6D rotation of non-root skeleton joint k (k = 1..N-1)
//   [D-C : D)        C soft foot-contact channels (one per skeleton ContactJoint)

#include <Anim/PoseClip.h>
#include <Rig/Skeleton.h>
#include <Anim/Usd.h>

namespace OaPosePack {

// USD SkelAnimation clip → packed OaPoseClip (quat→6D, channel-pack, derive
// contacts via forward kinematics on foot height + speed). InUsd's joints must
// cover every skeleton joint (matched by leaf name).
[[nodiscard]] OaResult<OaPoseClip> Pack(const OaUsdSkelClip& InUsd, const OaSkeleton& InSkel);

// Packed OaPoseClip → USD SkelAnimation clip (6D→quat; non-root translations
// taken from the skeleton rest pose; bind/rest filled from the skeleton;
// contacts dropped). The inverse of Pack up to the 6D↔quat round-trip and the
// discarded contact channels.
[[nodiscard]] OaResult<OaUsdSkelClip> Unpack(const OaPoseClip& InClip, const OaSkeleton& InSkel);

} // namespace OaPosePack
