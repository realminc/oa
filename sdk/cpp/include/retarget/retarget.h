#pragma once

// Retarget — transfer packed motion between rigs through the HumanIK rest pose.
//
// v1 is a **rest-relative rotation transfer**: for every joint, take its rotation
// relative to the *source* reference (tPose/aPose) and re-apply it on top of the
// *destination* reference. This is the HumanIK → MetaHuman → HumanIK path — it
// carries an animation authored on one mannequin's rest onto another's without a
// full IK/effector solve (that heavier solve is the documented next step).
//
// Operates directly on PoseClip canonical channels (root trans+6D, per-joint
// 6D, trailing contacts). src and dst must share the channel layout (joint
// count + order); contacts and root translation pass through untouched.

#include <anim/poseClip.h>
#include <rig/skeleton.h>
#include <retarget/humanIk.h>

namespace oa {

namespace Retarget {

// Retarget `inClip` (authored for `inSrc` with rest `inSrcRef`) onto `inDst`
// with rest `inDstRef`. Returns a new clip on the same channel budget.
[[nodiscard]] oa::Result<PoseClip> retargetClip(const Skeleton& inSrc,
                                                const Skeleton& inDst,
                                                const PoseClip& inClip,
                                                const RefPose&  inSrcRef,
                                                const RefPose&  inDstRef);

} // namespace Retarget

}  // namespace oa
