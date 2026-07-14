#pragma once

// OaRetarget — transfer packed motion between rigs through the HumanIK rest pose.
//
// v1 is a **rest-relative rotation transfer**: for every joint, take its rotation
// relative to the *source* reference (tPose/aPose) and re-apply it on top of the
// *destination* reference. This is the HumanIK → MetaHuman → HumanIK path — it
// carries an animation authored on one mannequin's rest onto another's without a
// full IK/effector solve (that heavier solve is the documented next step).
//
// Operates directly on OaPoseClip canonical channels (root trans+6D, per-joint
// 6D, trailing contacts). Src and Dst must share the channel layout (joint
// count + order); contacts and root translation pass through untouched.

#include <Anim/PoseClip.h>
#include <Rig/Skeleton.h>
#include <Retarget/HumanIk.h>

namespace OaRetarget {

// Retarget `InClip` (authored for `InSrc` with rest `InSrcRef`) onto `InDst`
// with rest `InDstRef`. Returns a new clip on the same channel budget.
[[nodiscard]] OaResult<OaPoseClip> RetargetClip(const OaSkeleton& InSrc,
                                                const OaSkeleton& InDst,
                                                const OaPoseClip& InClip,
                                                const OaRefPose&  InSrcRef,
                                                const OaRefPose&  InDstRef);

} // namespace OaRetarget
