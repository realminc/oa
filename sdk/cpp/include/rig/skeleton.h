#pragma once

// Skeleton — a UE-Skeleton-asset-style joint definition for the 3D anim stack.
// Data derived from Epic MetaHuman/CitySample + Autodesk HumanIK — see NOTICE.md.
//
// The skeleton is the *identity* layer that PoseClip lacks: it names every
// joint, fixes the parent hierarchy and joint order (⇒ fixed D_pose), and
// carries each joint's rest transform (an Joint: local offset + rest
// orientation) so the IO layer can convert between USD skelAnimation (joints +
// bind/rest + trans/quat) and the model's packed canonical channels (root
// trans+6D, per-joint 6D, contacts).
//
// It also carries per-limb physics priors — bone length (cm, = |rest offset to
// parent|) and a mass approximation (kg) — so motion/physics work can reason
// about segment inertia and joint distances without a separate rig.
//
// Identity + rest data come from oa3d's rtg definition
// (oa3d/oa/maya/resources/rtgdefinition/unreal.py): `RTG_DEFINITION_UNREAL` for
// the HumanIK ids and joint set, and `RTG_POSES_SKEL_MANNY` (tPose) for the real
// UE5 mannequin rest translate/orient — so a clip retargeted through HumanIK ⇄
// MetaHuman lands on exactly these joints. Masses remain ~80 kg Dempster-ish
// approximations.
//
// Builders: skMetaHuman() (the clean UE body), skHumanIk() (the HumanIK
// characterization rig used as the retarget hub), and skHumanMl3d() (the
// HumanML3D/SMPL 22-joint body for the text-to-motion datasets).

#include <core/joint.h>
#include <oa/core/filesystem.h>
#include <oa/core/types.h>

// One joint in the skeleton. joints are stored in a flat array in hierarchy
// order (every parent precedes its children), so parentIndex < own index, and
// the root joint is index 0 with parentIndex == -1.
namespace oa {

struct SkelJoint {
	oa::String name;                  // UE bone name, e.g. "thigh_l"
	oa::I32    parentIndex = -1;      // index into Skeleton::joints; -1 for root
	oa::I32    humanIkId   = 0;       // HumanIK slot id from the rtg definition
	oa::F32    mass        = 0.0f;    // segment mass approximation, kg
	Joint  rest;                  // rest transform: translate = local offset (cm),
	                                // jointOrient = rest orientation, rotate = identity

	// Channel spec (from the rig's LockSkeleton): which DOFs are actually animated.
	// locked channels are not stored/predicted — they stay at rest.
	bool     hasTranslate = false;  // live translate channel (root + pelvis only)
	oa::U8     rotDof       = 3;      // 3 = full rotation (6D), 1 = hinge (rotateZ), 0 = none

	// Bone length (cm) = |local offset to parent|. root / zero-offset → 0.
	[[nodiscard]] oa::F32 length() const noexcept {
		return oa::vlm::length(rest.translate);
	}

	// Canonical channels this joint contributes: translate (3 if live) + rotation
	// (6 for full 6D, 1 for a hinge angle, 0 if fully locked).
	[[nodiscard]] oa::I32 channelCount() const noexcept {
		const oa::I32 t = hasTranslate ? 3 : 0;
		const oa::I32 r = (rotDof == 3) ? 6 : (rotDof == 1 ? 1 : 0);
		return t + r;
	}
};

struct Skeleton {
	static constexpr oa::U32 formatVersion = 2;   // v2: rest carries orientation

	oa::String             name       = "metahuman_body";
	oa::U32                skeletonId = 0;   // matches PoseClip::skeletonId
	oa::Vector<SkelJoint>   joints;
	// indices into joints of the foot joints whose contacts are packed (in
	// channel order). Two for a biped: { foot_l, foot_r }.
	oa::Vector<oa::I32>         contactJoints;

	[[nodiscard]] oa::I32 jointCount() const noexcept {
		return static_cast<oa::I32>(joints.size());
	}

	// Channel budget for PoseClip::poseDim under the compact canonical layout:
	// each joint contributes only its live channels (see SkelJoint::channelCount),
	// in joint order, followed by one soft-contact channel per contactJoints entry.
	//   root/pelvis : translate(3) + 6D(6) = 9 each
	//   regular     : 6D(6)
	//   hinge       : rotateZ angle(1)
	//   contacts    : C trailing
	[[nodiscard]] oa::I32 poseDim() const noexcept {
		oa::I32 d = 0;
		for (const SkelJoint& j : joints) {
			d += j.channelCount();
		}
		return d + static_cast<oa::I32>(contactJoints.size());
	}

	// Canonical channel offset where joint inJoint's block begins (sum of the
	// channel counts of all earlier joints). Contacts trail all joint blocks.
	[[nodiscard]] oa::I32 channelOffset(oa::I32 inJoint) const noexcept {
		oa::I32 off = 0;
		for (oa::I32 k = 0; k < inJoint && k < jointCount(); ++k) {
			off += joints[static_cast<oa::Usize>(k)].channelCount();
		}
		return off;
	}

	// offset of the first contact channel (= total joint channels).
	[[nodiscard]] oa::I32 contactOffset() const noexcept {
		return channelOffset(jointCount());
	}

	// find a joint index by UE bone name; -1 if absent.
	[[nodiscard]] oa::I32 indexOf(oa::StringView inName) const noexcept;

	// Bind-pose (rest) world position of a joint via forward kinematics over the
	// per-joint rest OaJoints (orientation-aware). cm, Z-up.
	[[nodiscard]] oa::vlm::Vec3 restWorld(oa::I32 inJoint) const noexcept;

	// Bind-pose world *orientation* of a joint (accumulated rest orientations).
	[[nodiscard]] oa::vlm::Quat restWorldRotation(oa::I32 inJoint) const noexcept;

	[[nodiscard]] bool isValid() const noexcept;

	// `.skel` JSON IO (human-readable / editable; the built-in builders below are
	// the authoritative defaults, so the model path never depends on parsing).
	[[nodiscard]] static oa::Result<Skeleton> readSkel(const oa::Path& inPath);
	[[nodiscard]] oa::Status writeSkel(const oa::Path& inPath) const;
};

// The canonical clean MetaHuman base skeleton (skeletonId 0): 64 joints =
// LIST_SKEL_MH minus individual toes (full body + neck/head + both arms with all
// fingers + legs to ball). rest seeded from manny's tPose; per-joint channel spec
// from the rig's LockSkeleton. Corrective/twist/IK joints are never part of it.
[[nodiscard]] const Skeleton& skMetaHuman();

// The HumanIK characterization skeleton (the named HumanIK slots → UE nodes),
// used as the hub rig for retargeting (skeletonId 1).
[[nodiscard]] const Skeleton& skHumanIk();

// The HumanML3D / SMPL 22-joint body (skeletonId 2) — the skeleton the HumanML3D
// text-to-motion datasets and AnimationGPT's CMP are authored on (t2m kinematic
// chain). joint order matches the 263-dim feature vector and `new_joints` arrays.
// rest offsets are placeholders (SkHumanMl3d.inc); the USD preview supplies
// per-frame world joint positions directly.
[[nodiscard]] const Skeleton& skHumanMl3d();

}  // namespace oa
