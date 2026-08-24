#include "../oaTest.h"

#include <retarget/retarget.h>
#include <retarget/humanIk.h>
#include <rig/skeleton.h>
#include <anim/poseClip.h>

namespace {
// A trivial valid clip: identity pose in the compact layout + a moving root.
oa::PoseClip makeClip(const oa::Skeleton& Sk, oa::U32 frames) {
	const oa::I32 D = Sk.poseDim();
	const oa::I32 N = Sk.jointCount();
	oa::Vec<oa::F32> s;
	s.resize(static_cast<oa::Usize>(frames) * D);
	for (oa::U32 f = 0; f < frames; ++f) {
		const oa::Usize base = static_cast<oa::Usize>(f) * D;
		for (oa::I32 j = 0; j < N; ++j) {
			const oa::SkelJoint& jt = Sk.joints[static_cast<oa::Usize>(j)];
			oa::Usize c = base + static_cast<oa::Usize>(Sk.channelOffset(j));
			if (jt.hasTranslate) { c += 3; }
			if (jt.rotDof == 3) { s[c + 0] = 1.0f; s[c + 4] = 1.0f; } // identity 6D
		}
		s[base + 0] = static_cast<oa::F32>(f);    // root X moves (translate ch 0)
	}
	return *oa::PoseClip::create(frames, static_cast<oa::U32>(D), 30.0f, Sk.skeletonId,
		oa::Span<const oa::F32>(s.data(), s.size()));
}
} // namespace

// The characterization map binds HumanIK slots to UE nodes (from the rtg def).
TEST(Retarget, Characterization) {
	const auto& map = oa::humanIkCharacterization();
	EXPECT_EQ(map.size(), 25u);
	bool hips = false, larm = false;
	for (const oa::HumanIkSlot& s : map) {
		if (s.slot == "Hips")    { hips = true; EXPECT_TRUE(s.node == "pelvis");     EXPECT_EQ(s.id, 1); }
		if (s.slot == "LeftArm") { larm = true; EXPECT_TRUE(s.node == "upperarm_l"); EXPECT_EQ(s.id, 9); }
	}
	EXPECT_TRUE(hips);
	EXPECT_TRUE(larm);
}

// Identity retarget (same skeleton + same reference pose) is a no-op within the
// 6D→quat→6D round-trip tolerance.
TEST(Retarget, IdentityIsNoOp) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	const oa::RefPose& ref = oa::refPoseFor(oa::Mannequin::Manny, oa::PoseKind::TPose);
	const oa::PoseClip clip = makeClip(sk, 3);

	auto out = oa::Retarget::retargetClip(sk, sk, clip, ref, ref);
	ASSERT_TRUE(out.isOk()) << out.getStatus().toString();
	ASSERT_EQ(out->samples.size(), clip.samples.size());
	for (oa::Usize i = 0; i < clip.samples.size(); ++i) {
		EXPECT_NEAR(out->samples[i], clip.samples[i], 1e-4f);
	}
}

// Cross retarget (manny tPose → quinn aPose) leaves root translation untouched
// but rewrites pose channels where the two reference orientations differ.
TEST(Retarget, CrossKeepsRootChangesArm) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	const oa::RefPose& mref = oa::refPoseFor(oa::Mannequin::Manny, oa::PoseKind::TPose);
	const oa::RefPose& qref = oa::refPoseFor(oa::Mannequin::Quinn, oa::PoseKind::APose);
	const oa::PoseClip clip = makeClip(sk, 2);

	auto out = oa::Retarget::retargetClip(sk, sk, clip, mref, qref);
	ASSERT_TRUE(out.isOk()) << out.getStatus().toString();
	EXPECT_EQ(out->poseDim, clip.poseDim);

	const oa::Usize armOff = static_cast<oa::Usize>(sk.channelOffset(sk.indexOf("upperarm_l")));
	bool armChanged = false;
	for (oa::U32 f = 0; f < clip.frameCount; ++f) {
		const oa::Usize base = static_cast<oa::Usize>(f) * clip.poseDim;
		// root translation preserved.
		EXPECT_NEAR(out->samples[base + 0], clip.samples[base + 0], 1e-4f);
		// Arm pose channel rewritten (quinn aPose orients the shoulder differently).
		for (oa::Usize d = 0; d < 6; ++d) {
			if (std::abs(out->samples[base + armOff + d] - clip.samples[base + armOff + d]) > 1e-3f) {
				armChanged = true;
			}
		}
	}
	EXPECT_TRUE(armChanged);
}
