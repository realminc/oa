#include "../OaTest.h"

#include <Retarget/Retarget.h>
#include <Retarget/HumanIk.h>
#include <Rig/Skeleton.h>
#include <Anim/PoseClip.h>

namespace {
// A trivial valid clip: identity pose in the compact layout + a moving root.
OaPoseClip MakeClip(const OaSkeleton& Sk, OaU32 Frames) {
	const OaI32 D = Sk.PoseDim();
	const OaI32 N = Sk.JointCount();
	OaVec<OaF32> s;
	s.Resize(static_cast<OaUsize>(Frames) * D);
	for (OaU32 f = 0; f < Frames; ++f) {
		const OaUsize base = static_cast<OaUsize>(f) * D;
		for (OaI32 j = 0; j < N; ++j) {
			const OaSkelJoint& jt = Sk.Joints[static_cast<OaUsize>(j)];
			OaUsize c = base + static_cast<OaUsize>(Sk.ChannelOffset(j));
			if (jt.HasTranslate) { c += 3; }
			if (jt.RotDof == 3) { s[c + 0] = 1.0f; s[c + 4] = 1.0f; } // identity 6D
		}
		s[base + 0] = static_cast<OaF32>(f);    // root X moves (translate ch 0)
	}
	return *OaPoseClip::Create(Frames, static_cast<OaU32>(D), 30.0f, Sk.SkeletonId,
		OaSpan<const OaF32>(s.Data(), s.Size()));
}
} // namespace

// The characterization map binds HumanIK slots to UE nodes (from the rtg def).
TEST(Retarget, Characterization) {
	const auto& map = OaHumanIkCharacterization();
	EXPECT_EQ(map.Size(), 25u);
	bool hips = false, larm = false;
	for (const OaHumanIkSlot& s : map) {
		if (s.Slot == "Hips")    { hips = true; EXPECT_TRUE(s.Node == "pelvis");     EXPECT_EQ(s.Id, 1); }
		if (s.Slot == "LeftArm") { larm = true; EXPECT_TRUE(s.Node == "upperarm_l"); EXPECT_EQ(s.Id, 9); }
	}
	EXPECT_TRUE(hips);
	EXPECT_TRUE(larm);
}

// Identity retarget (same skeleton + same reference pose) is a no-op within the
// 6D→quat→6D round-trip tolerance.
TEST(Retarget, IdentityIsNoOp) {
	const OaSkeleton& sk = OaSkMetaHuman();
	const OaRefPose& ref = OaRefPoseFor(OaMannequin::Manny, OaPoseKind::TPose);
	const OaPoseClip clip = MakeClip(sk, 3);

	auto out = OaRetarget::RetargetClip(sk, sk, clip, ref, ref);
	ASSERT_TRUE(out.IsOk()) << out.GetStatus().ToString();
	ASSERT_EQ(out->Samples.Size(), clip.Samples.Size());
	for (OaUsize i = 0; i < clip.Samples.Size(); ++i) {
		EXPECT_NEAR(out->Samples[i], clip.Samples[i], 1e-4f);
	}
}

// Cross retarget (manny tPose → quinn aPose) leaves root translation untouched
// but rewrites pose channels where the two reference orientations differ.
TEST(Retarget, CrossKeepsRootChangesArm) {
	const OaSkeleton& sk = OaSkMetaHuman();
	const OaRefPose& mref = OaRefPoseFor(OaMannequin::Manny, OaPoseKind::TPose);
	const OaRefPose& qref = OaRefPoseFor(OaMannequin::Quinn, OaPoseKind::APose);
	const OaPoseClip clip = MakeClip(sk, 2);

	auto out = OaRetarget::RetargetClip(sk, sk, clip, mref, qref);
	ASSERT_TRUE(out.IsOk()) << out.GetStatus().ToString();
	EXPECT_EQ(out->PoseDim, clip.PoseDim);

	const OaUsize armOff = static_cast<OaUsize>(sk.ChannelOffset(sk.IndexOf("upperarm_l")));
	bool armChanged = false;
	for (OaU32 f = 0; f < clip.FrameCount; ++f) {
		const OaUsize base = static_cast<OaUsize>(f) * clip.PoseDim;
		// Root translation preserved.
		EXPECT_NEAR(out->Samples[base + 0], clip.Samples[base + 0], 1e-4f);
		// Arm pose channel rewritten (quinn aPose orients the shoulder differently).
		for (OaUsize d = 0; d < 6; ++d) {
			if (std::abs(out->Samples[base + armOff + d] - clip.Samples[base + armOff + d]) > 1e-3f) {
				armChanged = true;
			}
		}
	}
	EXPECT_TRUE(armChanged);
}
