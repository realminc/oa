#include "../OaTest.h"

#include <Rig/Skeleton.h>

// Channel budget is unchanged by the reseed (25 joints ⇒ PoseDim 155) so trained
// checkpoints stay compatible.
TEST(Skeleton, MetaHumanBudget) {
	const OaSkeleton& sk = OaSkMetaHuman();
	EXPECT_TRUE(sk.IsValid());
	EXPECT_EQ(sk.JointCount(), 64);            // LIST_SKEL_MH minus individual toes
	EXPECT_EQ(sk.PoseDim(), 272);              // compact: 18 + 24·1 + 38·6 + 2
	EXPECT_EQ(sk.IndexOf("root"), 0);
	EXPECT_EQ(sk.ContactJoints.Size(), 2u);
	// Hinge joints carry a single rotateZ channel; regular joints carry 6D.
	EXPECT_EQ(sk.Joints[static_cast<OaUsize>(sk.IndexOf("calf_l"))].RotDof, 1);
	EXPECT_EQ(sk.Joints[static_cast<OaUsize>(sk.IndexOf("thigh_l"))].RotDof, 3);
	EXPECT_TRUE(sk.Joints[static_cast<OaUsize>(sk.IndexOf("pelvis"))].HasTranslate);
	EXPECT_FALSE(sk.Joints[static_cast<OaUsize>(sk.IndexOf("spine_01"))].HasTranslate);
}

// Bone lengths come straight from manny's real rest offsets (cm).
TEST(Skeleton, MannyBoneLengths) {
	const OaSkeleton& sk = OaSkMetaHuman();
	auto len = [&](const char* n) { return sk.Joints[static_cast<OaUsize>(sk.IndexOf(n))].Length(); };
	EXPECT_NEAR(len("calf_l"), 43.34f, 0.1f);
	EXPECT_NEAR(len("foot_l"), 42.22f, 0.1f);
	EXPECT_NEAR(len("calf_r"), 43.34f, 0.1f);
}

// Manny-tPose PIN: forward kinematics over the rest orientations must reproduce a
// standing pose — pelvis at hip height, spine/head climbing above it, feet near
// the floor. This is the test that locks the Euler order (Maya XYZ) and the
// jointOrient compose order; if either is wrong the body won't stand up.
TEST(Skeleton, MannyRestStanding) {
	const OaSkeleton& sk = OaSkMetaHuman();
	auto z = [&](const char* n) { return sk.RestWorld(sk.IndexOf(n)).Z; };
	EXPECT_NEAR(z("pelvis"), 95.9f, 0.5f);
	EXPECT_GT(z("spine_04"), z("pelvis"));   // spine climbs
	EXPECT_GT(z("head"), z("pelvis"));       // head above hips
	EXPECT_GT(z("head"), 140.0f);            // roughly human height
	EXPECT_LT(z("foot_l"), 25.0f);           // foot near the floor
	EXPECT_LT(z("foot_l"), z("pelvis"));
	EXPECT_LT(z("foot_r"), 25.0f);
}

// `.skel` JSON round-trip preserves identity + rest (translate + orient quat).
TEST(Skeleton, SkelRoundTrip) {
	const OaSkeleton& sk = OaSkMetaHuman();
	const OaPath dir = OaPaths::Temp() / "oa_rig";
	ASSERT_TRUE(OaFilesystem::CreateDirectories(dir).IsOk());
	const OaPath p = dir / "metahuman.skel.json";
	ASSERT_TRUE(sk.WriteSkel(p).IsOk());

	auto loaded = OaSkeleton::ReadSkel(p);
	ASSERT_TRUE(loaded.IsOk()) << loaded.GetStatus().ToString();
	const OaSkeleton& sk2 = *loaded;
	ASSERT_EQ(sk2.JointCount(), sk.JointCount());
	EXPECT_EQ(sk2.PoseDim(), sk.PoseDim());
	for (OaI32 j = 0; j < sk.JointCount(); ++j) {
		EXPECT_TRUE(sk2.Joints[j].Name == sk.Joints[j].Name);
		EXPECT_EQ(sk2.Joints[j].ParentIndex, sk.Joints[j].ParentIndex);
		EXPECT_NEAR(sk2.Joints[j].Rest.Translate.X, sk.Joints[j].Rest.Translate.X, 1e-3f);
		EXPECT_NEAR(sk2.Joints[j].Rest.JointOrient.W, sk.Joints[j].Rest.JointOrient.W, 1e-3f);
	}
}

// The HumanIK characterization rig is the same body under HumanIK slot names.
TEST(Skeleton, HumanIkNames) {
	const OaSkeleton& sk = OaSkHumanIk();
	EXPECT_TRUE(sk.IsValid());
	EXPECT_EQ(sk.JointCount(), 64);
	EXPECT_GE(sk.IndexOf("Hips"), 0);
	EXPECT_GE(sk.IndexOf("LeftArm"), 0);
	EXPECT_GE(sk.IndexOf("LeftHandThumb3"), 0);   // fingers now carry HumanIK slots too
	EXPECT_EQ(sk.IndexOf("pelvis"), -1);   // renamed away
}
