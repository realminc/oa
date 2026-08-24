#include "../oaTest.h"

#include <rig/skeleton.h>

// Channel budget is unchanged by the reseed (25 joints ⇒ poseDim 155) so trained
// checkpoints stay compatible.
TEST(Skeleton, MetaHumanBudget) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	EXPECT_TRUE(sk.isValid());
	EXPECT_EQ(sk.jointCount(), 64);            // LIST_SKEL_MH minus individual toes
	EXPECT_EQ(sk.poseDim(), 272);              // compact: 18 + 24·1 + 38·6 + 2
	EXPECT_EQ(sk.indexOf("root"), 0);
	EXPECT_EQ(sk.contactJoints.size(), 2u);
	// Hinge joints carry a single rotateZ channel; regular joints carry 6D.
	EXPECT_EQ(sk.joints[static_cast<oa::Usize>(sk.indexOf("calf_l"))].rotDof, 1);
	EXPECT_EQ(sk.joints[static_cast<oa::Usize>(sk.indexOf("thigh_l"))].rotDof, 3);
	EXPECT_TRUE(sk.joints[static_cast<oa::Usize>(sk.indexOf("pelvis"))].hasTranslate);
	EXPECT_FALSE(sk.joints[static_cast<oa::Usize>(sk.indexOf("spine_01"))].hasTranslate);
}

// Bone lengths come straight from manny's real rest offsets (cm).
TEST(Skeleton, MannyBoneLengths) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	auto len = [&](const char* n) { return sk.joints[static_cast<oa::Usize>(sk.indexOf(n))].length(); };
	EXPECT_NEAR(len("calf_l"), 43.34f, 0.1f);
	EXPECT_NEAR(len("foot_l"), 42.22f, 0.1f);
	EXPECT_NEAR(len("calf_r"), 43.34f, 0.1f);
}

// Manny-tPose PIN: forward kinematics over the rest orientations must reproduce a
// standing pose — pelvis at hip height, spine/head climbing above it, feet near
// the floor. This is the test that locks the Euler order (Maya XYZ) and the
// jointOrient compose order; if either is wrong the body won't stand up.
TEST(Skeleton, MannyRestStanding) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	auto z = [&](const char* n) { return sk.restWorld(sk.indexOf(n)).z; };
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
	const oa::Skeleton& sk = oa::skMetaHuman();
	const oa::Path dir = oa::Paths::temp() / "oa_rig";
	ASSERT_TRUE(oa::Filesystem::createDirectories(dir).isOk());
	const oa::Path p = dir / "metahuman.skel.json";
	ASSERT_TRUE(sk.writeSkel(p).isOk());

	auto loaded = oa::Skeleton::readSkel(p);
	ASSERT_TRUE(loaded.isOk()) << loaded.getStatus().toString();
	const oa::Skeleton& sk2 = *loaded;
	ASSERT_EQ(sk2.jointCount(), sk.jointCount());
	EXPECT_EQ(sk2.poseDim(), sk.poseDim());
	for (oa::I32 j = 0; j < sk.jointCount(); ++j) {
		EXPECT_TRUE(sk2.joints[j].name == sk.joints[j].name);
		EXPECT_EQ(sk2.joints[j].parentIndex, sk.joints[j].parentIndex);
		EXPECT_NEAR(sk2.joints[j].rest.translate.x, sk.joints[j].rest.translate.x, 1e-3f);
		EXPECT_NEAR(sk2.joints[j].rest.jointOrient.w, sk.joints[j].rest.jointOrient.w, 1e-3f);
	}
}

// The HumanIK characterization rig is the same body under HumanIK slot names.
TEST(Skeleton, HumanIkNames) {
	const oa::Skeleton& sk = oa::skHumanIk();
	EXPECT_TRUE(sk.isValid());
	EXPECT_EQ(sk.jointCount(), 64);
	EXPECT_GE(sk.indexOf("Hips"), 0);
	EXPECT_GE(sk.indexOf("LeftArm"), 0);
	EXPECT_GE(sk.indexOf("LeftHandThumb3"), 0);   // fingers now carry HumanIK slots too
	EXPECT_EQ(sk.indexOf("pelvis"), -1);   // renamed away
}
