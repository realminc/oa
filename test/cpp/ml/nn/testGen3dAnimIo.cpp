#include "../../oaTest.h"

#include <anim/fbxWriter.h>
#include <anim/poseClip.h>
#include <anim/posePack.h>
#include <rig/skeleton.h>
#include <anim/usd.h>

#include <oa/core/vlm.h>

// ── Skeleton: built-in MetaHuman body + `.skel` JSON round-trip ──────────────
TEST(Gen3dAnimIo, SkeletonBuiltinAndSkelRoundTrip) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	EXPECT_TRUE(sk.isValid());
	EXPECT_EQ(sk.jointCount(), 64);
	EXPECT_EQ(sk.indexOf("root"), 0);
	EXPECT_EQ(sk.contactJoints.size(), 2u);
	// Compact layout: root+pelvis 9 each, 24 hinges ×1, 38 regular ×6, +2 contacts.
	EXPECT_EQ(sk.poseDim(), 272);

	const oa::Path dir = oa::Paths::temp() / "oa_gen3danim_io";
	ASSERT_TRUE(oa::Filesystem::createDirectories(dir).isOk());
	const oa::Path skelPath = dir / "metahuman_body.skel.json";
	ASSERT_TRUE(sk.writeSkel(skelPath).isOk());

	auto loaded = oa::Skeleton::readSkel(skelPath);
	ASSERT_TRUE(loaded.isOk()) << loaded.getStatus().toString();
	const oa::Skeleton& sk2 = *loaded;
	ASSERT_EQ(sk2.jointCount(), sk.jointCount());
	EXPECT_EQ(sk2.poseDim(), sk.poseDim());
	for (oa::I32 j = 0; j < sk.jointCount(); ++j) {
		EXPECT_EQ(sk2.joints[j].name, sk.joints[j].name);
		EXPECT_EQ(sk2.joints[j].parentIndex, sk.joints[j].parentIndex);
		EXPECT_EQ(sk2.joints[j].humanIkId, sk.joints[j].humanIkId);
		EXPECT_NEAR(sk2.joints[j].rest.translate.z, sk.joints[j].rest.translate.z, 1e-4f);
		EXPECT_NEAR(sk2.joints[j].mass, sk.joints[j].mass, 1e-4f);
	}
}

// Build a small valid USD clip on the built-in skeleton: identity rotations
// everywhere, then a couple of joints given real (unit-quat) rotation, and a
// moving root. Goes through Unpack to get correct joint paths / bind / rest.
static oa::UsdSkelClip makeSampleUsd(const oa::Skeleton& sk, oa::U32 frames) {
	const oa::I32 D = sk.poseDim();
	const oa::I32 N = sk.jointCount();

	// Identity pose in the compact layout: walk each joint's ChannelSpec.
	oa::Vec<oa::F32> s;
	s.resize(static_cast<oa::Usize>(frames) * D);
	for (oa::U32 f = 0; f < frames; ++f) {
		const oa::Usize base = static_cast<oa::Usize>(f) * D;
		for (oa::I32 j = 0; j < N; ++j) {
			const oa::SkelJoint& jt = sk.joints[static_cast<oa::Usize>(j)];
			oa::Usize c = base + static_cast<oa::Usize>(sk.channelOffset(j));
			if (jt.hasTranslate) { c += 3; }              // leave translate at 0
			if (jt.rotDof == 3) { s[c + 0] = 1.0f; s[c + 4] = 1.0f; } // identity 6D
			// hinge (rotDof==1): angle 0 already
		}
		s[base + 0] = static_cast<oa::F32>(f) * 5.0f;       // root X moves (translate ch 0)
	}
	auto clip = oa::PoseClip::create(frames, static_cast<oa::U32>(D), 30.0f, sk.skeletonId,
		oa::Span<const oa::F32>(s.data(), s.size()));
	auto usd = oa::PosePack::unpack(*clip, sk);
	oa::UsdSkelClip out = *usd;

	// Inject real rotations on the legs so contacts + 6D have something to chew.
	const oa::I32 thighL = sk.indexOf("thigh_l");
	const oa::I32 thighR = sk.indexOf("thigh_r");
	for (oa::U32 f = 0; f < frames; ++f) {
		const oa::F32 a = 20.0f * static_cast<oa::F32>(f);
		out.rotations[static_cast<oa::Usize>(f) * N + thighL] =
			oa::vlm::quaternionFromEuler(0.0f, a, 0.0f);
		out.rotations[static_cast<oa::Usize>(f) * N + thighR] =
			oa::vlm::quaternionFromEuler(0.0f, -a, 0.0f);
	}
	return out;
}

// ── Pack ⇄ Unpack: 6D channel round-trip is stable ──────────────────────────
TEST(Gen3dAnimIo, PackUnpackChannelStable) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	const oa::U32 frames = 4;
	oa::UsdSkelClip usd0 = makeSampleUsd(sk, frames);
	ASSERT_TRUE(usd0.isValid());
	ASSERT_GE(usd0.jointPaths.size(), 3u);
	EXPECT_EQ(usd0.jointPaths[0], "root");
	EXPECT_EQ(usd0.jointPaths[1], "root/pelvis");
	EXPECT_EQ(usd0.jointPaths[2], "root/pelvis/spine_01");

	auto a = oa::PosePack::pack(usd0, sk);
	ASSERT_TRUE(a.isOk()) << a.getStatus().toString();
	auto usdB = oa::PosePack::unpack(*a, sk);
	ASSERT_TRUE(usdB.isOk()) << usdB.getStatus().toString();
	auto b = oa::PosePack::pack(*usdB, sk);
	ASSERT_TRUE(b.isOk()) << b.getStatus().toString();

	const oa::I32 D = sk.poseDim();
	const oa::I32 C = static_cast<oa::I32>(sk.contactJoints.size());
	ASSERT_EQ(a->samples.size(), b->samples.size());
	// compare all non-contact channels (root trans + every 6D rotation). Contacts
	// are FK-derived and need not match identically, but the pose channels must.
	for (oa::U32 f = 0; f < frames; ++f) {
		for (oa::I32 d = 0; d < D - C; ++d) {
			const oa::Usize i = static_cast<oa::Usize>(f) * D + d;
			EXPECT_NEAR(b->samples[i], a->samples[i], 1e-4f) << "frame " << f << " ch " << d;
		}
	}
}

// ── USD: WriteUsda → ReadUsda recovers the clip ─────────────────────────────
TEST(Gen3dAnimIo, UsdFileRoundTrip) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	oa::UsdSkelClip usd = makeSampleUsd(sk, 4);
	ASSERT_TRUE(usd.isValid());

	const oa::Path dir = oa::Paths::temp() / "oa_gen3danim_io";
	ASSERT_TRUE(oa::Filesystem::createDirectories(dir).isOk());
	const oa::Path usdaPath = dir / "sample.usda";
	ASSERT_TRUE(oa::Usd::writeUsda(usdaPath, usd, "Sample").isOk());
	auto written = oa::Filesystem::readText(usdaPath);
	ASSERT_TRUE(written.isOk());
	EXPECT_NE(written->find("uniform token[] joints = [\"root\", \"root/pelvis\", \"root/pelvis/spine_01\""),
	          oa::String::Npos);

	auto read = oa::Usd::readUsda(usdaPath);
	ASSERT_TRUE(read.isOk()) << read.getStatus().toString();
	const oa::UsdSkelClip& back = *read;

	EXPECT_EQ(back.jointCount(), usd.jointCount());
	EXPECT_EQ(back.frameCount, usd.frameCount);
	EXPECT_NEAR(back.fps, usd.fps, 1e-4f);
	ASSERT_EQ(back.translations.size(), usd.translations.size());
	for (oa::Usize i = 0; i < usd.translations.size(); ++i) {
		EXPECT_NEAR(back.translations[i].x, usd.translations[i].x, 1e-3f);
		EXPECT_NEAR(back.translations[i].y, usd.translations[i].y, 1e-3f);
		EXPECT_NEAR(back.translations[i].z, usd.translations[i].z, 1e-3f);
	}
	for (oa::Usize i = 0; i < usd.rotations.size(); ++i) {
		// compare quaternion components (written/read at 9 sig digits).
		EXPECT_NEAR(back.rotations[i].x, usd.rotations[i].x, 1e-4f);
		EXPECT_NEAR(back.rotations[i].y, usd.rotations[i].y, 1e-4f);
		EXPECT_NEAR(back.rotations[i].z, usd.rotations[i].z, 1e-4f);
		EXPECT_NEAR(back.rotations[i].w, usd.rotations[i].w, 1e-4f);
	}
}

// ── FBX: writer emits a non-trivial file ────────────────────────────────────
TEST(Gen3dAnimIo, FbxWriteSucceeds) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	oa::UsdSkelClip usd = makeSampleUsd(sk, 6);
	ASSERT_TRUE(usd.isValid());

	const oa::Path dir = oa::Paths::temp() / "oa_gen3danim_io";
	ASSERT_TRUE(oa::Filesystem::createDirectories(dir).isOk());
	const oa::Path fbxPath = dir / "sample.fbx";
	ASSERT_TRUE(oa::Fbx::writeFbx(fbxPath, usd).isOk());

	auto txt = oa::Filesystem::readText(fbxPath);
	ASSERT_TRUE(txt.isOk());
	EXPECT_NE(txt->find("FBXVersion: 7500"), oa::String::Npos);
	EXPECT_NE(txt->find("Model::thigh_l"), oa::String::Npos);
	EXPECT_NE(txt->find("Take 001"), oa::String::Npos);
}
