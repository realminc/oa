#include "../../OaTest.h"

#include <Anim/FbxWriter.h>
#include <Anim/PoseClip.h>
#include <Anim/PosePack.h>
#include <Rig/Skeleton.h>
#include <Anim/Usd.h>

#include <Oa/Core/Vlm.h>

// ── Skeleton: built-in MetaHuman body + `.skel` JSON round-trip ──────────────
TEST(Gen3dAnimIo, SkeletonBuiltinAndSkelRoundTrip) {
	const OaSkeleton& sk = OaMetaHumanBodySkeleton();
	EXPECT_TRUE(sk.IsValid());
	EXPECT_EQ(sk.JointCount(), 64);
	EXPECT_EQ(sk.IndexOf("root"), 0);
	EXPECT_EQ(sk.ContactJoints.Size(), 2u);
	// Compact layout: root+pelvis 9 each, 24 hinges ×1, 38 regular ×6, +2 contacts.
	EXPECT_EQ(sk.PoseDim(), 272);

	const OaPath dir = OaFileIo::GetTempDirectory() / "oa_gen3danim_io";
	ASSERT_TRUE(OaFileIo::CreateDirectories(dir).IsOk());
	const OaPath skelPath = dir / "metahuman_body.skel.json";
	ASSERT_TRUE(sk.WriteSkel(skelPath).IsOk());

	auto loaded = OaSkeleton::ReadSkel(skelPath);
	ASSERT_TRUE(loaded.IsOk()) << loaded.GetStatus().ToString();
	const OaSkeleton& sk2 = *loaded;
	ASSERT_EQ(sk2.JointCount(), sk.JointCount());
	EXPECT_EQ(sk2.PoseDim(), sk.PoseDim());
	for (OaI32 j = 0; j < sk.JointCount(); ++j) {
		EXPECT_EQ(sk2.Joints[j].Name, sk.Joints[j].Name);
		EXPECT_EQ(sk2.Joints[j].ParentIndex, sk.Joints[j].ParentIndex);
		EXPECT_EQ(sk2.Joints[j].HumanIkId, sk.Joints[j].HumanIkId);
		EXPECT_NEAR(sk2.Joints[j].Rest.Translate.Z, sk.Joints[j].Rest.Translate.Z, 1e-4f);
		EXPECT_NEAR(sk2.Joints[j].Mass, sk.Joints[j].Mass, 1e-4f);
	}
}

// Build a small valid USD clip on the built-in skeleton: identity rotations
// everywhere, then a couple of joints given real (unit-quat) rotation, and a
// moving root. Goes through Unpack to get correct joint paths / bind / rest.
static OaUsdSkelClip MakeSampleUsd(const OaSkeleton& sk, OaU32 frames) {
	const OaI32 D = sk.PoseDim();
	const OaI32 N = sk.JointCount();

	// Identity pose in the compact layout: walk each joint's ChannelSpec.
	OaVec<OaF32> s;
	s.Resize(static_cast<OaUsize>(frames) * D);
	for (OaU32 f = 0; f < frames; ++f) {
		const OaUsize base = static_cast<OaUsize>(f) * D;
		for (OaI32 j = 0; j < N; ++j) {
			const OaSkelJoint& jt = sk.Joints[static_cast<OaUsize>(j)];
			OaUsize c = base + static_cast<OaUsize>(sk.ChannelOffset(j));
			if (jt.HasTranslate) { c += 3; }              // leave translate at 0
			if (jt.RotDof == 3) { s[c + 0] = 1.0f; s[c + 4] = 1.0f; } // identity 6D
			// hinge (RotDof==1): angle 0 already
		}
		s[base + 0] = static_cast<OaF32>(f) * 5.0f;       // root X moves (translate ch 0)
	}
	auto clip = OaPoseClip::Create(frames, static_cast<OaU32>(D), 30.0f, sk.SkeletonId,
		OaSpan<const OaF32>(s.Data(), s.Size()));
	auto usd = OaPosePack::Unpack(*clip, sk);
	OaUsdSkelClip out = *usd;

	// Inject real rotations on the legs so contacts + 6D have something to chew.
	const OaI32 thighL = sk.IndexOf("thigh_l");
	const OaI32 thighR = sk.IndexOf("thigh_r");
	for (OaU32 f = 0; f < frames; ++f) {
		const OaF32 a = 20.0f * static_cast<OaF32>(f);
		out.Rotations[static_cast<OaUsize>(f) * N + thighL] = Vlm::QuaternionFromEuler(0.0f, a, 0.0f);
		out.Rotations[static_cast<OaUsize>(f) * N + thighR] = Vlm::QuaternionFromEuler(0.0f, -a, 0.0f);
	}
	return out;
}

// ── Pack ⇄ Unpack: 6D channel round-trip is stable ──────────────────────────
TEST(Gen3dAnimIo, PackUnpackChannelStable) {
	const OaSkeleton& sk = OaMetaHumanBodySkeleton();
	const OaU32 frames = 4;
	OaUsdSkelClip usd0 = MakeSampleUsd(sk, frames);
	ASSERT_TRUE(usd0.IsValid());
	ASSERT_GE(usd0.JointPaths.Size(), 3u);
	EXPECT_EQ(usd0.JointPaths[0], "root");
	EXPECT_EQ(usd0.JointPaths[1], "root/pelvis");
	EXPECT_EQ(usd0.JointPaths[2], "root/pelvis/spine_01");

	auto a = OaPosePack::Pack(usd0, sk);
	ASSERT_TRUE(a.IsOk()) << a.GetStatus().ToString();
	auto usdB = OaPosePack::Unpack(*a, sk);
	ASSERT_TRUE(usdB.IsOk()) << usdB.GetStatus().ToString();
	auto b = OaPosePack::Pack(*usdB, sk);
	ASSERT_TRUE(b.IsOk()) << b.GetStatus().ToString();

	const OaI32 D = sk.PoseDim();
	const OaI32 C = static_cast<OaI32>(sk.ContactJoints.Size());
	ASSERT_EQ(a->Samples.Size(), b->Samples.Size());
	// Compare all non-contact channels (root trans + every 6D rotation). Contacts
	// are FK-derived and need not match identically, but the pose channels must.
	for (OaU32 f = 0; f < frames; ++f) {
		for (OaI32 d = 0; d < D - C; ++d) {
			const OaUsize i = static_cast<OaUsize>(f) * D + d;
			EXPECT_NEAR(b->Samples[i], a->Samples[i], 1e-4f) << "frame " << f << " ch " << d;
		}
	}
}

// ── USD: WriteUsda → ReadUsda recovers the clip ─────────────────────────────
TEST(Gen3dAnimIo, UsdFileRoundTrip) {
	const OaSkeleton& sk = OaMetaHumanBodySkeleton();
	OaUsdSkelClip usd = MakeSampleUsd(sk, 4);
	ASSERT_TRUE(usd.IsValid());

	const OaPath dir = OaFileIo::GetTempDirectory() / "oa_gen3danim_io";
	ASSERT_TRUE(OaFileIo::CreateDirectories(dir).IsOk());
	const OaPath usdaPath = dir / "sample.usda";
	ASSERT_TRUE(OaUsd::WriteUsda(usdaPath, usd, "Sample").IsOk());
	auto written = OaFileIo::ReadText(usdaPath);
	ASSERT_TRUE(written.IsOk());
	EXPECT_NE(written->find("uniform token[] joints = [\"root\", \"root/pelvis\", \"root/pelvis/spine_01\""),
	          OaString::npos);

	auto read = OaUsd::ReadUsda(usdaPath);
	ASSERT_TRUE(read.IsOk()) << read.GetStatus().ToString();
	const OaUsdSkelClip& back = *read;

	EXPECT_EQ(back.JointCount(), usd.JointCount());
	EXPECT_EQ(back.FrameCount, usd.FrameCount);
	EXPECT_NEAR(back.Fps, usd.Fps, 1e-4f);
	ASSERT_EQ(back.Translations.Size(), usd.Translations.Size());
	for (OaUsize i = 0; i < usd.Translations.Size(); ++i) {
		EXPECT_NEAR(back.Translations[i].X, usd.Translations[i].X, 1e-3f);
		EXPECT_NEAR(back.Translations[i].Y, usd.Translations[i].Y, 1e-3f);
		EXPECT_NEAR(back.Translations[i].Z, usd.Translations[i].Z, 1e-3f);
	}
	for (OaUsize i = 0; i < usd.Rotations.Size(); ++i) {
		// Compare quaternion components (written/read at 9 sig digits).
		EXPECT_NEAR(back.Rotations[i].X, usd.Rotations[i].X, 1e-4f);
		EXPECT_NEAR(back.Rotations[i].Y, usd.Rotations[i].Y, 1e-4f);
		EXPECT_NEAR(back.Rotations[i].Z, usd.Rotations[i].Z, 1e-4f);
		EXPECT_NEAR(back.Rotations[i].W, usd.Rotations[i].W, 1e-4f);
	}
}

// ── FBX: writer emits a non-trivial file ────────────────────────────────────
TEST(Gen3dAnimIo, FbxWriteSucceeds) {
	const OaSkeleton& sk = OaMetaHumanBodySkeleton();
	OaUsdSkelClip usd = MakeSampleUsd(sk, 6);
	ASSERT_TRUE(usd.IsValid());

	const OaPath dir = OaFileIo::GetTempDirectory() / "oa_gen3danim_io";
	ASSERT_TRUE(OaFileIo::CreateDirectories(dir).IsOk());
	const OaPath fbxPath = dir / "sample.fbx";
	ASSERT_TRUE(OaFbx::WriteFbx(fbxPath, usd).IsOk());

	auto txt = OaFileIo::ReadText(fbxPath);
	ASSERT_TRUE(txt.IsOk());
	EXPECT_NE(txt->find("FBXVersion: 7500"), OaString::npos);
	EXPECT_NE(txt->find("Model::thigh_l"), OaString::npos);
	EXPECT_NE(txt->find("Take 001"), OaString::npos);
}
