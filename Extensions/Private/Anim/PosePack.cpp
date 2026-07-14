#include <Anim/PosePack.h>

#include <Core/Transform.h>

#include <algorithm>
#include <cmath>

namespace {

using namespace Vlm;

// 6D rotation = first two columns of the rotation matrix (Zhou et al. 2019).
void QuatTo6D(const VlmQuat& q, OaF32 out[6]) {
	const VlmMat4 m = QuaternionToMatrix(q);
	out[0] = m.M[0][0]; out[1] = m.M[1][0]; out[2] = m.M[2][0]; // column 0
	out[3] = m.M[0][1]; out[4] = m.M[1][1]; out[5] = m.M[2][1]; // column 1
}

VlmQuat Mat3ToQuat(const VlmMat4& m) {
	const OaF32 m00 = m.M[0][0], m01 = m.M[0][1], m02 = m.M[0][2];
	const OaF32 m10 = m.M[1][0], m11 = m.M[1][1], m12 = m.M[1][2];
	const OaF32 m20 = m.M[2][0], m21 = m.M[2][1], m22 = m.M[2][2];
	const OaF32 tr = m00 + m11 + m22;
	VlmQuat q;
	if (tr > 0.0f) {
		OaF32 s = std::sqrt(tr + 1.0f) * 2.0f;
		q.W = 0.25f * s; q.X = (m21 - m12) / s; q.Y = (m02 - m20) / s; q.Z = (m10 - m01) / s;
	} else if (m00 > m11 && m00 > m22) {
		OaF32 s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
		q.W = (m21 - m12) / s; q.X = 0.25f * s; q.Y = (m01 + m10) / s; q.Z = (m02 + m20) / s;
	} else if (m11 > m22) {
		OaF32 s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
		q.W = (m02 - m20) / s; q.X = (m01 + m10) / s; q.Y = 0.25f * s; q.Z = (m12 + m21) / s;
	} else {
		OaF32 s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
		q.W = (m10 - m01) / s; q.X = (m02 + m20) / s; q.Y = (m12 + m21) / s; q.Z = 0.25f * s;
	}
	return q;
}

// 6D → unit quaternion via Gram-Schmidt (inverse of QuatTo6D).
VlmQuat SixDToQuat(const OaF32 d[6]) {
	VlmVec3 a1 = { d[0], d[1], d[2] };
	VlmVec3 a2 = { d[3], d[4], d[5] };
	VlmVec3 b1 = Normalize(a1);
	VlmVec3 b2 = Normalize(Sub(a2, Scale(b1, Dot(b1, a2))));
	VlmVec3 b3 = Cross(b1, b2);
	VlmMat4 m = VlmMat4::Identity();
	m.M[0][0] = b1.X; m.M[1][0] = b1.Y; m.M[2][0] = b1.Z;
	m.M[0][1] = b2.X; m.M[1][1] = b2.Y; m.M[2][1] = b2.Z;
	m.M[0][2] = b3.X; m.M[1][2] = b3.Y; m.M[2][2] = b3.Z;
	return Mat3ToQuat(m);
}

// Hinge encode/decode. A hinge joint animates only rotateZ relative to its rest
// orientation. The stored USD quat is the full local orientation
// (restOrient ⊗ rotate); the live delta is rotate = restOrient⁻¹ ⊗ q, which for a
// clean clip is a pure spin about local Z → a single angle (radians).
OaF32 HingeAngleOf(const VlmQuat& InLocal, const VlmQuat& InRestOrient) {
	const VlmQuat delta = VlmQuatMul(VlmQuatConjugate(InRestOrient), InLocal);
	return 2.0f * std::atan2(delta.Z, delta.W);
}
VlmQuat HingeQuatFrom(OaF32 InAngle, const VlmQuat& InRestOrient) {
	const OaF32 h = InAngle * 0.5f;
	const VlmQuat delta = { 0.0f, 0.0f, std::sin(h), std::cos(h) }; // spin about local Z
	return VlmQuatMul(InRestOrient, delta);
}

OaF32 Clamp01(OaF32 v) {
	return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

OaF32 Comp(const VlmVec3& v, OaI32 i) {
	return i == 0 ? v.X : (i == 1 ? v.Y : v.Z);
}

// Leaf bone name = path segment after the last '/'.
OaStringView LeafName(OaStringView path) {
	OaUsize slash = OaStringView::npos;
	for (OaUsize i = 0; i < path.Size(); ++i) {
		if (path[i] == '/') { slash = i; }
	}
	return slash == OaStringView::npos ? path : path.SubStr(slash + 1);
}

} // namespace

OaResult<OaPoseClip> OaPosePack::Pack(const OaUsdSkelClip& InUsd, const OaSkeleton& InSkel) {
	if (!InUsd.IsValid()) {
		return OaStatus::InvalidArgument("OaPosePack::Pack: invalid USD clip");
	}
	if (!InSkel.IsValid()) {
		return OaStatus::InvalidArgument("OaPosePack::Pack: invalid skeleton");
	}

	const OaI32 N = InSkel.JointCount();
	const OaI32 usdN = InUsd.JointCount();

	// Map skeleton joint → USD joint by leaf bone name. Missing is allowed (the
	// clip may lack a base joint, or carry junk joints we ignore) → rest fallback.
	OaVec<OaI32> usdOf;
	usdOf.Resize(static_cast<OaUsize>(N));
	for (OaI32 s = 0; s < N; ++s) {
		const OaStringView want = InSkel.Joints[static_cast<OaUsize>(s)].Name;
		OaI32 found = -1;
		for (OaI32 u = 0; u < usdN; ++u) {
			if (LeafName(InUsd.JointPaths[static_cast<OaUsize>(u)]) == want) { found = u; break; }
		}
		usdOf[static_cast<OaUsize>(s)] = found;
	}

	const OaU32 frames = InUsd.FrameCount;
	const OaI32 D = InSkel.PoseDim();
	const OaI32 C = static_cast<OaI32>(InSkel.ContactJoints.Size());

	// Up axis and the two horizontal axes for contact derivation.
	const OaI32 up = (InUsd.UpAxis == 1) ? 1 : 2;
	const OaI32 h0 = (up == 2) ? 0 : 0;
	const OaI32 h1 = (up == 2) ? 1 : 2;

	// Local transform of skeleton joint s at frame f (USD value, or rest fallback).
	auto localTrans = [&](OaU32 f, OaI32 s) -> VlmVec3 {
		const OaI32 u = usdOf[static_cast<OaUsize>(s)];
		return (u >= 0) ? InUsd.Translations[static_cast<OaUsize>(f) * usdN + u]
		                : InSkel.Joints[static_cast<OaUsize>(s)].Rest.Translate;
	};
	auto localRot = [&](OaU32 f, OaI32 s) -> VlmQuat {
		const OaI32 u = usdOf[static_cast<OaUsize>(s)];
		return (u >= 0) ? InUsd.Rotations[static_cast<OaUsize>(f) * usdN + u]
		                : InSkel.Joints[static_cast<OaUsize>(s)].Rest.OrientedRotation();
	};

	// Forward kinematics → world position of every joint (contacts only). Skeleton
	// order guarantees each parent is solved first.
	OaVec<VlmVec3> worldPos; worldPos.Resize(static_cast<OaUsize>(frames) * N);
	OaVec<VlmQuat> worldRot; worldRot.Resize(static_cast<OaUsize>(frames) * N);
	for (OaU32 f = 0; f < frames; ++f) {
		for (OaI32 s = 0; s < N; ++s) {
			const VlmVec3 lt = localTrans(f, s);
			const VlmQuat lr = localRot(f, s);
			const OaI32 parent = InSkel.Joints[static_cast<OaUsize>(s)].ParentIndex;
			const OaUsize wi = static_cast<OaUsize>(f) * N + s;
			if (parent < 0) {
				worldRot[wi] = lr; worldPos[wi] = lt;
			} else {
				const OaUsize pi = static_cast<OaUsize>(f) * N + parent;
				worldRot[wi] = VlmQuatMul(worldRot[pi], lr);
				worldPos[wi] = Add(worldPos[pi], RotateVector(worldRot[pi], lt));
			}
		}
	}

	// Contact derivation: floor = global min foot height along the up axis; a foot
	// is "planted" when near the floor and horizontally slow. Soft value in [0,1].
	constexpr OaF32 kHeightBand = 6.0f;
	constexpr OaF32 kSpeedThresh = 40.0f;
	OaF32 floorUp = 1e30f;
	for (OaI32 ci = 0; ci < C; ++ci) {
		const OaI32 s = InSkel.ContactJoints[static_cast<OaUsize>(ci)];
		for (OaU32 f = 0; f < frames; ++f) {
			floorUp = std::min(floorUp, Comp(worldPos[static_cast<OaUsize>(f) * N + s], up));
		}
	}

	OaVec<OaF32> samples;
	samples.Resize(static_cast<OaUsize>(frames) * D);
	for (OaU32 f = 0; f < frames; ++f) {
		const OaUsize base = static_cast<OaUsize>(f) * D;

		// Per-joint live channels at the joint's canonical offset.
		for (OaI32 s = 0; s < N; ++s) {
			const OaSkelJoint& j = InSkel.Joints[static_cast<OaUsize>(s)];
			OaUsize c = base + static_cast<OaUsize>(InSkel.ChannelOffset(s));
			if (j.HasTranslate) {
				const VlmVec3 t = localTrans(f, s);
				samples[c++] = t.X; samples[c++] = t.Y; samples[c++] = t.Z;
			}
			const VlmQuat q = localRot(f, s);
			if (j.RotDof == 3) {
				OaF32 r6[6];
				QuatTo6D(q, r6);
				for (int i = 0; i < 6; ++i) { samples[c++] = r6[i]; }
			} else if (j.RotDof == 1) {
				samples[c++] = HingeAngleOf(q, j.Rest.JointOrient);
			}
		}

		// Contacts (trailing).
		const OaUsize cBase = base + static_cast<OaUsize>(InSkel.ContactOffset());
		for (OaI32 ci = 0; ci < C; ++ci) {
			const OaI32 s = InSkel.ContactJoints[static_cast<OaUsize>(ci)];
			const OaUsize wi = static_cast<OaUsize>(f) * N + s;
			const OaF32 height = Comp(worldPos[wi], up) - floorUp;
			VlmVec3 prev = worldPos[wi];
			VlmVec3 next = worldPos[wi];
			if (f > 0) { prev = worldPos[static_cast<OaUsize>(f - 1) * N + s]; }
			if (f + 1 < frames) { next = worldPos[static_cast<OaUsize>(f + 1) * N + s]; }
			const OaF32 dt = (f > 0 && f + 1 < frames) ? 2.0f / InUsd.Fps : 1.0f / InUsd.Fps;
			const OaF32 dh0 = Comp(next, h0) - Comp(prev, h0);
			const OaF32 dh1 = Comp(next, h1) - Comp(prev, h1);
			const OaF32 speed = std::sqrt(dh0 * dh0 + dh1 * dh1) / std::max(dt, 1e-6f);
			const OaF32 heightTerm = Clamp01(1.0f - height / kHeightBand);
			const OaF32 speedTerm  = Clamp01(1.0f - speed / kSpeedThresh);
			samples[cBase + static_cast<OaUsize>(ci)] = heightTerm * speedTerm;
		}
	}

	return OaPoseClip::Create(frames, static_cast<OaU32>(D), InUsd.Fps,
		InSkel.SkeletonId, OaSpan<const OaF32>(samples.Data(), samples.Size()));
}

OaResult<OaUsdSkelClip> OaPosePack::Unpack(const OaPoseClip& InClip, const OaSkeleton& InSkel) {
	if (!InClip.IsValid()) {
		return OaStatus::InvalidArgument("OaPosePack::Unpack: invalid clip");
	}
	if (!InSkel.IsValid()) {
		return OaStatus::InvalidArgument("OaPosePack::Unpack: invalid skeleton");
	}
	if (static_cast<OaI32>(InClip.PoseDim) != InSkel.PoseDim()) {
		return OaStatus::InvalidArgument("OaPosePack::Unpack: PoseDim != skeleton channel budget");
	}

	const OaI32 N = InSkel.JointCount();
	const OaU32 frames = InClip.FrameCount;

	OaUsdSkelClip usd;
	usd.FrameCount = frames;
	usd.Fps = InClip.Fps;
	usd.UpAxis = 1;  // Y-up to match Maya USD exports (requires retargeted data)

	// Full UsdSkel joint paths + bind (world rest) / rest (local) transforms.
	usd.JointPaths.Resize(static_cast<OaUsize>(N));
	usd.BindTransforms.Resize(static_cast<OaUsize>(N));
	usd.RestTransforms.Resize(static_cast<OaUsize>(N));
	for (OaI32 s = 0; s < N; ++s) {
		OaString path;
		OaVec<OaI32> chain;
		for (OaI32 cur = s; cur >= 0; cur = InSkel.Joints[static_cast<OaUsize>(cur)].ParentIndex) {
			chain.PushBack(cur);
		}
		for (OaI32 i = static_cast<OaI32>(chain.Size()) - 1; i >= 0; --i) {
			if (!path.empty()) { path.PushBack('/'); }
			path += InSkel.Joints[static_cast<OaUsize>(chain[static_cast<OaUsize>(i)])].Name;
		}
		usd.JointPaths[static_cast<OaUsize>(s)] = path;
		const OaSkelJoint& j = InSkel.Joints[static_cast<OaUsize>(s)];
		usd.RestTransforms[static_cast<OaUsize>(s)] = j.Rest.LocalMatrix();
		usd.BindTransforms[static_cast<OaUsize>(s)] = OaTrsMatrix(
			{ 1.0f, 1.0f, 1.0f },
			InSkel.RestWorldRotation(s),
			InSkel.RestWorld(s));
	}

	usd.Translations.Resize(static_cast<OaUsize>(frames) * N);
	usd.Rotations.Resize(static_cast<OaUsize>(frames) * N);
	for (OaU32 f = 0; f < frames; ++f) {
		const OaUsize base = static_cast<OaUsize>(f) * InClip.PoseDim;
		for (OaI32 s = 0; s < N; ++s) {
			const OaSkelJoint& j = InSkel.Joints[static_cast<OaUsize>(s)];
			OaUsize c = base + static_cast<OaUsize>(InSkel.ChannelOffset(s));

			VlmVec3 t = j.Rest.Translate;
			if (j.HasTranslate) {
				t = { InClip.Samples[c], InClip.Samples[c + 1], InClip.Samples[c + 2] };
				c += 3;
			}

			VlmQuat q = j.Rest.OrientedRotation();
			if (j.RotDof == 3) {
				OaF32 r6[6];
				for (int i = 0; i < 6; ++i) { r6[i] = InClip.Samples[c + static_cast<OaUsize>(i)]; }
				q = SixDToQuat(r6);
			} else if (j.RotDof == 1) {
				q = HingeQuatFrom(InClip.Samples[c], j.Rest.JointOrient);
			}

			usd.Translations[static_cast<OaUsize>(f) * N + s] = t;
			usd.Rotations[static_cast<OaUsize>(f) * N + s] = q;
		}
	}

	if (!usd.IsValid()) {
		return OaStatus::InvalidArgument("OaPosePack::Unpack: assembled USD clip invalid");
	}
	return usd;
}
