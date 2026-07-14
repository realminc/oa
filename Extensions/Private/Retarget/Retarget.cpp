#include <Retarget/Retarget.h>

#include <Core/Transform.h>

namespace {

using namespace Vlm;

// 6D rotation = first two columns of the rotation matrix (Zhou et al. 2019).
// Mirrors the helpers in OaPosePack so retarget speaks the same channel dialect.
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

// Re-express a packed local rotation from src-rest to dst-rest:
//   delta    = srcRef⁻¹ ⊗ animLocal     (animation relative to the source rest)
//   newLocal = dstRef   ⊗ delta         (re-applied on top of the dest rest)
void RetargetChannel(OaF32* InOut6D, const VlmQuat& InSrcRef, const VlmQuat& InDstRef) {
	OaF32 d[6];
	for (int i = 0; i < 6; ++i) { d[i] = InOut6D[i]; }
	const VlmQuat animLocal = SixDToQuat(d);
	const VlmQuat delta    = VlmQuatMul(VlmQuatConjugate(InSrcRef), animLocal);
	const VlmQuat newLocal = VlmQuatMul(InDstRef, delta);
	QuatTo6D(VlmQuatNormalize(newLocal), InOut6D);
}

} // namespace

OaResult<OaPoseClip> OaRetarget::RetargetClip(const OaSkeleton& InSrc,
                                              const OaSkeleton& InDst,
                                              const OaPoseClip& InClip,
                                              const OaRefPose&  InSrcRef,
                                              const OaRefPose&  InDstRef) {
	if (!InSrc.IsValid() || !InDst.IsValid()) {
		return OaStatus::InvalidArgument("OaRetarget::RetargetClip: invalid skeleton");
	}
	if (!InClip.IsValid()) {
		return OaStatus::InvalidArgument("OaRetarget::RetargetClip: invalid clip");
	}
	const OaI32 N = InSrc.JointCount();
	if (InDst.JointCount() != N) {
		return OaStatus::InvalidArgument("OaRetarget::RetargetClip: src/dst joint count mismatch");
	}
	if (static_cast<OaI32>(InClip.PoseDim) != InSrc.PoseDim() ||
	    InSrc.PoseDim() != InDst.PoseDim()) {
		return OaStatus::InvalidArgument("OaRetarget::RetargetClip: PoseDim mismatch");
	}

	// Per-joint source/dest rest orientations (by bone name, identity if absent).
	OaVec<VlmQuat> srcRef; srcRef.Resize(static_cast<OaUsize>(N));
	OaVec<VlmQuat> dstRef; dstRef.Resize(static_cast<OaUsize>(N));
	for (OaI32 s = 0; s < N; ++s) {
		srcRef[static_cast<OaUsize>(s)] = InSrcRef.OrientOf(InSrc.Joints[static_cast<OaUsize>(s)].Name);
		dstRef[static_cast<OaUsize>(s)] = InDstRef.OrientOf(InDst.Joints[static_cast<OaUsize>(s)].Name);
	}

	OaVec<OaF32> out = InClip.Samples;   // copy; translations + hinges + contacts pass through
	const OaI32 D = static_cast<OaI32>(InClip.PoseDim);
	for (OaU32 f = 0; f < InClip.FrameCount; ++f) {
		const OaUsize base = static_cast<OaUsize>(f) * D;
		for (OaI32 s = 0; s < N; ++s) {
			const OaSkelJoint& j = InDst.Joints[static_cast<OaUsize>(s)];
			OaUsize c = base + static_cast<OaUsize>(InDst.ChannelOffset(s));
			if (j.HasTranslate) { c += 3; }     // translate passes through
			// Only full-6D joints are re-oriented; a hinge angle is already a
			// rest-relative spin and transfers unchanged.
			if (j.RotDof == 3) {
				RetargetChannel(&out[c], srcRef[static_cast<OaUsize>(s)], dstRef[static_cast<OaUsize>(s)]);
			}
		}
	}

	return OaPoseClip::Create(InClip.FrameCount, InClip.PoseDim, InClip.Fps,
		InDst.SkeletonId, OaSpan<const OaF32>(out.Data(), out.Size()));
}
