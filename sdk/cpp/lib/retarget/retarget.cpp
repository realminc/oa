#include <retarget/retarget.h>

#include <oa/core/fnTransform.h>

namespace {

// Re-express a packed local rotation from src-rest to dst-rest:
//   delta    = srcRef⁻¹ ⊗ animLocal     (animation relative to the source rest)
//   newLocal = dstRef   ⊗ delta         (re-applied on top of the dest rest)
void retargetChannel(
	oa::F32* inOut6D,
	const oa::vlm::Quat& inSrcRef,
	const oa::vlm::Quat& inDstRef
) {
	oa::F32 d[6];
	for (int i = 0; i < 6; ++i) { d[i] = inOut6D[i]; }
	const oa::vlm::Quat animLocal = oa::FnTransform::quaternionFromSixD(d);
	const oa::vlm::Quat delta = inSrcRef.conjugate() * animLocal;
	const oa::vlm::Quat newLocal = inDstRef * delta;
	oa::FnTransform::quaternionToSixD(newLocal.normalized(), inOut6D);
}

} // namespace

oa::Result<oa::PoseClip> oa::Retarget::retargetClip(const oa::Skeleton& inSrc,
                                              const oa::Skeleton& inDst,
                                              const oa::PoseClip& inClip,
                                              const oa::RefPose&  inSrcRef,
                                              const oa::RefPose&  inDstRef) {
	if (!inSrc.isValid() || !inDst.isValid()) {
		return oa::Status::invalidArgument("oa::Retarget::RetargetClip: invalid skeleton");
	}
	if (!inClip.isValid()) {
		return oa::Status::invalidArgument("oa::Retarget::RetargetClip: invalid clip");
	}
	const oa::I32 N = inSrc.jointCount();
	if (inDst.jointCount() != N) {
		return oa::Status::invalidArgument("oa::Retarget::RetargetClip: src/dst joint count mismatch");
	}
	if (static_cast<oa::I32>(inClip.poseDim) != inSrc.poseDim() ||
	    inSrc.poseDim() != inDst.poseDim()) {
		return oa::Status::invalidArgument("oa::Retarget::RetargetClip: poseDim mismatch");
	}

	// Per-joint source/dest rest orientations (by bone name, identity if absent).
	oa::Vector<oa::vlm::Quat> srcRef; srcRef.resize(static_cast<oa::Usize>(N));
	oa::Vector<oa::vlm::Quat> dstRef; dstRef.resize(static_cast<oa::Usize>(N));
	for (oa::I32 s = 0; s < N; ++s) {
		srcRef[static_cast<oa::Usize>(s)] = inSrcRef.orientOf(inSrc.joints[static_cast<oa::Usize>(s)].name);
		dstRef[static_cast<oa::Usize>(s)] = inDstRef.orientOf(inDst.joints[static_cast<oa::Usize>(s)].name);
	}

	oa::Vector<oa::F32> out = inClip.samples;   // copy; translations + hinges + contacts pass through
	const oa::I32 D = static_cast<oa::I32>(inClip.poseDim);
	for (oa::U32 f = 0; f < inClip.frameCount; ++f) {
		const oa::Usize base = static_cast<oa::Usize>(f) * D;
		for (oa::I32 s = 0; s < N; ++s) {
			const oa::SkelJoint& j = inDst.joints[static_cast<oa::Usize>(s)];
			oa::Usize c = base + static_cast<oa::Usize>(inDst.channelOffset(s));
			if (j.hasTranslate) { c += 3; }     // translate passes through
			// Only full-6D joints are re-oriented; a hinge angle is already a
			// rest-relative spin and transfers unchanged.
			if (j.rotDof == 3) {
				retargetChannel(&out[c], srcRef[static_cast<oa::Usize>(s)], dstRef[static_cast<oa::Usize>(s)]);
			}
		}
	}

	return oa::PoseClip::create(inClip.frameCount, inClip.poseDim, inClip.fps,
		inDst.skeletonId, oa::Span<const oa::F32>(out.data(), out.size()));
}
