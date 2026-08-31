#include <anim/posePack.h>

#include <oa/core/fnTransform.h>

#include <algorithm>
#include <cmath>

namespace {

// Hinge encode/decode. A hinge joint animates only rotateZ relative to its rest
// orientation. The stored USD quat is the full local orientation
// (restOrient ⊗ rotate); the live delta is rotate = restOrient⁻¹ ⊗ q, which for a
// clean clip is a pure spin about local Z → a single angle (radians).
oa::F32 hingeAngleOf(const oa::vlm::Quat& inLocal, const oa::vlm::Quat& inRestOrient) {
	const oa::vlm::Quat delta = inRestOrient.conjugate() * inLocal;
	return 2.0f * std::atan2(delta.z, delta.w);
}
oa::vlm::Quat hingeQuatFrom(oa::F32 inAngle, const oa::vlm::Quat& inRestOrient) {
	const oa::F32 h = inAngle * 0.5f;
	const oa::vlm::Quat delta = { 0.0f, 0.0f, std::sin(h), std::cos(h) }; // spin about local Z
	return inRestOrient * delta;
}

oa::F32 clamp01(oa::F32 v) {
	return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

oa::F32 comp(const oa::vlm::Vec3& v, oa::I32 i) {
	return i == 0 ? v.x : (i == 1 ? v.y : v.z);
}

// Leaf bone name = path segment after the last '/'.
oa::StringView leafName(oa::StringView path) {
	oa::Usize slash = oa::StringView::Npos;
	for (oa::Usize i = 0; i < path.size(); ++i) {
		if (path[i] == '/') { slash = i; }
	}
	return slash == oa::StringView::Npos ? path : path.subStr(slash + 1);
}

} // namespace

oa::Result<oa::PoseClip> oa::PosePack::pack(const oa::UsdSkelClip& inUsd, const oa::Skeleton& inSkel) {
	if (!inUsd.isValid()) {
		return oa::Status::invalidArgument("oa::PosePack::Pack: invalid USD clip");
	}
	if (!inSkel.isValid()) {
		return oa::Status::invalidArgument("oa::PosePack::Pack: invalid skeleton");
	}

	const oa::I32 N = inSkel.jointCount();
	const oa::I32 usdN = inUsd.jointCount();

	// map skeleton joint → USD joint by leaf bone name. Missing is allowed (the
	// clip may lack a base joint, or carry junk joints we ignore) → rest fallback.
	oa::Vector<oa::I32> usdOf;
	usdOf.resize(static_cast<oa::Usize>(N));
	for (oa::I32 s = 0; s < N; ++s) {
		const oa::StringView want = inSkel.joints[static_cast<oa::Usize>(s)].name;
		oa::I32 found = -1;
		for (oa::I32 u = 0; u < usdN; ++u) {
			if (leafName(inUsd.jointPaths[static_cast<oa::Usize>(u)]) == want) { found = u; break; }
		}
		usdOf[static_cast<oa::Usize>(s)] = found;
	}

	const oa::U32 frames = inUsd.frameCount;
	const oa::I32 D = inSkel.poseDim();
	const oa::I32 C = static_cast<oa::I32>(inSkel.contactJoints.size());

	// Up axis and the two horizontal axes for contact derivation.
	const oa::I32 up = (inUsd.upAxis == 1) ? 1 : 2;
	const oa::I32 h0 = (up == 2) ? 0 : 0;
	const oa::I32 h1 = (up == 2) ? 1 : 2;

	// local transform of skeleton joint s at frame f (USD value, or rest fallback).
	auto localTrans = [&](oa::U32 f, oa::I32 s) -> oa::vlm::Vec3 {
		const oa::I32 u = usdOf[static_cast<oa::Usize>(s)];
		return (u >= 0) ? inUsd.translations[static_cast<oa::Usize>(f) * usdN + u]
		                : inSkel.joints[static_cast<oa::Usize>(s)].rest.getTransform().getPosition();
	};
	auto localRot = [&](oa::U32 f, oa::I32 s) -> oa::vlm::Quat {
		const oa::I32 u = usdOf[static_cast<oa::Usize>(s)];
		return (u >= 0) ? inUsd.rotations[static_cast<oa::Usize>(f) * usdN + u]
		                : inSkel.joints[static_cast<oa::Usize>(s)].rest.getOrientedRotation();
	};

	// forward kinematics → world position of every joint (contacts only). Skeleton
	// order guarantees each parent is solved first.
	oa::Vector<oa::vlm::Vec3> worldPos; worldPos.resize(static_cast<oa::Usize>(frames) * N);
	oa::Vector<oa::vlm::Quat> worldRot; worldRot.resize(static_cast<oa::Usize>(frames) * N);
	for (oa::U32 f = 0; f < frames; ++f) {
		for (oa::I32 s = 0; s < N; ++s) {
			const oa::vlm::Vec3 lt = localTrans(f, s);
			const oa::vlm::Quat lr = localRot(f, s);
			const oa::I32 parent = inSkel.joints[static_cast<oa::Usize>(s)].parentIndex;
			const oa::Usize wi = static_cast<oa::Usize>(f) * N + s;
			if (parent < 0) {
				worldRot[wi] = lr; worldPos[wi] = lt;
			} else {
				const oa::Usize pi = static_cast<oa::Usize>(f) * N + parent;
				worldRot[wi] = worldRot[pi] * lr;
				worldPos[wi] = oa::vlm::add(
					worldPos[pi], oa::vlm::rotateVector(worldRot[pi], lt));
			}
		}
	}

	// Contact derivation: floor = global min foot height along the up axis; a foot
	// is "planted" when near the floor and horizontally slow. Soft value in [0,1].
	constexpr oa::F32 kHeightBand = 6.0f;
	constexpr oa::F32 kSpeedThresh = 40.0f;
	oa::F32 floorUp = 1e30f;
	for (oa::I32 ci = 0; ci < C; ++ci) {
		const oa::I32 s = inSkel.contactJoints[static_cast<oa::Usize>(ci)];
		for (oa::U32 f = 0; f < frames; ++f) {
			floorUp = std::min(floorUp, comp(worldPos[static_cast<oa::Usize>(f) * N + s], up));
		}
	}

	oa::Vector<oa::F32> samples;
	samples.resize(static_cast<oa::Usize>(frames) * D);
	for (oa::U32 f = 0; f < frames; ++f) {
		const oa::Usize base = static_cast<oa::Usize>(f) * D;

		// Per-joint live channels at the joint's canonical offset.
		for (oa::I32 s = 0; s < N; ++s) {
			const oa::SkelJoint& j = inSkel.joints[static_cast<oa::Usize>(s)];
			oa::Usize c = base + static_cast<oa::Usize>(inSkel.channelOffset(s));
			if (j.hasTranslate) {
				const oa::vlm::Vec3 t = localTrans(f, s);
				samples[c++] = t.x; samples[c++] = t.y; samples[c++] = t.z;
			}
			const oa::vlm::Quat q = localRot(f, s);
			if (j.rotDof == 3) {
				oa::F32 r6[6];
				oa::FnTransform::quaternionToSixD(q, r6);
				for (int i = 0; i < 6; ++i) { samples[c++] = r6[i]; }
			} else if (j.rotDof == 1) {
				samples[c++] = hingeAngleOf(q, j.rest.getOrientation());
			}
		}

		// contacts (trailing).
		const oa::Usize cBase = base + static_cast<oa::Usize>(inSkel.contactOffset());
		for (oa::I32 ci = 0; ci < C; ++ci) {
			const oa::I32 s = inSkel.contactJoints[static_cast<oa::Usize>(ci)];
			const oa::Usize wi = static_cast<oa::Usize>(f) * N + s;
			const oa::F32 height = comp(worldPos[wi], up) - floorUp;
			oa::vlm::Vec3 prev = worldPos[wi];
			oa::vlm::Vec3 next = worldPos[wi];
			if (f > 0) { prev = worldPos[static_cast<oa::Usize>(f - 1) * N + s]; }
			if (f + 1 < frames) { next = worldPos[static_cast<oa::Usize>(f + 1) * N + s]; }
			const oa::F32 dt = (f > 0 && f + 1 < frames) ? 2.0f / inUsd.fps : 1.0f / inUsd.fps;
			const oa::F32 dh0 = comp(next, h0) - comp(prev, h0);
			const oa::F32 dh1 = comp(next, h1) - comp(prev, h1);
			const oa::F32 speed = std::sqrt(dh0 * dh0 + dh1 * dh1) / std::max(dt, 1e-6f);
			const oa::F32 heightTerm = clamp01(1.0f - height / kHeightBand);
			const oa::F32 speedTerm  = clamp01(1.0f - speed / kSpeedThresh);
			samples[cBase + static_cast<oa::Usize>(ci)] = heightTerm * speedTerm;
		}
	}

	return oa::PoseClip::create(frames, static_cast<oa::U32>(D), inUsd.fps,
		inSkel.skeletonId, oa::Span<const oa::F32>(samples.data(), samples.size()));
}

oa::Result<oa::UsdSkelClip> oa::PosePack::unpack(const oa::PoseClip& inClip, const oa::Skeleton& inSkel) {
	if (!inClip.isValid()) {
		return oa::Status::invalidArgument("oa::PosePack::Unpack: invalid clip");
	}
	if (!inSkel.isValid()) {
		return oa::Status::invalidArgument("oa::PosePack::Unpack: invalid skeleton");
	}
	if (static_cast<oa::I32>(inClip.poseDim) != inSkel.poseDim()) {
		return oa::Status::invalidArgument("oa::PosePack::Unpack: poseDim != skeleton channel budget");
	}

	const oa::I32 N = inSkel.jointCount();
	const oa::U32 frames = inClip.frameCount;

	oa::UsdSkelClip usd;
	usd.frameCount = frames;
	usd.fps = inClip.fps;
	usd.upAxis = 1;  // Y-up to match Maya USD exports (requires retargeted data)

	// Full UsdSkel joint paths + bind (world rest) / rest (local) transforms.
	usd.jointPaths.resize(static_cast<oa::Usize>(N));
	usd.bindTransforms.resize(static_cast<oa::Usize>(N));
	usd.restTransforms.resize(static_cast<oa::Usize>(N));
	for (oa::I32 s = 0; s < N; ++s) {
		oa::String path;
		oa::Vector<oa::I32> chain;
		for (oa::I32 cur = s; cur >= 0; cur = inSkel.joints[static_cast<oa::Usize>(cur)].parentIndex) {
			chain.pushBack(cur);
		}
		for (oa::I32 i = static_cast<oa::I32>(chain.size()) - 1; i >= 0; --i) {
			if (!path.empty()) { path.pushBack('/'); }
			path += inSkel.joints[static_cast<oa::Usize>(chain[static_cast<oa::Usize>(i)])].name;
		}
		usd.jointPaths[static_cast<oa::Usize>(s)] = path;
		const oa::SkelJoint& j = inSkel.joints[static_cast<oa::Usize>(s)];
		usd.restTransforms[static_cast<oa::Usize>(s)] = j.rest.getMatrix();
		usd.bindTransforms[static_cast<oa::Usize>(s)] = oa::vlm::composeTrs(
			inSkel.restWorld(s),
			inSkel.restWorldRotation(s),
			{1.0F, 1.0F, 1.0F});
	}

	usd.translations.resize(static_cast<oa::Usize>(frames) * N);
	usd.rotations.resize(static_cast<oa::Usize>(frames) * N);
	for (oa::U32 f = 0; f < frames; ++f) {
		const oa::Usize base = static_cast<oa::Usize>(f) * inClip.poseDim;
		for (oa::I32 s = 0; s < N; ++s) {
			const oa::SkelJoint& j = inSkel.joints[static_cast<oa::Usize>(s)];
			oa::Usize c = base + static_cast<oa::Usize>(inSkel.channelOffset(s));

			oa::vlm::Vec3 t = j.rest.getTransform().getPosition();
			if (j.hasTranslate) {
				t = { inClip.samples[c], inClip.samples[c + 1], inClip.samples[c + 2] };
				c += 3;
			}

			oa::vlm::Quat q = j.rest.getOrientedRotation();
			if (j.rotDof == 3) {
				oa::F32 r6[6];
				for (int i = 0; i < 6; ++i) { r6[i] = inClip.samples[c + static_cast<oa::Usize>(i)]; }
				q = oa::FnTransform::quaternionFromSixD(r6);
			} else if (j.rotDof == 1) {
				q = hingeQuatFrom(inClip.samples[c], j.rest.getOrientation());
			}

			usd.translations[static_cast<oa::Usize>(f) * N + s] = t;
			usd.rotations[static_cast<oa::Usize>(f) * N + s] = q;
		}
	}

	if (!usd.isValid()) {
		return oa::Status::invalidArgument("oa::PosePack::Unpack: assembled USD clip invalid");
	}
	return usd;
}
