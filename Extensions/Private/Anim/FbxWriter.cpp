#include <Anim/FbxWriter.h>

#include <Anim/Usd.h>

#include <Oa/Core/Vlm.h>

#include <cmath>
#include <sstream>

namespace {

constexpr long long kFbxTimePerSec = 46186158000LL; // FBX KTime units/sec (TimeMode 6)
constexpr const char* kTake = "Take 001";

OaString Fv(OaF32 v) {
	std::ostringstream s;
	s.precision(9);
	s << static_cast<double>(v);
	return OaString(s.str());
}

// Index of the last '/' in a string, or OaString::npos.
OaUsize LastSlash(const OaString& s) {
	OaUsize slash = OaString::npos;
	for (OaUsize i = 0; i < s.Size(); ++i) {
		if (s[i] == '/') { slash = i; }
	}
	return slash;
}

// One animation curve node (T or R) with its three channel curves.
struct AnimNode {
	OaI32       Joint = 0;
	bool        IsRot = false;
	const char* Prop  = "";   // "Lcl Translation" | "Lcl Rotation"
	const char* Kind  = "";   // "T" | "R"
	long long   NodeId = 0;
	long long   CurveId[3] = { 0, 0, 0 };
	OaVec<OaF32> Col[3];
};

// Leaf bone name from a UsdSkel path.
OaString Leaf(const OaString& path) {
	const OaUsize slash = LastSlash(path);
	return slash == OaString::npos ? path : path.substr(slash + 1);
}

} // namespace

OaStatus OaFbx::WriteFbx(const OaPath& InPath, const OaUsdSkelClip& InClip) {
	if (!InClip.IsValid()) {
		return OaStatus::InvalidArgument("OaFbx::WriteFbx: invalid clip");
	}
	const OaI32 n = InClip.JointCount();
	const OaU32 frames = InClip.FrameCount;
	const OaF32 fps = InClip.Fps;

	// Parent index per joint, recovered from the slash-delimited paths.
	OaVec<OaI32> parent; parent.Resize(static_cast<OaUsize>(n));
	OaVec<OaString> leaf; leaf.Resize(static_cast<OaUsize>(n));
	for (OaI32 j = 0; j < n; ++j) {
		const OaString& path = InClip.JointPaths[static_cast<OaUsize>(j)];
		leaf[static_cast<OaUsize>(j)] = Leaf(path);
		const OaUsize slash = LastSlash(path);
		if (slash == OaString::npos) {
			parent[static_cast<OaUsize>(j)] = -1;
		} else {
			const OaString parentPath = path.substr(0, slash);
			OaI32 p = -1;
			for (OaI32 k = 0; k < n; ++k) {
				if (InClip.JointPaths[static_cast<OaUsize>(k)] == parentPath) { p = k; break; }
			}
			parent[static_cast<OaUsize>(j)] = p;
		}
	}

	// FBX key times.
	OaVec<long long> times; times.Resize(static_cast<OaUsize>(frames));
	for (OaU32 i = 0; i < frames; ++i) {
		times[i] = static_cast<long long>(std::llround(static_cast<double>(i) * kFbxTimePerSec / fps));
	}
	const long long stopTime = frames ? times[frames - 1] : 0;

	long long nextId = 1000000000LL;
	auto NewId = [&]() { return ++nextId; };

	OaVec<long long> boneId; boneId.Resize(static_cast<OaUsize>(n));
	OaVec<long long> attrId; attrId.Resize(static_cast<OaUsize>(n));
	for (OaI32 j = 0; j < n; ++j) { boneId[static_cast<OaUsize>(j)] = NewId(); }
	for (OaI32 j = 0; j < n; ++j) { attrId[static_cast<OaUsize>(j)] = NewId(); }

	// Build animation nodes: T where a joint translates, R where it rotates.
	OaVec<AnimNode> anim;
	for (OaI32 j = 0; j < n; ++j) {
		const VlmVec3 t0 = InClip.Translations[static_cast<OaUsize>(j)];
		const VlmQuat q0 = InClip.Rotations[static_cast<OaUsize>(j)];
		bool hasT = false, hasR = false;
		for (OaU32 f = 1; f < frames; ++f) {
			const VlmVec3 t = InClip.Translations[static_cast<OaUsize>(f) * n + j];
			const VlmQuat q = InClip.Rotations[static_cast<OaUsize>(f) * n + j];
			if (t.X != t0.X || t.Y != t0.Y || t.Z != t0.Z) { hasT = true; }
			if (q.X != q0.X || q.Y != q0.Y || q.Z != q0.Z || q.W != q0.W) { hasR = true; }
		}
		for (int kind = 0; kind < 2; ++kind) {
			const bool isRot = (kind == 1);
			if ((isRot && !hasR) || (!isRot && !hasT)) { continue; }
			AnimNode node;
			node.Joint = j;
			node.IsRot = isRot;
			node.Prop = isRot ? "Lcl Rotation" : "Lcl Translation";
			node.Kind = isRot ? "R" : "T";
			for (int c = 0; c < 3; ++c) { node.Col[c].Resize(static_cast<OaUsize>(frames)); }
			for (OaU32 f = 0; f < frames; ++f) {
				if (isRot) {
					// quat → Euler (deg). QuaternionToEuler returns (yaw=Z, pitch=Y, roll=X);
					// FBX Lcl Rotation channels are (X, Y, Z) = (roll, pitch, yaw).
					const VlmVec3 e = Vlm::QuaternionToEuler(InClip.Rotations[static_cast<OaUsize>(f) * n + j]);
					node.Col[0][f] = e.Z;
					node.Col[1][f] = e.Y;
					node.Col[2][f] = e.X;
				} else {
					const VlmVec3 t = InClip.Translations[static_cast<OaUsize>(f) * n + j];
					node.Col[0][f] = t.X;
					node.Col[1][f] = t.Y;
					node.Col[2][f] = t.Z;
				}
			}
			anim.PushBack(std::move(node));
		}
	}
	for (AnimNode& a : anim) {
		a.NodeId = NewId();
		for (int c = 0; c < 3; ++c) { a.CurveId[c] = NewId(); }
	}
	const long long stackId = NewId();
	const long long layerId = NewId();
	const long long docId   = NewId();
	const OaUsize nNodes  = anim.Size();
	const OaUsize nCurves = nNodes * 3;
	const bool hasAnim = nNodes > 0;

	std::ostringstream o;
	o.precision(9);

	// ── FBXHeaderExtension ──
	o << "; FBX 7.5.0 project file\n";
	o << "; ----------------------------------------------------\n\n";
	o << "FBXHeaderExtension:  {\n";
	o << "\tFBXHeaderVersion: 1003\n\tFBXVersion: 7500\n";
	o << "\tCreationTimeStamp:  {\n\t\tVersion: 1000\n";
	o << "\t\tYear: 2026\n\t\tMonth: 6\n\t\tDay: 16\n\t\tHour: 0\n\t\tMinute: 0\n\t\tSecond: 0\n\t\tMillisecond: 0\n\t}\n";
	o << "\tCreator: \"FBX SDK/FBX Plugins version 2020.2\"\n";
	o << "\tSceneInfo: \"SceneInfo::GlobalInfo\", \"UserData\" {\n";
	o << "\t\tType: \"UserData\"\n\t\tVersion: 100\n\t\tMetaData:  {\n";
	o << "\t\t\tVersion: 100\n\t\t\tTitle: \"OA Gen3dAnim\"\n\t\t\tSubject: \"\"\n";
	o << "\t\t\tAuthor: \"\"\n\t\t\tKeywords: \"\"\n\t\t\tRevision: \"\"\n\t\t\tComment: \"\"\n\t\t}\n\t}\n}\n";

	// ── GlobalSettings (Z-up, cm, TimeMode 6) ──
	o << "GlobalSettings:  {\n\tVersion: 1000\n\tProperties70:  {\n";
	o << "\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\",2\n";
	o << "\t\tP: \"UpAxisSign\", \"int\", \"Integer\", \"\",1\n";
	o << "\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\",1\n";
	o << "\t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\",-1\n";
	o << "\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n";
	o << "\t\tP: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1\n";
	o << "\t\tP: \"OriginalUpAxis\", \"int\", \"Integer\", \"\",2\n";
	o << "\t\tP: \"OriginalUpAxisSign\", \"int\", \"Integer\", \"\",1\n";
	o << "\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1\n";
	o << "\t\tP: \"OriginalUnitScaleFactor\", \"double\", \"Number\", \"\",1\n";
	o << "\t\tP: \"TimeMode\", \"enum\", \"\", \"\",6\n";
	o << "\t\tP: \"TimeSpanStart\", \"KTime\", \"Time\", \"\",0\n";
	o << "\t\tP: \"TimeSpanStop\", \"KTime\", \"Time\", \"\"," << stopTime << "\n";
	o << "\t}\n}\n";

	// ── Documents ──
	o << "Documents:  {\n\tCount: 1\n\tDocument: " << docId << ", \"\", \"Scene\" {\n";
	o << "\t\tProperties70:  {\n";
	o << "\t\t\tP: \"SourceObject\", \"object\", \"\", \"\"\n";
	o << "\t\t\tP: \"ActiveAnimStackName\", \"KString\", \"\", \"\", \"" << (hasAnim ? kTake : "") << "\"\n";
	o << "\t\t}\n\t\tRootNode: 0\n\t}\n}\n";
	o << "References:  {\n}\n";

	// ── Definitions ──
	const OaUsize total = 1 + static_cast<OaUsize>(n) * 2 + 2 + nNodes + nCurves;
	o << "Definitions:  {\n\tVersion: 100\n\tCount: " << total << "\n";
	o << "\tObjectType: \"GlobalSettings\" {\n\t\tCount: 1\n\t}\n";
	o << "\tObjectType: \"AnimationStack\" {\n\t\tCount: 1\n\t\tPropertyTemplate: \"FbxAnimStack\" {\n\t\t\tProperties70:  {\n";
	o << "\t\t\t\tP: \"Description\", \"KString\", \"\", \"\", \"\"\n";
	o << "\t\t\t\tP: \"LocalStart\", \"KTime\", \"Time\", \"\",0\n";
	o << "\t\t\t\tP: \"LocalStop\", \"KTime\", \"Time\", \"\",0\n";
	o << "\t\t\t\tP: \"ReferenceStart\", \"KTime\", \"Time\", \"\",0\n";
	o << "\t\t\t\tP: \"ReferenceStop\", \"KTime\", \"Time\", \"\",0\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"AnimationLayer\" {\n\t\tCount: 1\n\t\tPropertyTemplate: \"FbxAnimLayer\" {\n\t\t\tProperties70:  {\n\t\t\t\tP: \"Weight\", \"Number\", \"\", \"A\",100\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"NodeAttribute\" {\n\t\tCount: " << n << "\n\t\tPropertyTemplate: \"FbxSkeleton\" {\n\t\t\tProperties70:  {\n";
	o << "\t\t\t\tP: \"Color\", \"ColorRGB\", \"Color\", \"\",0.8,0.8,0.8\n";
	o << "\t\t\t\tP: \"Size\", \"double\", \"Number\", \"\",100\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"Model\" {\n\t\tCount: " << n << "\n\t\tPropertyTemplate: \"FbxNode\" {\n\t\t\tProperties70:  {\n";
	o << "\t\t\t\tP: \"RotationActive\", \"bool\", \"\", \"\",0\n";
	o << "\t\t\t\tP: \"InheritType\", \"enum\", \"\", \"\",0\n";
	o << "\t\t\t\tP: \"ScalingMax\", \"Vector3D\", \"Vector\", \"\",0,0,0\n";
	o << "\t\t\t\tP: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",-1\n";
	o << "\t\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0\n";
	o << "\t\t\t\tP: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",0,0,0\n";
	o << "\t\t\t\tP: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1\n";
	o << "\t\t\t\tP: \"Visibility\", \"Visibility\", \"\", \"A\",1\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"AnimationCurveNode\" {\n\t\tCount: " << nNodes << "\n\t\tPropertyTemplate: \"FbxAnimCurveNode\" {\n\t\t\tProperties70:  {\n\t\t\t\tP: \"d\", \"Compound\", \"\", \"\"\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"AnimationCurve\" {\n\t\tCount: " << nCurves << "\n\t}\n";
	o << "}\n";

	// ── Objects ──
	o << "Objects:  {\n";
	for (OaI32 j = 0; j < n; ++j) {
		const OaString& name = leaf[static_cast<OaUsize>(j)];
		if (parent[static_cast<OaUsize>(j)] < 0) {
			o << "\tNodeAttribute: " << attrId[static_cast<OaUsize>(j)] << ", \"NodeAttribute::" << name << "\", \"Root\" {\n";
			o << "\t\tTypeFlags: \"Null\", \"Skeleton\", \"Root\"\n\t}\n";
		} else {
			o << "\tNodeAttribute: " << attrId[static_cast<OaUsize>(j)] << ", \"NodeAttribute::" << name << "\", \"LimbNode\" {\n";
			o << "\t\tProperties70:  {\n\t\t\tP: \"Size\", \"double\", \"Number\", \"\",6\n\t\t}\n";
			o << "\t\tTypeFlags: \"Skeleton\"\n\t}\n";
		}
	}
	for (OaI32 j = 0; j < n; ++j) {
		const OaString& name = leaf[static_cast<OaUsize>(j)];
		const VlmVec3 rest = InClip.Translations[static_cast<OaUsize>(j)]; // frame 0 local
		o << "\tModel: " << boneId[static_cast<OaUsize>(j)] << ", \"Model::" << name << "\", \"LimbNode\" {\n";
		o << "\t\tVersion: 232\n\t\tProperties70:  {\n";
		o << "\t\t\tP: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n";
		o << "\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A+\"," << Fv(rest.X) << "," << Fv(rest.Y) << "," << Fv(rest.Z) << "\n";
		o << "\t\t\tP: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A+\",0,0,0\n";
		o << "\t\t}\n\t\tShading: Y\n\t\tCulling: \"CullingOff\"\n\t}\n";
	}
	if (hasAnim) {
		o << "\tAnimationStack: " << stackId << ", \"AnimStack::" << kTake << "\", \"\" {\n";
		o << "\t\tProperties70:  {\n";
		o << "\t\t\tP: \"LocalStart\", \"KTime\", \"Time\", \"\",0\n";
		o << "\t\t\tP: \"LocalStop\", \"KTime\", \"Time\", \"\"," << stopTime << "\n";
		o << "\t\t\tP: \"ReferenceStart\", \"KTime\", \"Time\", \"\",0\n";
		o << "\t\t\tP: \"ReferenceStop\", \"KTime\", \"Time\", \"\"," << stopTime << "\n\t\t}\n\t}\n";
		o << "\tAnimationLayer: " << layerId << ", \"AnimLayer::Base Layer\", \"\" {\n\t}\n";
	}
	for (const AnimNode& a : anim) {
		o << "\tAnimationCurveNode: " << a.NodeId << ", \"AnimCurveNode::" << a.Kind << "\", \"\" {\n";
		o << "\t\tProperties70:  {\n";
		const char* chans[3] = { "d|X", "d|Y", "d|Z" };
		for (int c = 0; c < 3; ++c) {
			o << "\t\t\tP: \"" << chans[c] << "\", \"Number\", \"\", \"A\"," << Fv(a.Col[c][0]) << "\n";
		}
		o << "\t\t}\n\t}\n";
		for (int c = 0; c < 3; ++c) {
			o << "\tAnimationCurve: " << a.CurveId[c] << ", \"AnimCurve::\", \"\" {\n";
			o << "\t\tDefault: " << Fv(a.Col[c][0]) << "\n\t\tKeyVer: 4009\n";
			o << "\t\tKeyTime: *" << frames << " {\n\t\t\ta: ";
			for (OaU32 f = 0; f < frames; ++f) { o << (f ? "," : "") << times[f]; }
			o << "\n\t\t}\n";
			o << "\t\tKeyValueFloat: *" << frames << " {\n\t\t\ta: ";
			for (OaU32 f = 0; f < frames; ++f) { o << (f ? "," : "") << Fv(a.Col[c][f]); }
			o << "\n\t\t}\n";
			o << "\t\tKeyAttrFlags: *1 {\n\t\t\ta: 24840\n\t\t}\n";
			o << "\t\tKeyAttrDataFloat: *4 {\n\t\t\ta: 0,0,0,0\n\t\t}\n";
			o << "\t\tKeyAttrRefCount: *1 {\n\t\t\ta: " << frames << "\n\t\t}\n\t}\n";
		}
	}
	o << "}\n";

	// ── Connections ──
	o << "Connections:  {\n";
	for (OaI32 j = 0; j < n; ++j) {
		const OaI32 p = parent[static_cast<OaUsize>(j)];
		const long long dst = (p < 0) ? 0 : boneId[static_cast<OaUsize>(p)];
		o << "\t;Model::" << leaf[static_cast<OaUsize>(j)] << ", Model::" << (p < 0 ? OaString("RootNode") : leaf[static_cast<OaUsize>(p)]) << "\n";
		o << "\tC: \"OO\"," << boneId[static_cast<OaUsize>(j)] << "," << dst << "\n";
		o << "\t;NodeAttribute::" << leaf[static_cast<OaUsize>(j)] << ", Model::" << leaf[static_cast<OaUsize>(j)] << "\n";
		o << "\tC: \"OO\"," << attrId[static_cast<OaUsize>(j)] << "," << boneId[static_cast<OaUsize>(j)] << "\n";
	}
	if (hasAnim) {
		o << "\t;AnimLayer::Base Layer, AnimStack::" << kTake << "\n";
		o << "\tC: \"OO\"," << layerId << "," << stackId << "\n";
	}
	for (const AnimNode& a : anim) {
		o << "\tC: \"OO\"," << a.NodeId << "," << layerId << "\n";
		o << "\tC: \"OP\"," << a.NodeId << "," << boneId[static_cast<OaUsize>(a.Joint)] << ", \"" << a.Prop << "\"\n";
		const char* chans[3] = { "d|X", "d|Y", "d|Z" };
		for (int c = 0; c < 3; ++c) {
			o << "\tC: \"OP\"," << a.CurveId[c] << "," << a.NodeId << ", \"" << chans[c] << "\"\n";
		}
	}
	o << "}\n";

	// ── Takes (legacy; Maya importer still reads it) ──
	if (hasAnim) {
		o << "Takes:  {\n\tCurrent: \"" << kTake << "\"\n\tTake: \"" << kTake << "\" {\n";
		o << "\t\tFileName: \"Take_001.tak\"\n";
		o << "\t\tLocalTime: 0," << stopTime << "\n";
		o << "\t\tReferenceTime: 0," << stopTime << "\n\t}\n}\n";
	}

	return OaFileIo::WriteText(InPath, OaString(o.str()));
}
