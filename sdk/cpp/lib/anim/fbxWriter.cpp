#include <anim/fbxWriter.h>
#include <core/streamText.h>

#include <anim/usd.h>

#include <core/streamText.h>

#include <oa/core/vlm.h>

#include <cmath>
#include <sstream>

namespace {

constexpr long long kFbxTimePerSec = 46186158000LL; // FBX KTime units/sec (TimeMode 6)
constexpr const char* kTake = "Take 001";

oa::String fv(oa::F32 v) {
	std::ostringstream s;
	s.precision(9);
	s << static_cast<double>(v);
	return oa::sdk::fromStdString(s.str());
}

// index of the last '/' in a string, or oa::String::Npos.
oa::Usize lastSlash(const oa::String& s) {
	oa::Usize slash = oa::String::Npos;
	for (oa::Usize i = 0; i < s.size(); ++i) {
		if (s[i] == '/') { slash = i; }
	}
	return slash;
}

// One animation curve node (T or R) with its three channel curves.
struct AnimNode {
	oa::I32       joint = 0;
	bool        isRot = false;
	const char* prop  = "";   // "Lcl Translation" | "Lcl rotation"
	const char* kind  = "";   // "T" | "R"
	long long   nodeId = 0;
	long long   curveId[3] = { 0, 0, 0 };
	oa::Vector<oa::F32> col[3];
};

// Leaf bone name from a UsdSkel path.
oa::String leaf(const oa::String& path) {
	const oa::Usize slash = lastSlash(path);
	return slash == oa::String::Npos ? path : path.substr(slash + 1);
}

} // namespace

oa::Status oa::Fbx::writeFbx(const oa::Path& inPath, const oa::UsdSkelClip& inClip) {
	if (!inClip.isValid()) {
		return oa::Status::invalidArgument("oa::Fbx::WriteFbx: invalid clip");
	}
	const oa::I32 n = inClip.jointCount();
	const oa::U32 frames = inClip.frameCount;
	const oa::F32 fps = inClip.fps;

	// parent index per joint, recovered from the slash-delimited paths.
	oa::Vector<oa::I32> parent; parent.resize(static_cast<oa::Usize>(n));
	oa::Vector<oa::String> leafNames; leafNames.resize(static_cast<oa::Usize>(n));
	for (oa::I32 j = 0; j < n; ++j) {
		const oa::String& path = inClip.jointPaths[static_cast<oa::Usize>(j)];
		leafNames[static_cast<oa::Usize>(j)] = leaf(path);
		const oa::Usize slash = lastSlash(path);
		if (slash == oa::String::Npos) {
			parent[static_cast<oa::Usize>(j)] = -1;
		} else {
			const oa::String parentPath = path.substr(0, slash);
			oa::I32 p = -1;
			for (oa::I32 k = 0; k < n; ++k) {
				if (inClip.jointPaths[static_cast<oa::Usize>(k)] == parentPath) { p = k; break; }
			}
			parent[static_cast<oa::Usize>(j)] = p;
		}
	}

	// FBX key times.
	oa::Vector<long long> times; times.resize(static_cast<oa::Usize>(frames));
	for (oa::U32 i = 0; i < frames; ++i) {
		times[i] = static_cast<long long>(std::llround(static_cast<double>(i) * kFbxTimePerSec / fps));
	}
	const long long stopTime = frames ? times[frames - 1] : 0;

	long long nextId = 1000000000LL;
	auto newId = [&]() { return ++nextId; };

	oa::Vector<long long> boneId; boneId.resize(static_cast<oa::Usize>(n));
	oa::Vector<long long> attrId; attrId.resize(static_cast<oa::Usize>(n));
	for (oa::I32 j = 0; j < n; ++j) { boneId[static_cast<oa::Usize>(j)] = newId(); }
	for (oa::I32 j = 0; j < n; ++j) { attrId[static_cast<oa::Usize>(j)] = newId(); }

	// Build animation nodes: T where a joint translates, R where it rotates.
	oa::Vector<AnimNode> anim;
	for (oa::I32 j = 0; j < n; ++j) {
		const oa::vlm::Vec3 t0 = inClip.translations[static_cast<oa::Usize>(j)];
		const oa::vlm::Quat q0 = inClip.rotations[static_cast<oa::Usize>(j)];
		bool hasT = false, hasR = false;
		for (oa::U32 f = 1; f < frames; ++f) {
			const oa::vlm::Vec3 t = inClip.translations[static_cast<oa::Usize>(f) * n + j];
			const oa::vlm::Quat q = inClip.rotations[static_cast<oa::Usize>(f) * n + j];
			if (t.x != t0.x || t.y != t0.y || t.z != t0.z) { hasT = true; }
			if (q.x != q0.x || q.y != q0.y || q.z != q0.z || q.w != q0.w) { hasR = true; }
		}
		for (int kind = 0; kind < 2; ++kind) {
			const bool isRot = (kind == 1);
			if ((isRot && !hasR) || (!isRot && !hasT)) { continue; }
			AnimNode node;
			node.joint = j;
			node.isRot = isRot;
			node.prop = isRot ? "Lcl rotation" : "Lcl Translation";
			node.kind = isRot ? "R" : "T";
			for (int c = 0; c < 3; ++c) { node.col[c].resize(static_cast<oa::Usize>(frames)); }
			for (oa::U32 f = 0; f < frames; ++f) {
				if (isRot) {
					// quat → euler (deg). QuaternionToEuler returns (yaw=Z, pitch=Y, roll=X);
					// FBX Lcl rotation channels are (X, Y, Z) = (roll, pitch, yaw).
					const oa::vlm::Vec3 e = oa::vlm::quaternionToEuler(
						inClip.rotations[static_cast<oa::Usize>(f) * n + j]);
					node.col[0][f] = e.z;
					node.col[1][f] = e.y;
					node.col[2][f] = e.x;
				} else {
					const oa::vlm::Vec3 t = inClip.translations[static_cast<oa::Usize>(f) * n + j];
					node.col[0][f] = t.x;
					node.col[1][f] = t.y;
					node.col[2][f] = t.z;
				}
			}
			anim.pushBack(std::move(node));
		}
	}
	for (AnimNode& a : anim) {
		a.nodeId = newId();
		for (int c = 0; c < 3; ++c) { a.curveId[c] = newId(); }
	}
	const long long stackId = newId();
	const long long layerId = newId();
	const long long docId   = newId();
	const oa::Usize nNodes  = anim.size();
	const oa::Usize nCurves = nNodes * 3;
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
	o << "\tSceneInfo: \"SceneInfo::GlobalInfo\", \"userData\" {\n";
	o << "\t\tType: \"userData\"\n\t\tVersion: 100\n\t\tMetaData:  {\n";
	o << "\t\t\tVersion: 100\n\t\t\tTitle: \"OA Gen3dAnim\"\n\t\t\tSubject: \"\"\n";
	o << "\t\t\tAuthor: \"\"\n\t\t\tKeywords: \"\"\n\t\t\tRevision: \"\"\n\t\t\tComment: \"\"\n\t\t}\n\t}\n}\n";

	// ── globalSettings (Z-up, cm, TimeMode 6) ──
	o << "GlobalSettings:  {\n\tVersion: 1000\n\tProperties70:  {\n";
	o << "\t\tP: \"upAxis\", \"int\", \"Integer\", \"\",2\n";
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
	o << "\t\tP: \"TimeSpanStart\", \"KTime\", \"time\", \"\",0\n";
	o << "\t\tP: \"TimeSpanStop\", \"KTime\", \"time\", \"\"," << stopTime << "\n";
	o << "\t}\n}\n";

	// ── Documents ──
	o << "Documents:  {\n\tCount: 1\n\tDocument: " << docId << ", \"\", \"Scene\" {\n";
	o << "\t\tProperties70:  {\n";
	o << "\t\t\tP: \"SourceObject\", \"object\", \"\", \"\"\n";
	o << "\t\t\tP: \"ActiveAnimStackName\", \"KString\", \"\", \"\", \"" << (hasAnim ? kTake : "") << "\"\n";
	o << "\t\t}\n\t\tRootNode: 0\n\t}\n}\n";
	o << "References:  {\n}\n";

	// ── Definitions ──
	const oa::Usize total = 1 + static_cast<oa::Usize>(n) * 2 + 2 + nNodes + nCurves;
	o << "Definitions:  {\n\tVersion: 100\n\tCount: " << total << "\n";
	o << "\tObjectType: \"GlobalSettings\" {\n\t\tCount: 1\n\t}\n";
	o << "\tObjectType: \"AnimationStack\" {\n\t\tCount: 1\n\t\tPropertyTemplate: \"FbxAnimStack\" {\n\t\t\tProperties70:  {\n";
	o << "\t\t\t\tP: \"description\", \"KString\", \"\", \"\", \"\"\n";
	o << "\t\t\t\tP: \"LocalStart\", \"KTime\", \"time\", \"\",0\n";
	o << "\t\t\t\tP: \"LocalStop\", \"KTime\", \"time\", \"\",0\n";
	o << "\t\t\t\tP: \"ReferenceStart\", \"KTime\", \"time\", \"\",0\n";
	o << "\t\t\t\tP: \"ReferenceStop\", \"KTime\", \"time\", \"\",0\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"AnimationLayer\" {\n\t\tCount: 1\n\t\tPropertyTemplate: \"FbxAnimLayer\" {\n\t\t\tProperties70:  {\n\t\t\t\tP: \"weight\", \"Number\", \"\", \"A\",100\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"NodeAttribute\" {\n\t\tCount: " << n << "\n\t\tPropertyTemplate: \"FbxSkeleton\" {\n\t\t\tProperties70:  {\n";
	o << "\t\t\t\tP: \"Color\", \"ColorRGB\", \"Color\", \"\",0.8,0.8,0.8\n";
	o << "\t\t\t\tP: \"size\", \"double\", \"Number\", \"\",100\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"Model\" {\n\t\tCount: " << n << "\n\t\tPropertyTemplate: \"FbxNode\" {\n\t\t\tProperties70:  {\n";
	o << "\t\t\t\tP: \"RotationActive\", \"bool\", \"\", \"\",0\n";
	o << "\t\t\t\tP: \"InheritType\", \"enum\", \"\", \"\",0\n";
	o << "\t\t\t\tP: \"ScalingMax\", \"Vector3D\", \"Vector\", \"\",0,0,0\n";
	o << "\t\t\t\tP: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",-1\n";
	o << "\t\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0\n";
	o << "\t\t\t\tP: \"Lcl rotation\", \"Lcl rotation\", \"\", \"A\",0,0,0\n";
	o << "\t\t\t\tP: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1\n";
	o << "\t\t\t\tP: \"Visibility\", \"Visibility\", \"\", \"A\",1\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"AnimationCurveNode\" {\n\t\tCount: " << nNodes << "\n\t\tPropertyTemplate: \"FbxAnimCurveNode\" {\n\t\t\tProperties70:  {\n\t\t\t\tP: \"d\", \"Compound\", \"\", \"\"\n\t\t\t}\n\t\t}\n\t}\n";
	o << "\tObjectType: \"AnimationCurve\" {\n\t\tCount: " << nCurves << "\n\t}\n";
	o << "}\n";

	// ── Objects ──
	o << "Objects:  {\n";
	for (oa::I32 j = 0; j < n; ++j) {
		const oa::String& name = leafNames[static_cast<oa::Usize>(j)];
		if (parent[static_cast<oa::Usize>(j)] < 0) {
			o << "\tNodeAttribute: " << attrId[static_cast<oa::Usize>(j)] << ", \"NodeAttribute::" << name << "\", \"root\" {\n";
			o << "\t\tTypeFlags: \"Null\", \"Skeleton\", \"root\"\n\t}\n";
		} else {
			o << "\tNodeAttribute: " << attrId[static_cast<oa::Usize>(j)] << ", \"NodeAttribute::" << name << "\", \"LimbNode\" {\n";
			o << "\t\tProperties70:  {\n\t\t\tP: \"size\", \"double\", \"Number\", \"\",6\n\t\t}\n";
			o << "\t\tTypeFlags: \"Skeleton\"\n\t}\n";
		}
	}
	for (oa::I32 j = 0; j < n; ++j) {
		const oa::String& name = leafNames[static_cast<oa::Usize>(j)];
		const oa::vlm::Vec3 rest = inClip.translations[static_cast<oa::Usize>(j)]; // frame 0 local
		o << "\tModel: " << boneId[static_cast<oa::Usize>(j)] << ", \"Model::" << name << "\", \"LimbNode\" {\n";
		o << "\t\tVersion: 232\n\t\tProperties70:  {\n";
		o << "\t\t\tP: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n";
		o << "\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A+\"," << fv(rest.x) << "," << fv(rest.y) << "," << fv(rest.z) << "\n";
		o << "\t\t\tP: \"Lcl rotation\", \"Lcl rotation\", \"\", \"A+\",0,0,0\n";
		o << "\t\t}\n\t\tShading: Y\n\t\tCulling: \"CullingOff\"\n\t}\n";
	}
	if (hasAnim) {
		o << "\tAnimationStack: " << stackId << ", \"AnimStack::" << kTake << "\", \"\" {\n";
		o << "\t\tProperties70:  {\n";
		o << "\t\t\tP: \"LocalStart\", \"KTime\", \"time\", \"\",0\n";
		o << "\t\t\tP: \"LocalStop\", \"KTime\", \"time\", \"\"," << stopTime << "\n";
		o << "\t\t\tP: \"ReferenceStart\", \"KTime\", \"time\", \"\",0\n";
		o << "\t\t\tP: \"ReferenceStop\", \"KTime\", \"time\", \"\"," << stopTime << "\n\t\t}\n\t}\n";
		o << "\tAnimationLayer: " << layerId << ", \"AnimLayer::base layer\", \"\" {\n\t}\n";
	}
	for (const AnimNode& a : anim) {
		o << "\tAnimationCurveNode: " << a.nodeId << ", \"AnimCurveNode::" << a.kind << "\", \"\" {\n";
		o << "\t\tProperties70:  {\n";
		const char* chans[3] = { "d|X", "d|Y", "d|Z" };
		for (int c = 0; c < 3; ++c) {
			o << "\t\t\tP: \"" << chans[c] << "\", \"Number\", \"\", \"A\"," << fv(a.col[c][0]) << "\n";
		}
		o << "\t\t}\n\t}\n";
		for (int c = 0; c < 3; ++c) {
			o << "\tAnimationCurve: " << a.curveId[c] << ", \"AnimCurve::\", \"\" {\n";
			o << "\t\tDefault: " << fv(a.col[c][0]) << "\n\t\tKeyVer: 4009\n";
			o << "\t\tKeyTime: *" << frames << " {\n\t\t\ta: ";
			for (oa::U32 f = 0; f < frames; ++f) { o << (f ? "," : "") << times[f]; }
			o << "\n\t\t}\n";
			o << "\t\tKeyValueFloat: *" << frames << " {\n\t\t\ta: ";
			for (oa::U32 f = 0; f < frames; ++f) { o << (f ? "," : "") << fv(a.col[c][f]); }
			o << "\n\t\t}\n";
			o << "\t\tKeyAttrFlags: *1 {\n\t\t\ta: 24840\n\t\t}\n";
			o << "\t\tKeyAttrDataFloat: *4 {\n\t\t\ta: 0,0,0,0\n\t\t}\n";
			o << "\t\tKeyAttrRefCount: *1 {\n\t\t\ta: " << frames << "\n\t\t}\n\t}\n";
		}
	}
	o << "}\n";

	// ── Connections ──
	o << "Connections:  {\n";
	for (oa::I32 j = 0; j < n; ++j) {
		const oa::I32 p = parent[static_cast<oa::Usize>(j)];
		const long long dst = (p < 0) ? 0 : boneId[static_cast<oa::Usize>(p)];
		o << "\t;Model::" << leafNames[static_cast<oa::Usize>(j)] << ", Model::" << (p < 0 ? oa::String("RootNode") : leafNames[static_cast<oa::Usize>(p)]) << "\n";
		o << "\tC: \"OO\"," << boneId[static_cast<oa::Usize>(j)] << "," << dst << "\n";
		o << "\t;NodeAttribute::" << leafNames[static_cast<oa::Usize>(j)] << ", Model::" << leafNames[static_cast<oa::Usize>(j)] << "\n";
		o << "\tC: \"OO\"," << attrId[static_cast<oa::Usize>(j)] << "," << boneId[static_cast<oa::Usize>(j)] << "\n";
	}
	if (hasAnim) {
		o << "\t;AnimLayer::base layer, AnimStack::" << kTake << "\n";
		o << "\tC: \"OO\"," << layerId << "," << stackId << "\n";
	}
	for (const AnimNode& a : anim) {
		o << "\tC: \"OO\"," << a.nodeId << "," << layerId << "\n";
		o << "\tC: \"OP\"," << a.nodeId << "," << boneId[static_cast<oa::Usize>(a.joint)] << ", \"" << a.prop << "\"\n";
		const char* chans[3] = { "d|X", "d|Y", "d|Z" };
		for (int c = 0; c < 3; ++c) {
			o << "\tC: \"OP\"," << a.curveId[c] << "," << a.nodeId << ", \"" << chans[c] << "\"\n";
		}
	}
	o << "}\n";

	// ── takes (legacy; Maya importer still reads it) ──
	if (hasAnim) {
		o << "Takes:  {\n\tCurrent: \"" << kTake << "\"\n\tTake: \"" << kTake << "\" {\n";
		o << "\t\tFileName: \"Take_001.tak\"\n";
		o << "\t\tLocalTime: 0," << stopTime << "\n";
		o << "\t\tReferenceTime: 0," << stopTime << "\n\t}\n}\n";
	}

	return oa::Filesystem::writeText(inPath, oa::sdk::fromStdString(o.str()));
}
